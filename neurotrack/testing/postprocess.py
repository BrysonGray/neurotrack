"""
Post-processing functions for reconstructed neuron paths.

This module provides functions to clean and refine raw inference outputs:
- Removing short branches
- Smoothing jagged paths
- Merging redundant/overlapping paths
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.spatial import KDTree
from scipy.ndimage import uniform_filter1d


def remove_short_paths(paths: List[np.ndarray], min_length: float) -> List[np.ndarray]:
    """
    Remove paths shorter than the specified threshold.
    
    This function filters out paths that don't meet the minimum length
    requirement, which are often spurious detections or reconstruction artifacts.
    
    Parameters
    ----------
    paths : List[np.ndarray]
        List of path arrays, each of shape (N, 3) where N is the number of points
    min_length : float
        Minimum path length in voxels/units. Paths shorter than this
        will be removed.
        
    Returns
    -------
    List[np.ndarray]
        Filtered list of paths meeting the length requirement
        
    Examples
    --------
    >>> paths = [np.array([[0,0,0], [1,1,1], [2,2,2]]),  # length ~3.46
    ...          np.array([[0,0,0], [0,0,1]])]          # length = 1
    >>> filtered = remove_short_paths(paths, min_length=2.0)
    >>> len(filtered)  # Only path 1 remains
    1
    """
    filtered_paths = []
    
    for path in paths:
        # Convert to numpy if tensor
        if isinstance(path, torch.Tensor):
            path = path.cpu().numpy()
        
        # Calculate path length
        if len(path) < 2:
            continue  # Skip isolated points
        
        # Compute cumulative Euclidean distance
        deltas = np.diff(path, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        total_length = np.sum(distances)
        
        # Keep only if longer than threshold
        if total_length >= min_length:
            filtered_paths.append(path)
    
    n_removed = len(paths) - len(filtered_paths)
    print(f"    Removed {n_removed} paths shorter than {min_length:.1f} units")
    
    return filtered_paths


def smooth_paths(paths: List[np.ndarray], window_size: int = 5, method: str = 'uniform') -> List[np.ndarray]:
    """
    Smooth jagged paths using moving average filtering.
    
    This reduces noise and jaggedness in the reconstructed paths while
    preserving overall shape. Connection points between paths are automatically
    detected and preserved to maintain connectivity.
    
    Parameters
    ----------
    paths : List[np.ndarray]
        List of numpy arrays, each of shape (N, 3) representing a path of 3D points
    window_size : int, optional
        Size of the smoothing window. Larger values produce smoother paths.
        Must be odd. Default is 5.
    method : str, optional
        Smoothing method. Options:
        - 'uniform': Simple moving average (default)
        - 'gaussian': Gaussian-weighted average (future)
        
    Returns
    -------
    List[np.ndarray]
        List of smoothed paths as numpy arrays
        
    Notes
    -----
    Very short segments (length < window_size) are returned unchanged.
    Connection points (shared coordinates between paths) are preserved to maintain
    connectivity between paths.
    """
    # Convert all paths to numpy arrays
    paths_np = []
    for path in paths:
        if isinstance(path, torch.Tensor):
            paths_np.append(path.cpu().numpy())
        else:
            paths_np.append(np.array(path))
    
    # Build a set of all connection points (points that appear in multiple paths)
    # We'll preserve these during smoothing
    point_counts = {}
    path_point_indices = []  # Track which indices in each path correspond to connection points
    
    for path_idx, path in enumerate(paths_np):
        indices_to_preserve = []
        for point_idx, point in enumerate(path):
            point_tuple = tuple(point[:3])  # Use only x, y, z for matching
            if point_tuple not in point_counts:
                point_counts[point_tuple] = []
            point_counts[point_tuple].append((path_idx, point_idx))
            indices_to_preserve.append(point_idx)
        path_point_indices.append(indices_to_preserve)
    
    # Identify connection points (appear in more than one path OR at path boundaries)
    connection_points = set()
    for point_tuple, occurrences in point_counts.items():
        # Preserve if:
        # 1. Point appears in multiple paths (connection point)
        # 2. Point is at the start or end of any path (boundary point)
        if len(occurrences) > 1:
            connection_points.add(point_tuple)
        else:
            # Check if it's a boundary point (first or last in its path)
            path_idx, point_idx = occurrences[0]
            if point_idx == 0 or point_idx == len(paths_np[path_idx]) - 1:
                connection_points.add(point_tuple)
    
    # Now smooth each path, preserving connection points
    smoothed_paths = []
    for path_idx, path in enumerate(paths_np):
        # Skip very short segments
        if len(path) < window_size:
            smoothed_paths.append(path)
            continue
        
        # Store original points that need to be preserved
        preserved_points = {}
        for point_idx, point in enumerate(path):
            point_tuple = tuple(point[:3])
            if point_tuple in connection_points:
                preserved_points[point_idx] = point.copy()
        
        # Apply smoothing to each coordinate dimension
        smoothed = np.copy(path)
        for dim in range(3):  # x, y, z
            smoothed[:, dim] = uniform_filter1d(
                path[:, dim], 
                size=window_size, 
                mode='nearest'
            )
        
        # Restore connection points to maintain connectivity
        for point_idx, original_point in preserved_points.items():
            smoothed[point_idx] = original_point
        
        smoothed_paths.append(smoothed)
    
    print(f"    Smoothed {len(smoothed_paths)} paths with window size {window_size}")
    print(f"    Preserved {len(connection_points)} connection points")
    
    return smoothed_paths


def merge_redundant_paths(paths: List[np.ndarray], 
                          overlap_threshold: float = 0.8,
                          distance_threshold: float = 2.0) -> List[np.ndarray]:
    """
    Join paths that significantly overlap with each other.
    
    Redundant paths can occur when the agent traces the same structure
    multiple times or when branches are very close together. This function
    identifies and merges such paths based on spatial overlap.
    
    Parameters
    ----------
    paths : List[np.ndarray]
        List of numpy arrays, each of shape (N, 3) representing a path of 3D points
    overlap_threshold : float, optional
        Fraction of points that must be within distance_threshold to
        consider paths redundant. Range [0, 1]. Default is 0.8.
    distance_threshold : float, optional
        Maximum distance in voxels/units for points to be considered
        overlapping. Default is 2.0.
        
    Returns
    -------
    List[np.ndarray]
        List of merged paths with redundant paths removed
        
    Algorithm
    ---------
    1. For each pair of paths, compute fraction of points within
       distance_threshold using KDTree queries
    2. If overlap fraction > overlap_threshold, mark as redundant
    3. Keep the longer path, discard the shorter one
    4. Repeat until no more merges possible
    
    """
    # Convert to list and ensure all are numpy arrays
    merged_paths = []
    for path in paths:
        if isinstance(path, torch.Tensor):
            merged_paths.append(path.cpu().numpy())
        else:
            merged_paths.append(path)
    
    # Iteratively find and merge redundant paths
    n_merged = 0
    changed = True
    
    while changed:
        changed = False
        i = 0
        
        while i < len(merged_paths):
            path_a = merged_paths[i]
            j = i + 1
            
            while j < len(merged_paths):
                path_b = merged_paths[j]
                
                # Build KDTree for path_b
                tree_b = KDTree(path_b)
                
                # Query distances from path_a to path_b
                distances_a, _ = tree_b.query(path_a)
                overlap_fraction_a = np.mean(distances_a <= distance_threshold)
                
                # Check if overlap is significant
                if overlap_fraction_a >= overlap_threshold:
                    # Keep the longer path, remove the shorter
                    len_a = len(path_a)
                    len_b = len(path_b)
                    
                    if len_a >= len_b:
                        # Remove path_b
                        merged_paths.pop(j)
                    else:
                        # Remove path_a
                        merged_paths.pop(i)
                        j = len(merged_paths)  # Exit inner loop, path_a is gone
                    
                    n_merged += 1
                    changed = True
                else:
                    j += 1
            
            i += 1
    
    print(f"    Merged {n_merged} redundant paths (threshold={overlap_threshold:.2f})")
    
    return merged_paths


def calculate_path_length(segment: np.ndarray) -> float:
    """
    Calculate the total length of a path segment.
    
    Parameters
    ----------
    segment : np.ndarray
        Array of shape (N, 3) representing 3D coordinates
        
    Returns
    -------
    float
        Total path length
    """
    if len(segment) < 2:
        return 0.0
    
    deltas = np.diff(segment, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return np.sum(distances)
