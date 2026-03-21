"""Metric primitives for reconstruction evaluation."""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial import KDTree

from neurotrack.data import loading as load
from neurotrack.data import tree


def _as_swc_array(swc_list: list) -> np.ndarray:
    """Convert SWC rows to a normalized ``(N, >=7)`` float array."""
    swc_array = np.asarray(swc_list, dtype=np.float32)
    if swc_array.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if swc_array.ndim == 1:
        if swc_array.shape[0] < 7:
            raise ValueError(f"SWC row must have at least 7 columns, got shape {swc_array.shape}")
        swc_array = swc_array.reshape(1, -1)
    elif swc_array.ndim != 2:
        raise ValueError(f"SWC array must be 2D, got shape {swc_array.shape}")
    if swc_array.shape[1] < 7:
        raise ValueError(f"SWC rows must have at least 7 columns, got shape {swc_array.shape}")
    return swc_array


def _get_special_node_coords(swc_list: list, node_type: str) -> np.ndarray:
    """Return endpoint or branchpoint coordinates from an SWC tree."""
    swc_array = _as_swc_array(swc_list)
    if swc_array.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    adjacency = load.adjacency_dict(swc_array)
    coords_by_id = {int(row[0]): row[2:5].astype(np.float32, copy=False) for row in swc_array}
    selected_coords = []

    for node_id in swc_array[:, 0].astype(int).tolist():
        degree = len(adjacency.get(int(node_id), []))
        if node_type == "endpoint" and degree <= 1:
            selected_coords.append(coords_by_id[int(node_id)])
        elif node_type == "branchpoint" and degree > 2:
            selected_coords.append(coords_by_id[int(node_id)])

    if not selected_coords:
        return np.empty((0, 3), dtype=np.float32)

    return np.stack(selected_coords, axis=0)


def _mean_nearest_neighbor_distance(source_coords: np.ndarray, target_coords: np.ndarray) -> float:
    """Return mean nearest-neighbor distance from ``source_coords`` to ``target_coords``."""
    if source_coords.shape[0] == 0 and target_coords.shape[0] == 0:
        return 0.0
    if source_coords.shape[0] == 0 or target_coords.shape[0] == 0:
        return float("nan")

    target_tree = KDTree(target_coords)
    distances, _ = target_tree.query(source_coords)
    return float(np.mean(distances))


def _symmetric_localization_error(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Return a symmetric nearest-neighbor localization error between point sets."""
    if coords_a.shape[0] == 0 and coords_b.shape[0] == 0:
        return 0.0
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return float("nan")

    forward = _mean_nearest_neighbor_distance(coords_a, coords_b)
    backward = _mean_nearest_neighbor_distance(coords_b, coords_a)
    return float((forward + backward) / 2.0)


def directed_divergence(tree_a, tree_b, threshold=4.0):
    """
    Calculate the directed divergence from tree_a to tree_b.
    For each point in tree_a, find the nearest point in tree_b and compute the average distance.

    Parameters
    ----------
    tree_a : list or array_like
        Nx7 SWC formatted list or array of points representing the first tree.
    tree_b : list or array_like
        Mx7 SWC formatted list or array of points representing the second tree.
    threshold : float
        Distance threshold to consider for divergence calculation.

    Returns
    -------
    avg_distance : float
        The average distance from each point in tree_a to the nearest point in tree_b.
    n_close : float
        The number of points in tree_a that are within the threshold distance to tree_b.
    n_far : float
        The number of points in tree_a that are farther than the threshold distance to tree_b.
    """
    tree_a = np.asarray(tree_a)
    tree_b = np.asarray(tree_b)
    kdtree_b = KDTree(tree_b[:, 2:5])  # Use only x, y, z coordinates
    distances, _ = kdtree_b.query(tree_a[:, 2:5]) # distances from points in tree_a to nearest points in tree_b
    avg_distance = np.mean(distances)
    n_close = np.sum(distances <= threshold)
    n_far = len(tree_a) - n_close
    return avg_distance, n_close, n_far


def spatial_distance(tree_a, tree_b, threshold=4.0):
    """
    Calculate the average of the directed divergence from A to B and from B to A. 

    Parameters
    ----------
    tree_a : list or array_like
        Nx7 SWC formatted list or array of points representing the first tree.
    tree_b : list or array_like
        Mx7 SWC formatted list or array of points representing the second tree.

    Returns
    -------
    float
        The average bi-directional distance.
    float
        The proportion of points within the threshold distance in both directions.
    """

    divergence_a_to_b, _ = directed_divergence(tree_a, tree_b, threshold)
    divergence_b_to_a, _ = directed_divergence(tree_b, tree_a, threshold)
    avg_divergence = (divergence_a_to_b + divergence_b_to_a) / 2

    return avg_divergence


def evaluate_reconstruction(pred_swc: list, gt_swc: list, threshold: float = 4.0) -> Dict[str, Any]:
    div_pred_to_gt, n_substantial_pred = directed_divergence(
        pred_swc,
        gt_swc,
        threshold=threshold,
    )

    div_gt_to_pred, n_substantial_gt = directed_divergence(
        gt_swc,
        pred_swc,
        threshold=threshold,
    )

    bidirectional_dist = (div_pred_to_gt + div_gt_to_pred) / 2
    precision = n_substantial_pred / len(pred_swc) if len(pred_swc) > 0 else 0
    coverage = n_substantial_gt / len(gt_swc) if len(gt_swc) > 0 else 0
    pred_endpoints = _get_special_node_coords(pred_swc, "endpoint")
    gt_endpoints = _get_special_node_coords(gt_swc, "endpoint")
    pred_branchpoints = _get_special_node_coords(pred_swc, "branchpoint")
    gt_branchpoints = _get_special_node_coords(gt_swc, "branchpoint")

    return {
        "directed_div_pred_to_gt": float(div_pred_to_gt),
        "n_substantial_pred_to_gt": int(n_substantial_pred),
        "directed_div_gt_to_pred": float(div_gt_to_pred),
        "n_substantial_gt_to_pred": int(n_substantial_gt),
        "bidirectional_distance": float(bidirectional_dist),
        "precision": float(precision),
        "coverage": float(coverage),
        "n_points_pred": len(pred_swc),
        "n_points_gt": len(gt_swc),
        "endpoint_localization_error": _symmetric_localization_error(pred_endpoints, gt_endpoints),
        "branchpoint_localization_error": _symmetric_localization_error(pred_branchpoints, gt_branchpoints),
        "endpoint_count_error": int(abs(pred_endpoints.shape[0] - gt_endpoints.shape[0])),
        "branchpoint_count_error": int(abs(pred_branchpoints.shape[0] - gt_branchpoints.shape[0])),
    }


def compute_coverage(pred_swc: list, gt_swc: list, threshold: float = 4.0) -> float:
    pred_coords = np.array([row[2:5] for row in pred_swc])
    gt_coords = np.array([row[2:5] for row in gt_swc])

    pred_tree = KDTree(pred_coords)
    distances, _ = pred_tree.query(gt_coords)
    coverage = np.mean(distances <= threshold)
    return float(coverage)


def compute_precision(pred_swc: list, gt_swc: list, threshold: float = 4.0) -> float:
    pred_coords = np.array([row[2:5] for row in pred_swc])
    gt_coords = np.array([row[2:5] for row in gt_swc])

    gt_tree = KDTree(gt_coords)
    distances, _ = gt_tree.query(pred_coords)
    precision = np.mean(distances <= threshold)
    return float(precision)

# L-Measures

def num_bifurcations() -> int:
    """Number of bifurcations in the tree"""
    pass

def num_branches() -> int:
    """Number of segments in the tree between the root, branch points or termination points"""
    pass

def num_tips() -> int:
    """Number of terminal points in the tree"""
    pass

def span() -> Tuple[float, float, float]:
    """The spatial extent of the tree in x, y, z dimensions"""
    pass

def total_length() -> float:
    """The total length of all segments in the tree"""
    pass

def max_euclidean_distance() -> float:
    """Maximum euclidean distance between any point of the reconstruction and the root"""
    pass

def max_path_distance() -> float:
    """Maximum path distance along the tree nodes between any node of the reconstruction and the root"""
    pass

def max_branch_order() -> int:
    """Maximum branch order, defined as 1 in the first branch stemming from the root and increasing by 1 every time there is a new bifurcation point."""
    pass

def average_contraction() -> float:
    """Contraction is defined as the ratio between euclidean distance and path length to the root at any node of the reconstruction. Average contraction is obtained by averaging over all the nodes of the tree"""
    pass

def average_fragmentation() -> float:
    """Average number of segments in each branch"""
    pass

def bifurcation_angle_local() -> float:
    """Angle between downstream nodes closest to a bifurcation"""
    pass

def bifurcation_angle_remote() -> float:
    """Angle between downstream branch or termination points closest to a bifurcation"""
    pass

def different_structure_average(distance_threshold: float = 2.0) -> float:
    """Average distance from neuron 1 to 2 and from neuron 2 to 1 for points that have a distance >= distance_threshold pixels"""
    pass

def percentage_different_structure(tree1, tree2, distance_threshold: float = 2.0) -> float:
    """Percentage of points in neuron 1 that are >= distance_threshold pixels from any point in neuron 2 and vice versa"""
    pass

def percent_different_structure_average(tree1, tree2, distance_threshold: float = 2.0) -> float:
    """Average of percent of different structure from neuron 1 to 2 and from neuron 2 to 1"""
    pass