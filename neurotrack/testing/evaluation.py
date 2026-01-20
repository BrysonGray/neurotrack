"""
Evaluation functions for comparing reconstructed neurons to ground truth.

This module provides tools for:
- Computing distance metrics between reconstructions and ground truth
- Aggregating and saving evaluation results
- Generating summary statistics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
from scipy.spatial import KDTree

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_prep import tree


def evaluate_reconstruction(pred_swc: list, 
                           gt_swc: list, 
                           threshold: float = 0.0) -> Dict[str, Any]:
    """
    Compare predicted reconstruction to ground truth using distance metrics.
    
    Uses existing functions from data_prep.tree:
    - directed_divergence(): Average distance from points in tree A to nearest in tree B
    - spatial_distance(): Bidirectional average distance
    
    Parameters
    ----------
    pred_swc : list
        Predicted neuron in SWC format (Nx7 list)
    gt_swc : list
        Ground truth neuron in SWC format (Nx7 list)
    threshold : float, optional
        Distance threshold for computing substantial divergence.
        Default is 0.0 (all points considered).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'directed_div_pred_to_gt': Average distance from prediction to ground truth
        - 'n_substantial_pred_to_gt': Number of points beyond threshold
        - 'directed_div_gt_to_pred': Average distance from ground truth to prediction
        - 'n_substantial_gt_to_pred': Number of points beyond threshold
        - 'bidirectional_distance': Mean of both directed divergences
        - 'proportion_within_threshold': Proportion of points within threshold
        - 'n_points_pred': Number of points in prediction
        - 'n_points_gt': Number of points in ground truth
        
    Examples
    --------
    >>> pred = [[1, 0, 0, 0, 0, 1, -1], [2, 0, 1, 0, 0, 1, 1]]
    >>> gt = [[1, 0, 0, 0, 0, 1, -1], [2, 0, 1.1, 0, 0, 1, 1]]
    >>> metrics = evaluate_reconstruction(pred, gt, threshold=0.5)
    >>> metrics['bidirectional_distance']
    0.05
    """
    # Compute directed divergences using existing functions
    div_pred_to_gt, n_substantial_pred = tree.directed_divergence(
        pred_swc, gt_swc, threshold=threshold
    )
    
    div_gt_to_pred, n_substantial_gt = tree.directed_divergence(
        gt_swc, pred_swc, threshold=threshold
    )
    
    # Compute bidirectional distance using existing function
    bidirectional_dist, proportion_within = tree.spatial_distance(
        pred_swc, gt_swc, threshold=threshold
    )
    
    return {
        'directed_div_pred_to_gt': float(div_pred_to_gt),
        'n_substantial_pred_to_gt': int(n_substantial_pred),
        'directed_div_gt_to_pred': float(div_gt_to_pred),
        'n_substantial_gt_to_pred': int(n_substantial_gt),
        'bidirectional_distance': float(bidirectional_dist),
        'proportion_within_threshold': float(proportion_within),
        'n_points_pred': len(pred_swc),
        'n_points_gt': len(gt_swc)
    }


def save_evaluation_results(results: List[Dict[str, Any]], 
                           output_path: str,
                           summary_path: Optional[str] = None):
    """
    Save evaluation metrics to CSV file(s).
    
    Parameters
    ----------
    results : list
        List of dictionaries, one per neuron, containing evaluation metrics
        and metadata (neuron_name, etc.)
    output_path : str
        Path to save detailed per-neuron results CSV
    summary_path : str, optional
        Path to save summary statistics JSON. If None, saves to same
        directory as output_path with '_summary.json' suffix.
        
    Notes
    -----
    The detailed CSV includes all metrics for each neuron.
    The summary JSON contains aggregate statistics (mean, std, median, etc.).
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    df.to_csv(output_path, index=False)
    print(f"  Saved detailed metrics to: {output_path}")
    
    # Compute and save summary statistics
    if summary_path is None:
        summary_path = output_path.replace('.csv', '_summary.json')
    
    # Select numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = {}
    for col in numeric_cols:
        summary[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'median': float(df[col].median()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    
    summary['n_neurons'] = len(df)
    
    # Save summary as JSON
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved summary statistics to: {summary_path}")


def compute_coverage(pred_swc: list, 
                    gt_swc: list, 
                    threshold: float = 2.0) -> float:
    """
    Compute what fraction of the ground truth is covered by the prediction.
    
    Coverage is defined as the proportion of ground truth points that have
    at least one predicted point within the threshold distance.
    
    Parameters
    ----------
    pred_swc : list
        Predicted neuron in SWC format
    gt_swc : list
        Ground truth neuron in SWC format
    threshold : float, optional
        Distance threshold for considering a point "covered". Default is 2.0.
        
    Returns
    -------
    float
        Coverage fraction in range [0, 1]
    """
    # Extract coordinates
    pred_coords = np.array([row[2:5] for row in pred_swc])
    gt_coords = np.array([row[2:5] for row in gt_swc])
    
    # Build KDTree for predicted points
    tree = KDTree(pred_coords)
    
    # Query nearest distances from ground truth to prediction
    distances, _ = tree.query(gt_coords)
    
    # Compute coverage
    coverage = np.mean(distances <= threshold)
    
    return float(coverage)


def compute_precision(pred_swc: list, 
                     gt_swc: list, 
                     threshold: float = 2.0) -> float:
    """
    Compute what fraction of the prediction is close to ground truth.
    
    Precision is defined as the proportion of predicted points that have
    at least one ground truth point within the threshold distance.
    
    Parameters
    ----------
    pred_swc : list
        Predicted neuron in SWC format
    gt_swc : list
        Ground truth neuron in SWC format
    threshold : float, optional
        Distance threshold for considering a point "correct". Default is 2.0.
        
    Returns
    -------
    float
        Precision fraction in range [0, 1]
    """
    # Extract coordinates
    pred_coords = np.array([row[2:5] for row in pred_swc])
    gt_coords = np.array([row[2:5] for row in gt_swc])
    
    # Build KDTree for ground truth points
    tree = KDTree(gt_coords)
    
    # Query nearest distances from prediction to ground truth
    distances, _ = tree.query(pred_coords)
    
    # Compute precision
    precision = np.mean(distances <= threshold)
    
    return float(precision)
