"""Metric primitives for reconstruction evaluation."""

from typing import Any, Dict

import numpy as np
from scipy.spatial import KDTree

from neurotrack.data import tree


def evaluate_reconstruction(pred_swc: list, gt_swc: list, threshold: float = 4.0) -> Dict[str, Any]:
    div_pred_to_gt, n_substantial_pred = tree.directed_divergence(
        pred_swc,
        gt_swc,
        threshold=threshold,
    )

    div_gt_to_pred, n_substantial_gt = tree.directed_divergence(
        gt_swc,
        pred_swc,
        threshold=threshold,
    )

    bidirectional_dist = (div_pred_to_gt + div_gt_to_pred) / 2
    precision = n_substantial_pred / len(pred_swc) if len(pred_swc) > 0 else 0
    coverage = n_substantial_gt / len(gt_swc) if len(gt_swc) > 0 else 0

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
