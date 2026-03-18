"""Metric primitives for reconstruction evaluation."""

from typing import Any, Dict

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
