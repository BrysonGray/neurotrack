"""Metric primitives for reconstruction evaluation."""

from itertools import combinations
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from neurotrack.data import loading as load


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


def _swc_topology_maps(swc_list: list) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, int], Dict[int, list], list]:
    """Build common topology maps for SWC node-wise metrics."""
    swc_array = _as_swc_array(swc_list)
    node_ids = swc_array[:, 0].astype(int).tolist()
    node_set = set(node_ids)
    rows_by_id = {int(row[0]): row for row in swc_array}
    parent_by_id: Dict[int, int] = {}
    children_by_id: Dict[int, list] = {node_id: [] for node_id in node_ids}

    for row in swc_array:
        node_id = int(row[0])
        parent_id = int(row[6])
        parent_by_id[node_id] = parent_id
        if parent_id in node_set:
            children_by_id[parent_id].append(node_id)

    roots = [node_id for node_id in node_ids if parent_by_id.get(node_id, -1) not in node_set]
    if not roots and node_ids:
        roots = [int(min(node_ids))]

    return swc_array, rows_by_id, parent_by_id, children_by_id, roots


def _path_length_and_component_root(
    rows_by_id: Dict[int, np.ndarray],
    children_by_id: Dict[int, list],
    roots: list,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Compute root-referenced geodesic lengths for each connected component."""
    path_length_by_id: Dict[int, float] = {}
    component_root_by_id: Dict[int, int] = {}
    visited = set()

    for root_id in roots:
        stack = [(int(root_id), 0.0)]
        while stack:
            node_id, path_length = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            path_length_by_id[node_id] = float(path_length)
            component_root_by_id[node_id] = int(root_id)

            node_coord = rows_by_id[node_id][2:5]
            for child_id in children_by_id.get(node_id, []):
                child_coord = rows_by_id[child_id][2:5]
                edge_len = float(np.linalg.norm(child_coord - node_coord))
                stack.append((int(child_id), path_length + edge_len))

    # Handle malformed graphs (e.g. cycles/disconnected from inferred roots) defensively.
    for node_id in rows_by_id.keys():
        if node_id not in visited:
            path_length_by_id[node_id] = 0.0
            component_root_by_id[node_id] = int(node_id)

    return path_length_by_id, component_root_by_id


def _directed_divergence_stats(tree_a, tree_b, threshold: float = 4.0) -> Dict[str, Any]:
    """Compute directed nearest-neighbor stats used by multiple metrics."""
    swc_a = _as_swc_array(tree_a)
    swc_b = _as_swc_array(tree_b)

    if swc_a.shape[0] == 0 and swc_b.shape[0] == 0:
        return {
            "avg_distance": 0.0,
            "n_close": 0,
            "n_far": 0,
            "different_structure_average": 0.0,
            "percentage_different_structure": 0.0,
            "distances": np.empty((0,), dtype=np.float32),
        }

    if swc_a.shape[0] == 0:
        return {
            "avg_distance": float("nan"),
            "n_close": 0,
            "n_far": 0,
            "different_structure_average": float("nan"),
            "percentage_different_structure": float("nan"),
            "distances": np.empty((0,), dtype=np.float32),
        }

    if swc_b.shape[0] == 0:
        return {
            "avg_distance": float("nan"),
            "n_close": 0,
            "n_far": int(swc_a.shape[0]),
            "different_structure_average": float("nan"),
            "percentage_different_structure": 1.0,
            "distances": np.full((swc_a.shape[0],), np.nan, dtype=np.float32),
        }

    kdtree_b = KDTree(swc_b[:, 2:5])
    distances, _ = kdtree_b.query(swc_a[:, 2:5])
    distances = np.asarray(distances, dtype=np.float32)

    close_mask = distances <= float(threshold)
    far_mask = distances >= float(threshold)
    n_close = int(np.sum(close_mask))
    n_far = int(swc_a.shape[0] - n_close)
    far_distances = distances[far_mask]

    return {
        "avg_distance": float(np.mean(distances)),
        "n_close": n_close,
        "n_far": n_far,
        "different_structure_average": float(np.mean(far_distances)) if far_distances.size > 0 else 0.0,
        "percentage_different_structure": float(np.mean(far_mask)),
        "distances": distances,
    }


def directed_divergence(tree_a, tree_b, threshold=4.0, return_details: bool = False):
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
    details : dict, optional
        Returned only when ``return_details=True``. Includes
        ``n_far``, ``different_structure_average``,
        and ``percentage_different_structure``.
    """
    stats = _directed_divergence_stats(tree_a, tree_b, threshold=float(threshold))
    if return_details:
        return float(stats["avg_distance"]), int(stats["n_close"]), stats
    return float(stats["avg_distance"]), int(stats["n_close"])


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

def num_bifurcations(swc_list: list) -> int:
    """Number of bifurcations in the tree"""
    _, _, _, children_by_id, _ = _swc_topology_maps(swc_list)
    return int(sum(len(children) > 1 for children in children_by_id.values()))


def num_branches(swc_list: list) -> int:
    """Number of segments in the tree between the root, branch points or termination points"""
    _, _, _, children_by_id, roots = _swc_topology_maps(swc_list)
    bifurcations = {node_id for node_id, children in children_by_id.items() if len(children) > 1}
    branch_starts = set(roots).union(bifurcations)
    return int(sum(len(children_by_id.get(node_id, [])) for node_id in branch_starts))


def num_tips(swc_list: list) -> int:
    """Number of terminal points in the tree"""
    _, _, _, children_by_id, _ = _swc_topology_maps(swc_list)
    return int(sum(len(children) == 0 for children in children_by_id.values()))


def span(swc_list: list) -> Tuple[float, float, float]:
    """The spatial extent of the tree in x, y, z dimensions"""
    swc_array = _as_swc_array(swc_list)
    if swc_array.shape[0] == 0:
        return 0.0, 0.0, 0.0
    mins = np.min(swc_array[:, 2:5], axis=0)
    maxs = np.max(swc_array[:, 2:5], axis=0)
    extent = maxs - mins
    return float(extent[0]), float(extent[1]), float(extent[2])


def total_length(swc_list: list) -> float:
    """The total length of all segments in the tree"""
    swc_array, rows_by_id, parent_by_id, _, _ = _swc_topology_maps(swc_list)
    if swc_array.shape[0] == 0:
        return 0.0

    total = 0.0
    node_set = set(rows_by_id.keys())
    for node_id, parent_id in parent_by_id.items():
        if parent_id in node_set:
            total += float(np.linalg.norm(rows_by_id[node_id][2:5] - rows_by_id[parent_id][2:5]))
    return float(total)


def max_euclidean_distance(swc_list: list, root_id: Optional[int] = None) -> float:
    """Maximum euclidean distance between any point of the reconstruction and the root"""
    swc_array, rows_by_id, _, children_by_id, roots = _swc_topology_maps(swc_list)
    if swc_array.shape[0] == 0:
        return 0.0

    if root_id is not None and int(root_id) in rows_by_id:
        roots = [int(root_id)]

    _, component_root_by_id = _path_length_and_component_root(rows_by_id, children_by_id, roots)
    max_dist = 0.0
    for node_id, row in rows_by_id.items():
        root_row = rows_by_id[component_root_by_id[node_id]]
        max_dist = max(max_dist, float(np.linalg.norm(row[2:5] - root_row[2:5])))
    return float(max_dist)


def max_path_distance(swc_list: list, root_id: Optional[int] = None) -> float:
    """Maximum path distance along the tree nodes between any node of the reconstruction and the root"""
    swc_array, rows_by_id, _, children_by_id, roots = _swc_topology_maps(swc_list)
    if swc_array.shape[0] == 0:
        return 0.0

    if root_id is not None and int(root_id) in rows_by_id:
        roots = [int(root_id)]

    path_length_by_id, _ = _path_length_and_component_root(rows_by_id, children_by_id, roots)
    return float(max(path_length_by_id.values(), default=0.0))


def max_branch_order(swc_list: list, root_id: Optional[int] = None) -> int:
    """Maximum branch order, defined as 1 in the first branch stemming from the root and increasing by 1 every time there is a new bifurcation point."""
    swc_array, rows_by_id, _, children_by_id, roots = _swc_topology_maps(swc_list)
    if swc_array.shape[0] == 0:
        return 0

    if root_id is not None and int(root_id) in rows_by_id:
        roots = [int(root_id)]

    max_order = 0
    visited = set()
    for root in roots:
        stack = [(int(root), 0)]
        while stack:
            node_id, branch_order = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            max_order = max(max_order, int(branch_order))

            is_bifurcation = len(children_by_id.get(node_id, [])) > 1
            next_order = branch_order + 1 if is_bifurcation else branch_order
            for child_id in children_by_id.get(node_id, []):
                stack.append((int(child_id), int(next_order)))

    return int(max_order)


def average_contraction(swc_list: list, root_id: Optional[int] = None) -> float:
    """Contraction is defined as the ratio between euclidean distance and path length to the root at any node of the reconstruction. Average contraction is obtained by averaging over all the nodes of the tree"""
    swc_array, rows_by_id, _, children_by_id, roots = _swc_topology_maps(swc_list)
    if swc_array.shape[0] <= 1:
        return 0.0

    if root_id is not None and int(root_id) in rows_by_id:
        roots = [int(root_id)]

    path_length_by_id, component_root_by_id = _path_length_and_component_root(rows_by_id, children_by_id, roots)
    contractions = []
    for node_id, path_len in path_length_by_id.items():
        if path_len <= 0:
            continue
        root_row = rows_by_id[component_root_by_id[node_id]]
        node_row = rows_by_id[node_id]
        euclidean = float(np.linalg.norm(node_row[2:5] - root_row[2:5]))
        contractions.append(euclidean / path_len)

    if not contractions:
        return 0.0
    return float(np.mean(contractions))


def average_fragmentation(swc_list: list) -> float:
    """Average number of segments in each branch"""
    _, _, _, children_by_id, roots = _swc_topology_maps(swc_list)
    if not children_by_id:
        return 0.0

    bifurcations = {node_id for node_id, children in children_by_id.items() if len(children) > 1}
    tips = {node_id for node_id, children in children_by_id.items() if len(children) == 0}
    branch_starts = set(roots).union(bifurcations)

    branch_fragment_counts = []
    for start_id in branch_starts:
        for child_id in children_by_id.get(start_id, []):
            fragments = 1
            node_id = int(child_id)
            while node_id not in bifurcations and node_id not in tips:
                next_children = children_by_id.get(node_id, [])
                if len(next_children) != 1:
                    break
                node_id = int(next_children[0])
                fragments += 1
            branch_fragment_counts.append(fragments)

    if not branch_fragment_counts:
        return 0.0
    return float(np.mean(branch_fragment_counts))


def _angle_degrees(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute stable angle in degrees between two vectors."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    cos_theta = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def bifurcation_angle_local(swc_list: list) -> float:
    """Angle between downstream nodes closest to a bifurcation"""
    swc_array, rows_by_id, _, children_by_id, _ = _swc_topology_maps(swc_list)
    if swc_array.shape[0] == 0:
        return float("nan")

    angles = []
    for node_id, children in children_by_id.items():
        if len(children) < 2:
            continue
        center = rows_by_id[node_id][2:5]
        for child_a, child_b in combinations(children, 2):
            vec_a = rows_by_id[int(child_a)][2:5] - center
            vec_b = rows_by_id[int(child_b)][2:5] - center
            angle = _angle_degrees(vec_a, vec_b)
            if not np.isnan(angle):
                angles.append(angle)

    if not angles:
        return float("nan")
    return float(np.mean(angles))


def bifurcation_angle_remote(swc_list: list) -> float:
    """Angle between downstream branch or termination points closest to a bifurcation"""
    swc_array, rows_by_id, _, children_by_id, _ = _swc_topology_maps(swc_list)
    if swc_array.shape[0] == 0:
        return float("nan")

    bifurcations = {node_id for node_id, children in children_by_id.items() if len(children) > 1}
    tips = {node_id for node_id, children in children_by_id.items() if len(children) == 0}
    angles = []

    for node_id, children in children_by_id.items():
        if len(children) < 2:
            continue

        remote_nodes = []
        for child_id in children:
            current = int(child_id)
            while current not in bifurcations and current not in tips:
                next_children = children_by_id.get(current, [])
                if len(next_children) != 1:
                    break
                current = int(next_children[0])
            remote_nodes.append(current)

        center = rows_by_id[node_id][2:5]
        for remote_a, remote_b in combinations(remote_nodes, 2):
            vec_a = rows_by_id[int(remote_a)][2:5] - center
            vec_b = rows_by_id[int(remote_b)][2:5] - center
            angle = _angle_degrees(vec_a, vec_b)
            if not np.isnan(angle):
                angles.append(angle)

    if not angles:
        return float("nan")
    return float(np.mean(angles))


def different_structure_average(tree1, tree2, distance_threshold: float = 2.0) -> float:
    """Average distance from neuron 1 to 2 and from neuron 2 to 1 for points that have a distance >= distance_threshold pixels"""
    _, _, details_1_to_2 = directed_divergence(
        tree1,
        tree2,
        threshold=float(distance_threshold),
        return_details=True,
    )
    _, _, details_2_to_1 = directed_divergence(
        tree2,
        tree1,
        threshold=float(distance_threshold),
        return_details=True,
    )

    vals = [
        float(details_1_to_2["different_structure_average"]),
        float(details_2_to_1["different_structure_average"]),
    ]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def percentage_different_structure(tree1, tree2, distance_threshold: float = 2.0) -> float:
    """Percentage of points in neuron 1 that are >= distance_threshold from neuron 2."""
    _, _, details = directed_divergence(
        tree1,
        tree2,
        threshold=float(distance_threshold),
        return_details=True,
    )
    return float(details["percentage_different_structure"])


def percent_different_structure_average(tree1, tree2, distance_threshold: float = 2.0) -> float:
    """Average of percent of different structure from neuron 1 to 2 and from neuron 2 to 1"""
    pct_1_to_2 = percentage_different_structure(tree1, tree2, distance_threshold=distance_threshold)
    pct_2_to_1 = percentage_different_structure(tree2, tree1, distance_threshold=distance_threshold)
    vals = [v for v in [pct_1_to_2, pct_2_to_1] if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


# Comprehensive evaluation function that returns multiple metrics in a single call
def evaluate_reconstruction(
    pred_swc: list,
    gt_swc: list,
    threshold: float = 4.0,
    return_l_measures: bool = False,
) -> Dict[str, Any]:
    div_pred_to_gt, _, pred_to_gt_details = directed_divergence(
        pred_swc,
        gt_swc,
        threshold=threshold,
        return_details=True,
    )

    div_gt_to_pred, _, gt_to_pred_details = directed_divergence(
        gt_swc,
        pred_swc,
        threshold=threshold,
        return_details=True,
    )

    bidirectional_dist = (div_pred_to_gt + div_gt_to_pred) / 2
    precision = 1.0 - float(pred_to_gt_details["percentage_different_structure"])
    coverage = 1.0 - float(gt_to_pred_details["percentage_different_structure"])
    pred_endpoints = _get_special_node_coords(pred_swc, "endpoint")
    gt_endpoints = _get_special_node_coords(gt_swc, "endpoint")
    pred_branchpoints = _get_special_node_coords(pred_swc, "branchpoint")
    gt_branchpoints = _get_special_node_coords(gt_swc, "branchpoint")

    results: Dict[str, Any] = {
        "directed_div_pred_to_gt": float(div_pred_to_gt),
        "directed_div_gt_to_pred": float(div_gt_to_pred),
        "bidirectional_distance": float(bidirectional_dist),
        "precision": float(precision),
        "coverage": float(coverage),
        "endpoint_localization_error": _symmetric_localization_error(pred_endpoints, gt_endpoints),
        "branchpoint_localization_error": _symmetric_localization_error(pred_branchpoints, gt_branchpoints),
        "endpoint_count_error": int(abs(pred_endpoints.shape[0] - gt_endpoints.shape[0])),
        "branchpoint_count_error": int(abs(pred_branchpoints.shape[0] - gt_branchpoints.shape[0])),
    }

    if return_l_measures:
        different_structure_vals = [
            float(pred_to_gt_details["different_structure_average"]),
            float(gt_to_pred_details["different_structure_average"]),
        ]
        different_structure_vals = [v for v in different_structure_vals if not np.isnan(v)]
        percentage_vals = [
            float(pred_to_gt_details["percentage_different_structure"]),
            float(gt_to_pred_details["percentage_different_structure"]),
        ]
        percentage_vals = [v for v in percentage_vals if not np.isnan(v)]

        results.update(
            {
                "num_bifurcations_pred": num_bifurcations(pred_swc),
                "num_bifurcations_gt": num_bifurcations(gt_swc),
                "num_branches_pred": num_branches(pred_swc),
                "num_branches_gt": num_branches(gt_swc),
                "num_tips_pred": num_tips(pred_swc),
                "num_tips_gt": num_tips(gt_swc),
                "span_pred": span(pred_swc),
                "span_gt": span(gt_swc),
                "total_length_pred": total_length(pred_swc),
                "total_length_gt": total_length(gt_swc),
                "max_euclidean_distance_pred": max_euclidean_distance(pred_swc),
                "max_euclidean_distance_gt": max_euclidean_distance(gt_swc),
                "max_path_distance_pred": max_path_distance(pred_swc),
                "max_path_distance_gt": max_path_distance(gt_swc),
                "max_branch_order_pred": max_branch_order(pred_swc),
                "max_branch_order_gt": max_branch_order(gt_swc),
                "average_contraction_pred": average_contraction(pred_swc),
                "average_contraction_gt": average_contraction(gt_swc),
                "average_fragmentation_pred": average_fragmentation(pred_swc),
                "average_fragmentation_gt": average_fragmentation(gt_swc),
                "bifurcation_angle_local_pred": bifurcation_angle_local(pred_swc),
                "bifurcation_angle_local_gt": bifurcation_angle_local(gt_swc),
                "bifurcation_angle_remote_pred": bifurcation_angle_remote(pred_swc),
                "bifurcation_angle_remote_gt": bifurcation_angle_remote(gt_swc),
                "different_structure_average": float(np.mean(different_structure_vals)) if different_structure_vals else float("nan"),
                "percentage_different_structure_pred_to_gt": float(
                    pred_to_gt_details["percentage_different_structure"]
                ),
                "percentage_different_structure_gt_to_pred": float(
                    gt_to_pred_details["percentage_different_structure"]
                ),
                "percent_different_structure_average": float(np.mean(percentage_vals)) if percentage_vals else float("nan"),
            }
        )

    return results