"""Shared post-processing flow for inference outputs."""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
from scipy.spatial import KDTree

from neurotrack.data import loading as data_loading
from neurotrack.data import save, tree


def _coord_key_xyz(point_xyz: np.ndarray, decimals: int = 5) -> tuple[float, float, float]:
    point = np.asarray(point_xyz, dtype=np.float32).reshape(-1)
    rounded = np.round(point[:3].astype(np.float64), decimals=decimals)
    return float(rounded[0]), float(rounded[1]), float(rounded[2])


def filter_paths_by_length(
    paths: List[np.ndarray],
    min_length: float,
    max_length: float = float("inf"),
) -> List[np.ndarray]:
    normalized_paths: List[np.ndarray] = []
    max_length_f = float(max_length)

    for path in paths:
        if isinstance(path, torch.Tensor):
            path = path.cpu().numpy()

        path = np.asarray(path)

        if len(path) == 0:
            continue

        normalized_paths.append(path.astype(np.float32, copy=False))

    if len(normalized_paths) == 0:
        return []

    # Build topology over the full prediction so removals can cascade to descendants.
    swc_list = save.paths_to_swc(normalized_paths)
    coord_to_node_id = {
        _coord_key_xyz(np.asarray(row[2:5], dtype=np.float32)): int(row[0])
        for row in swc_list
    }

    # Coordinates may be shared across trunk/branch sections.  For short-path
    # removal we should cascade from the first path-unique node, not from a
    # shared anchor on a trunk.
    coord_to_path_ids: Dict[tuple[float, float, float], set[int]] = {}
    for path_idx, path in enumerate(normalized_paths):
        for node_xyz in path:
            key = _coord_key_xyz(node_xyz[:3])
            coord_to_path_ids.setdefault(key, set()).add(path_idx)

    def _first_unique_node_id(path_idx: int, path: np.ndarray) -> int | None:
        for node_xyz in path:
            key = _coord_key_xyz(node_xyz[:3])
            if coord_to_path_ids.get(key) == {path_idx}:
                return coord_to_node_id.get(key)
        return None

    nodes_to_remove: set[int] = set()
    drop_path_indices: set[int] = set()
    n_trimmed_paths = 0
    n_short_paths = 0
    n_short_paths_dropped_only = 0

    for path_idx, path in enumerate(normalized_paths):
        if len(path) < 2:
            # Treat one-point paths as short branches.  If the only node is
            # shared, drop just this path instead of cascading from a trunk.
            node_id = _first_unique_node_id(path_idx, path)
            if node_id is not None:
                nodes_to_remove.update(
                    data_loading.get_downstream_swc_node_ids(swc_list, node_id, include_start=True)
                )
            else:
                drop_path_indices.add(path_idx)
                n_short_paths_dropped_only += 1
            n_short_paths += 1
            continue

        deltas = np.diff(path, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        total_length = float(np.sum(distances))

        if total_length < float(min_length):
            # Remove short branches from their first unique node, not from a
            # potentially shared branch anchor.
            start_node_id = _first_unique_node_id(path_idx, path)
            if start_node_id is not None:
                nodes_to_remove.update(
                    data_loading.get_downstream_swc_node_ids(
                        swc_list,
                        start_node_id,
                        include_start=True,
                    )
                )
            else:
                drop_path_indices.add(path_idx)
                n_short_paths_dropped_only += 1
            n_short_paths += 1
            continue

        if np.isfinite(max_length_f):
            cumulative_lengths = np.cumsum(distances)
            over_threshold_indices = np.where(cumulative_lengths > max_length_f)[0]
            if over_threshold_indices.size > 0:
                first_over_idx = int(over_threshold_indices[0]) + 1
                trim_node_id = coord_to_node_id.get(_coord_key_xyz(path[first_over_idx, :3]))
                if trim_node_id is not None:
                    nodes_to_remove.update(
                        data_loading.get_downstream_swc_node_ids(
                            swc_list,
                            trim_node_id,
                            include_start=True,
                        )
                    )
                    n_trimmed_paths += 1

    filtered_paths: List[np.ndarray] = []
    n_residual_short_paths = 0
    for path_idx, path in enumerate(normalized_paths):
        if path_idx in drop_path_indices:
            continue
        kept_nodes = []
        for node_xyz in path:
            node_id = coord_to_node_id.get(_coord_key_xyz(node_xyz))
            if node_id is not None and node_id in nodes_to_remove:
                continue
            kept_nodes.append(np.asarray(node_xyz, dtype=np.float32)[:3].copy())
        if len(kept_nodes) == 0:
            continue

        kept_arr = np.asarray(kept_nodes, dtype=np.float32)
        if len(kept_arr) < 2:
            n_residual_short_paths += 1
            continue
        kept_len = float(np.sum(np.linalg.norm(np.diff(kept_arr, axis=0), axis=1)))
        if kept_len < float(min_length):
            n_residual_short_paths += 1
            continue
        filtered_paths.append(kept_arr)

    normalized_paths = filtered_paths

    if not normalized_paths and len(paths) > 0:
        longest_idx = None
        longest_score = (-1.0, -1)
        for idx, path in enumerate(paths):
            if isinstance(path, torch.Tensor):
                path = path.cpu().numpy()

            path = np.asarray(path)
            if len(path) == 0:
                continue

            if len(path) < 2:
                score = (0.0, len(path))
            else:
                deltas = np.diff(path, axis=0)
                distances = np.linalg.norm(deltas, axis=1)
                score = (float(np.sum(distances)), len(path))

            if score > longest_score:
                longest_score = score
                longest_idx = idx

        if longest_idx is not None:
            fallback_path = paths[longest_idx]
            if isinstance(fallback_path, torch.Tensor):
                fallback_path = fallback_path.cpu().numpy()
            normalized_paths.append(np.asarray(fallback_path))

    n_removed = len(paths) - len(normalized_paths)
    max_label = "inf" if not np.isfinite(max_length_f) else f"{max_length_f:.1f}"
    print(
        "    Removed "
        f"{n_removed} paths outside [{float(min_length):.1f}, {max_label}] units"
    )
    if np.isfinite(max_length_f):
        print(f"    Trimmed {n_trimmed_paths} path(s) at first node past max length threshold")
    if n_short_paths > 0:
        print(f"    Removed {n_short_paths} short branch path(s) with descendants")
    if n_short_paths_dropped_only > 0:
        print(
            "    Dropped "
            f"{n_short_paths_dropped_only} short shared-anchor path(s) "
            "without descendant cascade"
        )
    if n_residual_short_paths > 0:
        print(f"    Dropped {n_residual_short_paths} residual short path stub(s) after filtering")

    return normalized_paths


def _restructure_paths(paths: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize path hierarchy/order via tree.restructure_neuron_tree."""
    paths_as_tensors = [_path_to_tensor(path) for path in paths]
    sections = tree.restructure_neuron_tree(paths_as_tensors, input_type="paths")
    return [
        section.detach().cpu().numpy() if isinstance(section, torch.Tensor) else np.asarray(section)
        for section in sections.values()
    ]


def smooth_paths(paths: List[np.ndarray], window_size: int = 5) -> List[np.ndarray]:
    paths_np = []
    for path in paths:
        if isinstance(path, torch.Tensor):
            paths_np.append(path.cpu().numpy())
        else:
            paths_np.append(np.array(path))

    point_counts: Dict[tuple, List[tuple]] = {}
    for path_idx, path in enumerate(paths_np):
        for point_idx, point in enumerate(path):
            point_tuple = tuple(point[:3].astype(float))
            if point_tuple not in point_counts:
                point_counts[point_tuple] = []
            point_counts[point_tuple].append((path_idx, point_idx))

    connection_points = set()
    for point_tuple, occurrences in point_counts.items():
        if len(occurrences) > 1:
            connection_points.add(point_tuple)
        else:
            path_idx, point_idx = occurrences[0]
            if point_idx == 0 or point_idx == len(paths_np[path_idx]) - 1:
                connection_points.add(point_tuple)

    smoothed_paths = []
    for path in paths_np:
        if len(path) < window_size:
            smoothed_paths.append(path)
            continue

        # Collect indices of all branch/connection points in this path.
        preserved_indices = set()
        for point_idx, point in enumerate(path):
            point_tuple = tuple(point[:3].astype(float))
            if point_tuple in connection_points:
                preserved_indices.add(point_idx)

        # Build segment boundaries, always including the path endpoints.
        boundaries = sorted(preserved_indices | {0, len(path) - 1})

        smoothed = np.copy(path)

        # Smooth each inter-branch segment independently so the filter kernel
        # never crosses a branch point.  Both endpoints of each segment are
        # pinned to their original positions after smoothing.
        for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
            seg_len = seg_end - seg_start + 1
            if seg_len < 3:
                # Nothing useful to smooth in a two-point segment.
                continue
            segment = path[seg_start : seg_end + 1]
            effective_window = min(window_size, seg_len)
            smoothed_seg = np.copy(segment)
            for dim in range(3):
                smoothed_seg[:, dim] = uniform_filter1d(
                    segment[:, dim], size=effective_window, mode="nearest"
                )
            # Pin both endpoints (branch / connection points).
            smoothed_seg[0] = segment[0]
            smoothed_seg[-1] = segment[-1]
            smoothed[seg_start : seg_end + 1] = smoothed_seg

        smoothed_paths.append(smoothed)

    print(f"    Smoothed {len(smoothed_paths)} paths with window size {window_size}")
    print(f"    Preserved {len(connection_points)} connection points")

    return smoothed_paths


def _clean_overlap_mask(overlap_mask: np.ndarray, smoothing_size: int = 0) -> np.ndarray:
    """Denoise a 1-D overlap mask with binary closing followed by opening.

    Closing fills short unique gaps (spurious ``False`` runs inside overlap);
    opening removes short overlap spikes (spurious ``True`` runs inside unique).
    ``smoothing_size <= 1`` returns the mask unchanged.
    """
    mask = np.asarray(overlap_mask, dtype=bool)
    if smoothing_size is None or int(smoothing_size) <= 1 or mask.size == 0:
        return mask
    structure = np.ones(int(smoothing_size), dtype=bool)
    # ``border_value=1`` for closing keeps a genuinely overlapping endpoint from
    # being eroded into a spurious unique point at the path ends; opening uses
    # the default ``border_value=0`` so it cannot invent overlap at the borders.
    cleaned = binary_closing(mask, structure=structure, border_value=1)
    cleaned = binary_opening(cleaned, structure=structure, border_value=0)
    return np.asarray(cleaned, dtype=bool)


def _merge_runs(
    path: np.ndarray,
    overlap_mask: np.ndarray,
    nearest_reference_idx: np.ndarray,
    reference_points: np.ndarray,
) -> List[np.ndarray]:
    """Split a redundant path into merged, re-rooted unique segments.

    Each maximal run of unique (non-overlapping) points becomes one segment.  A
    run preceded by overlap is re-rooted onto the longer path: the nearest
    reference point to the last preceding overlap point is prepended as an
    anchor, so it dedups onto that node in the final SWC and the run attaches as
    a branch at the divergence point.  Overlap points are otherwise dropped; a
    run that begins at the path start keeps its original free root.  Tails are
    never re-attached (that would create a cycle), so a run that diverges and
    rejoins ends as a free branch tip.
    """
    mask = np.asarray(overlap_mask, dtype=bool)
    n = len(path)
    n_cols = path.shape[1] if path.ndim == 2 else 3
    segments: List[np.ndarray] = []

    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        run_start = i
        while i < n and not mask[i]:
            i += 1
        run = np.asarray(path[run_start:i], dtype=np.float32)

        if run_start > 0:
            # Preceded by overlap: anchor onto the longer path at the divergence.
            anchor = np.asarray(
                reference_points[int(nearest_reference_idx[run_start - 1])],
                dtype=np.float32,
            ).reshape(1, -1)[:, :n_cols]
            segments.append(np.vstack([anchor, run]))
        else:
            # Begins in unique territory: keep its original free root.
            segments.append(run.copy())

    return segments


def _confidence_clip_paths(
    merged_paths: List[np.ndarray],
    input_paths: List[np.ndarray],
    merge_threshold: float,
    confidence_threshold: int,
) -> List[np.ndarray]:
    """Trim low-confidence distal tails from the merged paths.

    The confidence of a node is the number of input paths that have at least one
    node within ``merge_threshold`` of it.  A node is kept only when it (or one
    of its SWC descendants) is supported by at least ``confidence_threshold``
    input paths; purely low-confidence distal tails are clipped, cascading to any
    branch that hangs off a removed tail.
    """
    if confidence_threshold <= 1 or not merged_paths:
        return merged_paths

    # Flatten merged nodes and score each by how many input paths pass nearby.
    # One KDTree over the merged nodes + one batched ball query per input path
    # counts distinct supporting paths (the per-path set dedups repeated hits).
    flat_nodes = [node[:3] for path in merged_paths for node in path]
    if not flat_nodes:
        return merged_paths
    flat = np.asarray(flat_nodes, dtype=np.float64)
    support = np.zeros(len(flat), dtype=np.int64)
    node_tree = KDTree(flat)
    for input_path in input_paths:
        input_path = np.asarray(input_path)
        if len(input_path) == 0:
            continue
        neighbor_lists = node_tree.query_ball_point(
            np.asarray(input_path[:, :3], dtype=np.float64), r=merge_threshold
        )
        covered: set[int] = set()
        for neighbors in neighbor_lists:
            covered.update(neighbors)
        for idx in covered:
            support[idx] += 1

    # Collapse to a per-coordinate score (duplicate anchor nodes share a key).
    coord_support: Dict[tuple, int] = {}
    for node, score in zip(flat, support):
        key = _coord_key_xyz(node)
        if int(score) > coord_support.get(key, -1):
            coord_support[key] = int(score)

    # Build SWC topology so "downstream" follows the real parent->child tree.
    swc_paths = [
        torch.from_numpy(np.asarray(path, dtype=np.float32))
        for path in merged_paths
        if len(path) > 0
    ]
    swc_list = save.paths_to_swc(swc_paths)
    if not swc_list:
        return merged_paths

    id_to_parent = {int(row[0]): int(row[6]) for row in swc_list}
    coord_to_node_id = {
        _coord_key_xyz(np.asarray(row[2:5], dtype=np.float32)): int(row[0])
        for row in swc_list
    }

    # Keep every confident node and all of its ancestors; everything else is a
    # low-confidence tail (or hangs off one).  The ancestor walk is memoized via
    # ``keep_ids`` so each node is visited at most once.
    keep_ids: set[int] = set()
    for row in swc_list:
        key = _coord_key_xyz(np.asarray(row[2:5], dtype=np.float32))
        if coord_support.get(key, 0) < confidence_threshold:
            continue
        current = int(row[0])
        while current != -1 and current not in keep_ids:
            keep_ids.add(current)
            current = id_to_parent.get(current, -1)

    # Rebuild each path as its kept root-side prefix.  ``keep_ids`` is
    # ancestor-closed, so the first dropped node ends the path and a branch whose
    # anchor was dropped collapses to nothing (cascading the removal).
    clipped_paths: List[np.ndarray] = []
    for path in merged_paths:
        kept = []
        for node in path:
            node_id = coord_to_node_id.get(_coord_key_xyz(node[:3]))
            if node_id is None or node_id not in keep_ids:
                break
            kept.append(np.asarray(node[:3], dtype=np.float32))
        if kept:
            clipped_paths.append(np.asarray(kept, dtype=np.float32))

    return clipped_paths


def merge_paths(
    paths: List[np.ndarray],
    merge_threshold: float = 2.0,
    mask_smoothing_size: int = 0,
    confidence_threshold: int = 0,
        merge_guard_max_paths: int = 0,
        merge_guard_max_nodes: int = 0,
        merge_timeout_seconds: float = 30.0,
) -> List[np.ndarray]:
    """Merge overlapping paths into longer paths while preserving unique runs.

    When ``confidence_threshold > 1`` a confidence filter runs after merging:
    each finalized node is scored by how many input paths have a node within
    ``merge_threshold``, and low-confidence distal tails (nodes with no
    descendant supported by at least ``confidence_threshold`` input paths) are
    clipped, cascading to any branch hanging off a removed tail.

        Guardrails:
        - ``merge_guard_max_paths`` / ``merge_guard_max_nodes``: when positive, skip
            merge entirely if the input exceeds the corresponding budget.
        - ``merge_timeout_seconds``: when positive, abort merge after the time budget
            and pass through remaining unprocessed paths unchanged.
    """

    merged_paths = []
    for path in paths:
        if isinstance(path, torch.Tensor):
            merged_paths.append(path.cpu().numpy())
        else:
            merged_paths.append(np.array(path))

    # Snapshot the numpy input paths as the confidence reference before merging.
    input_paths_np = list(merged_paths)

    n_merge_modified = 0
    n_paths = len(merged_paths)
    n_total_nodes = int(sum(len(path) for path in merged_paths))
    if int(merge_guard_max_paths) > 0 and n_paths > int(merge_guard_max_paths):
        print(
            "    Skipped merge_paths: "
            f"n_paths={n_paths} exceeds merge_guard_max_paths={int(merge_guard_max_paths)}"
        )
        return merged_paths
    if int(merge_guard_max_nodes) > 0 and n_total_nodes > int(merge_guard_max_nodes):
        print(
            "    Skipped merge_paths: "
            f"n_nodes={n_total_nodes} exceeds merge_guard_max_nodes={int(merge_guard_max_nodes)}"
        )
        return merged_paths

    order = sorted(range(n_paths), key=lambda i: len(merged_paths[i]), reverse=True)
    finalized: Dict[int, List[np.ndarray]] = {}
    reference_points_list: List[np.ndarray] = []
    pending_points: List[np.ndarray] = []
    current_len = None
    reference_points = None
    tree_others = None
    merge_start_time = time.perf_counter()

    for order_idx, oi in enumerate(order):
        if (
            float(merge_timeout_seconds) > 0.0
            and (order_idx % 32 == 0)
            and (time.perf_counter() - merge_start_time) > float(merge_timeout_seconds)
        ):
            remaining = len(order) - order_idx
            print(
                "    merge_paths timeout reached "
                f"({float(merge_timeout_seconds):.1f}s); "
                f"passing through {remaining} remaining path(s) without merge"
            )
            for rem_oi in order[order_idx:]:
                rem_path = merged_paths[rem_oi]
                finalized[rem_oi] = [rem_path] if len(rem_path) > 0 else []
            break

        path = merged_paths[oi]
        path_len = len(path)
        if current_len is None:
            current_len = path_len
        if path_len < current_len:
            # The equal-length group is complete; only now does it join
            # the reference, preserving the strict "longer" rule.
            reference_points_list.extend(pending_points)
            pending_points = []
            current_len = path_len
            reference_points = None
            tree_others = None

        if path_len == 0:
            finalized[oi] = []
            continue
        if not reference_points_list:
            # Longest group: nothing strictly longer to merge into.
            finalized[oi] = [path]
            pending_points.append(path)
            continue

        # Reuse the same KDTree for all paths in an equal-length group since
        # the reference set does not change until we move to the next group.
        if tree_others is None:
            reference_points = np.vstack(reference_points_list)
            tree_others = KDTree(reference_points)
        distances, nn_idx = tree_others.query(path)
        overlap_mask = _clean_overlap_mask(distances <= merge_threshold, mask_smoothing_size)

        if not np.any(overlap_mask):
            finalized[oi] = [path]
            pending_points.append(path)
            continue

        segments = _merge_runs(path, overlap_mask, nn_idx, reference_points)
        finalized[oi] = segments
        pending_points.extend(segments)
        # Only count as modified if path actually changed:
        # - If 0 segments: path was completely absorbed (successful merge)
        # - If 1+ segments but path changed: it was split/re-anchored (actual modification)
        # - If output identical to input: false positive (no real change)
        path_was_modified = (
            len(segments) == 0  # Completely absorbed
            or len(segments) > 1  # Split into multiple
            or (len(segments) == 1 and not np.array_equal(segments[0], path))  # Single segment but changed
        )
        if path_was_modified:
            n_merge_modified += 1

    # Emit longest-first so a trunk always precedes the branches anchored
    # onto it (paths_to_swc assigns parent by first occurrence of a
    # shared coordinate).
    rebuilt_paths = []
    for oi in order:
        rebuilt_paths.extend(finalized.get(oi, []))
    merged_paths = rebuilt_paths

    print(f"    Merged {n_merge_modified} overlapping path(s) onto longer paths")

    if int(confidence_threshold) > 1:
        n_nodes_before = sum(len(path) for path in merged_paths)
        merged_paths = _confidence_clip_paths(
            merged_paths, input_paths_np, merge_threshold, int(confidence_threshold)
        )
        n_nodes_after = sum(len(path) for path in merged_paths)
        print(
            f"    Confidence-clipped {n_nodes_before - n_nodes_after} low-confidence "
            f"node(s) (threshold={int(confidence_threshold)})"
        )

    return merged_paths


def _path_to_tensor(path: Any) -> torch.Tensor:
    if isinstance(path, list):
        return torch.stack([
            torch.from_numpy(np.asarray(point)) if isinstance(point, (np.ndarray, list)) else point
            for point in path
        ])
    if isinstance(path, np.ndarray):
        return torch.from_numpy(path)
    return path


def process_results(results: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    enable_length_filter = bool(params.get("enable_length_filter", True))
    min_branch_length = float(params.get("min_branch_length", 5.0))
    max_branch_length = float(params.get("max_branch_length", float("inf")))
    enable_resample = bool(params.get("enable_resample", True))
    resampling_step_size = float(params.get("resampling_step_size", 4.0))
    enable_smooth_paths = bool(params.get("enable_smooth_paths", True))
    smoothing_window = int(params.get("smoothing_window", 5))
    enable_merge = bool(params.get("enable_merge", True))
    merge_threshold = float(params.get("merge_threshold", 1.0))
    confidence_threshold = int(params.get("confidence_threshold", 0))
    mask_smoothing_size = int(params.get("mask_smoothing_size", 0))
    merge_guard_max_paths = int(params.get("merge_guard_max_paths", 0))
    merge_guard_max_nodes = int(params.get("merge_guard_max_nodes", 0))
    merge_timeout_seconds = float(params.get("merge_timeout_seconds", 30.0))
    max_branch_label = "inf" if not np.isfinite(max_branch_length) else f"{max_branch_length}"

    processed_results: List[Dict[str, Any]] = []
    for result in results:
        neuron_name = result.get("neuron_name", "unknown")
        raw_paths = result.get("paths", [])
        print(f"Processing neuron '{neuron_name}'\n\
              Params\n\
              ------\n\
              enable_length_filter: {enable_length_filter}\n\
              min_branch_length: {min_branch_length}\n\
              max_branch_length: {max_branch_label}\n\
              enable_resample: {enable_resample}\n\
              resampling_step_size: {resampling_step_size}\n\
              enable_smooth_paths: {enable_smooth_paths}\n\
              smoothing_window: {smoothing_window}\n\
              enable_merge: {enable_merge}\n\
              merge_threshold: {merge_threshold}\n\
              confidence_threshold: {confidence_threshold}\n\
              merge_guard_max_paths: {merge_guard_max_paths}\n\
              merge_guard_max_nodes: {merge_guard_max_nodes}\n\
              merge_timeout_seconds: {merge_timeout_seconds}\n\
              mask_smoothing_size: {mask_smoothing_size}\n")
        try:
            paths = _restructure_paths(raw_paths)

            if enable_resample:
                paths = tree.resample_tree(paths, step_size=resampling_step_size)
            if enable_smooth_paths:
                paths = smooth_paths(paths, window_size=smoothing_window)
            if enable_merge:
                paths = merge_paths(
                    paths,
                    merge_threshold=merge_threshold,
                    mask_smoothing_size=mask_smoothing_size,
                    confidence_threshold=confidence_threshold,
                    merge_guard_max_paths=merge_guard_max_paths,
                    merge_guard_max_nodes=merge_guard_max_nodes,
                    merge_timeout_seconds=merge_timeout_seconds,
                )
                # Merge can split/re-anchor sections; re-normalize hierarchy before
                # descendant-cascade length filtering.
                paths = _restructure_paths(paths)
            if enable_length_filter:
                paths = filter_paths_by_length(
                    paths,
                    min_length=min_branch_length,
                    max_length=max_branch_length,
                )

            post_paths = [
                torch.from_numpy(path.astype(np.float32))
                if isinstance(path, np.ndarray)
                else path
                for path in paths
                if len(path) > 0
            ]
            swc_list = save.paths_to_swc(post_paths)

            processed_results.append({
                "neuron_name": neuron_name,
                "swc_list": swc_list,
                "processed_paths": post_paths,
                "n_raw_paths": len(raw_paths),
                "n_processed_paths": len(post_paths),
                "n_swc_nodes": len(swc_list),
            })
        except Exception as exc:
            print(f"\n[postprocess ERROR] '{neuron_name}': {exc}")
            traceback.print_exc()
            processed_results.append({
                "neuron_name": neuron_name,
                "swc_list": [],
                "processed_paths": [],
                "n_raw_paths": len(raw_paths),
                "n_processed_paths": 0,
                "n_swc_nodes": 0,
                "error": str(exc),
            })

    return processed_results


def write_processed_swc(processed_results: List[Dict[str, Any]], out_dir: Path | str) -> Dict[str, Any]:
    run_out_dir = Path(out_dir)
    swc_out_dir = run_out_dir / "reconstructions"
    swc_out_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    n_failed = 0
    per_neuron: List[Dict[str, Any]] = []

    for result in processed_results:
        neuron_name = result.get("neuron_name", "unknown")
        neuron_basename = str(neuron_name).strip() or "unknown"

        if "error" in result:
            n_failed += 1
            print(f"[write_swc SKIP] '{neuron_name}': {result['error']}")
            per_neuron.append({
                "neuron_name": neuron_name,
                "saved": False,
                "reason": result["error"],
            })
            continue

        swc_list = result.get("swc_list", [])
        if len(swc_list) == 0:
            n_failed += 1
            print(f"[write_swc SKIP] '{neuron_name}': empty SWC after post-processing")
            per_neuron.append({
                "neuron_name": neuron_name,
                "saved": False,
                "reason": "empty_swc_after_postprocess",
            })
            continue

        swc_path = swc_out_dir / f"{neuron_basename}.swc"
        save.write_swc(swc_list, str(swc_path))
        n_saved += 1

        per_neuron.append({
            "neuron_name": neuron_name,
            "saved": True,
            "swc_path": str(swc_path),
            "n_raw_paths": int(result.get("n_raw_paths", 0)),
            "n_processed_paths": int(result.get("n_processed_paths", 0)),
            "n_swc_nodes": int(result.get("n_swc_nodes", 0)),
        })

    return {
        "swc_out_dir": swc_out_dir,
    }


__all__ = [
    "filter_paths_by_length",
    "smooth_paths",
    "merge_paths",
    "process_results",
    "write_processed_swc",
]
