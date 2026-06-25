"""Shared post-processing flow for inference outputs."""

import json
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

    nodes_to_remove: set[int] = set()
    n_trimmed_paths = 0
    n_short_paths = 0

    for path in normalized_paths:
        if len(path) < 2:
            # Treat one-point paths as short branches.
            node_id = coord_to_node_id.get(_coord_key_xyz(path[0, :3]))
            if node_id is not None:
                nodes_to_remove.update(
                    data_loading.get_downstream_swc_node_ids(swc_list, node_id, include_start=True)
                )
            n_short_paths += 1
            continue

        deltas = np.diff(path, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        total_length = float(np.sum(distances))

        if total_length < float(min_length):
            start_node_id = coord_to_node_id.get(_coord_key_xyz(path[0, :3]))
            if start_node_id is not None:
                nodes_to_remove.update(
                    data_loading.get_downstream_swc_node_ids(
                        swc_list,
                        start_node_id,
                        include_start=True,
                    )
                )
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
    for path in normalized_paths:
        kept_nodes = []
        for node_xyz in path:
            node_id = coord_to_node_id.get(_coord_key_xyz(node_xyz))
            if node_id is not None and node_id in nodes_to_remove:
                continue
            kept_nodes.append(np.asarray(node_xyz, dtype=np.float32)[:3].copy())
        if len(kept_nodes) > 0:
            filtered_paths.append(np.asarray(kept_nodes, dtype=np.float32))

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

    return normalized_paths


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


def _build_connectivity_graph(paths: List[np.ndarray]) -> Dict[int, List[int]]:
    graph: Dict[int, List[int]] = {i: [] for i in range(len(paths))}

    start_point_to_paths: Dict[tuple, List[int]] = {}
    for i, path in enumerate(paths):
        if len(path) > 0:
            start_tuple = tuple(np.round(path[0][:3].astype(float), decimals=3))
            start_point_to_paths.setdefault(start_tuple, []).append(i)

    for i, path in enumerate(paths):
        for point in path[1:]:
            point_tuple = tuple(np.round(point[:3].astype(float), decimals=3))
            if point_tuple in start_point_to_paths:
                child_indices = start_point_to_paths[point_tuple]
                for child_idx in child_indices:
                    if child_idx != i:
                        graph[i].append(child_idx)

    return graph


def _get_all_descendants(idx: int, graph: Dict[int, List[int]], visited=None) -> set:
    if visited is None:
        visited = set()
    if idx in visited:
        return set()

    visited.add(idx)
    descendants = {idx}

    for child in graph.get(idx, []):
        descendants.update(_get_all_descendants(child, graph, visited))

    return descendants


def _best_clip_index(
    overlap_mask: np.ndarray,
    keep_weight: float = 1.0,
    remove_weight: float = 1.0,
) -> int:
    """Choose where to clip a path that partially overlaps longer paths.

    Points ``[0, k)`` are kept and points ``[k, n)`` are removed.  ``k`` is the
    index that maximizes ``keep_weight * (non-overlapping points kept)`` plus
    ``remove_weight * (overlapping points removed)``.  That objective reduces to
    the maximum prefix sum of per-point weights (``+keep_weight`` for a unique
    point, ``-remove_weight`` for an overlapping point), an O(n) scan.

    ``k == 0`` drops the whole path; ``k == len(mask)`` keeps it unchanged.  On
    ties the largest ``k`` is returned, so non-overlapping points are preserved
    rather than discarded — a path whose overlap is at the *head* (e.g. a branch
    rooted on the longer path) is left intact instead of being clipped to
    nothing, since a suffix clip cannot keep a unique tail behind it.
    """
    mask = np.asarray(overlap_mask, dtype=bool)
    if mask.size == 0:
        return 0
    weights = np.where(mask, -float(remove_weight), float(keep_weight))
    prefix_sums = np.concatenate(([0.0], np.cumsum(weights)))
    best_indices = np.flatnonzero(prefix_sums == prefix_sums.max())
    return int(best_indices[-1])


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


def merge_redundant_paths(
    paths: List[np.ndarray],
    overlap_threshold: float = 0.8,
    distance_threshold: float = 2.0,
    redundancy_action: str = "remove",
    mask_smoothing_size: int = 0,
) -> List[np.ndarray]:
    """Drop, clip, or merge paths that overlap with longer paths.

    Parameters
    ----------
    redundancy_action : str
        ``"remove"`` (default) drops a redundant path together with its
        descendants.  ``"clip"`` instead trims a redundant path to its unique
        prefix, removing only the overlapping tail (the cut point is chosen by
        :func:`_best_clip_index`).  Clipping is local to the flagged path: it
        never edits the longer paths it overlaps, and any branch that started on
        the removed tail is left in place (it simply becomes its own root in the
        final SWC) rather than being deleted.  A path whose overlap is entirely
        at the head is left unchanged, since a suffix clip cannot preserve a
        unique tail behind it.  ``"merge"`` re-roots a redundant path onto the
        longer path: overlapping points are dropped and each remaining unique
        run is re-attached at its divergence point, so no unique points are
        lost.  A single path may split into several branches, and a fully
        redundant path vanishes.  Merge runs as one longest->shortest pass so
        anchors target already-finalized (and therefore stable) longer paths.
    mask_smoothing_size : int
        When > 1 and ``redundancy_action == "merge"``, the 1-D overlap mask is
        denoised with binary closing then opening using a structuring element of
        this length before it is segmented into runs.
    """
    action = str(redundancy_action).lower()
    if action not in ("remove", "clip", "merge"):
        raise ValueError(
            f"redundancy_action must be 'remove', 'clip', or 'merge', got {redundancy_action!r}"
        )

    merged_paths = []
    for path in paths:
        if isinstance(path, torch.Tensor):
            merged_paths.append(path.cpu().numpy())
        else:
            merged_paths.append(np.array(path))

    n_merged = 0
    n_clipped = 0
    n_points_clipped = 0
    n_merge_modified = 0
    changed = True

    while changed:
        changed = False
        n_paths = len(merged_paths)

        indexed_paths = [(i, len(merged_paths[i]), merged_paths[i]) for i in range(n_paths)]
        indexed_paths.sort(key=lambda item: item[1])

        if action == "remove":
            connectivity_graph = _build_connectivity_graph(merged_paths)
            paths_to_remove = set()

            for idx, path_len, path in indexed_paths:
                if idx in paths_to_remove:
                    continue

                # Only compare against paths that are strictly longer.  A path
                # should only be considered redundant if a longer path covers the
                # same territory.  Using the union of ALL other paths (including
                # shorter siblings and children) could cause a path near a busy
                # branch region to appear covered by the combined cloud of nearby
                # shorter paths — then the cascade removes its unique children.
                other_paths = [
                    other_path
                    for other_idx, other_len, other_path in indexed_paths
                    if other_idx != idx and other_idx not in paths_to_remove and other_len > path_len
                ]
                if not other_paths:
                    continue

                all_other_points = np.vstack(other_paths)
                tree_others = KDTree(all_other_points)
                distances, _ = tree_others.query(path)
                overlap_fraction = np.mean(distances <= distance_threshold)

                if overlap_fraction >= overlap_threshold:
                    # Cascade to all descendants to preserve tree topology:
                    # children whose start point lies within the removed path
                    # would otherwise become disjoint orphans.
                    descendants = _get_all_descendants(idx, connectivity_graph)
                    paths_to_remove.update(descendants)
                    changed = True

            if paths_to_remove:
                for remove_idx in sorted(paths_to_remove, reverse=True):
                    merged_paths.pop(remove_idx)
                    n_merged += 1
        elif action == "clip":  # clip
            # Trim each redundant path to its unique prefix instead of dropping
            # it.  Working purely by index keeps the operation local to the
            # flagged path: it never removes points from the longer paths it
            # overlaps (even where they share exact coordinates), and it leaves
            # any unique branches that hang off the removed tail in place rather
            # than deleting them.
            clip_at: Dict[int, int] = {}

            for idx, path_len, path in indexed_paths:
                # Compare against strictly longer paths only (same rationale as
                # the "remove" branch).
                other_paths = [
                    other_path
                    for other_idx, other_len, other_path in indexed_paths
                    if other_idx != idx and other_len > path_len
                ]
                if not other_paths:
                    continue

                all_other_points = np.vstack(other_paths)
                tree_others = KDTree(all_other_points)
                distances, _ = tree_others.query(path)
                overlap_mask = distances <= distance_threshold
                overlap_fraction = float(np.mean(overlap_mask))

                if overlap_fraction < overlap_threshold:
                    continue

                clip_idx = _best_clip_index(overlap_mask)
                if clip_idx >= len(path):
                    # Net-unique path: nothing worth clipping.
                    continue

                clip_at[idx] = clip_idx
                n_clipped += 1
                changed = True

            if clip_at:
                rebuilt_paths: List[np.ndarray] = []
                for pi, path in enumerate(merged_paths):
                    limit = clip_at.get(pi, len(path))
                    if limit >= len(path):
                        rebuilt_paths.append(path)
                        continue
                    n_points_clipped += len(path) - limit
                    if limit > 0:
                        rebuilt_paths.append(path[:limit].copy())
                merged_paths = rebuilt_paths
        else:  # merge
            # Single longest->shortest pass.  Compare each path against the
            # points of the already-finalized (strictly longer) paths, drop its
            # overlapping points, and re-root each remaining unique run onto the
            # nearest longer-path point at its divergence so it becomes a branch.
            # Paths can split into several branches; fully redundant paths vanish.
            # Finalized points seed the reference for shorter paths, keeping
            # anchors stable and guaranteeing termination in one pass.
            order = sorted(range(n_paths), key=lambda i: len(merged_paths[i]), reverse=True)
            finalized: Dict[int, List[np.ndarray]] = {}
            reference_points_list: List[np.ndarray] = []
            pending_points: List[np.ndarray] = []
            current_len = None

            for oi in order:
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

                if path_len == 0:
                    finalized[oi] = []
                    continue
                if not reference_points_list:
                    # Longest group: nothing strictly longer to merge into.
                    finalized[oi] = [path]
                    pending_points.append(path)
                    continue

                reference_points = np.vstack(reference_points_list)
                tree_others = KDTree(reference_points)
                distances, nn_idx = tree_others.query(path)
                overlap_mask = _clean_overlap_mask(
                    distances <= distance_threshold, mask_smoothing_size
                )

                if float(np.mean(overlap_mask)) < overlap_threshold:
                    finalized[oi] = [path]
                    pending_points.append(path)
                    continue

                segments = _merge_runs(path, overlap_mask, nn_idx, reference_points)
                finalized[oi] = segments
                pending_points.extend(segments)
                n_merge_modified += 1

            # Emit longest-first so a trunk always precedes the branches anchored
            # onto it (paths_to_swc assigns parent by first occurrence of a
            # shared coordinate).
            rebuilt_paths = []
            for oi in order:
                rebuilt_paths.extend(finalized.get(oi, []))
            merged_paths = rebuilt_paths

    if action == "remove":
        print(f"    Merged {n_merged} redundant paths (threshold={overlap_threshold:.2f})")
    elif action == "clip":
        print(
            f"    Clipped {n_clipped} redundant path(s) "
            f"({n_points_clipped} point(s) removed, threshold={overlap_threshold:.2f})"
        )
    else:
        print(
            f"    Merged {n_merge_modified} overlapping path(s) onto longer paths "
            f"(threshold={overlap_threshold:.2f})"
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
    enable_remove_overlaps = bool(params.get("enable_remove_overlaps", True))
    overlap_threshold = float(params.get("overlap_threshold", 0.5))
    overlap_distance_threshold = float(params.get("overlap_distance_threshold", 1.0))
    redundancy_action = str(params.get("redundancy_action", "remove")).lower()
    mask_smoothing_size = int(params.get("mask_smoothing_size", 0))
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
              enable_remove_overlaps: {enable_remove_overlaps}\n\
              overlap_threshold: {overlap_threshold}\n\
              overlap_distance_threshold: {overlap_distance_threshold}\n\
              redundancy_action: {redundancy_action}\n\
              mask_smoothing_size: {mask_smoothing_size}\n")
        try:
            paths_as_tensors = [_path_to_tensor(path) for path in raw_paths]
            sections = tree.restructure_neuron_tree(paths_as_tensors, input_type="paths")
            paths = [
                section.detach().cpu().numpy() if isinstance(section, torch.Tensor) else np.asarray(section)
                for section in sections.values()
            ]

            if enable_length_filter:
                paths = filter_paths_by_length(
                    paths,
                    min_length=min_branch_length,
                    max_length=max_branch_length,
                )
            if enable_resample:
                paths = tree.resample_tree(paths, step_size=resampling_step_size)
            if enable_smooth_paths:
                paths = smooth_paths(paths, window_size=smoothing_window)
            if enable_remove_overlaps:
                paths = merge_redundant_paths(
                    paths,
                    overlap_threshold=overlap_threshold,
                    distance_threshold=overlap_distance_threshold,
                    redundancy_action=redundancy_action,
                    mask_smoothing_size=mask_smoothing_size,
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
        neuron_basename = Path(neuron_name).stem

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
    "merge_redundant_paths",
    "process_results",
    "write_processed_swc",
]
