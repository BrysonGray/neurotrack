"""Shared post-processing flow for inference outputs."""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.spatial import KDTree

from neurotrack.data import save, tree


def remove_short_paths(paths: List[np.ndarray], min_length: float) -> List[np.ndarray]:
    filtered_paths: List[np.ndarray] = []

    for path in paths:
        if isinstance(path, torch.Tensor):
            path = path.cpu().numpy()

        if len(path) < 2:
            continue

        deltas = np.diff(path, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        total_length = np.sum(distances)

        if total_length >= min_length:
            filtered_paths.append(path)

    n_removed = len(paths) - len(filtered_paths)
    print(f"    Removed {n_removed} paths shorter than {min_length:.1f} units")

    return filtered_paths


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


def merge_redundant_paths(
    paths: List[np.ndarray],
    overlap_threshold: float = 0.8,
    distance_threshold: float = 2.0,
) -> List[np.ndarray]:
    merged_paths = []
    for path in paths:
        if isinstance(path, torch.Tensor):
            merged_paths.append(path.cpu().numpy())
        else:
            merged_paths.append(np.array(path))

    n_merged = 0
    changed = True

    while changed:
        changed = False
        n_paths = len(merged_paths)
        connectivity_graph = _build_connectivity_graph(merged_paths)
        paths_to_remove = set()

        indexed_paths = [(i, len(merged_paths[i]), merged_paths[i]) for i in range(n_paths)]
        indexed_paths.sort(key=lambda item: item[1])

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
                merged_paths[j]
                for j, (_, other_len, _) in enumerate(indexed_paths)
                if j != idx and j not in paths_to_remove and other_len > path_len
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

    print(f"    Merged {n_merged} redundant paths (threshold={overlap_threshold:.2f})")

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
    min_branch_length = float(params.get("min_branch_length", 5.0))
    resampling_step_size = float(params.get("resampling_step_size", 4.0))
    smoothing_window = int(params.get("smoothing_window", 5))
    overlap_threshold = float(params.get("overlap_threshold", 0.5))
    overlap_distance_threshold = float(params.get("overlap_distance_threshold", 1.0))

    processed_results: List[Dict[str, Any]] = []
    for result in results:
        neuron_name = result.get("neuron_name", "unknown")
        raw_paths = result.get("paths", [])
        print(f"Processing neuron '{neuron_name}'\n\
              Params\n\
              ------\n\
              min_branch_length: {min_branch_length}\n\
              resampling_step_size: {resampling_step_size}\n\
              smoothing_window: {smoothing_window}\n\
              overlap_threshold: {overlap_threshold}\n\
              overlap_distance_threshold: {overlap_distance_threshold}\n")
        try:
            paths_as_tensors = [_path_to_tensor(path) for path in raw_paths]
            sections = tree.restructure_neuron_tree(paths_as_tensors, input_type="paths")
            paths = [
                section.detach().cpu().numpy() if isinstance(section, torch.Tensor) else np.asarray(section)
                for section in sections.values()
            ]

            paths = remove_short_paths(paths, min_length=min_branch_length)
            paths = tree.resample_tree(paths, step_size=resampling_step_size)
            paths = smooth_paths(paths, window_size=smoothing_window)
            paths = merge_redundant_paths(
                paths,
                overlap_threshold=overlap_threshold,
                distance_threshold=overlap_distance_threshold,
            )

            post_paths = [
                torch.from_numpy(path.astype(np.float32))
                if isinstance(path, np.ndarray)
                else path
                for path in paths
                if len(path) > 1
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
    swc_out_dir = run_out_dir / "processed_swc"
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

        swc_path = swc_out_dir / f"{neuron_basename}_reconstructed.swc"
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
    "remove_short_paths",
    "smooth_paths",
    "merge_redundant_paths",
    "process_results",
    "write_processed_swc",
]
