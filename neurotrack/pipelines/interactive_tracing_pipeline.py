"""Pipeline orchestrator for interactively selecting, tracing, post-processing,
and evaluating neuron reconstructions.

Developer note: prediction tracing is rebuilt only from the editable prediction
graph plus the current seeds. Reference post-processing stays isolated so
reference edits can be evaluated without mutating prediction trace state.
"""

from __future__ import annotations

import json
import importlib
import threading
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile as tf
import torch

from neurotrack.data import NeuronPatchDataset
from neurotrack.data.image import Image
from neurotrack.data import loading as data_loading
from neurotrack.data import save as data_save
from neurotrack.data.seed_io import load_seeds_json, save_seeds_json
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.evaluation.metrics import evaluate_reconstruction
from neurotrack.core.pipeline_config import PostprocessConfig, flexible_image_key_lookup
from neurotrack.inference.postprocess import process_results
from neurotrack.inference.runtime import load_models
from neurotrack.inference.tracing import trace_image as sac_trace_image
from neurotrack.visualization.editor_state import AnnotationGraph
from neurotrack.evaluation.io import (
    compute_pipeline_summary,
    upsert_evaluation_results_csv,
)
from neurotrack.visualization.ortho_viewer import (
    interactive_seed_selection_session,
    prompt_select_model_weights,
    prompt_save_json_path,
    prompt_select_directory,
    prompt_seed_session_paths,
)


def _discover_images(image_dir: Path):
    image_paths = sorted([*image_dir.rglob("*.tif"), *image_dir.rglob("*.tiff")])
    return [p for p in image_paths if p.is_file()]


def _load_optional_session_config(config_path: Optional[str]) -> Dict[str, Optional[str]]:
    if config_path is None:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Seed-selection config must be a JSON object.")
    return payload


def _output_stem(name_or_path: str) -> str:
    key = str(name_or_path).strip()
    if not key:
        return "unknown"

    path = Path(key)
    if path.parent == Path("."):
        return path.name

    parts = [part for part in path.parts if part not in ("", ".")]
    if len(parts) == 1:
        return parts[0]

    parent_key = "__".join(parts[:-1])
    leaf_key = Path(parts[-1]).stem
    return f"{parent_key}__{leaf_key}"



def _normalize_seed_array(seed_array: np.ndarray, shape: tuple[int, int, int]) -> List[List[float]]:
    arr = np.asarray(seed_array, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Seeds must be an array with shape (N, 3) in (z, y, x) order.")
    arr[:, 0] = np.clip(arr[:, 0], 0, shape[0] - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, shape[1] - 1)
    arr[:, 2] = np.clip(arr[:, 2], 0, shape[2] - 1)
    return arr.tolist()


def _format_eval_value(value: Any, decimals: int = 2) -> str:
    """Format evaluation values for the interactive report."""
    if value is None:
        return "N/A"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        formatted = [_format_eval_value(v, decimals=decimals) for v in value]
        open_bracket, close_bracket = ("(", ")") if isinstance(value, tuple) else ("[", "]")
        return f"{open_bracket}{', '.join(formatted)}{close_bracket}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        if not np.isfinite(value_f):
            return "N/A"
        return f"{value_f:.{decimals}f}"
    return str(value)


def _format_eval_report(image_key: str, result: Dict) -> str:
    """Format evaluation metrics as a human-readable multi-line string."""
    sep = "-" * 44
    lines = [
        f"Evaluation: {image_key}",
        sep,
        f"  Bidirectional Distance:    {_format_eval_value(result.get('bidirectional_distance'))}",
        f"  Directed Div pred\u2192gt:  {_format_eval_value(result.get('directed_div_pred_to_gt'))}",
        f"  Directed Div gt\u2192pred:  {_format_eval_value(result.get('directed_div_gt_to_pred'))}",
        f"  Precision:                {_format_eval_value(result.get('precision'))}",
        f"  Coverage:                 {_format_eval_value(result.get('coverage'))}",
        f"  Endpoint Loc Error:       {_format_eval_value(result.get('endpoint_localization_error'))}",
        f"  Endpoint Count Error:     {_format_eval_value(result.get('endpoint_count_error'), decimals=0)}",
        f"  Branchpoint Loc Error:    {_format_eval_value(result.get('branchpoint_localization_error'))}",
        f"  Branchpoint Count Error:  {_format_eval_value(result.get('branchpoint_count_error'), decimals=0)}",
    ]

    # Optional legacy/extended fields: show only when provided.
    if "n_substantial_pred_to_gt" in result:
        lines.append(
            f"  Substantial pred\u2192gt:     {_format_eval_value(result.get('n_substantial_pred_to_gt'), decimals=0)}"
        )
    if "n_substantial_gt_to_pred" in result:
        lines.append(
            f"  Substantial gt\u2192pred:     {_format_eval_value(result.get('n_substantial_gt_to_pred'), decimals=0)}"
        )
    if "n_points_pred" in result or "n_points_gt" in result:
        lines.append(
            f"  Pred Nodes: {_format_eval_value(result.get('n_points_pred'), decimals=0)}"
            f"  |  GT Nodes: {_format_eval_value(result.get('n_points_gt'), decimals=0)}"
        )

    l_measure_pairs = [
        ("num_bifurcations", "Num Bifurcations"),
        ("num_branches", "Num Branches"),
        ("num_tips", "Num Tips"),
        ("span", "Span"),
        ("total_length", "Total Length"),
        ("max_euclidean_distance", "Max Euclidean Distance"),
        ("max_path_distance", "Max Path Distance"),
        ("max_branch_order", "Max Branch Order"),
        ("average_contraction", "Average Contraction"),
        ("average_fragmentation", "Average Fragmentation"),
        ("bifurcation_angle_local", "Bifurcation Angle Local"),
        ("bifurcation_angle_remote", "Bifurcation Angle Remote"),
    ]
    pairwise_l_measure_keys = [
        "different_structure_average",
        "percentage_different_structure_pred_to_gt",
        "percentage_different_structure_gt_to_pred",
        "percent_different_structure_average",
    ]

    has_l_measures = any(
        f"{prefix}_pred" in result or f"{prefix}_gt" in result
        for prefix, _ in l_measure_pairs
    ) or any(k in result for k in pairwise_l_measure_keys)

    if has_l_measures:
        lines.append(sep)
        lines.append("L-Measures")
        for prefix, label in l_measure_pairs:
            pred_key = f"{prefix}_pred"
            gt_key = f"{prefix}_gt"
            if pred_key in result or gt_key in result:
                lines.append(
                    f"  {label} (pred/gt): "
                    f"{_format_eval_value(result.get(pred_key))}"
                    f" / {_format_eval_value(result.get(gt_key))}"
                )
        if "different_structure_average" in result:
            lines.append(
                f"  Different Structure Avg: {_format_eval_value(result.get('different_structure_average'))}"
            )
        if "percentage_different_structure_pred_to_gt" in result:
            lines.append(
                "  % Different Structure pred→gt: "
                f"{_format_eval_value(result.get('percentage_different_structure_pred_to_gt'))}"
            )
        if "percentage_different_structure_gt_to_pred" in result:
            lines.append(
                "  % Different Structure gt→pred: "
                f"{_format_eval_value(result.get('percentage_different_structure_gt_to_pred'))}"
            )
        if "percent_different_structure_average" in result:
            lines.append(
                "  % Different Structure Avg: "
                f"{_format_eval_value(result.get('percent_different_structure_average'))}"
            )

    if "gt_file" in result:
        lines.append(f"  Reference File: {result['gt_file']}")

    return "\n".join(lines)


def _coerce_paths_xyz(paths: List[List[List[float]]]) -> List[np.ndarray]:
    coerced: List[np.ndarray] = []
    for path in paths:
        path_np = np.asarray(path, dtype=np.float32)
        if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] > 0:
            coerced.append(path_np[:, :3].copy())
    return coerced


def _coord_key_xyz(point_xyz: np.ndarray, decimals: int = 5) -> Tuple[float, float, float]:
    """Build a stable hashable XYZ key for node lookup across path/SWC conversions."""
    point = np.asarray(point_xyz, dtype=np.float32).reshape(-1)
    if point.shape[0] < 3:
        raise ValueError("XYZ point must contain at least three values.")
    rounded = np.round(point[:3].astype(np.float64), decimals=decimals)
    return float(rounded[0]), float(rounded[1]), float(rounded[2])


def _find_closest_node_xyz(
    paths_xyz: List[np.ndarray],
    query_xyz: np.ndarray,
) -> Optional[Tuple[int, int, np.ndarray]]:
    best_dist_sq = float("inf")
    best_path_idx: Optional[int] = None
    best_node_idx: Optional[int] = None
    best_node_xyz: Optional[np.ndarray] = None

    query = np.asarray(query_xyz, dtype=np.float32).reshape(-1)
    if query.shape[0] < 3:
        return None
    query = query[:3]

    for path_idx, path in enumerate(paths_xyz):
        if path.shape[0] == 0:
            continue
        deltas = path - query[None, :]
        dist_sq = np.sum(deltas * deltas, axis=1)
        node_idx = int(np.argmin(dist_sq))
        node_dist_sq = float(dist_sq[node_idx])
        if node_dist_sq < best_dist_sq:
            best_dist_sq = node_dist_sq
            best_path_idx = path_idx
            best_node_idx = node_idx
            best_node_xyz = path[node_idx].copy()

    if best_path_idx is None or best_node_idx is None or best_node_xyz is None:
        return None
    return best_path_idx, best_node_idx, best_node_xyz


def _trim_paths_downstream(
    paths_xyz: List[np.ndarray],
    selected_path_idx: int,
    selected_node_idx: int,
    selected_node_xyz: np.ndarray,
    atol: float = 1e-3,
) -> List[np.ndarray]:
    if selected_path_idx < 0 or selected_path_idx >= len(paths_xyz):
        return [path.copy() for path in paths_xyz]

    selected_path = paths_xyz[selected_path_idx]
    if selected_path.shape[0] == 0:
        return [path.copy() for path in paths_xyz]

    clipped_node_idx = int(np.clip(selected_node_idx, 0, selected_path.shape[0] - 1))
    selected_node = np.asarray(selected_node_xyz, dtype=np.float32)[:3]
    if clipped_node_idx >= 0 and clipped_node_idx < selected_path.shape[0]:
        selected_node = selected_path[clipped_node_idx].astype(np.float32, copy=True)

    swc_list = data_save.paths_to_swc(paths_xyz)
    if len(swc_list) == 0:
        return [path.copy() for path in paths_xyz]

    coord_to_node_id: Dict[Tuple[float, float, float], int] = {}
    for row in swc_list:
        node_key = _coord_key_xyz(np.asarray(row[2:5], dtype=np.float32))
        coord_to_node_id[node_key] = int(row[0])

    selected_node_id = coord_to_node_id.get(_coord_key_xyz(selected_node))
    if selected_node_id is None:
        # Fallback for tiny float drift if keying fails.
        swc_xyz = np.asarray([row[2:5] for row in swc_list], dtype=np.float32)
        if swc_xyz.size == 0:
            return [path.copy() for path in paths_xyz]
        dist_sq = np.sum((swc_xyz - selected_node[None, :]) ** 2, axis=1)
        best_idx = int(np.argmin(dist_sq))
        if float(dist_sq[best_idx]) > float(atol) * float(atol):
            return [path.copy() for path in paths_xyz]
        selected_node_id = int(swc_list[best_idx][0])

    downstream_ids = data_loading.get_downstream_swc_node_ids(
        swc_list=swc_list,
        start_node_id=int(selected_node_id),
        include_start=False,
    )
    if len(downstream_ids) == 0:
        return [path.copy() for path in paths_xyz]

    output: List[np.ndarray] = []
    for path in paths_xyz:
        kept_nodes: List[np.ndarray] = []
        for node_xyz in path:
            node_id = coord_to_node_id.get(_coord_key_xyz(node_xyz))
            if node_id is not None and node_id in downstream_ids:
                continue
            kept_nodes.append(np.asarray(node_xyz, dtype=np.float32)[:3].copy())
        if len(kept_nodes) > 0:
            output.append(np.asarray(kept_nodes, dtype=np.float32))
    return output


def _draw_mask_from_paths_xyz(
    paths_xyz: List[np.ndarray],
    shape_zyx: Tuple[int, int, int],
    width: float,
    mask_dtype: np.dtype,
) -> np.ndarray:
    shape = tuple(int(v) for v in shape_zyx)
    torch_dtype = torch.uint8 if np.dtype(mask_dtype) == np.uint8 else torch.float32
    mask_image = Image(torch.zeros((1,) + shape, dtype=torch_dtype))

    for path in paths_xyz:
        if path.ndim != 2 or path.shape[1] < 3 or path.shape[0] < 2:
            continue
        # Reversing columns with ::-1 creates a negative-stride view; torch.as_tensor
        # cannot consume it directly, so materialize as a contiguous float32 array.
        path_zyx = np.ascontiguousarray(path[:, ::-1], dtype=np.float32)
        if path_zyx.shape[0] > 2:
            # Drop consecutive duplicate/near-duplicate vertices to reduce no-op draw calls.
            deltas = np.abs(np.diff(path_zyx, axis=0))
            keep_mask = np.ones((path_zyx.shape[0],), dtype=bool)
            keep_mask[1:] = np.any(deltas > 1e-5, axis=1)
            path_zyx = path_zyx[keep_mask]
        if path_zyx.shape[0] < 2:
            continue

        path_zyx_t = torch.from_numpy(path_zyx)
        for idx in range(path_zyx_t.shape[0] - 1):
            segment = path_zyx_t[idx: idx + 2]
            mask_image.draw_line_segment(segment, width=width, channel=0, mask=True)

    return mask_image.data[0].detach().cpu().numpy()


class _PredictionGraphInitializationAdapter:
    """Convert an editable prediction graph into runtime initialization inputs."""

    def __init__(self, prediction_paths: Optional[List[List[List[float]]]]) -> None:
        self._prediction_graph = AnnotationGraph.from_paths(prediction_paths)

    def build_initial_path_mask(
        self,
        image_shape_zyx: Tuple[int, int, int],
        width: float,
    ) -> Optional[np.ndarray]:
        paths_xyz = self._prediction_graph.to_paths()
        if len(paths_xyz) == 0:
            return None
        return _draw_mask_from_paths_xyz(paths_xyz, image_shape_zyx, width=width, mask_dtype=np.uint8)


class _TraceRuntime:
    """Stateful tracer for current image set used by GUI session callbacks."""

    def __init__(self, trace_params: Dict[str, object]):
        self.trace_params = trace_params
        self._lock = threading.Lock()
        self._actor, self._q_net = load_models(trace_params)

        rng_seed = int(trace_params.get("rng_seed", 0))
        self._dataset = NeuronPatchDataset(
            img_dir=str(trace_params["img_dir"]),
            swc_dir=trace_params.get("swc_dir", None),
            alpha=1.0,
            step_width=float(trace_params.get("step_width", 4.0)),
            rng=np.random.default_rng(rng_seed),
            crop_patches=False,
            patches_per_image=1,
            seed_points_by_image={},
            soma_sample_radius=float(trace_params.get("soma_sample_radius", 0.0)),
            random_offset=float(trace_params.get("random_offset", 0.0)),
            seed_jitter_count=int(trace_params.get("seed_jitter_count", 0)),
            seed_jitter_radius=float(trace_params.get("seed_jitter_radius", 0.0)),
            seed_jitter_weight_strategy=str(
                trace_params.get("seed_jitter_weight_strategy", trace_params.get("weight_strategy", "uniform"))
            ),
            seed_jitter_nonce=int(trace_params.get("seed_jitter_nonce", 0)),
            inference_mode=True,
        )

        self._env = NeuronTrackingEnvironment(
            dataset=self._dataset,
            radius=17,
            step_width=float(trace_params.get("step_width", 4.0)),
            stall_threshold=float(trace_params.get("stall_threshold", 1.0)),
            max_len=int(trace_params.get("max_len", 9999999)),
            max_paths=int(trace_params.get("max_paths", 9999999)),
            branching=bool(trace_params.get("branching", True)),
            repeat_starts=bool(trace_params.get("repeat_starts", False)),
            start_idx=0,
            inference_mode=True,
        )

        self._dataset_keys_by_index: List[str] = [
            path.relative_to(self._dataset.img_dir).as_posix() for path in self._dataset.img_files
        ]
        self._dataset_index_by_key: Dict[str, int] = {
            key: idx for idx, key in enumerate(self._dataset_keys_by_index)
        }

    def _resolve_dataset_index(self, image_index: int, image_relative_key: str) -> Tuple[int, str]:
        """Resolve dataset index from a relative image key, falling back to provided index."""
        key = str(image_relative_key)
        exact_idx = self._dataset_index_by_key.get(key)
        if exact_idx is not None:
            return exact_idx, key

        # Try tolerant matching when keys differ by root/suffix or extension details.
        key_match_map = {candidate: candidate for candidate in self._dataset_index_by_key.keys()}
        matched_key = flexible_image_key_lookup(key_match_map, key, default=None)
        if isinstance(matched_key, str):
            matched_idx = self._dataset_index_by_key.get(matched_key)
            if matched_idx is not None:
                return matched_idx, matched_key

        fallback_idx = int(image_index) % len(self._dataset.img_files)
        fallback_key = self._dataset_keys_by_index[fallback_idx]
        return fallback_idx, fallback_key

    def trace_image(
        self,
        image_index: int,
        image_relative_key: str,
        seed_rows: List[List[float]],
        prediction_paths: Optional[List[List[List[float]]]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, object]:
        with self._lock:
            dataset_index, resolved_key = self._resolve_dataset_index(
                image_index=image_index,
                image_relative_key=image_relative_key,
            )
            normalized_seed_rows = [[float(coord) for coord in row] for row in seed_rows]
            self._dataset.seed_points_by_image[image_relative_key] = normalized_seed_rows
            self._dataset.seed_points_by_image[resolved_key] = normalized_seed_rows

            initial_path_mask = None
            if prediction_paths is not None:
                sample = self._dataset[dataset_index]
                sample_image = sample.get("image", None) if isinstance(sample, dict) else None
                if sample_image is not None:
                    image_shape_zyx = tuple(int(v) for v in np.asarray(sample_image).shape[-3:])
                    initial_path_mask = _PredictionGraphInitializationAdapter(prediction_paths).build_initial_path_mask(
                        image_shape_zyx=image_shape_zyx,
                        width=float(self.trace_params.get("step_width", 4.0)),
                    )

            result = sac_trace_image(
                env=self._env,
                actor=self._actor,
                dataset_idx=dataset_index,
                Q_net=self._q_net,
                n_trials=int(self.trace_params.get("n_trials", 1)),
                show=False,
                show_live=False,
                stochastic=bool(self.trace_params.get("stochastic_actions", False)),
                cancel_event=cancel_event,
                initial_path_mask=initial_path_mask,
                retry_on_no_long_paths=bool(self.trace_params.get("retry_on_no_long_paths", True)),
                retry_initial_radius=float(self.trace_params.get("retry_initial_radius", 5.0)),
                retry_radius_step=float(self.trace_params.get("retry_radius_step", 5.0)),
                retry_max_radius=float(self.trace_params.get("retry_max_radius", 50.0)),
                retry_attempts_per_radius=int(self.trace_params.get("retry_attempts_per_radius", 50)),
            )
            labeled_neuron = result.get("labeled_neuron", None)
            if labeled_neuron is not None and hasattr(labeled_neuron, "detach"):
                labeled_neuron = labeled_neuron.detach().cpu().numpy()
            elif labeled_neuron is not None:
                labeled_neuron = np.asarray(labeled_neuron)
            return {
                "paths": result["paths"],
                "labeled_neuron": labeled_neuron,
                "timing_ms": result.get("timing_ms", None),
            }

    def get_effective_seed_points(
        self,
        image_index: int,
        image_relative_key: str,
        seed_rows: List[List[float]],
    ) -> Optional[np.ndarray]:
        """Return the exact seed points consumed by the dataset for this image/index."""
        with self._lock:
            dataset_index, resolved_key = self._resolve_dataset_index(
                image_index=image_index,
                image_relative_key=image_relative_key,
            )
            normalized_seed_rows = [[float(coord) for coord in row] for row in seed_rows]
            self._dataset.seed_points_by_image[image_relative_key] = normalized_seed_rows
            self._dataset.seed_points_by_image[resolved_key] = normalized_seed_rows
            sample = self._dataset[dataset_index]
            seed_points = sample.get("seed_points", None)
            if seed_points is None:
                return None
            if hasattr(seed_points, "detach"):
                return seed_points.detach().cpu().numpy()
            return np.asarray(seed_points, dtype=np.float32)


class _TraceSessionManager:
    """Thread-safe trace orchestration for per-image and background trace-all actions."""

    def __init__(
        self,
        image_paths: List[Path],
        image_root: Path,
        trace_params: Optional[Dict[str, object]],
        postprocess_config: Optional[PostprocessConfig] = None,
        report_stem: Optional[str] = None,
    ) -> None:
        self.image_paths = image_paths
        self.image_root = image_root
        self.trace_params: Dict[str, object] = {} if trace_params is None else dict(trace_params)
        self.postprocess_config: PostprocessConfig = postprocess_config or PostprocessConfig()
        self._runtime = None
        self.enabled = False

        self.trace_results_by_key: Dict[str, List[List[List[float]]]] = {}
        self.traced_seed_rows_by_key: Dict[str, List[List[float]]] = {}
        self._trace_output_dir: Optional[Path] = None
        self._temp_dir = tempfile.TemporaryDirectory(prefix="neurotrack_trace_session_")
        self._temp_root = Path(self._temp_dir.name)
        self._message = ""
        self._token = 0
        self._overlay_token = 0
        self._postprocess_token = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None
        self._state_lock = threading.Lock()
        self._progress_completed = 0
        self._progress_total = 0
        self._trace_timing_by_key: Dict[str, Dict[str, object]] = {}

        # post-processing and evaluation state
        self.postprocess_results_by_key: Dict[str, Dict[str, object]] = {}
        self.reference_postprocess_results_by_key: Dict[str, Dict[str, object]] = {}
        self.eval_results_by_key: Dict[str, Dict[str, object]] = {}
        self._pre_postprocess_trace_cache_by_key: Dict[str, List[List[List[float]]]] = {}
        self._pre_postprocess_reference_cache_by_key: Dict[str, List[List[float]]] = {}
        self.filtered_swc_by_key: Dict[str, List[List[float]]] = {}
        self._gt_swc_cache_by_key: Dict[str, List[List[float]]] = {}
        self._gt_swc_dir: Optional[Path] = None
        self._gt_swc_files_by_stem: Optional[Dict[str, Path]] = None
        if self.trace_params.get("swc_dir"):
            self._gt_swc_dir = Path(str(self.trace_params["swc_dir"]))
        self._postprocess_output_dir: Optional[Path] = None
        self._eval_output_dir: Optional[Path] = None
        self._filtered_swc_output_dir: Optional[Path] = None
        self._report_stem = str(report_stem).strip() if report_stem is not None else image_root.name
        if len(self._report_stem) == 0:
            self._report_stem = "session"
        _ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_stem = f"{self._report_stem}_{_ts}"

        if self.trace_params.get("sac_weights"):
            self.set_model_weights_path(str(self.trace_params["sac_weights"]))

    def _set_state(self, message: str, increment_token: bool = False):
        with self._state_lock:
            self._message = message
            if increment_token:
                self._token += 1

    def _increment_overlay_token(self) -> None:
        with self._state_lock:
            self._overlay_token += 1

    def _increment_postprocess_token(self) -> None:
        with self._state_lock:
            self._postprocess_token += 1

    def close(self):
        self.cancel_trace_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._temp_dir.cleanup()

    def _write_temp_trace(self, image_key: str, paths: List[List[List[float]]]):
        out_path = self._temp_root / Path(image_key).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coordinate_order": "xyz",
            "image_key": image_key,
            "paths": paths,
        }
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _ensure_output_dir(self, default_dir: Path) -> Path:
        if self._trace_output_dir is None:
            selected_dir = prompt_select_directory(default_path=str(default_dir))
            if selected_dir is None:
                raise ValueError("Trace output directory is required before saving traces.")
            self._trace_output_dir = Path(selected_dir)
        self._trace_output_dir.mkdir(parents=True, exist_ok=True)
        return self._trace_output_dir

    def save_trace(self, image_key: str, default_dir: Path):
        if image_key not in self.trace_results_by_key:
            self._set_state(f"No trace available to save for {image_key}.", increment_token=True)
            return
        out_dir = self._ensure_output_dir(default_dir=default_dir)
        out_dir = out_dir / "reconstructions"
        out_dir.mkdir(parents=True, exist_ok=True)
        swc_list = data_save.paths_to_swc(_coerce_paths_xyz(self.trace_results_by_key[image_key]))
        if not swc_list:
            self._set_state("Empty SWC — nothing to save.", increment_token=True)
            return
        dst = out_dir / f"{_output_stem(image_key)}.swc"
        dst.parent.mkdir(parents=True, exist_ok=True)
        data_save.write_swc(swc_list, str(dst))
        self._set_state(f"Saved trace: {dst}", increment_token=True)

    def save_all_traces(self, default_dir: Path):
        if len(self.trace_results_by_key) == 0:
            self._set_state("No traces available to save.", increment_token=True)
            return
        out_dir = self._ensure_output_dir(default_dir=default_dir)
        out_dir = out_dir / "reconstructions"
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_key, paths in self.trace_results_by_key.items():
            swc_list = data_save.paths_to_swc(_coerce_paths_xyz(paths))
            if not swc_list:
                continue
            dst = out_dir / f"{_output_stem(image_key)}.swc"
            dst.parent.mkdir(parents=True, exist_ok=True)
            data_save.write_swc(swc_list, str(dst))
        self._set_state(f"Saved all traces to: {out_dir}", increment_token=True)

    def _clear_derived_results(self, image_key: str) -> None:
        self.postprocess_results_by_key.pop(image_key, None)
        self.eval_results_by_key.pop(image_key, None)
        self._pre_postprocess_trace_cache_by_key.pop(image_key, None)

    @staticmethod
    def _normalize_paths_payload(paths: List[List[List[float]]]) -> List[List[List[float]]]:
        normalized: List[List[List[float]]] = []
        for path in _coerce_paths_xyz(paths):
            normalized.append(path.tolist())
        return normalized

    @staticmethod
    def _normalize_seed_rows_payload(seed_rows: List[List[float]]) -> List[List[float]]:
        normalized: List[List[float]] = []
        for row in seed_rows or []:
            row_np = np.asarray(row, dtype=np.float32).reshape(-1)
            if row_np.shape[0] < 3:
                continue
            normalized.append([float(row_np[0]), float(row_np[1]), float(row_np[2])])
        return normalized

    @staticmethod
    def _seed_row_key(seed_row: List[float], decimals: int = 4) -> Tuple[float, float, float]:
        row_np = np.asarray(seed_row, dtype=np.float32).reshape(-1)
        rounded = np.round(row_np[:3].astype(np.float64), decimals=decimals)
        return float(rounded[0]), float(rounded[1]), float(rounded[2])

    def _pending_seed_rows(self, image_key: str, seed_rows: List[List[float]]) -> List[List[float]]:
        normalized_rows = self._normalize_seed_rows_payload(seed_rows)
        consumed_rows = self.traced_seed_rows_by_key.get(image_key, [])
        if not consumed_rows:
            return normalized_rows
        consumed_keys = {self._seed_row_key(row) for row in consumed_rows}
        return [row for row in normalized_rows if self._seed_row_key(row) not in consumed_keys]

    def _mark_seed_rows_traced(self, image_key: str, seed_rows: List[List[float]]) -> None:
        normalized_rows = self._normalize_seed_rows_payload(seed_rows)
        if not normalized_rows:
            return
        existing_rows = self._normalize_seed_rows_payload(self.traced_seed_rows_by_key.get(image_key, []))
        seen_keys = {self._seed_row_key(row) for row in existing_rows}
        merged_rows = list(existing_rows)
        for row in normalized_rows:
            row_key = self._seed_row_key(row)
            if row_key in seen_keys:
                continue
            merged_rows.append(row)
            seen_keys.add(row_key)
        self.traced_seed_rows_by_key[image_key] = merged_rows

    def _append_trace_paths(self, image_key: str, new_paths: List[List[List[float]]]) -> List[List[List[float]]]:
        existing_paths = self._normalize_paths_payload(self.trace_results_by_key.get(image_key, []))
        appended_paths = existing_paths + self._normalize_paths_payload(new_paths)
        self.trace_results_by_key[image_key] = appended_paths
        return appended_paths

    def discard_trace(self, image_key: str) -> None:
        if image_key not in self.trace_results_by_key:
            self._set_state(f"No trace to discard for {image_key}.", increment_token=True)
            return
        self.trace_results_by_key.pop(image_key, None)
        self.traced_seed_rows_by_key.pop(image_key, None)
        self._clear_derived_results(image_key)
        trace_tmp_path = self._temp_root / Path(image_key).with_suffix(".json")
        if trace_tmp_path.exists():
            trace_tmp_path.unlink(missing_ok=True)
        self._increment_overlay_token()
        self._set_state(f"Discarded current predicted trace for {image_key}.", increment_token=True)

    def undo_postprocess(self, image_key: str, target: str = "prediction") -> Optional[object]:
        target = str(target or "prediction")
        if target == "prediction":
            original = self._pre_postprocess_trace_cache_by_key.get(image_key)
            if original is None:
                self._set_state(f"No cached pre-processed trace to restore for {image_key}.", increment_token=True)
                return None

            restored = self._normalize_paths_payload(original)
            self.trace_results_by_key[image_key] = restored
            self._write_temp_trace(image_key=image_key, paths=restored)
            self.eval_results_by_key.pop(image_key, None)
            self._pre_postprocess_trace_cache_by_key.pop(image_key, None)
            self._increment_overlay_token()
            self._increment_postprocess_token()
            self._set_state(f"Restored original predicted trace for {image_key}.", increment_token=True)
            return restored

        original_rows = self._pre_postprocess_reference_cache_by_key.get(image_key)
        if original_rows is None:
            self._set_state(f"No cached pre-processed annotation to restore for {image_key}.", increment_token=True)
            return None

        restored_rows = [[float(v) for v in row[:7]] for row in original_rows]
        self.filtered_swc_by_key[image_key] = restored_rows
        self.reference_postprocess_results_by_key.pop(image_key, None)
        self._pre_postprocess_reference_cache_by_key.pop(image_key, None)
        self._increment_postprocess_token()
        self._set_state(f"Restored original reference annotation for {image_key}.", increment_token=True)
        return restored_rows

    def _store_trace_timing(self, image_key: str, timing_ms: Optional[object]) -> None:
        if isinstance(timing_ms, dict):
            self._trace_timing_by_key[image_key] = dict(timing_ms)

    @staticmethod
    def _format_timing_summary(timing_ms: Optional[object]) -> str:
        if not isinstance(timing_ms, dict):
            return ""
        total = timing_ms.get("total", None)
        steps = timing_ms.get("steps", None)
        actor_forward = timing_ms.get("actor_forward", None)
        env_step = timing_ms.get("env_step", None)
        get_state = timing_ms.get("get_state", None)
        if total is None or steps is None:
            return ""
        return (
            f" [timing: total={float(total):.1f}ms, steps={int(steps)}, "
            f"actor={float(actor_forward or 0.0):.1f}ms, "
            f"env_step={float(env_step or 0.0):.1f}ms, "
            f"get_state={float(get_state or 0.0):.1f}ms]"
        )

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def get_gt_swc_path(self) -> Optional[str]:
        return None if self._gt_swc_dir is None else str(self._gt_swc_dir)

    def get_postprocess_output_dir(self) -> Optional[str]:
        return None if self._postprocess_output_dir is None else str(self._postprocess_output_dir)

    def get_eval_output_dir(self) -> Optional[str]:
        return None if self._eval_output_dir is None else str(self._eval_output_dir)

    def set_postprocess_output_dir(self, path: str) -> Optional[str]:
        """Set the directory where post-processed SWC files are written."""
        if not path or not str(path).strip():
            return None
        p = Path(path) / self._run_stem
        p.mkdir(parents=True, exist_ok=True)
        self._postprocess_output_dir = p
        self._set_state(f"Post-process output set to: {p}", increment_token=True)
        return str(p)

    def clear_postprocess_output_dir(self) -> Optional[str]:
        self._postprocess_output_dir = None
        self._set_state("Post-process output cleared.", increment_token=True)
        return None

    def set_eval_output_dir(self, path: str) -> Optional[str]:
        """Set the directory where evaluation reports are written."""
        if not path or not str(path).strip():
            return None
        p = Path(path) / self._run_stem
        p.mkdir(parents=True, exist_ok=True)
        self._eval_output_dir = p
        self._set_state(f"Eval output set to: {p}", increment_token=True)
        return str(p)

    def clear_eval_output_dir(self) -> Optional[str]:
        self._eval_output_dir = None
        self._set_state("Eval output cleared.", increment_token=True)
        return None

    def update_postprocess_config(self, overrides: Dict[str, object]) -> None:
        """Update postprocess/eval config parameters from the UI."""
        if "enable_length_filter" in overrides:
            self.postprocess_config.enable_length_filter = bool(overrides["enable_length_filter"])
        if "min_branch_length" in overrides:
            self.postprocess_config.min_branch_length = float(overrides["min_branch_length"])
        if "max_branch_length" in overrides:
            self.postprocess_config.max_branch_length = float(overrides["max_branch_length"])
        if "enable_resample" in overrides:
            self.postprocess_config.enable_resample = bool(overrides["enable_resample"])
        if "resampling_step_size" in overrides:
            self.postprocess_config.resampling_step_size = float(overrides["resampling_step_size"])
        if "enable_smooth_paths" in overrides:
            self.postprocess_config.enable_smooth_paths = bool(overrides["enable_smooth_paths"])
        if "smoothing_window" in overrides:
            self.postprocess_config.smoothing_window = int(overrides["smoothing_window"])
        if "enable_merge" in overrides:
            self.postprocess_config.enable_merge = bool(overrides["enable_merge"])
        if "merge_threshold" in overrides:
            self.postprocess_config.merge_threshold = float(overrides["merge_threshold"])
        if "confidence_threshold" in overrides:
            self.postprocess_config.confidence_threshold = int(overrides["confidence_threshold"])
        if "mask_smoothing_size" in overrides:
            self.postprocess_config.mask_smoothing_size = int(overrides["mask_smoothing_size"])
        if "merge_timeout_seconds" in overrides:
            self.postprocess_config.merge_timeout_seconds = float(overrides["merge_timeout_seconds"])
        if "distance_threshold" in overrides:
            self.postprocess_config.distance_threshold = float(overrides["distance_threshold"])

    def get_scales_path(self) -> Optional[str]:
        return self.postprocess_config.scales_path

    def set_scales_path(self, path: str) -> Optional[str]:
        """Update the scales JSON path in postprocess_config and invalidate the cache."""
        if not path or not str(path).strip():
            return None
        p = Path(path)
        if not p.exists():
            self._set_state(f"Scales JSON not found: {p}", increment_token=True)
            return None
        self.postprocess_config.scales_path = str(p)
        self.postprocess_config._scales_cache = None  # invalidate cache
        self._set_state(f"Scales JSON set: {p}", increment_token=True)
        return str(p)

    def clear_scales_path(self) -> Optional[str]:
        self.postprocess_config.scales_path = None
        self.postprocess_config._scales_cache = None
        self._set_state("Scales JSON path cleared.", increment_token=True)
        return None

    def set_gt_swc_dir(self, path: str) -> Optional[str]:
        if not path or not str(path).strip():
            return None
        p = Path(path)
        if not p.exists():
            self._set_state(f"GT SWC directory not found: {p}", increment_token=True)
            return None
        self._gt_swc_dir = p
        self._gt_swc_cache_by_key = {}
        self._gt_swc_files_by_stem = None
        self._set_state(f"GT SWC directory set: {p}", increment_token=True)
        return str(p)

    def clear_gt_swc_dir(self) -> Optional[str]:
        self._gt_swc_dir = None
        self._gt_swc_cache_by_key = {}
        self._gt_swc_files_by_stem = None
        self._set_state("GT SWC directory cleared.", increment_token=True)
        return None

    def _get_gt_swc_files_by_stem(self) -> Dict[str, Path]:
        if self._gt_swc_files_by_stem is None:
            if self._gt_swc_dir is None:
                self._gt_swc_files_by_stem = {}
            else:
                self._gt_swc_files_by_stem = {
                    f.stem: f for f in sorted(self._gt_swc_dir.rglob("*.swc"))
                }
        return self._gt_swc_files_by_stem

    def _resolve_gt_swc_file_for_image(self, image_key: str) -> Optional[Path]:
        if self._gt_swc_dir is None:
            return None
        gt_files_by_stem = self._get_gt_swc_files_by_stem()
        neuron_stem = Path(image_key).stem
        gt_file: Optional[Path] = flexible_image_key_lookup(gt_files_by_stem, neuron_stem, default=None)
        if gt_file is None:
            for stem, candidate in gt_files_by_stem.items():
                if neuron_stem in stem or stem in neuron_stem:
                    gt_file = candidate
                    break
        return gt_file

    def get_tree_swc_rows(self, image_key: str) -> List[List[float]]:
        if image_key in self.filtered_swc_by_key:
            return [list(row) for row in self.filtered_swc_by_key[image_key]]
        if image_key in self._gt_swc_cache_by_key:
            return [list(row) for row in self._gt_swc_cache_by_key[image_key]]
        gt_file = self._resolve_gt_swc_file_for_image(image_key)
        if gt_file is None:
            return []
        try:
            rows = data_loading.swc(str(gt_file), verbose=False)
        except Exception:
            rows = []
        normalized = [[float(v) for v in row[:7]] for row in rows if len(row) >= 7]
        self._gt_swc_cache_by_key[image_key] = normalized
        return [list(row) for row in normalized]

    def set_filtered_swc_rows(self, image_key: str, swc_rows: List[List[float]]) -> None:
        self.filtered_swc_by_key[image_key] = [
            [float(v) for v in row[:7]] for row in swc_rows if len(row) >= 7
        ]

    def get_filtered_swc_output_dir(self) -> Optional[str]:
        return None if self._filtered_swc_output_dir is None else str(self._filtered_swc_output_dir)

    def set_filtered_swc_output_dir(self, path: str) -> Optional[str]:
        if not path or not str(path).strip():
            return None
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._filtered_swc_output_dir = p
        self._set_state(f"Filtered SWC output set to: {p}", increment_token=True)
        return str(p)

    def clear_filtered_swc_output_dir(self) -> Optional[str]:
        self._filtered_swc_output_dir = None
        self._set_state("Filtered SWC output cleared.", increment_token=True)
        return None

    def select_filtered_swc_output_dir(self, default_dir: Path) -> Optional[str]:
        selected_dir = prompt_select_directory(default_path=str(default_dir))
        if selected_dir is None:
            return self.get_filtered_swc_output_dir()
        return self.set_filtered_swc_output_dir(selected_dir)

    def save_filtered_swc(self, image_key: str, swc_rows: List[List[float]], default_dir: Path) -> Optional[str]:
        if self._filtered_swc_output_dir is None:
            selected_dir = prompt_select_directory(default_path=str(default_dir))
            if selected_dir is None:
                return self.get_filtered_swc_output_dir()
            self._filtered_swc_output_dir = Path(selected_dir)

        out_dir = self._filtered_swc_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        rel_path = Path(image_key)
        rel_parent = rel_path.parent
        stem = rel_path.stem
        output_path = out_dir / rel_parent / f"{stem}_filtered.swc"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_save.write_swc(swc_rows, str(output_path))
        self.set_filtered_swc_rows(image_key=image_key, swc_rows=swc_rows)
        self._set_state(f"Saved filtered SWC: {output_path}", increment_token=True)
        return str(out_dir)

    def run_postprocess(self, image_key: str, target: str = "prediction") -> Optional[Dict[str, object]]:
        """Post-process the active annotation target for *image_key* and replace it in memory."""
        target = str(target or "prediction")
        if target not in {"prediction", "reference"}:
            self._set_state(f"Unsupported post-process target for {image_key}: {target}", increment_token=True)
            return None

        if target == "prediction":
            if image_key not in self.trace_results_by_key:
                self._set_state(
                    f"No trace to post-process for {image_key}. Trace the image first.",
                    increment_token=True,
                )
                return None
            if image_key not in self._pre_postprocess_trace_cache_by_key:
                self._pre_postprocess_trace_cache_by_key[image_key] = self._normalize_paths_payload(
                    self.trace_results_by_key[image_key]
                )
            raw_paths = self._normalize_paths_payload(self._pre_postprocess_trace_cache_by_key[image_key])
            raw_paths_xyz = _coerce_paths_xyz(raw_paths)
            if len(raw_paths_xyz) == 0:
                self._set_state("Empty prediction — post-processing skipped.", increment_token=True)
                return None
        else:
            if image_key not in self._pre_postprocess_reference_cache_by_key:
                self._pre_postprocess_reference_cache_by_key[image_key] = [
                    [float(v) for v in row[:7]] for row in self.get_tree_swc_rows(image_key)
                ]
            raw_reference_rows = self._pre_postprocess_reference_cache_by_key[image_key]
            raw_paths_xyz = _coerce_paths_xyz(AnnotationGraph.from_swc_rows(raw_reference_rows).to_paths())
            if len(raw_paths_xyz) == 0:
                self._set_state("Empty reference annotation — post-processing skipped.", increment_token=True)
                return None

        raw_result = {
            "neuron_name": image_key,
            "paths": raw_paths_xyz,
        }
        try:
            self._set_state(f"Post-processing {image_key} ({target})...", increment_token=False)
            processed = process_results([raw_result], self.postprocess_config.scaled_params_for_image(image_key))
            if processed:
                result = processed[0]
                processed_paths = self._normalize_paths_payload(result.get("processed_paths", []))
                if len(processed_paths) == 0:
                    self._set_state("Post-processing produced no paths.", increment_token=True)
                    return None

                n = result.get("n_processed_paths", 0)
                if target == "prediction":
                    self.postprocess_results_by_key[image_key] = result
                    self.trace_results_by_key[image_key] = processed_paths
                    self._write_temp_trace(image_key=image_key, paths=processed_paths)
                    self._increment_overlay_token()
                    self._increment_postprocess_token()
                    self.eval_results_by_key.pop(image_key, None)
                    self._set_state(
                        (
                            f"Post-processing complete: {n} paths for {image_key} ({target}). "
                            "Use 'Undo Post-Process' to restore the original trace."
                        ),
                        increment_token=True,
                    )
                else:
                    self.reference_postprocess_results_by_key[image_key] = result
                    processed_swc_rows = [
                        [float(v) for v in row[:7]]
                        for row in data_save.paths_to_swc(_coerce_paths_xyz(processed_paths))
                    ]
                    if len(processed_swc_rows) == 0:
                        self._set_state("Post-processing produced no reference rows.", increment_token=True)
                        return None
                    self.filtered_swc_by_key[image_key] = processed_swc_rows
                    self._increment_postprocess_token()
                    self._set_state(
                        (
                            f"Post-processing complete: {n} paths for {image_key} ({target}). "
                            "Use 'Undo Post-Process' to restore the original annotation."
                        ),
                        increment_token=True,
                    )
                return result
        except Exception as exc:
            self._set_state(f"Post-processing failed: {exc}", increment_token=True)
        return None

    def run_postprocess_all(self, target: str = "prediction") -> List[Dict[str, object]]:
        """Post-process every currently available annotation for the chosen target."""
        target = str(target or "prediction")
        if target == "prediction":
            image_keys = sorted(self.trace_results_by_key.keys())
            if len(image_keys) == 0:
                self._set_state("No traces available to post-process.", increment_token=True)
                return []
        else:
            image_keys = [path.relative_to(self.image_root).as_posix() for path in self.image_paths]

        processed_results: List[Dict[str, object]] = []
        for image_key in image_keys:
            result = self.run_postprocess(image_key, target=target)
            if result is not None:
                processed_results.append(result)

        self._set_state(
            f"Post-processing complete for {len(processed_results)} {target} annotation(s).",
            increment_token=True,
        )
        return processed_results

    def run_evaluation(self, image_key: str) -> Optional[Dict[str, object]]:
        """Evaluate the current predicted trace for *image_key* against the GT SWC."""
        if image_key not in self.trace_results_by_key:
            self._set_state(
                f"No predicted trace available for {image_key}. Trace the image first.",
                increment_token=True,
            )
            return None
        if self._gt_swc_dir is None:
            self._set_state("Ground truth SWC directory not set.", increment_token=True)
            return None
        # Build a name→path map that covers all .swc files (including subdirectories).
        gt_files_by_stem = {f.stem: f for f in self._gt_swc_dir.rglob("*.swc")}
        # image_key may be a multi-level relative path; extract just the stem.
        neuron_stem = Path(image_key).stem
        gt_file: Optional[Path] = flexible_image_key_lookup(
            gt_files_by_stem, neuron_stem, default=None
        )
        if gt_file is None:
            # Substring fallback: useful when SWC filenames carry extra suffixes.
            for stem, candidate in gt_files_by_stem.items():
                if neuron_stem in stem or stem in neuron_stem:
                    gt_file = candidate
                    break
        if gt_file is None:
            self._set_state(
                f"No matching GT SWC found for '{neuron_stem}' in {self._gt_swc_dir}",
                increment_token=True,
            )
            return None
        try:
            self._set_state(f"Evaluating {image_key}...", increment_token=False)
            gt_swc = data_loading.swc(str(gt_file), verbose=False)
            pred_paths_xyz = _coerce_paths_xyz(self.trace_results_by_key.get(image_key, []))
            pred_swc = data_save.paths_to_swc(pred_paths_xyz)
            if not pred_swc:
                self._set_state("Empty prediction — evaluation skipped.", increment_token=True)
                return None
            result = evaluate_reconstruction(
                pred_swc, gt_swc,
                threshold=self.postprocess_config.distance_threshold / self.postprocess_config.get_scale_for_image(image_key),
                return_l_measures=True,
            )
            result["neuron_name"] = image_key
            result["image_key"] = image_key
            result["gt_file"] = str(gt_file)
            self.eval_results_by_key[image_key] = result
            self._set_state(f"Evaluation complete for {image_key}", increment_token=True)
            return result
        except Exception as exc:
            self._set_state(f"Evaluation failed: {exc}", increment_token=True)
            return None

    def evaluate_all(self) -> List[Dict[str, object]]:
        """Evaluate every currently available trace in the session."""
        if len(self.trace_results_by_key) == 0:
            self._set_state("No traces available to evaluate.", increment_token=True)
            return []
        if self._gt_swc_dir is None:
            self._set_state("Ground truth SWC directory not set.", increment_token=True)
            return []

        evaluated: List[Dict[str, object]] = []
        for image_key in sorted(self.trace_results_by_key.keys()):
            result = self.run_evaluation(image_key)
            if result is not None:
                evaluated.append(result)

        self._set_state(
            f"Evaluation complete for {len(evaluated)} trace(s).",
            increment_token=True,
        )
        return evaluated

    def save_postprocessed(self, image_key: str, default_dir: Path) -> None:
        """Write the current predicted SWC for *image_key* to disk."""
        if image_key not in self.trace_results_by_key:
            self._set_state(
                f"No predicted trace data to save for {image_key}.",
                increment_token=True,
            )
            return
        if self._postprocess_output_dir is None:
            selected = prompt_select_directory(default_path=str(default_dir))
            if selected is None:
                return
            self._postprocess_output_dir = Path(selected)
        self._postprocess_output_dir.mkdir(parents=True, exist_ok=True)
        pred_paths_xyz = _coerce_paths_xyz(self.trace_results_by_key.get(image_key, []))
        swc_list = data_save.paths_to_swc(pred_paths_xyz)
        if not swc_list:
            self._set_state("Empty SWC — nothing to save.", increment_token=True)
            return
        neuron_stem = _output_stem(image_key)
        swc_path = self._postprocess_output_dir / f"{neuron_stem}_reconstructed.swc"
        data_save.write_swc(swc_list, str(swc_path))
        self._set_state(f"Saved predicted SWC: {swc_path}", increment_token=True)

    def save_eval_report(self, default_dir: Path) -> None:
        """Write the cached session evaluation report to CSV and JSON summary."""
        if len(self.eval_results_by_key) == 0:
            self._set_state("No cached evaluation data to save.", increment_token=True)
            return
        if self._eval_output_dir is None:
            selected = prompt_select_directory(default_path=str(default_dir))
            if selected is None:
                return
            self._eval_output_dir = Path(selected)
        self._eval_output_dir.mkdir(parents=True, exist_ok=True)
        report_stem = self._report_stem
        metrics_path = self._eval_output_dir / f"{report_stem}_metrics.csv"
        summary_path = self._eval_output_dir / f"{report_stem}_summary.json"

        rows: List[Dict[str, object]] = []
        for image_key, eval_result in sorted(self.eval_results_by_key.items()):
            row = {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in eval_result.items()
            }
            row.setdefault("neuron_name", image_key)
            row.pop("image_key", None)
            row.setdefault("skipped", False)
            postprocess_result = self.postprocess_results_by_key.get(image_key, {})
            row.setdefault("n_raw_paths", int(postprocess_result.get("n_raw_paths", 0)))
            row.setdefault("n_processed_paths", int(postprocess_result.get("n_processed_paths", 0)))
            rows.append(row)

        upsert_evaluation_results_csv(rows, str(metrics_path))

        summary = compute_pipeline_summary(
            postprocessed_results=[],
            evaluation_results=rows,
            has_ground_truth=True,
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._set_state(
            f"Saved eval report: {metrics_path} and {summary_path}",
            increment_token=True,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, current_key: str, target: str = "prediction") -> Dict[str, object]:
        target = str(target or "prediction")
        with self._state_lock:
            # Deep-copying the reference SWC rows is the most expensive part of a
            # status poll. The viewer only consumes ``reference_swc_rows`` /
            # reference ``postprocess_paths`` when the active target is reference,
            # so skip the copy entirely for the common prediction-editing case and
            # compute it at most once when it is actually needed.
            tree_rows_cache: Optional[List[List[float]]] = None

            def _tree_rows() -> List[List[float]]:
                nonlocal tree_rows_cache
                if tree_rows_cache is None:
                    tree_rows_cache = self.get_tree_swc_rows(current_key)
                return tree_rows_cache

            is_prediction = target == "prediction"
            status: Dict[str, object] = {
                "running": self._running,
                "message": self._message,
                "token": self._token,
                "overlay_token": self._overlay_token,
                "postprocess_token": self._postprocess_token,
                "overlay_paths": self.trace_results_by_key.get(current_key, []),
                "reference_swc_rows": [] if is_prediction else _tree_rows(),
                "trace_timing_ms": self._trace_timing_by_key.get(current_key, None),
                "trace_output_dir": None if self._trace_output_dir is None else str(self._trace_output_dir),
                "model_weights_path": self.get_model_weights_path(),
                "progress_completed": self._progress_completed,
                "progress_total": self._progress_total,
                "eval_report_text": None,
                "gt_swc_path": self.get_gt_swc_path(),
                "postprocess_target": target,
                "postprocess_paths": self.trace_results_by_key.get(current_key, [])
                if is_prediction
                else _tree_rows(),
                "can_undo_postprocess": (
                    current_key in self._pre_postprocess_trace_cache_by_key
                    if target == "prediction"
                    else current_key in self._pre_postprocess_reference_cache_by_key
                ),
            }
            eval_result = self.eval_results_by_key.get(current_key)
            if eval_result is not None:
                status["eval_report_text"] = _format_eval_report(current_key, eval_result)
            return status

    def get_model_weights_path(self) -> Optional[str]:
        value = self.trace_params.get("sac_weights", None)
        if value is None:
            return None
        return str(value)

    def set_model_weights_path(self, model_weights_path: str) -> Optional[str]:
        if model_weights_path is None or len(str(model_weights_path).strip()) == 0:
            return None
        path = Path(model_weights_path)
        if not path.exists():
            self._set_state(f"Model weights not found: {path}", increment_token=True)
            return None

        self.trace_params["sac_weights"] = str(path)
        try:
            self._runtime = _TraceRuntime(trace_params=self.trace_params)
            self.enabled = True
            self._set_state(f"Model weights loaded: {path}", increment_token=True)
            return str(path)
        except Exception as exc:
            self.enabled = False
            self._runtime = None
            self._set_state(f"Failed to load model weights: {exc}", increment_token=True)
            return None

    def clear_model_weights_path(self) -> Optional[str]:
        self.trace_params.pop("sac_weights", None)
        self.enabled = False
        self._runtime = None
        self._set_state("Model weights cleared.", increment_token=True)
        return None

    def set_trace_output_dir(self, path: str) -> Optional[str]:
        """Set the trace output directory directly (e.g. from a config file)."""
        if not path or not str(path).strip():
            return None
        p = Path(path) / self._run_stem
        p.mkdir(parents=True, exist_ok=True)
        self._trace_output_dir = p
        self._set_state(f"Trace output set to: {p}", increment_token=True)
        return str(p)

    def clear_trace_output_dir(self) -> Optional[str]:
        self._trace_output_dir = None
        self._set_state("Trace output cleared.", increment_token=True)
        return None

    def select_trace_output_dir(self, default_dir: Path) -> Optional[str]:
        selected_dir = prompt_select_directory(default_path=str(default_dir))
        if selected_dir is None:
            return None
        self._trace_output_dir = Path(selected_dir)
        self._trace_output_dir.mkdir(parents=True, exist_ok=True)
        self._set_state(f"Trace output set to: {self._trace_output_dir}", increment_token=True)
        return str(self._trace_output_dir)

    def update_trace_params(self, overrides: Dict[str, object]) -> None:
        """Update runtime trace parameters and reload the runtime if weights are available."""
        changed = False
        for key, value in overrides.items():
            if self.trace_params.get(key) != value:
                self.trace_params[key] = value
                changed = True
        if not changed:
            return
        if self.trace_params.get("sac_weights"):
            try:
                self._runtime = _TraceRuntime(trace_params=self.trace_params)
                self.enabled = True
            except Exception as exc:
                self.enabled = False
                self._runtime = None
                self._set_state(f"Runtime reload failed after param change: {exc}", increment_token=True)
        return str(self._trace_output_dir)

    def get_trace_output_dir(self) -> Optional[str]:
        return None if self._trace_output_dir is None else str(self._trace_output_dir)

    def trace_current(self, image_index: int, image_key: str, seed_rows: List[List[float]]) -> Optional[List[List[List[float]]]]:
        if not self.enabled or self._runtime is None:
            self._set_state("Tracing is disabled (missing model config).", increment_token=True)
            return None
        if self._running:
            self._set_state("Trace All is running. Cancel it before tracing a single image.", increment_token=True)
            return self.trace_results_by_key.get(image_key, [])

        normalized_seed_rows = self._normalize_seed_rows_payload(seed_rows)
        pending_seed_rows = self._pending_seed_rows(image_key=image_key, seed_rows=seed_rows)
        if len(normalized_seed_rows) > 0 and len(pending_seed_rows) == 0 and image_key in self.trace_results_by_key:
            self._set_state(f"No new seeds to trace for {image_key}.", increment_token=True)
            return self.trace_results_by_key.get(image_key, [])

        self._set_state(f"Tracing {image_key}...", increment_token=False)
        result = self._runtime.trace_image(
            image_index=image_index,
            image_relative_key=image_key,
            seed_rows=pending_seed_rows,
            prediction_paths=self._normalize_paths_payload(self.trace_results_by_key.get(image_key, [])),
            cancel_event=None,
        )
        self._store_trace_timing(image_key=image_key, timing_ms=result.get("timing_ms", None))
        paths = self._append_trace_paths(image_key=image_key, new_paths=result["paths"])
        self._mark_seed_rows_traced(image_key=image_key, seed_rows=pending_seed_rows)
        self._increment_overlay_token()
        self._write_temp_trace(image_key=image_key, paths=paths)
        self._clear_derived_results(image_key)
        self._set_state(
            f"Trace complete: {image_key}{self._format_timing_summary(self._trace_timing_by_key.get(image_key))}",
            increment_token=True,
        )
        return paths

    def start_trace_all(self, seeds_by_key: Dict[str, List[List[float]]]):
        if not self.enabled or self._runtime is None:
            self._set_state("Tracing is disabled (missing model config).", increment_token=True)
            return
        if self._running:
            self._set_state("Trace All is already running.", increment_token=True)
            return

        self._cancel_event = threading.Event()
        self._running = True
        with self._state_lock:
            self._progress_total = len(self.image_paths)
            self._progress_completed = 0

        def _worker():
            try:
                total = len(self.image_paths)
                for idx, image_path in enumerate(self.image_paths):
                    if self._cancel_event is not None and self._cancel_event.is_set():
                        self._set_state("Trace All cancelled.", increment_token=True)
                        return

                    key = image_path.relative_to(self.image_root).as_posix()
                    self._set_state(f"Tracing {idx + 1}/{total}: {key}", increment_token=False)
                    seed_rows = flexible_image_key_lookup(seeds_by_key, key, default=[])
                    normalized_seed_rows = self._normalize_seed_rows_payload(seed_rows)
                    pending_seed_rows = self._pending_seed_rows(image_key=key, seed_rows=seed_rows)
                    if len(normalized_seed_rows) > 0 and len(pending_seed_rows) == 0 and key in self.trace_results_by_key:
                        with self._state_lock:
                            self._progress_completed = idx + 1
                        self._set_state(f"Skipped {idx + 1}/{total}: {key} (no new seeds)", increment_token=True)
                        continue
                    result = self._runtime.trace_image(
                        image_index=idx,
                        image_relative_key=key,
                        seed_rows=pending_seed_rows,
                        prediction_paths=self._normalize_paths_payload(self.trace_results_by_key.get(key, [])),
                        cancel_event=self._cancel_event,
                    )
                    self._store_trace_timing(image_key=key, timing_ms=result.get("timing_ms", None))
                    paths = self._append_trace_paths(image_key=key, new_paths=result["paths"])
                    self._mark_seed_rows_traced(image_key=key, seed_rows=pending_seed_rows)
                    self._increment_overlay_token()
                    self._write_temp_trace(image_key=key, paths=paths)
                    self._clear_derived_results(key)
                    with self._state_lock:
                        self._progress_completed = idx + 1
                    self._set_state(
                        f"Completed {idx + 1}/{total}: {key}"
                        f"{self._format_timing_summary(self._trace_timing_by_key.get(key))}",
                        increment_token=True,
                    )
            except RuntimeError as exc:
                self._set_state(str(exc), increment_token=True)
            except Exception as exc:
                self._set_state(f"Trace All failed: {exc}", increment_token=True)
            finally:
                self._running = False

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def cancel_trace_all(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._set_state("Cancelling Trace All...", increment_token=False)

    def get_effective_seed_overlay(
        self,
        image_index: int,
        image_key: str,
        seed_rows: List[List[float]],
    ) -> Optional[np.ndarray]:
        """Compute display-only effective seeds (including configured jitter) for the viewer."""
        if self._runtime is None:
            if len(seed_rows) == 0:
                return None
            return np.asarray(seed_rows, dtype=np.float32)
        try:
            return self._runtime.get_effective_seed_points(
                image_index=image_index,
                image_relative_key=image_key,
                seed_rows=seed_rows,
            )
        except Exception:
            if len(seed_rows) == 0:
                return None
            return np.asarray(seed_rows, dtype=np.float32)


class _SessionState:
    """Mutable navigation and seed-management state for the interactive tracing session.

    Centralises the variables that were previously scattered across ``nonlocal``
    closures, making session state explicit and inspectable without relying on
    closure-captured mutable bindings.
    """

    def __init__(
        self,
        image_paths: List[Path],
        image_root: Path,
        existing_seeds: Dict[str, list],
        seeds_output_path: Optional[str],
        seeds_input_path: Optional[str],
    ) -> None:
        self.image_paths = image_paths
        self.image_root = image_root
        self.current_index: int = 0
        self.selected_seeds: Dict[str, list] = dict(existing_seeds)
        self.seeds_output_path: Optional[str] = seeds_output_path
        self.seeds_input_path: Optional[str] = seeds_input_path
        self.display_image_dir: Optional[str] = str(image_root)
        self.current_volume_shape: tuple[int, int, int] = (1, 1, 1)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def current_relative_key(self) -> str:
        return self.image_paths[self.current_index].relative_to(self.image_root).as_posix()

    def rows_from_seed_array(self, seed_array: np.ndarray) -> List[List[float]]:
        return _normalize_seed_array(seed_array=seed_array, shape=self.current_volume_shape)

    def build_context(self, index: int, trace_manager: _TraceSessionManager) -> Dict[str, object]:
        image_path = self.image_paths[index]
        relative_key = image_path.relative_to(self.image_root).as_posix()
        image_array = tf.imread(image_path)
        self.current_volume_shape = tuple(np.asarray(image_array).shape[-3:])
        initial_rows = flexible_image_key_lookup(self.selected_seeds, relative_key, default=[])
        initial_seeds = np.asarray(initial_rows, dtype=np.float32) if initial_rows else None
        effective_seed_overlay = trace_manager.get_effective_seed_overlay(
            image_index=index,
            image_key=relative_key,
            seed_rows=initial_rows,
        )
        trace_status = trace_manager.get_status(current_key=relative_key)
        return {
            "image_data": image_array,
            "neuron_name": relative_key,
            "initial_seeds": initial_seeds,
            "effective_seed_overlay": effective_seed_overlay,
            "show_prev_button": index > 0,
            "show_next_button": index < len(self.image_paths) - 1,
            "finished_paths": trace_manager.trace_results_by_key.get(relative_key, []),
            "tree_swc_rows": trace_manager.get_tree_swc_rows(relative_key),
            "postprocess_paths": trace_status.get("postprocess_paths", None),
            "eval_report_text": trace_status.get("eval_report_text", None),
            "seeds_output_path": self.seeds_output_path,
            "trace_output_path": trace_manager.get_trace_output_dir(),
            "model_weights_path": trace_manager.get_model_weights_path(),
            "gt_swc_path": trace_manager.get_gt_swc_path(),
            "scales_path": trace_manager.get_scales_path(),
            "filtered_swc_output_dir": trace_manager.get_filtered_swc_output_dir(),
            "image_dir": self.display_image_dir,
            "seeds_input_path": self.seeds_input_path,
        }

    def on_prev_image(
        self, seed_array: np.ndarray, trace_manager: _TraceSessionManager
    ) -> Optional[Dict[str, object]]:
        self.selected_seeds[self.current_relative_key()] = self.rows_from_seed_array(seed_array)
        if self.current_index <= 0:
            return None
        self.current_index -= 1
        return self.build_context(self.current_index, trace_manager)

    def on_next_image(
        self, seed_array: np.ndarray, trace_manager: _TraceSessionManager
    ) -> Optional[Dict[str, object]]:
        self.selected_seeds[self.current_relative_key()] = self.rows_from_seed_array(seed_array)
        if self.current_index >= len(self.image_paths) - 1:
            return None
        self.current_index += 1
        return self.build_context(self.current_index, trace_manager)

    # ------------------------------------------------------------------
    # Seed persistence
    # ------------------------------------------------------------------

    def ensure_output_path(self) -> str:
        if self.seeds_output_path is None:
            self.seeds_output_path = prompt_save_json_path(
                default_path=str(self.image_root / "seeds.json")
            )
        if self.seeds_output_path is None:
            raise ValueError("A seeds output path is required to save selected seeds.")
        return self.seeds_output_path

    def select_seeds_output_path(self) -> Optional[str]:
        # Use getSaveFileName with DontConfirmOverwrite so selecting an existing file
        # only sets the path without Qt showing an overwrite-confirmation dialog.
        qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
        options = qt_widgets_mod.QFileDialog.Options()
        options |= qt_widgets_mod.QFileDialog.DontConfirmOverwrite
        selected, _ = qt_widgets_mod.QFileDialog.getSaveFileName(
            None,
            "Select seeds output JSON",
            str(self.image_root / "seeds.json"),
            "JSON Files (*.json)",
            options=options,
        )
        if selected:
            self.seeds_output_path = selected
        return self.seeds_output_path

    def clear_seeds_output_path(self) -> Optional[str]:
        self.seeds_output_path = None
        return self.seeds_output_path

    def select_image_dir(self) -> Optional[str]:
        qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
        selected = qt_widgets_mod.QFileDialog.getExistingDirectory(
            None, "Select image directory", str(self.image_root)
        )
        if selected:
            self.display_image_dir = selected
        return self.display_image_dir

    def clear_image_dir(self) -> Optional[str]:
        self.display_image_dir = None
        return self.display_image_dir

    def _load_existing_output_seeds(self, out_path: str) -> Dict[str, List[List[float]]]:
        path = Path(out_path)
        if not path.exists():
            return {}
        try:
            return load_seeds_json(path)
        except Exception as exc:
            raise ValueError(f"Failed to load existing seeds JSON '{path}': {exc}") from exc

    def write_seeds_merge(self, out_path: str, updates: Dict[str, List[List[float]]]) -> None:
        existing = self._load_existing_output_seeds(out_path)
        merged = dict(existing)
        merged.update(updates)
        save_seeds_json(seeds_json_path=out_path, seeds_by_relative_path=merged)

    def _confirm_seed_overwrite(self, out_path: str, update_keys: List[str]) -> bool:
        """Return True if it is safe to proceed with a seeds write.

        Loads the existing output file (if any), finds keys that would be
        overwritten, and—when there are conflicts—shows a Qt confirmation
        dialog.  Returns False if the user cancels.
        """
        existing = self._load_existing_output_seeds(out_path)
        conflicting = [k for k in update_keys if k in existing]
        if not conflicting:
            return True
        qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
        count = len(conflicting)
        noun = "image" if count == 1 else "images"
        listed = "\n".join(f"  \u2022 {k}" for k in conflicting[:10])
        more = f"\n  \u2026 and {count - 10} more" if count > 10 else ""
        msg = (
            f"The following {count} {noun} already ha"
            + ("s" if count == 1 else "ve")
            + " seeds in the output file:\n\n"
            + listed
            + more
            + "\n\nOverwrite those entries with the current seeds?\n"
            "(Other entries in the file will not be affected.)"
        )
        reply = qt_widgets_mod.QMessageBox.question(
            None,
            "Overwrite Seeds?",
            msg,
            qt_widgets_mod.QMessageBox.Yes | qt_widgets_mod.QMessageBox.No,
            qt_widgets_mod.QMessageBox.No,
        )
        return reply == qt_widgets_mod.QMessageBox.Yes

    def save_current(self, seed_array: np.ndarray) -> None:
        relative_key = self.current_relative_key()
        self.selected_seeds[relative_key] = self.rows_from_seed_array(seed_array)
        out_path = self.ensure_output_path()
        if not self._confirm_seed_overwrite(out_path, [relative_key]):
            return
        self.write_seeds_merge(out_path=out_path, updates={relative_key: self.selected_seeds[relative_key]})
        print(f"Saved seeds for {relative_key} to: {out_path}")

    def save_all(self) -> None:
        out_path = self.ensure_output_path()
        if not self._confirm_seed_overwrite(out_path, list(self.selected_seeds.keys())):
            return
        self.write_seeds_merge(out_path=out_path, updates=self.selected_seeds)
        print(f"Saved all seeds to: {out_path}")

    def trace_current(
        self, seed_array: np.ndarray, trace_manager: _TraceSessionManager
    ) -> Optional[List[List[List[float]]]]:
        relative_key = self.current_relative_key()
        rows_from_ui = self.rows_from_seed_array(seed_array)
        if rows_from_ui:
            self.selected_seeds[relative_key] = rows_from_ui
        return trace_manager.trace_current(
            image_index=self.current_index,
            image_key=relative_key,
            seed_rows=self.selected_seeds.get(relative_key, []),
        )

    # ------------------------------------------------------------------
    # Seeds input loading
    # ------------------------------------------------------------------

    def select_seeds_input_path(self) -> tuple:
        """Prompt for a seeds JSON and load it into the active session.

        Returns ``(seeds_input_path, initial_seeds_array_or_none)`` so the
        viewer can immediately refresh the seed overlay for the current image.
        """
        qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
        selected, _ = qt_widgets_mod.QFileDialog.getOpenFileName(
            None, "Select existing seeds JSON", str(self.image_root), "JSON Files (*.json)"
        )
        if not selected:
            return self.seeds_input_path, None
        self.seeds_input_path = selected
        loaded = load_seeds_json(self.seeds_input_path)
        for key, rows in loaded.items():
            self.selected_seeds[key] = rows
        print(f"Loaded seeds from: {self.seeds_input_path}")
        current_rows = flexible_image_key_lookup(self.selected_seeds, self.current_relative_key(), default=[])
        current_seeds_array = np.asarray(current_rows, dtype=np.float32) if current_rows else None
        return self.seeds_input_path, current_seeds_array

    def clear_seeds_input_path(self) -> Optional[str]:
        self.seeds_input_path = None
        return self.seeds_input_path


def run_interactive_tracing_session(
    image_dir: Optional[str] = None,
    seeds_output_path: Optional[str] = None,
    seeds_input_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, list]:
    """Run an interactive tracing session: seed selection, tracing, post-processing, and evaluation."""
    config = _load_optional_session_config(config_path)

    def _first_config_value(*keys: str) -> Optional[str]:
        for key in keys:
            value = config.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if len(value_str) > 0:
                return value_str
        return None

    image_dir = image_dir or _first_config_value("image_dir", "img_dir", "img_path")
    if seeds_input_path is None:
        seeds_input_path = _first_config_value("seeds_input_path", "seeds_path")
    if seeds_output_path is None:
        seeds_output_path = _first_config_value("seeds_output_path")
    report_stem = _first_config_value("name", "test_name", "session_name", "run_name")

    image_dir, seeds_input_path, seeds_output_path = prompt_seed_session_paths(
        image_dir=image_dir,
        seeds_input_path=seeds_input_path,
        seeds_output_path=seeds_output_path,
    )

    image_root = Path(image_dir)
    image_paths = _discover_images(image_root)
    if len(image_paths) == 0:
        raise ValueError(f"No TIFF files found in image directory: {image_root}")

    existing_seeds = load_seeds_json(seeds_input_path) if seeds_input_path else {}

    # ---- tracing parameters (environment / SAC runtime) ----
    trace_params: Dict[str, object] = {
        "img_dir": str(image_root),
        "swc_dir": config.get("swc_dir"),
        "sac_weights": config.get("sac_weights"),
        "policy_output_mode": config.get("policy_output_mode", "direct_vector"),
        "rng_seed": config.get("rng_seed", 0),
        "step_width": config.get("step_width", 4.0),
        "stall_threshold": config.get("stall_threshold", 1.0),
        "soma_sample_radius": config.get("soma_sample_radius", 0.0),
        "random_offset": config.get("random_offset", 0.0),
        "seed_jitter_count": config.get("seed_jitter_count", 0),
        "seed_jitter_radius": config.get("seed_jitter_radius", 0.0),
        "seed_jitter_weight_strategy": config.get("seed_jitter_weight_strategy", config.get("weight_strategy", "uniform")),
        "max_len": config.get("max_len", 10000),
        "max_paths": config.get("max_paths", 1000),
        "branching": config.get("branching", True),
        "repeat_starts": config.get("repeat_starts", False),
        "n_trials": config.get("n_trials", 1),
        "stochastic_actions": config.get("stochastic_actions", False),
        "retry_on_no_long_paths": config.get("retry_on_no_long_paths", True),
        "retry_initial_radius": config.get("retry_initial_radius", 5.0),
        "retry_radius_step": config.get("retry_radius_step", 5.0),
        "retry_max_radius": config.get("retry_max_radius", 50.0),
        "retry_attempts_per_radius": config.get("retry_attempts_per_radius", 50),
    }

    # ---- post-processing / evaluation parameters (separate from trace_params) ----
    postprocess_config = PostprocessConfig.from_config(config)

    session = _SessionState(
        image_paths=image_paths,
        image_root=image_root,
        existing_seeds=existing_seeds,
        seeds_output_path=seeds_output_path,
        seeds_input_path=seeds_input_path,
    )

    trace_manager = _TraceSessionManager(
        image_paths=image_paths,
        image_root=image_root,
        trace_params=trace_params,
        postprocess_config=postprocess_config,
        report_stem=report_stem,
    )

    # Pre-set optional output directories from config so UI labels are populated immediately.
    trace_output_dir = _first_config_value("trace_output_dir", "out_dir")
    postprocess_output_dir = _first_config_value("postprocess_output_dir", "out_dir")
    eval_output_dir = _first_config_value("eval_output_dir", "out_dir")
    if trace_output_dir is not None:
        trace_manager.set_trace_output_dir(trace_output_dir)
    if postprocess_output_dir is not None:
        trace_manager.set_postprocess_output_dir(postprocess_output_dir)
    if eval_output_dir is not None:
        trace_manager.set_eval_output_dir(eval_output_dir)

    # ---- small local closures for operations that need both session + external state ----

    def _select_model_weights_path() -> Optional[str]:
        current_weights = trace_manager.get_model_weights_path()
        selected = prompt_select_model_weights(default_path=current_weights)
        if selected is None:
            return current_weights
        loaded = trace_manager.set_model_weights_path(selected)
        return loaded if loaded is not None else current_weights

    def _select_gt_swc_path() -> Optional[str]:
        selected = prompt_select_directory(
            default_path=config.get("swc_dir") or str(image_root)
        )
        if selected:
            trace_manager.set_gt_swc_dir(selected)
        return trace_manager.get_gt_swc_path()

    def _clear_gt_swc_path() -> Optional[str]:
        return trace_manager.clear_gt_swc_dir()

    def _select_scales_path() -> Optional[str]:
        qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
        selected, _ = qt_widgets_mod.QFileDialog.getOpenFileName(
            None, "Select scales JSON",
            str(Path(trace_manager.get_scales_path()).parent)
            if trace_manager.get_scales_path() else str(image_root),
            "JSON Files (*.json)",
        )
        if selected:
            trace_manager.set_scales_path(selected)
        return trace_manager.get_scales_path()

    def _clear_scales_path() -> Optional[str]:
        return trace_manager.clear_scales_path()

    def _select_postprocess_output_dir() -> Optional[str]:
        selected = prompt_select_directory(default_path=str(image_root / "postprocessed"))
        if selected is None:
            return trace_manager.get_postprocess_output_dir()
        return trace_manager.set_postprocess_output_dir(selected)

    def _clear_postprocess_output_dir() -> Optional[str]:
        return trace_manager.clear_postprocess_output_dir()

    def _select_eval_output_dir() -> Optional[str]:
        selected = prompt_select_directory(default_path=str(image_root / "evaluation"))
        if selected is None:
            return trace_manager.get_eval_output_dir()
        return trace_manager.set_eval_output_dir(selected)

    def _clear_eval_output_dir() -> Optional[str]:
        return trace_manager.clear_eval_output_dir()

    def _select_filtered_swc_output_dir() -> Optional[str]:
        return trace_manager.select_filtered_swc_output_dir(default_dir=image_root / "filtered_swc")

    def _clear_filtered_swc_output_dir() -> Optional[str]:
        return trace_manager.clear_filtered_swc_output_dir()

    def _save_filtered_swc(image_key: str, swc_rows: List[List[float]]) -> Optional[str]:
        return trace_manager.save_filtered_swc(
            image_key=image_key,
            swc_rows=swc_rows,
            default_dir=image_root / "filtered_swc",
        )

    def _on_filtered_swc_changed(image_key: str, swc_rows: List[List[float]]) -> None:
        trace_manager.set_filtered_swc_rows(image_key=image_key, swc_rows=swc_rows)

    def _on_prediction_paths_changed(image_key: str, prediction_paths: List[List[List[float]]]) -> None:
        trace_manager.trace_results_by_key[image_key] = trace_manager._normalize_paths_payload(prediction_paths)
        trace_manager._increment_overlay_token()
        trace_manager._clear_derived_results(image_key)

    def _clear_model_weights_path() -> Optional[str]:
        return trace_manager.clear_model_weights_path()

    try:
        initial_context = session.build_context(session.current_index, trace_manager)
        final_seeds = interactive_seed_selection_session(
            initial_context=initial_context,
            on_prev_image=lambda arr: session.on_prev_image(arr, trace_manager),
            on_next_image=lambda arr: session.on_next_image(arr, trace_manager),
            on_get_effective_seed_overlay=lambda arr: trace_manager.get_effective_seed_overlay(
                image_index=session.current_index,
                image_key=session.current_relative_key(),
                seed_rows=session.rows_from_seed_array(arr),
            ),
            on_save_current=session.save_current,
            on_save_all=session.save_all,
            show_trace_controls=True,
            on_trace_current=lambda arr: session.trace_current(arr, trace_manager),
            on_trace_all=lambda: trace_manager.start_trace_all(session.selected_seeds),
            on_cancel_trace=trace_manager.cancel_trace_all,
            get_trace_status=lambda target: trace_manager.get_status(
                current_key=session.current_relative_key(), target=target
            ),
            on_save_trace=lambda: trace_manager.save_trace(
                session.current_relative_key(), default_dir=image_root / "trace_outputs"
            ),
            on_save_all_traces=lambda: trace_manager.save_all_traces(
                default_dir=image_root / "trace_outputs"
            ),
            on_discard_trace=lambda: trace_manager.discard_trace(session.current_relative_key()),
            on_select_seeds_output_path=session.select_seeds_output_path,
            on_select_trace_output_path=lambda: trace_manager.select_trace_output_dir(
                default_dir=image_root / "trace_outputs"
            ),
            on_clear_seeds_output_path=session.clear_seeds_output_path,
            on_clear_trace_output_path=trace_manager.clear_trace_output_dir,
            on_select_model_weights_path=_select_model_weights_path,
            on_clear_model_weights_path=_clear_model_weights_path,
            on_select_image_dir=session.select_image_dir,
            on_clear_image_dir=session.clear_image_dir,
            on_select_seeds_input_path=session.select_seeds_input_path,
            on_clear_seeds_input_path=session.clear_seeds_input_path,
            trace_step_width=float(trace_params.get("step_width", 4.0)),
            trace_n_trials=int(trace_params.get("n_trials", 1)),
            trace_max_len=int(trace_params.get("max_len", 10000)),
            trace_max_paths=int(trace_params.get("max_paths", 1000)),
            trace_branching=bool(trace_params.get("branching", True)),
            trace_repeat_starts=bool(trace_params.get("repeat_starts", False)),
            trace_stochastic_actions=bool(trace_params.get("stochastic_actions", False)),
            trace_seed_jitter_count=int(trace_params.get("seed_jitter_count", 0)),
            trace_seed_jitter_radius=float(trace_params.get("seed_jitter_radius", 0.0)),
            trace_seed_jitter_weight_strategy=str(
                trace_params.get("seed_jitter_weight_strategy", trace_params.get("weight_strategy", "uniform"))
            ),
            on_trace_params_changed=trace_manager.update_trace_params,
            show_postprocess_controls=True,
            on_run_postprocess=lambda target: trace_manager.run_postprocess(
                session.current_relative_key(), target=target
            ),
            on_run_postprocess_all=lambda target: trace_manager.run_postprocess_all(target=target),
            on_undo_postprocess=lambda target: trace_manager.undo_postprocess(
                session.current_relative_key(), target=target
            ),
            on_run_evaluation=lambda: trace_manager.run_evaluation(session.current_relative_key()),
            on_run_evaluation_all=trace_manager.evaluate_all,
            on_save_eval_report=lambda: trace_manager.save_eval_report(default_dir=image_root / "evaluation"),
            on_select_gt_swc_path=_select_gt_swc_path,
            on_clear_gt_swc_path=_clear_gt_swc_path,
            on_select_scales_path=_select_scales_path,
            on_clear_scales_path=_clear_scales_path,
            postprocess_output_dir=trace_manager.get_postprocess_output_dir(),
            postprocess_min_branch_length=postprocess_config.min_branch_length,
            postprocess_max_branch_length=postprocess_config.max_branch_length,
            postprocess_enable_length_filter=postprocess_config.enable_length_filter,
            postprocess_resampling_step_size=postprocess_config.resampling_step_size,
            postprocess_enable_resample=postprocess_config.enable_resample,
            postprocess_smoothing_window=postprocess_config.smoothing_window,
            postprocess_enable_smooth_paths=postprocess_config.enable_smooth_paths,
            postprocess_merge_threshold=postprocess_config.merge_threshold,
            postprocess_confidence_threshold=postprocess_config.confidence_threshold,
            postprocess_enable_merge=postprocess_config.enable_merge,
            postprocess_mask_smoothing_size=postprocess_config.mask_smoothing_size,
            postprocess_merge_timeout_seconds=postprocess_config.merge_timeout_seconds,
            on_select_postprocess_output_dir=_select_postprocess_output_dir,
            on_clear_postprocess_output_dir=_clear_postprocess_output_dir,
            on_postprocess_params_changed=trace_manager.update_postprocess_config,
            eval_output_dir=trace_manager.get_eval_output_dir(),
            eval_distance_threshold=postprocess_config.distance_threshold,
            on_select_eval_output_dir=_select_eval_output_dir,
            on_clear_eval_output_dir=_clear_eval_output_dir,
            on_eval_params_changed=trace_manager.update_postprocess_config,
            filtered_swc_output_dir=trace_manager.get_filtered_swc_output_dir(),
            on_select_filtered_swc_output_dir=_select_filtered_swc_output_dir,
            on_clear_filtered_swc_output_dir=_clear_filtered_swc_output_dir,
            on_save_filtered_swc=_save_filtered_swc,
            on_filtered_swc_changed=_on_filtered_swc_changed,
            on_prediction_paths_changed=_on_prediction_paths_changed,
        )
        session.selected_seeds[session.current_relative_key()] = _normalize_seed_array(
            final_seeds.detach().cpu().numpy(),
            session.current_volume_shape,
        )
    finally:
        trace_manager.close()

    return session.selected_seeds