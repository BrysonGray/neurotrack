"""Pipeline orchestrator for interactively selecting, tracing, post-processing,
and evaluating neuron reconstructions.
"""

from __future__ import annotations

import json
import importlib
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _format_eval_report(image_key: str, result: Dict) -> str:
    """Format evaluation metrics as a human-readable multi-line string."""
    sep = "-" * 44
    lines = [
        f"Evaluation: {image_key}",
        sep,
        f"  Bidirectional Distance:    {result.get('bidirectional_distance', 'N/A'):.4f}",
        f"  Directed Div pred\u2192gt:  {result.get('directed_div_pred_to_gt', 'N/A'):.4f}"
        f"  (N={result.get('n_substantial_pred_to_gt', 'N/A')})",
        f"  Directed Div gt\u2192pred:  {result.get('directed_div_gt_to_pred', 'N/A'):.4f}"
        f"  (N={result.get('n_substantial_gt_to_pred', 'N/A')})",
        f"  Precision: {result.get('precision', 'N/A'):.4f}",
        f"  Coverage: {result.get('coverage', 'N/A'):.4f}",
        f"  Pred Nodes: {result.get('n_points_pred', 'N/A')}",
        f"  |  GT Nodes: {result.get('n_points_gt', 'N/A')}",

    ]
    if "gt_file" in result:
        lines.append(f"  GT File: {result['gt_file']}")
    return "\n".join(lines)


def _coerce_paths_xyz(paths: List[List[List[float]]]) -> List[np.ndarray]:
    coerced: List[np.ndarray] = []
    for path in paths:
        path_np = np.asarray(path, dtype=np.float32)
        if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] > 0:
            coerced.append(path_np[:, :3].copy())
    return coerced


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
    atol_sq = float(atol) * float(atol)

    trimmed_selected_path = selected_path[: clipped_node_idx + 1].copy()
    trimmed_paths = [path.copy() for path in paths_xyz]
    trimmed_paths[selected_path_idx] = trimmed_selected_path

    anchors: List[np.ndarray] = [selected_node]
    if clipped_node_idx + 1 < selected_path.shape[0]:
        anchors.extend([node.copy() for node in selected_path[clipped_node_idx + 1:]])

    keep_mask = [True] * len(paths_xyz)
    changed = True
    while changed:
        changed = False
        for path_idx, path in enumerate(paths_xyz):
            if path_idx == selected_path_idx or not keep_mask[path_idx] or path.shape[0] == 0:
                continue
            root = path[0]
            is_downstream = any(float(np.sum((root - anchor) ** 2)) <= atol_sq for anchor in anchors)
            if not is_downstream:
                continue
            keep_mask[path_idx] = False
            anchors.extend([node.copy() for node in path])
            changed = True

    output: List[np.ndarray] = []
    for path_idx, path in enumerate(trimmed_paths):
        if path_idx == selected_path_idx:
            output.append(path)
        elif keep_mask[path_idx]:
            output.append(path)
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
        path_zyx = path[:, ::-1].astype(np.float32, copy=False)
        for idx in range(path_zyx.shape[0] - 1):
            segment = torch.as_tensor(path_zyx[idx: idx + 2], dtype=torch.float32)
            mask_image.draw_line_segment(segment, width=width, channel=0, mask=False)

    return mask_image.data[0].detach().cpu().numpy()


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
            rng=np.random.default_rng(rng_seed),
            crop_patches=False,
            patches_per_image=1,
            inference_mode=True,
        )

        self._env = NeuronTrackingEnvironment(
            dataset=self._dataset,
            radius=17,
            step_width=float(trace_params.get("step_width", 4.0)),
            max_len=int(trace_params.get("max_len", 10000)),
            max_paths=int(trace_params.get("max_paths", 1000)),
            branching=bool(trace_params.get("branching", True)),
            repeat_starts=bool(trace_params.get("repeat_starts", False)),
            start_idx=0,
            inference_mode=True,
            auto_seed_selection_mode=str(trace_params.get("auto_seed_selection_mode", "remote_endnode")),
            seed_points_by_image={},
        )

    def trace_image(
        self,
        image_index: int,
        image_relative_key: str,
        seed_rows: List[List[float]],
        cancel_event: Optional[threading.Event] = None,
        initial_path_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        with self._lock:
            self._env.seed_points_by_image = {image_relative_key: seed_rows}
            result = sac_trace_image(
                env=self._env,
                actor=self._actor,
                dataset_idx=image_index,
                Q_net=self._q_net,
                n_trials=int(self.trace_params.get("n_trials", 1)),
                show=False,
                show_live=False,
                stochastic=bool(self.trace_params.get("stochastic_actions", False)),
                cancel_event=cancel_event,
                initial_path_mask=initial_path_mask,
            )
            labeled_neuron = result.get("labeled_neuron", None)
            if labeled_neuron is not None and hasattr(labeled_neuron, "detach"):
                labeled_neuron = labeled_neuron.detach().cpu().numpy()
            elif labeled_neuron is not None:
                labeled_neuron = np.asarray(labeled_neuron)
            return {
                "paths": result["paths"],
                "labeled_neuron": labeled_neuron,
            }


class _TraceSessionManager:
    """Thread-safe trace orchestration for per-image and background trace-all actions."""

    def __init__(
        self,
        image_paths: List[Path],
        image_root: Path,
        trace_params: Optional[Dict[str, object]],
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        self.image_paths = image_paths
        self.image_root = image_root
        self.trace_params: Dict[str, object] = {} if trace_params is None else dict(trace_params)
        self.postprocess_config: PostprocessConfig = postprocess_config or PostprocessConfig()
        self._runtime = None
        self.enabled = False
        if self.trace_params.get("sac_weights"):
            self.set_model_weights_path(str(self.trace_params["sac_weights"]))

        self.trace_results_by_key: Dict[str, List[List[List[float]]]] = {}
        self.trace_mask_cache_by_key: Dict[str, np.ndarray] = {}
        self._revision_state_by_key: Dict[str, Dict[str, object]] = {}
        self._trace_output_dir: Optional[Path] = None
        self._temp_dir = tempfile.TemporaryDirectory(prefix="neurotrack_trace_session_")
        self._temp_root = Path(self._temp_dir.name)
        self._message = ""
        self._token = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None
        self._state_lock = threading.Lock()
        self._progress_completed = 0
        self._progress_total = 0

        # post-processing and evaluation state
        self.postprocess_results_by_key: Dict[str, Dict[str, object]] = {}
        self.eval_results_by_key: Dict[str, Dict[str, object]] = {}
        self._gt_swc_dir: Optional[Path] = None
        if self.trace_params.get("swc_dir"):
            self._gt_swc_dir = Path(str(self.trace_params["swc_dir"]))
        self._postprocess_output_dir: Optional[Path] = None
        self._eval_output_dir: Optional[Path] = None

    def _set_state(self, message: str, increment_token: bool = False):
        with self._state_lock:
            self._message = message
            if increment_token:
                self._token += 1

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
        src = self._temp_root / Path(image_key).with_suffix(".json")
        if not src.exists():
            self._write_temp_trace(image_key=image_key, paths=self.trace_results_by_key[image_key])
        dst = out_dir / Path(image_key).with_suffix(".json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        self._set_state(f"Saved trace: {dst}", increment_token=True)

    def save_all_traces(self, default_dir: Path):
        if len(self.trace_results_by_key) == 0:
            self._set_state("No traces available to save.", increment_token=True)
            return
        out_dir = self._ensure_output_dir(default_dir=default_dir)
        for image_key, paths in self.trace_results_by_key.items():
            src = self._temp_root / Path(image_key).with_suffix(".json")
            if not src.exists():
                self._write_temp_trace(image_key=image_key, paths=paths)
            dst = out_dir / Path(image_key).with_suffix(".json")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        self._set_state(f"Saved all traces to: {out_dir}", increment_token=True)

    def _clear_revision_state(self, image_key: str) -> None:
        self._revision_state_by_key.pop(image_key, None)

    def _clear_derived_results(self, image_key: str) -> None:
        self.postprocess_results_by_key.pop(image_key, None)
        self.eval_results_by_key.pop(image_key, None)

    def select_revision_node(
        self,
        image_key: str,
        selected_point_zyx: np.ndarray,
    ) -> Optional[Dict[str, object]]:
        paths_raw = self.trace_results_by_key.get(image_key, [])
        if len(paths_raw) == 0:
            self._set_state(
                f"No trace available for revision on {image_key}. Run tracing first.",
                increment_token=True,
            )
            return None

        point_zyx = np.asarray(selected_point_zyx, dtype=np.float32).reshape(-1)
        if point_zyx.shape[0] < 3:
            self._set_state("Revision point must have three coordinates (z, y, x).", increment_token=True)
            return None
        query_xyz = np.array([point_zyx[2], point_zyx[1], point_zyx[0]], dtype=np.float32)

        paths_xyz = _coerce_paths_xyz(paths_raw)
        closest = _find_closest_node_xyz(paths_xyz=paths_xyz, query_xyz=query_xyz)
        if closest is None:
            self._set_state("Could not identify a nearby node on the predicted tree.", increment_token=True)
            return None

        path_idx, node_idx, node_xyz = closest
        self._revision_state_by_key[image_key] = {
            "selected_path_index": int(path_idx),
            "selected_node_index": int(node_idx),
            "selected_node_xyz": node_xyz.tolist(),
            "preview_paths": None,
        }
        self._set_state(
            (
                "Revision node selected at "
                f"(x={node_xyz[0]:.1f}, y={node_xyz[1]:.1f}, z={node_xyz[2]:.1f}) for {image_key}."
            ),
            increment_token=True,
        )
        return {"selected_node_xyz": node_xyz.tolist()}

    def preview_trace_revision(self, image_key: str) -> Optional[List[List[List[float]]]]:
        state = self._revision_state_by_key.get(image_key)
        if state is None:
            self._set_state("Select a revision point before previewing.", increment_token=True)
            return None

        paths_xyz = _coerce_paths_xyz(self.trace_results_by_key.get(image_key, []))
        if len(paths_xyz) == 0:
            self._set_state(f"No trace available to preview for {image_key}.", increment_token=True)
            return None

        selected_path_idx = int(state.get("selected_path_index", -1))
        selected_node_idx = int(state.get("selected_node_index", -1))
        selected_node_xyz = np.asarray(state.get("selected_node_xyz", []), dtype=np.float32)
        if selected_node_xyz.size < 3:
            self._set_state("Invalid selected node for revision preview.", increment_token=True)
            return None

        trimmed_paths_xyz = _trim_paths_downstream(
            paths_xyz=paths_xyz,
            selected_path_idx=selected_path_idx,
            selected_node_idx=selected_node_idx,
            selected_node_xyz=selected_node_xyz,
        )
        preview_paths = [path.tolist() for path in trimmed_paths_xyz if path.shape[0] > 0]
        state["preview_paths"] = preview_paths
        self._set_state(
            f"Revision preview ready for {image_key} ({len(preview_paths)} path(s)).",
            increment_token=True,
        )
        return preview_paths

    def launch_trace_revision(
        self,
        image_index: int,
        image_key: str,
        volume_shape: tuple[int, int, int],
    ) -> Optional[List[List[List[float]]]]:
        if not self.enabled or self._runtime is None:
            self._set_state("Tracing is disabled (missing model config).", increment_token=True)
            return None
        if self._running:
            self._set_state("Trace All is running. Cancel it before launching revision retrace.", increment_token=True)
            return None

        state = self._revision_state_by_key.get(image_key)
        if state is None:
            self._set_state("Select a revision point before launching retrace.", increment_token=True)
            return None

        preview_paths = state.get("preview_paths")
        if not isinstance(preview_paths, list):
            preview_paths = self.preview_trace_revision(image_key=image_key)
            state = self._revision_state_by_key.get(image_key)
        if preview_paths is None:
            return None

        selected_node_xyz = np.asarray(state.get("selected_node_xyz", []), dtype=np.float32)
        if selected_node_xyz.size < 3:
            self._set_state("Invalid selected node for revision retrace.", increment_token=True)
            return None

        cached_mask = self.trace_mask_cache_by_key.get(image_key)
        mask_dtype = np.uint8 if cached_mask is None else cached_mask.dtype
        trimmed_mask = _draw_mask_from_paths_xyz(
            paths_xyz=_coerce_paths_xyz(preview_paths),
            shape_zyx=volume_shape,
            width=float(self.trace_params.get("step_width", 4.0)),
            mask_dtype=mask_dtype,
        )
        revision_seed_rows = [[
            float(selected_node_xyz[2]),
            float(selected_node_xyz[1]),
            float(selected_node_xyz[0]),
        ]]

        self._set_state(f"Launching revision retrace for {image_key}...", increment_token=False)
        result = self._runtime.trace_image(
            image_index=image_index,
            image_relative_key=image_key,
            seed_rows=revision_seed_rows,
            cancel_event=None,
            initial_path_mask=trimmed_mask,
        )
        paths = result["paths"]
        self.trace_results_by_key[image_key] = paths
        labeled_neuron = result.get("labeled_neuron", None)
        if labeled_neuron is not None:
            self.trace_mask_cache_by_key[image_key] = np.asarray(labeled_neuron)
        self._write_temp_trace(image_key=image_key, paths=paths)
        self._clear_derived_results(image_key)
        self._clear_revision_state(image_key)
        self._set_state(f"Revision retrace complete: {image_key}", increment_token=True)
        return paths

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
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._postprocess_output_dir = p
        self._set_state(f"Post-process output set to: {p}", increment_token=True)
        return str(p)

    def set_eval_output_dir(self, path: str) -> Optional[str]:
        """Set the directory where evaluation reports are written."""
        if not path or not str(path).strip():
            return None
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._eval_output_dir = p
        self._set_state(f"Eval output set to: {p}", increment_token=True)
        return str(p)

    def update_postprocess_config(self, overrides: Dict[str, object]) -> None:
        """Update postprocess/eval config parameters from the UI."""
        if "min_branch_length" in overrides:
            self.postprocess_config.min_branch_length = float(overrides["min_branch_length"])
        if "resampling_step_size" in overrides:
            self.postprocess_config.resampling_step_size = float(overrides["resampling_step_size"])
        if "smoothing_window" in overrides:
            self.postprocess_config.smoothing_window = int(overrides["smoothing_window"])
        if "overlap_threshold" in overrides:
            self.postprocess_config.overlap_threshold = float(overrides["overlap_threshold"])
        if "overlap_distance_threshold" in overrides:
            self.postprocess_config.overlap_distance_threshold = float(overrides["overlap_distance_threshold"])
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

    def set_gt_swc_dir(self, path: str) -> Optional[str]:
        if not path or not str(path).strip():
            return None
        p = Path(path)
        if not p.exists():
            self._set_state(f"GT SWC directory not found: {p}", increment_token=True)
            return None
        self._gt_swc_dir = p
        self._set_state(f"GT SWC directory set: {p}", increment_token=True)
        return str(p)

    def run_postprocess(self, image_key: str) -> Optional[Dict[str, object]]:
        """Post-process raw trace paths for *image_key* and store the result."""
        if image_key not in self.trace_results_by_key:
            self._set_state(
                f"No trace to post-process for {image_key}. Trace the image first.",
                increment_token=True,
            )
            return None
        raw_paths = self.trace_results_by_key[image_key]
        raw_result = {
            "neuron_name": image_key,
            "paths": [np.asarray(p, dtype=np.float32) for p in raw_paths],
        }
        try:
            self._set_state(f"Post-processing {image_key}...", increment_token=False)
            processed = process_results([raw_result], self.postprocess_config.scaled_params_for_image(image_key))
            if processed:
                result = processed[0]
                self.postprocess_results_by_key[image_key] = result
                n = result.get("n_processed_paths", 0)
                self._set_state(
                    f"Post-processing complete: {n} paths for {image_key}",
                    increment_token=True,
                )
                return result
        except Exception as exc:
            self._set_state(f"Post-processing failed: {exc}", increment_token=True)
        return None

    def run_evaluation(self, image_key: str) -> Optional[Dict[str, object]]:
        """Evaluate the post-processed result for *image_key* against the GT SWC."""
        if image_key not in self.postprocess_results_by_key:
            self._set_state(
                f"Run post-processing for {image_key} before evaluation.",
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
            pred_swc = self.postprocess_results_by_key[image_key].get("swc_list", [])
            if not pred_swc:
                self._set_state("Empty prediction — evaluation skipped.", increment_token=True)
                return None
            result = evaluate_reconstruction(
                pred_swc, gt_swc,
                threshold=self.postprocess_config.distance_threshold / self.postprocess_config.get_scale_for_image(image_key),
            )
            result["image_key"] = image_key
            result["gt_file"] = str(gt_file)
            self.eval_results_by_key[image_key] = result
            self._set_state(f"Evaluation complete for {image_key}", increment_token=True)
            return result
        except Exception as exc:
            self._set_state(f"Evaluation failed: {exc}", increment_token=True)
            return None

    def save_postprocessed(self, image_key: str, default_dir: Path) -> None:
        """Write the post-processed SWC for *image_key* to disk."""
        if image_key not in self.postprocess_results_by_key:
            self._set_state(
                f"No post-processed data to save for {image_key}.",
                increment_token=True,
            )
            return
        if self._postprocess_output_dir is None:
            selected = prompt_select_directory(default_path=str(default_dir))
            if selected is None:
                return
            self._postprocess_output_dir = Path(selected)
        self._postprocess_output_dir.mkdir(parents=True, exist_ok=True)
        result = self.postprocess_results_by_key[image_key]
        swc_list = result.get("swc_list", [])
        if not swc_list:
            self._set_state("Empty SWC — nothing to save.", increment_token=True)
            return
        neuron_stem = Path(image_key).stem
        swc_path = self._postprocess_output_dir / f"{neuron_stem}_reconstructed.swc"
        data_save.write_swc(swc_list, str(swc_path))
        self._set_state(f"Saved post-processed SWC: {swc_path}", increment_token=True)

    def save_eval_report(self, image_key: str, default_dir: Path) -> None:
        """Write the evaluation report for *image_key* as JSON to disk."""
        if image_key not in self.eval_results_by_key:
            self._set_state(
                f"No evaluation data to save for {image_key}.",
                increment_token=True,
            )
            return
        if self._eval_output_dir is None:
            selected = prompt_select_directory(default_path=str(default_dir))
            if selected is None:
                return
            self._eval_output_dir = Path(selected)
        self._eval_output_dir.mkdir(parents=True, exist_ok=True)
        result = {k: (v.item() if hasattr(v, "item") else v) for k, v in self.eval_results_by_key[image_key].items()}
        neuron_stem = Path(image_key).stem
        report_path = self._eval_output_dir / f"{neuron_stem}_eval_report.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        self._set_state(f"Saved eval report: {report_path}", increment_token=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, current_key: str) -> Dict[str, object]:
        with self._state_lock:
            status: Dict[str, object] = {
                "running": self._running,
                "message": self._message,
                "token": self._token,
                "overlay_paths": self.trace_results_by_key.get(current_key, []),
                "trace_output_dir": None if self._trace_output_dir is None else str(self._trace_output_dir),
                "model_weights_path": self.get_model_weights_path(),
                "progress_completed": self._progress_completed,
                "progress_total": self._progress_total,
                "postprocess_paths": None,
                "eval_report_text": None,
                "gt_swc_path": self.get_gt_swc_path(),
            }
            pp_result = self.postprocess_results_by_key.get(current_key)
            if pp_result is not None:
                raw_post = pp_result.get("processed_paths", [])
                status["postprocess_paths"] = [
                    p.tolist() if hasattr(p, "tolist") else list(p)
                    for p in raw_post
                ]
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

    def select_trace_output_dir(self, default_dir: Path) -> Optional[str]:
        selected_dir = prompt_select_directory(default_path=str(default_dir))
        if selected_dir is None:
            return None
        self._trace_output_dir = Path(selected_dir)
        self._trace_output_dir.mkdir(parents=True, exist_ok=True)
        self._set_state(f"Trace output set to: {self._trace_output_dir}", increment_token=True)

    def update_trace_params(self, overrides: Dict[str, object]) -> None:
        """Update runtime trace parameters and reload the runtime if weights are available."""
        self.trace_params.update(overrides)
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

        self._set_state(f"Tracing {image_key}...", increment_token=False)
        result = self._runtime.trace_image(
            image_index=image_index,
            image_relative_key=image_key,
            seed_rows=seed_rows,
            cancel_event=None,
        )
        paths = result["paths"]
        self.trace_results_by_key[image_key] = paths
        labeled_neuron = result.get("labeled_neuron", None)
        if labeled_neuron is not None:
            self.trace_mask_cache_by_key[image_key] = np.asarray(labeled_neuron)
        self._write_temp_trace(image_key=image_key, paths=paths)
        self._clear_derived_results(image_key)
        self._clear_revision_state(image_key)
        self._set_state(f"Trace complete: {image_key}", increment_token=True)
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
                    result = self._runtime.trace_image(
                        image_index=idx,
                        image_relative_key=key,
                        seed_rows=seed_rows,
                        cancel_event=self._cancel_event,
                    )
                    paths = result["paths"]
                    self.trace_results_by_key[key] = paths
                    labeled_neuron = result.get("labeled_neuron", None)
                    if labeled_neuron is not None:
                        self.trace_mask_cache_by_key[key] = np.asarray(labeled_neuron)
                    self._write_temp_trace(image_key=key, paths=paths)
                    self._clear_derived_results(key)
                    self._clear_revision_state(key)
                    with self._state_lock:
                        self._progress_completed = idx + 1
                    self._set_state(f"Completed {idx + 1}/{total}: {key}", increment_token=True)
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
        return {
            "image_data": image_array,
            "neuron_name": relative_key,
            "initial_seeds": initial_seeds,
            "show_prev_button": index > 0,
            "show_next_button": index < len(self.image_paths) - 1,
            "finished_paths": trace_manager.trace_results_by_key.get(relative_key, []),
            "seeds_output_path": self.seeds_output_path,
            "trace_output_path": trace_manager.get_trace_output_dir(),
            "model_weights_path": trace_manager.get_model_weights_path(),
            "gt_swc_path": trace_manager.get_gt_swc_path(),
            "scales_path": trace_manager.get_scales_path(),
            "image_dir": str(self.image_root),
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
        selected = prompt_save_json_path(default_path=str(self.image_root / "seeds.json"))
        if selected:
            self.seeds_output_path = selected
        return self.seeds_output_path

    def save_current(self, seed_array: np.ndarray) -> None:
        relative_key = self.current_relative_key()
        self.selected_seeds[relative_key] = self.rows_from_seed_array(seed_array)
        out_path = self.ensure_output_path()
        save_seeds_json(seeds_json_path=out_path, seeds_by_relative_path=self.selected_seeds)
        print(f"Saved seeds for {relative_key} to: {out_path}")

    def save_all(self) -> None:
        out_path = self.ensure_output_path()
        save_seeds_json(seeds_json_path=out_path, seeds_by_relative_path=self.selected_seeds)
        print(f"Saved all seeds to: {out_path}")

    # ------------------------------------------------------------------
    # Trace callback
    # ------------------------------------------------------------------

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

    def select_trace_revision_node(
        self,
        selected_point_zyx: np.ndarray,
        trace_manager: _TraceSessionManager,
    ) -> Optional[Dict[str, object]]:
        return trace_manager.select_revision_node(
            image_key=self.current_relative_key(),
            selected_point_zyx=selected_point_zyx,
        )

    def preview_trace_revision(
        self,
        trace_manager: _TraceSessionManager,
    ) -> Optional[List[List[List[float]]]]:
        return trace_manager.preview_trace_revision(image_key=self.current_relative_key())

    def launch_trace_revision(
        self,
        trace_manager: _TraceSessionManager,
    ) -> Optional[List[List[List[float]]]]:
        return trace_manager.launch_trace_revision(
            image_index=self.current_index,
            image_key=self.current_relative_key(),
            volume_shape=self.current_volume_shape,
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


def run_interactive_tracing_session(
    image_dir: Optional[str] = None,
    seeds_output_path: Optional[str] = None,
    seeds_input_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, list]:
    """Run an interactive tracing session: seed selection, tracing, post-processing, and evaluation."""
    config = _load_optional_session_config(config_path)
    image_dir = image_dir or config.get("image_dir")
    seeds_input_path = seeds_input_path if seeds_input_path is not None else config.get("seeds_input_path")
    seeds_output_path = seeds_output_path if seeds_output_path is not None else config.get("seeds_output_path")

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
        "rng_seed": config.get("rng_seed", 0),
        "step_width": config.get("step_width", 4.0),
        "max_len": config.get("max_len", 10000),
        "max_paths": config.get("max_paths", 1000),
        "branching": config.get("branching", True),
        "repeat_starts": config.get("repeat_starts", False),
        "n_trials": config.get("n_trials", 1),
        "stochastic_actions": config.get("stochastic_actions", False),
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
    )

    # ---- small local closures for operations that need both session + external state ----

    def _select_model_weights_path() -> Optional[str]:
        current_weights = trace_manager.get_model_weights_path()
        selected = prompt_select_model_weights(default_path=current_weights)
        if selected is None:
            return current_weights
        loaded = trace_manager.set_model_weights_path(selected)
        return loaded if loaded is not None else current_weights

    def _select_image_dir() -> Optional[str]:
        """Prompt for a new image directory (informational only — session scope is fixed at startup)."""
        qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
        selected = qt_widgets_mod.QFileDialog.getExistingDirectory(
            None, "Select image directory", str(image_root)
        )
        return selected or None

    def _select_gt_swc_path() -> Optional[str]:
        selected = prompt_select_directory(
            default_path=config.get("swc_dir") or str(image_root)
        )
        if selected:
            trace_manager.set_gt_swc_dir(selected)
        return trace_manager.get_gt_swc_path()

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

    try:
        initial_context = session.build_context(session.current_index, trace_manager)
        final_seeds = interactive_seed_selection_session(
            initial_context=initial_context,
            on_prev_image=lambda arr: session.on_prev_image(arr, trace_manager),
            on_next_image=lambda arr: session.on_next_image(arr, trace_manager),
            on_save_current=session.save_current,
            on_save_all=session.save_all,
            show_trace_controls=True,
            on_trace_current=lambda arr: session.trace_current(arr, trace_manager),
            on_trace_all=lambda: trace_manager.start_trace_all(session.selected_seeds),
            on_cancel_trace=trace_manager.cancel_trace_all,
            on_trace_revision_select_point=lambda arr: session.select_trace_revision_node(arr, trace_manager),
            on_trace_revision_preview=lambda: session.preview_trace_revision(trace_manager),
            on_trace_revision_launch=lambda: session.launch_trace_revision(trace_manager),
            get_trace_status=lambda: trace_manager.get_status(current_key=session.current_relative_key()),
            on_save_trace=lambda: trace_manager.save_trace(
                session.current_relative_key(), default_dir=image_root / "trace_outputs"
            ),
            on_save_all_traces=lambda: trace_manager.save_all_traces(
                default_dir=image_root / "trace_outputs"
            ),
            on_select_seeds_output_path=session.select_seeds_output_path,
            on_select_trace_output_path=lambda: trace_manager.select_trace_output_dir(
                default_dir=image_root / "trace_outputs"
            ),
            on_select_model_weights_path=_select_model_weights_path,
            on_select_image_dir=_select_image_dir,
            on_select_seeds_input_path=session.select_seeds_input_path,
            trace_step_width=float(trace_params.get("step_width", 4.0)),
            trace_n_trials=int(trace_params.get("n_trials", 1)),
            trace_max_len=int(trace_params.get("max_len", 10000)),
            trace_max_paths=int(trace_params.get("max_paths", 1000)),
            trace_branching=bool(trace_params.get("branching", True)),
            trace_repeat_starts=bool(trace_params.get("repeat_starts", False)),
            trace_stochastic_actions=bool(trace_params.get("stochastic_actions", False)),
            trace_auto_seed_mode=str(trace_params.get("auto_seed_selection_mode", "remote_endnode")),
            on_trace_params_changed=trace_manager.update_trace_params,
            show_postprocess_controls=True,
            on_run_postprocess=lambda: trace_manager.run_postprocess(session.current_relative_key()),
            on_run_evaluation=lambda: trace_manager.run_evaluation(session.current_relative_key()),
            on_save_postprocessed=lambda: trace_manager.save_postprocessed(
                session.current_relative_key(), default_dir=image_root / "postprocessed"
            ),
            on_save_eval_report=lambda: trace_manager.save_eval_report(
                session.current_relative_key(), default_dir=image_root / "evaluation"
            ),
            on_select_gt_swc_path=_select_gt_swc_path,
            on_select_scales_path=_select_scales_path,
            postprocess_output_dir=trace_manager.get_postprocess_output_dir(),
            postprocess_min_branch_length=postprocess_config.min_branch_length,
            postprocess_resampling_step_size=postprocess_config.resampling_step_size,
            postprocess_smoothing_window=postprocess_config.smoothing_window,
            postprocess_overlap_threshold=postprocess_config.overlap_threshold,
            postprocess_overlap_distance_threshold=postprocess_config.overlap_distance_threshold,
            on_select_postprocess_output_dir=lambda: trace_manager.set_postprocess_output_dir(
                prompt_select_directory(default_path=str(image_root / "postprocessed")) or ""
            ),
            on_postprocess_params_changed=trace_manager.update_postprocess_config,
            eval_output_dir=trace_manager.get_eval_output_dir(),
            eval_distance_threshold=postprocess_config.distance_threshold,
            on_select_eval_output_dir=lambda: trace_manager.set_eval_output_dir(
                prompt_select_directory(default_path=str(image_root / "evaluation")) or ""
            ),
            on_eval_params_changed=trace_manager.update_postprocess_config,
        )
        session.selected_seeds[session.current_relative_key()] = _normalize_seed_array(
            final_seeds.detach().cpu().numpy(),
            session.current_volume_shape,
        )
    finally:
        trace_manager.close()

    if session.seeds_output_path is not None:
        save_seeds_json(
            seeds_json_path=session.seeds_output_path,
            seeds_by_relative_path=session.selected_seeds,
        )
        print(f"Saved seeds JSON to: {session.seeds_output_path}")
    else:
        print("Session ended without writing seeds to disk (no output path selected).")

    return session.selected_seeds