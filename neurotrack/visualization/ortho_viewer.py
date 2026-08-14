#!/usr/bin/env python

"""
Interactive orthoview-based tracing UI.

Provides a Qt dialog with synchronized XY / XZ / YZ orthoviews for manual
seed placement, tracing inference overlay review, trace post-processing, evaluating, and editing.

Author: Bryson Gray
2024
"""

import importlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
import torch

from neurotrack.visualization.editor_state import AnnotationTarget, ViewerSessionState
from neurotrack.visualization.selection_utils import (
    annotation_node_visible_in_view,
    annotation_plot_coords,
    hit_test_annotation_node,
    hit_test_seed,
    seed_plot_coords,
    seed_visible_in_view,
    visible_seed_indices,
)
from neurotrack.visualization._qt_utils import (
    _is_jupyter_notebook,
    _has_gui_display,
    _try_import_ui_dependencies,
    _ensure_qapplication,
)


def _extract_first_channel_numpy(image_data: np.ndarray) -> np.ndarray:
    """Convert image array to a numpy volume (Z, Y, X) using first channel if needed."""
    image_np = np.asarray(image_data)
    if image_np.ndim == 4:
        return image_np[0]
    return image_np


def _normalize_swc_rows(swc_rows: Optional[object]) -> np.ndarray:
    """Return SWC rows as float array with shape (N, 7) or empty array."""
    if swc_rows is None:
        return np.empty((0, 7), dtype=np.float32)
    arr = np.asarray(swc_rows, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7:
        return np.empty((0, 7), dtype=np.float32)
    return arr[:, :7].copy()


class _OrthoViewDialog:
    """Qt dialog with synchronized XY/XZ/YZ orthoviews and controls."""

    @property
    def selected_seed_index(self) -> Optional[int]:
        return self._editor_state.selection.selected_seed_index

    @selected_seed_index.setter
    def selected_seed_index(self, value: Optional[int]) -> None:
        self._editor_state.selection.selected_seed_index = value

    def __init__(
        self,
        image_data: np.ndarray,
        mode: str,
        finished_paths=None,
        tree_swc_rows: Optional[object] = None,
        neuron_name: str = "",
        initial_seeds: Optional[np.ndarray] = None,
        effective_seed_overlay: Optional[np.ndarray] = None,
        show_prev_button: bool = False,
        show_next_button: bool = False,
        show_save_buttons: bool = False,
        on_save_current: Optional[Callable[[np.ndarray], None]] = None,
        on_save_all: Optional[Callable[[], None]] = None,
        show_trace_controls: bool = False,
        on_trace_current: Optional[Callable[[np.ndarray], Optional[List[np.ndarray]]]] = None,
        on_trace_all: Optional[Callable[[], None]] = None,
        on_cancel_trace: Optional[Callable[[], None]] = None,
        get_trace_status: Optional[Callable[[str], Dict[str, object]]] = None,
        on_save_trace: Optional[Callable[[], None]] = None,
        on_save_all_traces: Optional[Callable[[], None]] = None,
        on_discard_trace: Optional[Callable[[], None]] = None,
        seeds_output_path: Optional[str] = None,
        trace_output_path: Optional[str] = None,
        on_select_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
        on_select_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
        on_clear_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
        on_clear_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
        model_weights_path: Optional[str] = None,
        on_select_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
        on_clear_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
        image_dir: Optional[str] = None,
        seeds_input_path: Optional[str] = None,
        on_select_image_dir: Optional[Callable[[], Optional[str]]] = None,
        on_select_seeds_input_path: Optional[Callable[[], Optional[str]]] = None,
        on_clear_image_dir: Optional[Callable[[], Optional[str]]] = None,
        on_clear_seeds_input_path: Optional[Callable[[], Optional[str]]] = None,
        on_prev_image: Optional[Callable[[np.ndarray], Optional[Dict[str, object]]]] = None,
        on_next_image: Optional[Callable[[np.ndarray], Optional[Dict[str, object]]]] = None,
        on_get_effective_seed_overlay: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
        trace_max_len: int = 10000,
        trace_max_paths: int = 1000,
        trace_branching: bool = True,
        trace_repeat_starts: bool = False,
        trace_seed_jitter_count: int = 0,
        trace_seed_jitter_radius: float = 0.0,
        trace_seed_jitter_weight_strategy: str = "uniform",
        on_trace_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
        gt_swc_path: Optional[str] = None,
        scales_path: Optional[str] = None,
        show_postprocess_controls: bool = False,
        on_run_postprocess: Optional[Callable[[str], None]] = None,
        on_run_postprocess_all: Optional[Callable[[str], None]] = None,
        on_undo_postprocess: Optional[Callable[[str], None]] = None,
        on_run_evaluation: Optional[Callable[[], None]] = None,
        on_run_evaluation_all: Optional[Callable[[], None]] = None,
        on_save_eval_report: Optional[Callable[[], None]] = None,
        on_select_gt_swc_path: Optional[Callable[[], object]] = None,
        on_clear_gt_swc_path: Optional[Callable[[], Optional[str]]] = None,
        on_select_scales_path: Optional[Callable[[], Optional[str]]] = None,
        on_clear_scales_path: Optional[Callable[[], Optional[str]]] = None,
        postprocess_output_dir: Optional[str] = None,
        postprocess_enable_length_filter: bool = True,
        postprocess_min_branch_length: float = 5.0,
        postprocess_max_branch_length: float = 1e9,
        postprocess_enable_resample: bool = True,
        postprocess_resampling_step_size: float = 4.0,
        postprocess_enable_smooth_paths: bool = True,
        postprocess_smoothing_window: int = 5,
        postprocess_enable_merge: bool = True,
        postprocess_join_roots_to_common_center: bool = True,
        postprocess_merge_threshold: float = 1.0,
        postprocess_confidence_threshold: int = 0,
        postprocess_mask_smoothing_size: int = 0,
        postprocess_merge_timeout_seconds: float = 30.0,
        on_select_postprocess_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_clear_postprocess_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_postprocess_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
        eval_output_dir: Optional[str] = None,
        eval_distance_threshold: float = 1.0,
        on_select_eval_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_clear_eval_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_eval_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
        filtered_swc_output_dir: Optional[str] = None,
        on_select_filtered_swc_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_clear_filtered_swc_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_save_filtered_swc: Optional[Callable[[str, List[List[float]]], Optional[str]]] = None,
        on_filtered_swc_changed: Optional[Callable[[str, List[List[float]]], None]] = None,
        on_prediction_paths_changed: Optional[Callable[[str, List[List[List[float]]]], None]] = None,
    ):
        _ensure_qapplication()
        QApplication, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, Qt, FigureCanvas = _try_import_ui_dependencies()

        self.QApplication = QApplication
        self.Qt = Qt
        self.mode = mode
        self.img_np = _extract_first_channel_numpy(image_data)
        self.shape = self.img_np.shape
        self._editor_state = ViewerSessionState()
        self.finished_paths: List[np.ndarray] = []
        self._tree_swc_committed = np.empty((0, 7), dtype=np.float32)
        self._set_prediction_paths(finished_paths)
        self._set_reference_swc_rows(tree_swc_rows)

        self.current_z = int(self.shape[0] // 2)
        self.current_y = int(self.shape[1] // 2)
        self.current_x = int(self.shape[2] // 2)
        self.maximized_view: Optional[str] = None
        self.projection_mode = "slice"
        self._mip_cache_by_view: Dict[str, Optional[np.ndarray]] = {"xy": None, "xz": None, "yz": None}
        self.seeds: List[Tuple[int, int, int]] = []
        if initial_seeds is not None:
            seed_arr = np.asarray(initial_seeds, dtype=np.float32)
            if seed_arr.ndim == 2 and seed_arr.shape[1] == 3:
                seed_arr[:, 0] = np.clip(seed_arr[:, 0], 0, self.shape[0] - 1)
                seed_arr[:, 1] = np.clip(seed_arr[:, 1], 0, self.shape[1] - 1)
                seed_arr[:, 2] = np.clip(seed_arr[:, 2], 0, self.shape[2] - 1)
                self.seeds = [
                    (int(round(z)), int(round(y)), int(round(x)))
                    for z, y, x in seed_arr
                ]
        self.effective_seed_overlay: List[Tuple[int, int, int]] = []
        if effective_seed_overlay is not None:
            overlay_arr = np.asarray(effective_seed_overlay, dtype=np.float32)
            if overlay_arr.ndim == 2 and overlay_arr.shape[1] == 3:
                overlay_arr[:, 0] = np.clip(overlay_arr[:, 0], 0, self.shape[0] - 1)
                overlay_arr[:, 1] = np.clip(overlay_arr[:, 1], 0, self.shape[1] - 1)
                overlay_arr[:, 2] = np.clip(overlay_arr[:, 2], 0, self.shape[2] - 1)
                self.effective_seed_overlay = [
                    (int(round(z)), int(round(y)), int(round(x)))
                    for z, y, x in overlay_arr
                ]
        self.zoom_limits = {}
        self.zoom_history = {"xy": [], "xz": [], "yz": []}
        self._active_view = "xy"
        self._drag_start = None
        self._drag_view = None
        self._drag_rect = None
        self._shift_held = False
        self._layout_dirty = True
        self._current_views = []
        self.axes_by_view = {}
        self.image_artists = {}
        self.crosshair_artists = {}
        self.overlay_artists = {}
        self.selection_artists = {}
        self._supports_blit = False
        self._blit_background_by_view: Dict[str, object] = {}
        self._blit_background_valid = False
        self.session_action = "finish"
        self._on_save_current = on_save_current
        self._on_save_all = on_save_all
        self._show_trace_controls = bool(show_trace_controls and mode == "seed")
        self._on_trace_current = on_trace_current
        self._on_trace_all = on_trace_all
        self._on_cancel_trace = on_cancel_trace
        self._get_trace_status = get_trace_status
        self._on_save_trace = on_save_trace
        self._on_save_all_traces = on_save_all_traces
        self._on_discard_trace = on_discard_trace
        self._on_select_seeds_output_path = on_select_seeds_output_path
        self._on_select_trace_output_path = on_select_trace_output_path
        self._on_clear_seeds_output_path = on_clear_seeds_output_path
        self._on_clear_trace_output_path = on_clear_trace_output_path
        self._seeds_output_path = seeds_output_path
        self._trace_output_path = trace_output_path
        self._model_weights_path = model_weights_path
        self._on_select_model_weights_path = on_select_model_weights_path
        self._on_clear_model_weights_path = on_clear_model_weights_path
        self._on_prev_image = on_prev_image
        self._on_next_image = on_next_image
        self._on_get_effective_seed_overlay = on_get_effective_seed_overlay
        self._show_postprocess_controls = bool(show_postprocess_controls and mode == "seed")
        self._on_run_postprocess = on_run_postprocess
        self._on_run_postprocess_all = on_run_postprocess_all
        self._on_undo_postprocess = on_undo_postprocess
        self._on_run_evaluation = on_run_evaluation
        self._on_run_evaluation_all = on_run_evaluation_all
        self._on_save_eval_report = on_save_eval_report
        self._gt_swc_path = gt_swc_path
        self._on_select_gt_swc_path = on_select_gt_swc_path
        self._on_clear_gt_swc_path = on_clear_gt_swc_path
        self._scales_path = scales_path
        self._on_select_scales_path = on_select_scales_path
        self._on_clear_scales_path = on_clear_scales_path
        self._image_dir = image_dir
        self._seeds_input_path = seeds_input_path
        self._on_select_image_dir = on_select_image_dir
        self._on_select_seeds_input_path = on_select_seeds_input_path
        self._on_clear_image_dir = on_clear_image_dir
        self._on_clear_seeds_input_path = on_clear_seeds_input_path
        self._on_trace_params_changed = on_trace_params_changed
        self._postprocess_output_dir = postprocess_output_dir
        self._eval_output_dir = eval_output_dir
        self._on_select_postprocess_output_dir = on_select_postprocess_output_dir
        self._on_clear_postprocess_output_dir = on_clear_postprocess_output_dir
        self._on_postprocess_params_changed = on_postprocess_params_changed
        self._on_select_eval_output_dir = on_select_eval_output_dir
        self._on_clear_eval_output_dir = on_clear_eval_output_dir
        self._on_eval_params_changed = on_eval_params_changed
        self._filtered_swc_output_dir = filtered_swc_output_dir
        self._on_select_filtered_swc_output_dir = on_select_filtered_swc_output_dir
        self._on_clear_filtered_swc_output_dir = on_clear_filtered_swc_output_dir
        self._on_save_filtered_swc = on_save_filtered_swc
        self._on_filtered_swc_changed = on_filtered_swc_changed
        self._on_prediction_paths_changed = on_prediction_paths_changed
        self._current_image_key = str(neuron_name or "")

        self._annotation_undo_stack: List[Tuple[str, np.ndarray, List[List[List[float]]]]] = []

        self._tree_swc_preview_source = np.empty((0, 7), dtype=np.float32)
        self._tree_swc_preview_filtered = np.empty((0, 7), dtype=np.float32)
        self._tree_preview_seed_points: List[Tuple[float, float, float]] = []
        self._tree_preview_removed_ids: set[int] = set()
        self._has_clip_preview = False
        self._active_tool = "zoom"
        self._tree_overlay_cache_source_id: Optional[int] = None
        self._tree_overlay_cache: Dict[str, np.ndarray] = {
            "child_idx": np.empty((0,), dtype=np.int64),
            "parent_idx": np.empty((0,), dtype=np.int64),
            "child_xyz": np.empty((0, 3), dtype=np.float32),
            "parent_xyz": np.empty((0, 3), dtype=np.float32),
        }
        self.selected_seed_index: Optional[int] = None

        self._trace_status_token = None
        self._trace_overlay_token = None
        self._trace_postprocess_token = None
        self._trace_controls_running_state: Optional[bool] = None
        self._last_trace_status_message = ""
        self._last_trace_progress_text = ""
        self._pending_trace_param_overrides: Optional[Dict[str, object]] = None
        self._trace_params_debounce_timer = None
        self.trace_overlay_visible = self._editor_state.show_prediction_overlay
        self.gt_overlay_visible = self._editor_state.show_reference_overlay
        self._qt_timer = None
        self._show_prev_button = bool(show_prev_button)
        self._show_next_button = bool(show_next_button)
        self._can_undo_postprocess = False

        self.dialog = QDialog()
        title = "Viewing Image" if mode == "seed" else "Inference Overlay"
        if neuron_name:
            title = f"{title}: {neuron_name}"
        self.dialog.setWindowTitle(title)
        self.dialog.resize(1400, 860)

        # Root layout: vertical (top controls, canvas row, footer)
        outer = QVBoxLayout(self.dialog)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Main horizontal row: left sidebar | center canvas
        # Uses QSplitter so each panel can be drag-resized by the user.
        _QtWidgetsMod = importlib.import_module("qtpy.QtWidgets")
        main_splitter = _QtWidgetsMod.QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(True)
        main_splitter.setHandleWidth(5)
        outer.addWidget(main_splitter, stretch=1)

        # -----------------------------------------------------------------
        # Create all buttons unconditionally so references are always valid
        # -----------------------------------------------------------------
        self.btn_all = QPushButton("All")
        self.btn_xy = QPushButton("XY")
        self.btn_xz = QPushButton("XZ")
        self.btn_yz = QPushButton("YZ")
        self.btn_back = QPushButton("◀ Back")
        self.btn_home = QPushButton("Home")
        self.btn_prev_image = QPushButton("◀ Prev Image")
        self.btn_next_image = QPushButton("Next Image ▶")
        self.btn_save_all_seeds = QPushButton("Save All Seeds")
        self.btn_trace_neuron = QPushButton("Trace Neuron")
        self.btn_trace_all = QPushButton("Trace All")
        self.btn_cancel_trace = QPushButton("Cancel Trace")
        self.btn_save_trace = QPushButton("Save Trace")
        self.btn_save_all_traces = QPushButton("Save All Traces")
        self.btn_discard_trace = QPushButton("Discard Trace")
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["Slice", "MIP"])
        _qt_w = importlib.import_module("qtpy.QtWidgets")
        self.chk_trace_overlay = _qt_w.QCheckBox("Show Prediction Overlay")
        self.chk_gt_overlay = _qt_w.QCheckBox("Show Reference Overlay")
        self.chk_trace_overlay.setChecked(self.trace_overlay_visible)
        self.chk_gt_overlay.setChecked(self.gt_overlay_visible)
        self.trace_progress_label = QLabel("")
        self.btn_run_postprocess = QPushButton("Run Post-Processing")
        self.btn_run_postprocess_all = QPushButton("Post-Process All")
        self.btn_undo_postprocess = QPushButton("Undo Post-Process")
        self.btn_run_evaluation = QPushButton("Run Evaluation")
        self.btn_run_evaluation_all = QPushButton("Evaluate All")
        self.btn_save_eval_report = QPushButton("Save Eval Report")
        if mode == "seed":
            self.btn_undo = QPushButton("Undo Last Seed")
            self.btn_clear = QPushButton("Clear Seeds")
            self.btn_save_seeds = QPushButton("Save Seeds")
            self.btn_set_seeds_output = QPushButton("Set Seeds Output")
            self.btn_clear_seeds_output = QPushButton("Unset Seeds Output")
            self.btn_set_trace_output = QPushButton("Set Trace Output")
            self.btn_clear_trace_output = QPushButton("Unset Trace Output")
            self.btn_set_model_weights = QPushButton("Set Model Weights")
            self.btn_clear_model_weights = QPushButton("Unset Model Weights")
            self.seeds_output_value_label = QLabel("")
            self.seeds_output_value_label.setWordWrap(True)
            self.seeds_output_value_label.setMinimumWidth(0)
            self.trace_output_value_label = QLabel("")
            self.trace_output_value_label.setWordWrap(True)
            self.trace_output_value_label.setMinimumWidth(0)
            self.model_weights_value_label = QLabel("")
            self.model_weights_value_label.setWordWrap(True)
            self.model_weights_value_label.setMinimumWidth(0)
            self.btn_set_image_dir = QPushButton("Set Image Dir")
            self.btn_clear_image_dir = QPushButton("Unset Image Dir")
            self.image_dir_value_label = QLabel("")
            self.image_dir_value_label.setWordWrap(True)
            self.image_dir_value_label.setMinimumWidth(0)
            self.btn_set_seeds_input = QPushButton("Set Seeds Input")
            self.btn_clear_seeds_input = QPushButton("Unset Seeds Input")
            self.seeds_input_value_label = QLabel("")
            self.seeds_input_value_label.setWordWrap(True)
            self.seeds_input_value_label.setMinimumWidth(0)
            self.btn_rejitter_seeds = QPushButton("Rejitter Seeds")
            # Advanced trace parameter widgets
            _qt_w = importlib.import_module("qtpy.QtWidgets")
            self._trace_max_len_spin = _qt_w.QSpinBox()
            self._trace_max_len_spin.setRange(1, 1000000)
            self._trace_max_len_spin.setValue(trace_max_len)
            self._trace_max_len_spin.setSizePolicy(
                _qt_w.QSizePolicy.Expanding, _qt_w.QSizePolicy.Fixed)
            self._trace_max_len_spin.setMinimumWidth(0)
            self._trace_max_paths_spin = _qt_w.QSpinBox()
            self._trace_max_paths_spin.setRange(1, 99999)
            self._trace_max_paths_spin.setValue(trace_max_paths)
            self._trace_max_paths_spin.setSizePolicy(
                _qt_w.QSizePolicy.Expanding, _qt_w.QSizePolicy.Fixed)
            self._trace_max_paths_spin.setMinimumWidth(0)
            self._trace_branching_check = _qt_w.QCheckBox("Branching")
            self._trace_branching_check.setChecked(trace_branching)
            self._trace_repeat_starts_check = _qt_w.QCheckBox("Repeat Starts")
            self._trace_repeat_starts_check.setChecked(trace_repeat_starts)
            self._trace_seed_jitter_count_spin = _qt_w.QSpinBox()
            self._trace_seed_jitter_count_spin.setRange(0, 10000)
            self._trace_seed_jitter_count_spin.setValue(int(max(0, trace_seed_jitter_count)))
            self._trace_seed_jitter_count_spin.setSizePolicy(
                _qt_w.QSizePolicy.Expanding, _qt_w.QSizePolicy.Fixed)
            self._trace_seed_jitter_count_spin.setMinimumWidth(0)
            self._trace_seed_jitter_radius_spin = _qt_w.QDoubleSpinBox()
            self._trace_seed_jitter_radius_spin.setRange(0.0, 1000.0)
            self._trace_seed_jitter_radius_spin.setSingleStep(0.5)
            self._trace_seed_jitter_radius_spin.setDecimals(2)
            self._trace_seed_jitter_radius_spin.setValue(float(max(0.0, trace_seed_jitter_radius)))
            self._trace_seed_jitter_radius_spin.setSizePolicy(
                _qt_w.QSizePolicy.Expanding, _qt_w.QSizePolicy.Fixed)
            self._trace_seed_jitter_radius_spin.setMinimumWidth(0)
            self._trace_seed_jitter_weight_combo = QComboBox()
            self._trace_seed_jitter_weight_combo.addItems([
                "uniform",
                "intensity_weighted",
                "boundary_weighted",
            ])
            self._trace_seed_jitter_weight_combo.setSizePolicy(
                _qt_w.QSizePolicy.Expanding, _qt_w.QSizePolicy.Fixed)
            self._trace_seed_jitter_weight_combo.setMinimumWidth(0)
            _jitter_weight_idx = self._trace_seed_jitter_weight_combo.findText(
                str(trace_seed_jitter_weight_strategy).strip().lower()
            )
            if _jitter_weight_idx >= 0:
                self._trace_seed_jitter_weight_combo.setCurrentIndex(_jitter_weight_idx)
            # Post-processing parameter widgets
            self.btn_set_postprocess_output = QPushButton("Set Post-Process Output")
            self.btn_clear_postprocess_output = QPushButton("Unset Post-Process Output")
            self.postprocess_output_value_label = QLabel("")
            self.postprocess_output_value_label.setWordWrap(True)
            self.postprocess_output_value_label.setMinimumWidth(0)
            self._pp_enable_length_filter_check = _qt_w.QCheckBox("Filter branches by length")
            self._pp_enable_length_filter_check.setChecked(postprocess_enable_length_filter)
            self._pp_min_branch_length_spin = _qt_w.QDoubleSpinBox()
            self._pp_min_branch_length_spin.setRange(0.0, 1000.0)
            self._pp_min_branch_length_spin.setSingleStep(0.5)
            self._pp_min_branch_length_spin.setDecimals(2)
            self._pp_min_branch_length_spin.setValue(postprocess_min_branch_length)
            self._pp_max_branch_length_spin = _qt_w.QDoubleSpinBox()
            self._pp_max_branch_length_spin.setRange(0.0, 1e9)
            self._pp_max_branch_length_spin.setSingleStep(10.0)
            self._pp_max_branch_length_spin.setDecimals(2)
            _max_branch_length_value = postprocess_max_branch_length
            if not np.isfinite(_max_branch_length_value):
                _max_branch_length_value = 1e9
            self._pp_max_branch_length_spin.setValue(float(np.clip(_max_branch_length_value, 0.0, 1e9)))
            self._pp_enable_resample_check = _qt_w.QCheckBox("Resample")
            self._pp_enable_resample_check.setChecked(postprocess_enable_resample)
            self._pp_resampling_step_size_spin = _qt_w.QDoubleSpinBox()
            self._pp_resampling_step_size_spin.setRange(0.1, 100.0)
            self._pp_resampling_step_size_spin.setSingleStep(0.5)
            self._pp_resampling_step_size_spin.setDecimals(2)
            self._pp_resampling_step_size_spin.setValue(postprocess_resampling_step_size)
            self._pp_enable_smooth_paths_check = _qt_w.QCheckBox("Smooth paths")
            self._pp_enable_smooth_paths_check.setChecked(postprocess_enable_smooth_paths)
            self._pp_smoothing_window_spin = _qt_w.QSpinBox()
            self._pp_smoothing_window_spin.setRange(1, 100)
            self._pp_smoothing_window_spin.setValue(postprocess_smoothing_window)
            self._pp_enable_merge_check = _qt_w.QCheckBox("Merge overlapping paths")
            self._pp_enable_merge_check.setChecked(postprocess_enable_merge)
            self._pp_join_roots_check = _qt_w.QCheckBox("Join roots")
            self._pp_join_roots_check.setChecked(postprocess_join_roots_to_common_center)
            self._pp_overlap_dist_threshold_spin = _qt_w.QDoubleSpinBox()
            self._pp_overlap_dist_threshold_spin.setRange(0.0, 100.0)
            self._pp_overlap_dist_threshold_spin.setSingleStep(0.1)
            self._pp_overlap_dist_threshold_spin.setDecimals(2)
            self._pp_overlap_dist_threshold_spin.setValue(postprocess_merge_threshold)
            self._pp_confidence_threshold_spin = _qt_w.QSpinBox()
            self._pp_confidence_threshold_spin.setRange(0, 1000)
            self._pp_confidence_threshold_spin.setValue(int(postprocess_confidence_threshold))
            self._pp_mask_smoothing_size_spin = _qt_w.QSpinBox()
            self._pp_mask_smoothing_size_spin.setRange(0, 100)
            self._pp_mask_smoothing_size_spin.setValue(int(postprocess_mask_smoothing_size))
            self._pp_merge_timeout_seconds_spin = _qt_w.QDoubleSpinBox()
            self._pp_merge_timeout_seconds_spin.setRange(0.0, 3600.0)
            self._pp_merge_timeout_seconds_spin.setSingleStep(1.0)
            self._pp_merge_timeout_seconds_spin.setDecimals(1)
            self._pp_merge_timeout_seconds_spin.setValue(float(postprocess_merge_timeout_seconds))
            # Postprocess/eval path widgets (always created in seed mode)
            self.btn_set_gt_swc = QPushButton("Set Reference SWC Dir")
            self.btn_clear_gt_swc = QPushButton("Unset Reference SWC Dir")
            self.gt_swc_value_label = QLabel("")
            self.gt_swc_value_label.setWordWrap(True)
            self.gt_swc_value_label.setMinimumWidth(0)
            self.btn_set_scales_path = QPushButton("Set Scales JSON")
            self.btn_clear_scales_path = QPushButton("Unset Scales JSON")
            self.scales_path_value_label = QLabel("")
            self.scales_path_value_label.setWordWrap(True)
            self.scales_path_value_label.setMinimumWidth(0)
            # Evaluation parameter widgets
            self.btn_set_eval_output = QPushButton("Set Eval Output")
            self.btn_clear_eval_output = QPushButton("Unset Eval Output")
            self.eval_output_value_label = QLabel("")
            self.eval_output_value_label.setWordWrap(True)
            self.eval_output_value_label.setMinimumWidth(0)
            self._eval_distance_threshold_spin = _qt_w.QDoubleSpinBox()
            self._eval_distance_threshold_spin.setRange(0.0, 100.0)
            self._eval_distance_threshold_spin.setSingleStep(0.1)
            self._eval_distance_threshold_spin.setDecimals(2)
            self._eval_distance_threshold_spin.setValue(eval_distance_threshold)
            self.eval_scales_path_value_label = QLabel("")
            self.eval_scales_path_value_label.setWordWrap(True)
            self.eval_scales_path_value_label.setMinimumWidth(0)
            self.btn_set_eval_scales_path = QPushButton("Set Scales JSON")
            self.btn_clear_eval_scales_path = QPushButton("Unset Scales JSON")

            # Tool and filter widgets (left panel)
            self._tool_button_group = _qt_w.QButtonGroup(self.dialog)
            self.radio_tool_zoom = _qt_w.QRadioButton("Zoom")
            self.radio_tool_select = _qt_w.QRadioButton("Select")
            self.annotation_target_combo = QComboBox()
            self.radio_tool_zoom.setChecked(True)
            self._tool_button_group.addButton(self.radio_tool_zoom)
            self._tool_button_group.addButton(self.radio_tool_select)
            self.btn_remove_selected = QPushButton("Remove Selected")
            self.btn_clip_selected = QPushButton("Clip Selected")
            self.btn_undo_annotation = QPushButton("Undo Annotation Edit")
            self.btn_undo_annotation.setEnabled(False)
            self._component_min_length_spin = _qt_w.QSpinBox()
            self._component_min_length_spin.setRange(1, 1000000)
            self._component_min_length_spin.setValue(50)
            self.btn_apply_component_filter = QPushButton("Filter By Length")
            self.btn_set_filtered_swc_output = QPushButton("Set Filtered SWC Output")
            self.btn_clear_filtered_swc_output = QPushButton("Unset Filtered SWC Output")
            self.filtered_swc_output_value_label = QLabel("")
            self.filtered_swc_output_value_label.setWordWrap(True)
            self.filtered_swc_output_value_label.setMinimumWidth(0)
            self.btn_save_filtered_swc = QPushButton("Save Filtered SWC")

        # -----------------------------------------------------------------
        # LEFT SIDEBAR — view controls, sliders, seed & trace actions
        # -----------------------------------------------------------------
        left_sidebar = QWidget()
        left_sidebar.setObjectName("leftSidebar")
        left_layout = QVBoxLayout(left_sidebar)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(4)

        _left_tab_widget = None
        _left_edit_lay = None
        _left_trace_lay = None
        _left_eval_lay = None
        if mode == "seed":
            _qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
            _qt_core_mod = importlib.import_module("qtpy.QtCore")
            _left_tab_widget = _qt_widgets_mod.QTabWidget()
            _left_tab_widget.setDocumentMode(True)
            _left_tab_widget.setTabPosition(_qt_widgets_mod.QTabWidget.West)
            _left_tab_widget.setSizePolicy(
                _qt_widgets_mod.QSizePolicy.MinimumExpanding,
                _qt_widgets_mod.QSizePolicy.Expanding,
            )

            def _make_left_tab_scroll():
                _sa = _qt_widgets_mod.QScrollArea()
                _sa.setWidgetResizable(True)
                _sa.setHorizontalScrollBarPolicy(_qt_core_mod.Qt.ScrollBarAlwaysOff)
                _sa.setSizePolicy(
                    _qt_widgets_mod.QSizePolicy.MinimumExpanding,
                    _qt_widgets_mod.QSizePolicy.Expanding,
                )
                _tw = QWidget()
                _tw.setSizePolicy(
                    _qt_widgets_mod.QSizePolicy.MinimumExpanding,
                    _qt_widgets_mod.QSizePolicy.Preferred,
                )
                _tl = QVBoxLayout(_tw)
                _tl.setContentsMargins(6, 6, 6, 6)
                _tl.setSpacing(4)
                _sa.setWidget(_tw)
                return _sa, _tl

            _left_edit_sa, _left_edit_lay = _make_left_tab_scroll()
            _left_trace_sa, _left_trace_lay = _make_left_tab_scroll()
            _left_eval_sa, _left_eval_lay = _make_left_tab_scroll()
            _left_tab_pages = [_left_trace_sa, _left_edit_sa, _left_eval_sa]
            _left_tab_widget.addTab(_left_trace_sa, "Trace")
            _left_tab_widget.addTab(_left_edit_sa, "Edit")
            _left_tab_widget.addTab(_left_eval_sa, "Evaluate")

        # Sliders
        left_layout.addWidget(QLabel("Slice Position:"))
        self.z_slider = self._make_slider(0, self.shape[0] - 1, self.current_z, "Z", left_layout)
        self.y_slider = self._make_slider(0, self.shape[1] - 1, self.current_y, "Y", left_layout)
        self.x_slider = self._make_slider(0, self.shape[2] - 1, self.current_x, "X", left_layout)

        if mode == "seed":
            left_layout.addWidget(QLabel("Tool:"))
            _tool_row = self._row_widget()
            _tool_row.layout().addWidget(self.radio_tool_zoom)
            _tool_row.layout().addWidget(self.radio_tool_select)
            left_layout.addWidget(_tool_row)

        if mode == "seed":
            self.btn_prev_image.setVisible(self._show_prev_button)
            self.btn_next_image.setVisible(self._show_next_button)
            self.btn_prev_image.setEnabled(self._show_prev_button)
            self.btn_next_image.setEnabled(self._show_next_button)

            if _left_trace_lay is not None and self._show_trace_controls:
                _left_trace_lay.addWidget(QLabel("Tracing:"))
                _left_trace_lay.addWidget(self.btn_trace_neuron)
                _left_trace_lay.addWidget(self.btn_trace_all)
                _left_trace_lay.addWidget(self.btn_cancel_trace)
                _left_trace_lay.addWidget(self.btn_save_trace)
                _left_trace_lay.addWidget(self.btn_save_all_traces)
                _left_trace_lay.addWidget(self.btn_discard_trace)
                _left_trace_lay.addWidget(self.trace_progress_label)
                _left_trace_lay.addWidget(self._sidebar_separator())

            if _left_edit_lay is not None:
                _left_edit_lay.addWidget(QLabel("Annotation:"))
                _left_edit_lay.addWidget(self.annotation_target_combo)
                _left_edit_lay.addWidget(self.btn_remove_selected)
                _left_edit_lay.addWidget(self.btn_clip_selected)
                _left_edit_lay.addWidget(self.btn_undo_annotation)
                _left_edit_lay.addWidget(self._sidebar_separator())
                _left_edit_lay.addWidget(QLabel("Post-Processing:"))
                _left_edit_lay.addWidget(QLabel("Output Directory:"))
                _left_edit_lay.addWidget(self.postprocess_output_value_label)
                _left_edit_lay.addWidget(self.btn_set_postprocess_output)
                _left_edit_lay.addWidget(self.btn_clear_postprocess_output)
                _left_edit_lay.addWidget(self._pp_enable_length_filter_check)
                _left_edit_lay.addWidget(QLabel("Min Branch Length:"))
                _left_edit_lay.addWidget(self._pp_min_branch_length_spin)
                _left_edit_lay.addWidget(QLabel("Max Branch Length:"))
                _left_edit_lay.addWidget(self._pp_max_branch_length_spin)
                _left_edit_lay.addWidget(self._pp_enable_resample_check)
                _left_edit_lay.addWidget(QLabel("Resampling Step Size:"))
                _left_edit_lay.addWidget(self._pp_resampling_step_size_spin)
                _left_edit_lay.addWidget(self._pp_enable_smooth_paths_check)
                _left_edit_lay.addWidget(QLabel("Smoothing Window:"))
                _left_edit_lay.addWidget(self._pp_smoothing_window_spin)
                _left_edit_lay.addWidget(self._pp_enable_merge_check)
                _left_edit_lay.addWidget(QLabel("Merge Threshold:"))
                _left_edit_lay.addWidget(self._pp_overlap_dist_threshold_spin)
                _left_edit_lay.addWidget(QLabel("Confidence Threshold:"))
                _left_edit_lay.addWidget(self._pp_confidence_threshold_spin)
                _left_edit_lay.addWidget(QLabel("Mask Smoothing Size:"))
                _left_edit_lay.addWidget(self._pp_mask_smoothing_size_spin)
                _left_edit_lay.addWidget(QLabel("Merge Timeout Seconds (0=off):"))
                _left_edit_lay.addWidget(self._pp_merge_timeout_seconds_spin)
                _left_edit_lay.addWidget(self._pp_join_roots_check)
                _left_edit_lay.addWidget(QLabel("Scales JSON (optional):"))
                _left_edit_lay.addWidget(self.scales_path_value_label)
                _left_edit_lay.addWidget(self.btn_set_scales_path)
                _left_edit_lay.addWidget(self.btn_clear_scales_path)
                if self._show_postprocess_controls:
                    _left_edit_lay.addWidget(self.btn_run_postprocess)
                    _left_edit_lay.addWidget(self.btn_run_postprocess_all)
                    _left_edit_lay.addWidget(self.btn_undo_postprocess)
                _left_edit_lay.addWidget(QLabel("Filtered SWC Output:"))
                _left_edit_lay.addWidget(self.filtered_swc_output_value_label)
                _left_edit_lay.addWidget(self.btn_set_filtered_swc_output)
                _left_edit_lay.addWidget(self.btn_clear_filtered_swc_output)
                _left_edit_lay.addWidget(self.btn_save_filtered_swc)
                _left_edit_lay.addStretch(1)

            if _left_trace_lay is not None:
                _left_trace_lay.addWidget(QLabel("Seed Controls:"))
                _left_trace_lay.addWidget(self.btn_undo)
                _left_trace_lay.addWidget(QLabel("Seed Jitter:"))
            _seed_jitter_count_row = self._row_widget()
            _seed_jitter_count_row.layout().addWidget(QLabel("Count:"))
            _seed_jitter_count_row.layout().addWidget(self._trace_seed_jitter_count_spin, stretch=1)
            _left_trace_lay.addWidget(_seed_jitter_count_row)
            _seed_jitter_radius_row = self._row_widget()
            _seed_jitter_radius_row.layout().addWidget(QLabel("Radius:"))
            _seed_jitter_radius_row.layout().addWidget(self._trace_seed_jitter_radius_spin, stretch=1)
            _left_trace_lay.addWidget(_seed_jitter_radius_row)
            _left_trace_lay.addWidget(QLabel("Weight Strategy:"))
            _left_trace_lay.addWidget(self._trace_seed_jitter_weight_combo)
            _left_trace_lay.addWidget(self.btn_rejitter_seeds)
            _left_trace_lay.addWidget(self.btn_clear)
            if show_save_buttons:
                _left_trace_lay.addWidget(self.btn_save_seeds)
                _left_trace_lay.addWidget(self.btn_save_all_seeds)

            _left_trace_lay.addWidget(self._sidebar_separator())
            _left_trace_lay.addWidget(QLabel("Input / Output:"))
            _trace_io_sp_btn = (_qt_widgets_mod.QSizePolicy.Ignored,
                                _qt_widgets_mod.QSizePolicy.Fixed)
            _trace_io_sp_lbl = (_qt_widgets_mod.QSizePolicy.Ignored,
                                _qt_widgets_mod.QSizePolicy.Preferred)
            for _tb in [
                self.btn_set_image_dir, self.btn_clear_image_dir,
                self.btn_set_model_weights, self.btn_clear_model_weights,
                self.btn_set_trace_output, self.btn_clear_trace_output,
                self.btn_set_seeds_output, self.btn_clear_seeds_output,
                self.btn_set_seeds_input, self.btn_clear_seeds_input,
            ]:
                _tb.setSizePolicy(*_trace_io_sp_btn)
                _tb.setMinimumWidth(0)
            for _lv in [
                self.image_dir_value_label,
                self.model_weights_value_label,
                self.trace_output_value_label,
                self.seeds_output_value_label,
                self.seeds_input_value_label,
            ]:
                _lv.setSizePolicy(*_trace_io_sp_lbl)
                _lv.setMinimumWidth(0)
            _left_trace_lay.addWidget(QLabel("Image Directory:"))
            _left_trace_lay.addWidget(self.image_dir_value_label)
            _left_trace_lay.addWidget(self.btn_set_image_dir)
            _left_trace_lay.addWidget(self.btn_clear_image_dir)
            _left_trace_lay.addWidget(QLabel("Model Weights:"))
            _left_trace_lay.addWidget(self.model_weights_value_label)
            _left_trace_lay.addWidget(self.btn_set_model_weights)
            _left_trace_lay.addWidget(self.btn_clear_model_weights)
            _left_trace_lay.addWidget(QLabel("Trace Output:"))
            _left_trace_lay.addWidget(self.trace_output_value_label)
            _left_trace_lay.addWidget(self.btn_set_trace_output)
            _left_trace_lay.addWidget(self.btn_clear_trace_output)
            _left_trace_lay.addWidget(QLabel("Seeds Output:"))
            _left_trace_lay.addWidget(self.seeds_output_value_label)
            _left_trace_lay.addWidget(self.btn_set_seeds_output)
            _left_trace_lay.addWidget(self.btn_clear_seeds_output)
            _left_trace_lay.addWidget(QLabel("Seeds Input (optional):"))
            _left_trace_lay.addWidget(self.seeds_input_value_label)
            _left_trace_lay.addWidget(self.btn_set_seeds_input)
            _left_trace_lay.addWidget(self.btn_clear_seeds_input)
            _left_trace_lay.addWidget(self._sidebar_separator())

            _adv_toggle = _qt_widgets_mod.QToolButton()
            _adv_toggle.setText("▶ Advanced")
            _adv_toggle.setCheckable(True)
            _adv_toggle.setChecked(False)
            _adv_toggle.setAutoRaise(True)
            _adv_toggle.setFocusPolicy(self.Qt.NoFocus)
            _adv_toggle.setSizePolicy(
                _qt_widgets_mod.QSizePolicy.Ignored,
                _qt_widgets_mod.QSizePolicy.Fixed,
            )
            _adv_toggle.setMinimumWidth(0)
            _left_trace_lay.addWidget(_adv_toggle)

            _adv_panel = QWidget()
            _adv_panel.setVisible(False)
            _adv_panel.setMaximumHeight(0)
            _adv_panel.setMinimumWidth(0)
            _adv_panel.setSizePolicy(
                _qt_widgets_mod.QSizePolicy.Ignored,
                _qt_widgets_mod.QSizePolicy.Preferred,
            )
            _adv_layout = QVBoxLayout(_adv_panel)
            _adv_layout.setContentsMargins(4, 0, 4, 0)
            _adv_layout.setSpacing(4)
            _adv_layout.addWidget(QLabel("Max Length:"))
            _adv_layout.addWidget(self._trace_max_len_spin)
            _adv_layout.addWidget(QLabel("Max Paths:"))
            _adv_layout.addWidget(self._trace_max_paths_spin)
            _adv_layout.addWidget(self._trace_branching_check)
            _adv_layout.addWidget(self._trace_repeat_starts_check)
            _left_trace_lay.addWidget(_adv_panel)

            def _toggle_adv_panel(checked, panel=_adv_panel, btn=_adv_toggle):
                panel.setVisible(checked)
                panel.setMaximumHeight(16777215 if checked else 0)
                btn.setText("▼ Advanced" if checked else "▶ Advanced")

            _adv_toggle.toggled.connect(_toggle_adv_panel)

            if _left_trace_lay is not None:
                _left_trace_lay.addStretch(1)

            if _left_eval_lay is not None:
                _left_eval_lay.addWidget(QLabel("Reference SWC Directory:"))
                _left_eval_lay.addWidget(self.gt_swc_value_label)
                _left_eval_lay.addWidget(self.btn_set_gt_swc)
                _left_eval_lay.addWidget(self.btn_clear_gt_swc)
                _left_eval_lay.addWidget(QLabel("Eval Output Directory:"))
                _left_eval_lay.addWidget(self.eval_output_value_label)
                _left_eval_lay.addWidget(self.btn_set_eval_output)
                _left_eval_lay.addWidget(self.btn_clear_eval_output)
                _left_eval_lay.addWidget(self._sidebar_separator())
                _left_eval_lay.addWidget(QLabel("Scales JSON (optional):"))
                _left_eval_lay.addWidget(self.eval_scales_path_value_label)
                _left_eval_lay.addWidget(self.btn_set_eval_scales_path)
                _left_eval_lay.addWidget(self.btn_clear_eval_scales_path)
                _left_eval_lay.addWidget(QLabel("Distance Threshold:"))
                _left_eval_lay.addWidget(self._eval_distance_threshold_spin)
                if self._show_postprocess_controls:
                    _left_eval_lay.addWidget(self._sidebar_separator())
                    _left_eval_lay.addWidget(self.btn_run_evaluation)
                    _left_eval_lay.addWidget(self.btn_run_evaluation_all)
                    _left_eval_lay.addWidget(self.btn_save_eval_report)
                    _left_eval_lay.addWidget(self._sidebar_separator())
                    _left_eval_lay.addWidget(QLabel("Evaluation Report:"))
                    self.eval_report_widget = _qt_widgets_mod.QTextEdit()
                    self.eval_report_widget.setReadOnly(True)
                    self.eval_report_widget.setPlaceholderText(
                        "Evaluation report will appear here after running evaluation."
                    )
                    self.eval_report_widget.setMinimumHeight(120)
                    self.eval_report_widget.setSizePolicy(
                        _qt_widgets_mod.QSizePolicy.Expanding,
                        _qt_widgets_mod.QSizePolicy.Expanding,
                    )
                    _left_eval_lay.addWidget(self.eval_report_widget, stretch=1)
                else:
                    self.eval_report_widget = None
                    _left_eval_lay.addStretch(1)

            left_layout.addWidget(self._sidebar_separator())
            left_layout.addWidget(_left_tab_widget, stretch=1)

        left_sidebar.adjustSize()
        left_min_width = max(left_sidebar.minimumSizeHint().width(), left_sidebar.sizeHint().width())
        if mode == "seed" and _left_tab_widget is not None:
            # Ensure initial sidebar width accommodates the widest tab content,
            # not only the currently active tab.
            _left_tab_widget.ensurePolished()
            _left_tab_widget.updateGeometry()
            tab_bar_width = _left_tab_widget.tabBar().sizeHint().width() if _left_tab_widget.tabBar() is not None else 0
            max_tab_page_width = 0
            for page in _left_tab_pages:
                if page is None:
                    continue
                page.ensurePolished()
                page.updateGeometry()
                max_tab_page_width = max(
                    max_tab_page_width,
                    page.minimumSizeHint().width(),
                    page.sizeHint().width(),
                    page.widget().minimumSizeHint().width() if page.widget() is not None else 0,
                    page.widget().sizeHint().width() if page.widget() is not None else 0,
                )
            left_min_width = max(left_min_width, int(tab_bar_width + max_tab_page_width + 24))
        left_sidebar.setMinimumWidth(left_min_width)

        main_splitter.addWidget(left_sidebar)

        # -----------------------------------------------------------------
        # CENTER — matplotlib canvas
        # -----------------------------------------------------------------
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.figure = plt.Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self._supports_blit = all(
            hasattr(self.canvas, name) for name in ("copy_from_bbox", "restore_region", "blit")
        )
        self._original_wheel_event = self.canvas.wheelEvent
        self.canvas.wheelEvent = self._canvas_wheel_event
        center_layout.addWidget(self.canvas, stretch=1)

        main_splitter.addWidget(center_widget)

        # Set initial sizes [left, center] and lock only center to stretch
        initial_left_width = max(left_sidebar.minimumWidth(), left_sidebar.sizeHint().width())
        initial_center_width = max(1, self.dialog.width() - initial_left_width)
        main_splitter.setSizes([initial_left_width, initial_center_width])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        # -----------------------------------------------------------------
        # TOP BAR — horizontal view controls
        # -----------------------------------------------------------------
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 6, 6, 6)
        top_layout.setSpacing(6)
        top_layout.addWidget(QLabel("Projection:"))
        top_layout.addWidget(self.projection_combo)
        top_layout.addWidget(self._sidebar_separator())
        top_layout.addWidget(QLabel("Maximize View:"))
        top_layout.addWidget(self.btn_all)
        top_layout.addWidget(self.btn_xy)
        top_layout.addWidget(self.btn_xz)
        top_layout.addWidget(self.btn_yz)
        top_layout.addWidget(self._sidebar_separator())
        top_layout.addWidget(self.btn_back)
        top_layout.addWidget(self.btn_home)
        if mode == "seed":
            top_layout.addWidget(self.chk_trace_overlay)
            top_layout.addWidget(self.chk_gt_overlay)
            top_layout.addWidget(self.btn_prev_image)
            top_layout.addWidget(self.btn_next_image)
        top_layout.addStretch(1)
        outer.insertWidget(0, top_bar)

        if mode == "seed":
            self._refresh_annotation_target_options()
            self._refresh_output_path_labels()

        # -----------------------------------------------------------------
        # FOOTER — info bar and status label
        # -----------------------------------------------------------------
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(6, 2, 6, 2)
        self.info_label = QLabel(
            "Space=Add seed/child, Right-click selected node=Add Branch, Delete/Backspace=Remove Selected, Scroll=Slice, Drag=Tool Action (Zoom/Select)"
            if mode == "seed"
            else "Scroll=Slice, Drag=Zoom Box, Enter=Finish"
        )
        footer_layout.addWidget(self.info_label)
        self.trace_status_label = QLabel("")
        footer_layout.addWidget(self.trace_status_label)
        footer_layout.addStretch(1)
        outer.addWidget(footer)

        button_list = [
            self.btn_all,
            self.btn_xy,
            self.btn_xz,
            self.btn_yz,
            self.btn_back,
            self.btn_home,
        ]
        if mode == "seed":
            button_list.extend([self.btn_undo, self.btn_clear])
            button_list.append(self.btn_rejitter_seeds)
            button_list.extend([self.btn_set_seeds_output, self.btn_set_trace_output, self.btn_set_model_weights])
            button_list.extend([self.btn_clear_seeds_output, self.btn_clear_trace_output, self.btn_clear_model_weights])
            button_list.extend([self.btn_set_image_dir, self.btn_set_seeds_input])
            button_list.extend([self.btn_clear_image_dir, self.btn_clear_seeds_input])
            button_list.extend([
                self.btn_remove_selected,
                self.btn_clip_selected,
                self.btn_undo_annotation,
                self.btn_set_filtered_swc_output,
                self.btn_clear_filtered_swc_output,
                self.btn_save_filtered_swc,
            ])
            button_list.extend([self.btn_prev_image, self.btn_next_image])
            button_list.extend([self.btn_set_postprocess_output, self.btn_set_gt_swc, self.btn_set_scales_path])
            button_list.extend([self.btn_clear_postprocess_output, self.btn_clear_gt_swc, self.btn_clear_scales_path])
            button_list.extend([self.btn_set_eval_output, self.btn_set_eval_scales_path])
            button_list.extend([self.btn_clear_eval_output, self.btn_clear_eval_scales_path])
            if show_save_buttons:
                button_list.extend([self.btn_save_seeds, self.btn_save_all_seeds])
            if self._show_trace_controls:
                button_list.extend([
                    self.btn_trace_neuron,
                    self.btn_trace_all,
                    self.btn_cancel_trace,
                    self.btn_save_trace,
                    self.btn_save_all_traces,
                    self.btn_discard_trace,
                ])
            if self._show_postprocess_controls:
                button_list.extend([
                    self.btn_run_postprocess,
                    self.btn_run_postprocess_all,
                    self.btn_undo_postprocess,
                    self.btn_run_evaluation,
                    self.btn_run_evaluation_all,
                    self.btn_save_eval_report,
                ])
        for button in button_list:
            button.setAutoDefault(False)
            button.setDefault(False)
            button.setFocusPolicy(self.Qt.NoFocus)

        self.axis_view_map = {}

        self.projection_combo.currentIndexChanged.connect(self._on_projection_changed)
        self.btn_all.clicked.connect(lambda: self._set_maximize(None))
        self.btn_xy.clicked.connect(lambda: self._set_maximize("xy"))
        self.btn_xz.clicked.connect(lambda: self._set_maximize("xz"))
        self.btn_yz.clicked.connect(lambda: self._set_maximize("yz"))
        self.btn_back.clicked.connect(self._zoom_back)
        self.btn_home.clicked.connect(self._zoom_home)
        self.z_slider.valueChanged.connect(self._on_slider_change)
        self.y_slider.valueChanged.connect(self._on_slider_change)
        self.x_slider.valueChanged.connect(self._on_slider_change)

        if mode == "seed":
            self.btn_undo.clicked.connect(self._undo_seed)
            self.btn_clear.clicked.connect(self._clear_seeds)
            self.btn_rejitter_seeds.clicked.connect(self._rejitter_seeds)
            self.annotation_target_combo.currentIndexChanged.connect(self._on_annotation_target_changed)
            self.radio_tool_zoom.toggled.connect(self._on_tool_toggled)
            self.btn_remove_selected.clicked.connect(self._remove_selected)
            self.btn_clip_selected.clicked.connect(self._clip_selected)
            self.btn_undo_annotation.clicked.connect(self._undo_annotation_edit)
            self.btn_set_filtered_swc_output.clicked.connect(self._select_filtered_swc_output_dir)
            self.btn_clear_filtered_swc_output.clicked.connect(self._clear_filtered_swc_output_dir)
            self.btn_save_filtered_swc.clicked.connect(self._save_filtered_swc)
            self.btn_set_seeds_output.clicked.connect(self._select_seeds_output_path)
            self.btn_clear_seeds_output.clicked.connect(self._clear_seeds_output_path)
            self.btn_set_trace_output.clicked.connect(self._select_trace_output_path)
            self.btn_clear_trace_output.clicked.connect(self._clear_trace_output_path)
            self.btn_set_model_weights.clicked.connect(self._select_model_weights_path)
            self.btn_clear_model_weights.clicked.connect(self._clear_model_weights_path)
            self.btn_set_image_dir.clicked.connect(self._select_image_dir)
            self.btn_clear_image_dir.clicked.connect(self._clear_image_dir)
            self.btn_set_seeds_input.clicked.connect(self._select_seeds_input_path)
            self.btn_clear_seeds_input.clicked.connect(self._clear_seeds_input_path)
            self.btn_prev_image.clicked.connect(self._go_prev_image)
            self.btn_next_image.clicked.connect(self._go_next_image)
            self._trace_max_len_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_max_paths_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_branching_check.toggled.connect(self._on_advanced_params_changed)
            self._trace_repeat_starts_check.toggled.connect(self._on_advanced_params_changed)
            self._trace_seed_jitter_count_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_seed_jitter_radius_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_seed_jitter_weight_combo.currentIndexChanged.connect(self._on_advanced_params_changed)
            self.btn_set_postprocess_output.clicked.connect(self._select_postprocess_output_dir)
            self.btn_clear_postprocess_output.clicked.connect(self._clear_postprocess_output_dir)
            self._pp_min_branch_length_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_max_branch_length_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_length_filter_check.toggled.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_length_filter_check.toggled.connect(self._update_postprocess_step_controls)
            self._pp_resampling_step_size_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_resample_check.toggled.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_resample_check.toggled.connect(self._update_postprocess_step_controls)
            self._pp_smoothing_window_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_smooth_paths_check.toggled.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_smooth_paths_check.toggled.connect(self._update_postprocess_step_controls)
            self._pp_overlap_dist_threshold_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_confidence_threshold_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_merge_check.toggled.connect(self._on_postprocess_params_changed_slot)
            self._pp_enable_merge_check.toggled.connect(self._update_postprocess_step_controls)
            self._pp_join_roots_check.toggled.connect(self._on_postprocess_params_changed_slot)
            self._pp_mask_smoothing_size_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_merge_timeout_seconds_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self.btn_set_eval_output.clicked.connect(self._select_eval_output_dir)
            self.btn_clear_eval_output.clicked.connect(self._clear_eval_output_dir)
            self._eval_distance_threshold_spin.valueChanged.connect(self._on_eval_params_changed_slot)
            self.btn_set_eval_scales_path.clicked.connect(self._select_scales_path)
            self.btn_clear_eval_scales_path.clicked.connect(self._clear_scales_path)
            if show_save_buttons:
                self.btn_save_seeds.clicked.connect(self._save_current_seeds)
                self.btn_save_all_seeds.clicked.connect(self._save_all_seeds)
            if self._show_trace_controls:
                self.btn_trace_neuron.clicked.connect(self._trace_current_neuron)
                self.btn_trace_all.clicked.connect(self._trace_all_neurons)
                self.btn_cancel_trace.clicked.connect(self._cancel_trace)
                self.btn_save_trace.clicked.connect(self._save_trace)
                self.btn_save_all_traces.clicked.connect(self._save_all_traces)
                self.btn_discard_trace.clicked.connect(self._discard_trace)
                self.chk_trace_overlay.toggled.connect(self._toggle_trace_overlay)
                self.chk_gt_overlay.toggled.connect(self._toggle_gt_overlay)
            self.btn_set_gt_swc.clicked.connect(self._select_gt_swc_path)
            self.btn_clear_gt_swc.clicked.connect(self._clear_gt_swc_path)
            self.btn_set_scales_path.clicked.connect(self._select_scales_path)
            self.btn_clear_scales_path.clicked.connect(self._clear_scales_path)
            if self._show_postprocess_controls:
                self.btn_run_postprocess.clicked.connect(self._run_postprocess)
                self.btn_run_postprocess_all.clicked.connect(self._run_postprocess_all)
                self.btn_undo_postprocess.clicked.connect(self._undo_postprocess)
                self.btn_run_evaluation.clicked.connect(self._run_evaluation)
                self.btn_run_evaluation_all.clicked.connect(self._run_evaluation_all)
                self.btn_save_eval_report.clicked.connect(self._save_eval_report)

        if self._show_trace_controls and self._get_trace_status is not None:
            qt_core = importlib.import_module("qtpy.QtCore")
            self._qt_timer = qt_core.QTimer(self.dialog)
            self._qt_timer.timeout.connect(self._poll_trace_status)
            self._qt_timer.start(250)
            self._set_trace_controls_busy(False)
            self._trace_controls_running_state = False
        if mode == "seed" and self._on_trace_params_changed is not None:
            qt_core = importlib.import_module("qtpy.QtCore")
            self._trace_params_debounce_timer = qt_core.QTimer(self.dialog)
            self._trace_params_debounce_timer.setSingleShot(True)
            self._trace_params_debounce_timer.timeout.connect(self._flush_pending_trace_params)
        if mode == "seed":
            self._update_postprocess_step_controls()
            self._refresh_seed_order_controls()

        self._mpl_press_cid = self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._mpl_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._mpl_release_cid = self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._mpl_key_cid = self.canvas.mpl_connect("key_press_event", self._on_mpl_keypress)
        self._mpl_key_release_cid = self.canvas.mpl_connect("key_release_event", self._on_mpl_keyrelease)
        self._mpl_draw_cid = self.canvas.mpl_connect("draw_event", self._on_canvas_draw)
        self.dialog.keyPressEvent = self._make_keypress_handler(self.dialog.keyPressEvent)
        self.dialog.keyReleaseEvent = self._make_keyrelease_handler(self.dialog.keyReleaseEvent)
        self.canvas.setFocusPolicy(self.Qt.StrongFocus)
        self.canvas.setFocus()
        self._redraw()

    def _make_slider(self, minimum: int, maximum: int, value: int, label: str, parent_layout):
        row = self._row_widget()
        row_layout = row.layout()
        row_layout.addWidget(self._label_widget(f"{label}:"))
        slider = self._slider_widget(minimum, maximum, value)
        value_label = self._label_widget(str(value))
        slider.valueChanged.connect(lambda v, l=value_label, name=label: l.setText(str(v)))
        row_layout.addWidget(slider, stretch=1)
        row_layout.addWidget(value_label)
        parent_layout.addWidget(row)
        return slider

    def _row_widget(self):
        _, QWidget, _, _, QHBoxLayout, _, _, _, _, _, _ = _try_import_ui_dependencies()
        row = QWidget()
        row.setLayout(QHBoxLayout())
        return row

    def _label_widget(self, text: str):
        _, _, _, _, _, _, QLabel, _, _, _, _ = _try_import_ui_dependencies()
        return QLabel(text)

    def _slider_widget(self, minimum: int, maximum: int, value: int):
        _, _, _, _, _, _, _, QSlider, _, Qt, _ = _try_import_ui_dependencies()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        return slider

    def _sidebar_separator(self):
        """Return a thin horizontal QFrame line for use as a visual divider in sidebars."""
        qt_widgets = importlib.import_module("qtpy.QtWidgets")
        line = qt_widgets.QFrame()
        line.setFrameShape(qt_widgets.QFrame.HLine)
        line.setFrameShadow(qt_widgets.QFrame.Sunken)
        return line

    def _make_keypress_handler(self, original_handler):
        def _handler(event):
            if event.key() == self.Qt.Key_Shift:
                self._shift_held = True
                original_handler(event)
                return
            if event.key() == self.Qt.Key_Escape:
                if self.mode == "seed":
                    self._clear_current_selection()
                    self._redraw_selection_only()
                # Prevent accidental dialog close from Escape.
                return
            if self.mode == "seed" and event.key() == self.Qt.Key_Space:
                self._add_current_seed()
                return
            if self.mode == "seed" and event.key() in (self.Qt.Key_Backspace, self.Qt.Key_Delete):
                self._remove_selected()
                return
            if event.key() in (self.Qt.Key_Return, self.Qt.Key_Enter):
                self.dialog.accept()
                return
            original_handler(event)

        return _handler

    def _make_keyrelease_handler(self, original_handler):
        def _handler(event):
            if event.key() == self.Qt.Key_Shift:
                self._shift_held = False
                original_handler(event)
                return
            original_handler(event)

        return _handler

    def _on_projection_changed(self, _index: int):
        self.projection_mode = "slice" if self.projection_combo.currentText().lower() == "slice" else "mip"
        self._redraw()

    def _set_maximize(self, view: Optional[str]):
        self.maximized_view = view
        self._layout_dirty = True
        self._redraw()

    def _on_slider_change(self, _value: int):
        self.current_z = int(self.z_slider.value())
        self.current_y = int(self.y_slider.value())
        self.current_x = int(self.x_slider.value())
        if self.projection_mode == "mip" and self._try_blit_crosshair_update():
            self._refresh_info_label()
            return
        # In MIP mode, slider moves only update crosshairs; overlays are unchanged.
        # Skip overlay re-plotting to keep interactive scrubbing responsive.
        self._redraw(skip_overlay_redraw=(self.projection_mode == "mip"))

    def _undo_seed(self):
        if self.seeds:
            self.seeds.pop()
            self.selected_seed_index = None
            self._refresh_seed_order_controls()
            self._refresh_effective_seed_overlay()
            self._redraw()

    def _remove_selected(self):
        removed = False

        if self._editor_state.selection.clip_preview_node_ids:
            target = self._editor_state.active_annotation
            graph = self._active_annotation_graph()
            self._push_annotation_undo_snapshot()
            graph.remove_nodes(self._editor_state.selection.clip_preview_node_ids)
            self._sync_annotation_graph_to_view(target)
            removed = True
        elif self.selected_seed_index is not None:
            idx = int(self.selected_seed_index)
            if 0 <= idx < len(self.seeds):
                self.seeds.pop(idx)
                removed = True
            self.selected_seed_index = None
        elif self._editor_state.selection.selected_annotation_node_ids:
            target = self._editor_state.active_annotation
            graph = self._active_annotation_graph()
            self._push_annotation_undo_snapshot()
            graph.remove_nodes(self._editor_state.selection.selected_annotation_node_ids)
            self._sync_annotation_graph_to_view(target)
            removed = True

        if not removed:
            return

        self._clear_current_selection()
        self._refresh_effective_seed_overlay()
        self._redraw()

    def _clip_selected(self):
        if self.selected_seed_index is not None:
            return
        selected_node_ids = self._editor_state.selection.selected_annotation_node_ids
        if len(selected_node_ids) == 0:
            return

        anchor_node_id = int(sorted(selected_node_ids)[0])
        graph = self._active_annotation_graph()
        preview_node_ids = graph.descendant_ids_including(anchor_node_id)
        if not preview_node_ids:
            return

        if preview_node_ids == self._editor_state.selection.clip_preview_node_ids:
            self._remove_selected()
            return

        self._editor_state.selection.clip_preview_node_ids = set(preview_node_ids)
        self._redraw_selection_only()

    def _remove_selected_seed(self):
        self._remove_selected()

    def _clear_seeds(self):
        self.seeds.clear()
        self.selected_seed_index = None
        self._refresh_seed_order_controls()
        self._refresh_effective_seed_overlay()
        self._redraw()

    def _refresh_seed_order_controls(self):
        # Seed ordering widgets were removed; keep selected index in range.
        seed_count = len(self.seeds)
        if seed_count <= 0:
            self.selected_seed_index = None
            self._refresh_edit_action_controls()
            return
        if self.selected_seed_index is None:
            self._refresh_edit_action_controls()
            return
        selected_index = int(np.clip(self.selected_seed_index, 0, seed_count - 1))
        self.selected_seed_index = selected_index
        self._refresh_edit_action_controls()

    def _refresh_edit_action_controls(self) -> None:
        if self.mode != "seed":
            return
        has_seed_selection = self.selected_seed_index is not None
        has_node_selection = len(self._editor_state.selection.selected_annotation_node_ids) > 0
        has_clip_preview = len(self._editor_state.selection.clip_preview_node_ids) > 0
        can_remove = has_seed_selection or has_node_selection or has_clip_preview
        can_clip = (not has_seed_selection) and len(self._editor_state.selection.selected_annotation_node_ids) > 0

        self.btn_remove_selected.setEnabled(can_remove)
        self.btn_clip_selected.setEnabled(can_clip)

    def _move_selected_seed_to_index(self, new_index: int):
        if self.selected_seed_index is None or not self.seeds:
            return
        old_index = int(self.selected_seed_index)
        if old_index < 0 or old_index >= len(self.seeds):
            self.selected_seed_index = None
            self._refresh_seed_order_controls()
            return

        new_index = int(np.clip(new_index, 0, len(self.seeds) - 1))
        if new_index == old_index:
            return

        selected_seed = self.seeds.pop(old_index)
        self.seeds.insert(new_index, selected_seed)
        self.selected_seed_index = new_index
        self._refresh_seed_order_controls()
        self._refresh_effective_seed_overlay()
        self._redraw()

    def _move_selected_seed_up(self):
        if self.selected_seed_index is None:
            return
        self._move_selected_seed_to_index(int(self.selected_seed_index) - 1)

    def _move_selected_seed_down(self):
        if self.selected_seed_index is None:
            return
        self._move_selected_seed_to_index(int(self.selected_seed_index) + 1)

    def _on_seed_order_changed(self, one_based_index: int):
        self._move_selected_seed_to_index(one_based_index - 1)

    def _on_tool_toggled(self, checked: bool):
        self._active_tool = "zoom" if checked else "select"
        self._editor_state.active_tool = self._active_tool

    def _invalidate_tree_overlay_cache(self) -> None:
        self._tree_overlay_cache_source_id = None
        self._tree_overlay_cache = {
            "child_idx": np.empty((0,), dtype=np.int64),
            "parent_idx": np.empty((0,), dtype=np.int64),
            "child_xyz": np.empty((0, 3), dtype=np.float32),
            "parent_xyz": np.empty((0, 3), dtype=np.float32),
        }

    def _ensure_tree_overlay_cache(self, swc_source: np.ndarray) -> None:
        source_id = id(swc_source)
        if self._tree_overlay_cache_source_id == source_id:
            return

        if swc_source.size == 0:
            self._invalidate_tree_overlay_cache()
            self._tree_overlay_cache_source_id = source_id
            return

        node_ids = swc_source[:, 0].astype(np.int64, copy=False)
        parent_ids = swc_source[:, 6].astype(np.int64, copy=False)

        order = np.argsort(node_ids)
        sorted_ids = node_ids[order]
        parent_pos = np.searchsorted(sorted_ids, parent_ids)
        within_bounds = (parent_ids != -1) & (parent_pos >= 0) & (parent_pos < sorted_ids.shape[0])
        has_parent = np.zeros_like(within_bounds, dtype=bool)
        if np.any(within_bounds):
            pos = parent_pos[within_bounds]
            has_parent[within_bounds] = sorted_ids[pos] == parent_ids[within_bounds]

        child_idx = np.flatnonzero(has_parent)
        if child_idx.size == 0:
            self._invalidate_tree_overlay_cache()
            self._tree_overlay_cache_source_id = source_id
            return

        parent_idx = order[parent_pos[child_idx]]
        self._tree_overlay_cache = {
            "child_idx": child_idx.astype(np.int64, copy=False),
            "parent_idx": parent_idx.astype(np.int64, copy=False),
            "child_xyz": swc_source[child_idx, 2:5].astype(np.float32, copy=False),
            "parent_xyz": swc_source[parent_idx, 2:5].astype(np.float32, copy=False),
        }
        self._tree_overlay_cache_source_id = source_id

    def _active_tree_for_filters(self) -> np.ndarray:
        if self._has_clip_preview:
            return self._tree_swc_preview_filtered.copy()
        return self._tree_swc_committed.copy()

    @staticmethod
    def _seed_points_from_swc(swc_rows: np.ndarray) -> List[Tuple[float, float, float]]:
        if swc_rows.size == 0:
            return []
        roots = swc_rows[swc_rows[:, 6] == -1]
        if roots.size == 0:
            return []
        return [tuple(map(float, row)) for row in roots[:, [4, 3, 2]].tolist()]

    def _filter_components_by_length(self, swc_rows: np.ndarray, min_length: int) -> np.ndarray:
        if swc_rows.size == 0:
            return np.empty((0, 7), dtype=np.float32)

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        id_to_idx = {int(row[0]): idx for idx, row in enumerate(swc_rows)}
        edges = []
        for row in swc_rows:
            node_id = int(row[0])
            parent_id = int(row[6])
            if parent_id != -1 and parent_id in id_to_idx:
                edges.append((id_to_idx[node_id], id_to_idx[parent_id]))

        n_nodes = swc_rows.shape[0]
        if edges:
            edge_array = np.asarray(edges, dtype=np.int64)
            adjacency = csr_matrix(
                (np.ones((len(edge_array),), dtype=np.float32), (edge_array[:, 0], edge_array[:, 1])),
                shape=(n_nodes, n_nodes),
            )
        else:
            adjacency = csr_matrix((n_nodes, n_nodes), dtype=np.float32)
        adjacency = adjacency + adjacency.T

        _, labels = connected_components(csgraph=adjacency, directed=False)
        component_sizes = np.bincount(labels)
        valid_components = np.where(component_sizes >= int(min_length))[0]
        keep_mask = np.isin(labels, valid_components)
        filtered = swc_rows[keep_mask].copy()

        if filtered.size == 0:
            return np.empty((0, 7), dtype=np.float32)

        remaining_ids = set(filtered[:, 0].astype(int).tolist())
        for row in filtered:
            parent_id = int(row[6])
            if parent_id != -1 and parent_id not in remaining_ids:
                row[6] = -1

        return filtered

    def _apply_component_filter(self):
        source = self._active_tree_for_filters()
        if source.size == 0:
            return
        min_length = int(self._component_min_length_spin.value())
        filtered = self._filter_components_by_length(source, min_length=min_length)
        self._set_reference_swc_rows(filtered)
        self._tree_swc_preview_source = np.empty((0, 7), dtype=np.float32)
        self._tree_swc_preview_filtered = np.empty((0, 7), dtype=np.float32)
        self._tree_preview_removed_ids = set()
        self._tree_preview_seed_points = []
        self._has_clip_preview = False
        self._invalidate_tree_overlay_cache()

        seed_points = self._seed_points_from_swc(filtered)
        self.seeds = [
            (
                int(np.clip(round(z), 0, self.shape[0] - 1)),
                int(np.clip(round(y), 0, self.shape[1] - 1)),
                int(np.clip(round(x), 0, self.shape[2] - 1)),
            )
            for z, y, x in seed_points
        ]
        self.selected_seed_index = None
        self._refresh_seed_order_controls()

        if self._on_filtered_swc_changed is not None:
            self._on_filtered_swc_changed(self._current_image_key, self._tree_swc_committed.tolist())
        self._redraw()

    def _select_filtered_swc_output_dir(self):
        if self._on_select_filtered_swc_output_dir is None:
            return
        selected = self._on_select_filtered_swc_output_dir()
        if selected is not None:
            self._filtered_swc_output_dir = selected
            self._refresh_output_path_labels()

    def _clear_filtered_swc_output_dir(self):
        if self._on_clear_filtered_swc_output_dir is not None:
            self._filtered_swc_output_dir = self._on_clear_filtered_swc_output_dir()
        else:
            self._filtered_swc_output_dir = None
        self._refresh_output_path_labels()

    def _save_filtered_swc(self):
        if self._on_save_filtered_swc is None:
            return
        saved_dir = self._on_save_filtered_swc(self._current_image_key, self._tree_swc_committed.tolist())
        if saved_dir is not None:
            self._filtered_swc_output_dir = saved_dir
            self._refresh_output_path_labels()

    def _finish(self):
        self.session_action = "finish"
        self.dialog.accept()

    def _go_prev_image(self):
        if self._on_prev_image is not None:
            new_context = self._on_prev_image(np.asarray(self.seeds, dtype=np.float32))
            if isinstance(new_context, dict):
                self.load_seed_context(new_context)
            return
        self.session_action = "prev"
        self.dialog.accept()

    def _go_next_image(self):
        if self._on_next_image is not None:
            new_context = self._on_next_image(np.asarray(self.seeds, dtype=np.float32))
            if isinstance(new_context, dict):
                self.load_seed_context(new_context)
            return
        self.session_action = "next"
        self.dialog.accept()

    def _set_seeds_from_array(self, seed_array: Optional[np.ndarray]):
        self.seeds = []
        self.selected_seed_index = None
        if seed_array is None:
            self._refresh_seed_order_controls()
            self._refresh_effective_seed_overlay()
            return
        arr = np.asarray(seed_array, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            self._refresh_seed_order_controls()
            self._refresh_effective_seed_overlay()
            return
        arr[:, 0] = np.clip(arr[:, 0], 0, self.shape[0] - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, self.shape[1] - 1)
        arr[:, 2] = np.clip(arr[:, 2], 0, self.shape[2] - 1)
        self.seeds = [(int(round(z)), int(round(y)), int(round(x))) for z, y, x in arr]
        self._refresh_seed_order_controls()

    def _set_effective_seed_overlay_from_array(self, seed_array: Optional[np.ndarray]):
        self.effective_seed_overlay = []
        if seed_array is None:
            return
        arr = np.asarray(seed_array, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return
        arr[:, 0] = np.clip(arr[:, 0], 0, self.shape[0] - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, self.shape[1] - 1)
        arr[:, 2] = np.clip(arr[:, 2], 0, self.shape[2] - 1)
        self.effective_seed_overlay = [
            (int(round(z)), int(round(y)), int(round(x))) for z, y, x in arr
        ]

    def _refresh_effective_seed_overlay(self):
        if self.mode != "seed" or self._on_get_effective_seed_overlay is None:
            return
        try:
            seed_array = np.asarray(self.seeds, dtype=np.float32)
            overlay = self._on_get_effective_seed_overlay(seed_array)
            self._set_effective_seed_overlay_from_array(overlay)
        except Exception:
            # Keep UI responsive even if the optional overlay callback fails.
            self.effective_seed_overlay = []

    def _rejitter_seeds(self):
        self._pending_trace_param_overrides = self.get_trace_params_overrides()
        self._pending_trace_param_overrides["seed_jitter_nonce"] = int(np.random.default_rng().integers(1, 2**31 - 1))
        self._flush_pending_trace_params()
        self._refresh_effective_seed_overlay()
        self._redraw()

    def _set_finished_paths(self, finished_paths):
        self._set_prediction_paths(finished_paths)

    def _update_slider_bounds(self):
        self.z_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.x_slider.blockSignals(True)
        self.z_slider.setMinimum(0)
        self.y_slider.setMinimum(0)
        self.x_slider.setMinimum(0)
        self.z_slider.setMaximum(max(0, self.shape[0] - 1))
        self.y_slider.setMaximum(max(0, self.shape[1] - 1))
        self.x_slider.setMaximum(max(0, self.shape[2] - 1))
        self.z_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.x_slider.blockSignals(False)

    def load_seed_context(self, context: Dict[str, object]):
        image_data = context.get("image_data")
        if image_data is None:
            return

        self.img_np = _extract_first_channel_numpy(np.asarray(image_data))
        self.shape = self.img_np.shape
        self._invalidate_mip_cache()
        self.current_z = int(self.shape[0] // 2)
        self.current_y = int(self.shape[1] // 2)
        self.current_x = int(self.shape[2] // 2)

        self.zoom_limits = {}
        self.zoom_history = {"xy": [], "xz": [], "yz": []}
        self._layout_dirty = True

        self._set_seeds_from_array(context.get("initial_seeds"))
        self._set_effective_seed_overlay_from_array(context.get("effective_seed_overlay"))
        self._set_finished_paths(context.get("finished_paths"))
        self._current_image_key = str(context.get("neuron_name", ""))
        self._set_reference_swc_rows(context.get("tree_swc_rows"))
        self._refresh_annotation_target_options()
        self._tree_swc_preview_source = np.empty((0, 7), dtype=np.float32)
        self._tree_swc_preview_filtered = np.empty((0, 7), dtype=np.float32)
        self._tree_preview_seed_points = []
        self._tree_preview_removed_ids = set()
        self._has_clip_preview = False
        self._invalidate_tree_overlay_cache()

        self._show_prev_button = bool(context.get("show_prev_button", False))
        self._show_next_button = bool(context.get("show_next_button", False))
        self.btn_prev_image.setVisible(self._show_prev_button)
        self.btn_next_image.setVisible(self._show_next_button)
        self.btn_prev_image.setEnabled(self._show_prev_button)
        self.btn_next_image.setEnabled(self._show_next_button)

        self._seeds_output_path = context.get("seeds_output_path")  # type: ignore[assignment]
        self._trace_output_path = context.get("trace_output_path")  # type: ignore[assignment]
        self._model_weights_path = context.get("model_weights_path")  # type: ignore[assignment]
        if "image_dir" in context:
            self._image_dir = context.get("image_dir")  # type: ignore[assignment]
        if "seeds_input_path" in context:
            self._seeds_input_path = context.get("seeds_input_path")  # type: ignore[assignment]
        if "gt_swc_path" in context:
            self._gt_swc_path = context.get("gt_swc_path")  # type: ignore[assignment]
        if "scales_path" in context:
            self._scales_path = context.get("scales_path")  # type: ignore[assignment]
        if "filtered_swc_output_dir" in context:
            self._filtered_swc_output_dir = context.get("filtered_swc_output_dir")  # type: ignore[assignment]
        self._refresh_output_path_labels()

        neuron_name = context.get("neuron_name", "")
        title = "Viewing Image"
        if isinstance(neuron_name, str) and len(neuron_name) > 0:
            title = f"{title}: {neuron_name}"
        self.dialog.setWindowTitle(title)
        if self.eval_report_widget is not None:
            eval_report_text = context.get("eval_report_text", None)
            self.eval_report_widget.setPlainText("")
            if eval_report_text is not None:
                self.eval_report_widget.setPlainText(str(eval_report_text))

        self._update_slider_bounds()
        self._sync_sliders_from_cursor()
        self._redraw()

    def _save_current_seeds(self):
        if self._on_save_current is None:
            return
        self._on_save_current(np.asarray(self.seeds, dtype=np.float32))

    def _save_all_seeds(self):
        if self._on_save_all is None:
            return
        self._on_save_all()

    def _trace_current_neuron(self):
        if self._on_trace_current is None:
            return
        self._flush_pending_trace_params()
        paths = self._on_trace_current(np.asarray(self.seeds, dtype=np.float32))
        if paths is not None:
            self._set_prediction_paths(paths)
            self._clear_transient_selection_state()
            self._redraw()

    def _trace_all_neurons(self):
        if self._on_trace_all is None:
            return
        self._flush_pending_trace_params()
        self._on_trace_all()

    def _cancel_trace(self):
        if self._on_cancel_trace is None:
            return
        self._on_cancel_trace()

    def _toggle_trace_overlay(self, checked: bool):
        self.trace_overlay_visible = bool(checked)
        self._editor_state.show_prediction_overlay = self.trace_overlay_visible
        self._redraw()

    def _toggle_gt_overlay(self, checked: bool):
        self.gt_overlay_visible = bool(checked)
        self._editor_state.show_reference_overlay = self.gt_overlay_visible
        self._redraw()

    def _save_trace(self):
        if self._on_save_trace is None:
            return
        self._on_save_trace()

    def _save_all_traces(self):
        if self._on_save_all_traces is None:
            return
        self._on_save_all_traces()

    def _discard_trace(self):
        if self._on_discard_trace is not None:
            self._on_discard_trace()
        self._set_prediction_paths([])
        self._clear_transient_selection_state()
        self._redraw()

    def _run_postprocess(self):
        # Ensure in-progress edits in spin boxes are committed before reading
        # values, then push a fresh override snapshot to the pipeline manager.
        self._commit_postprocess_editor_values()
        self._on_postprocess_params_changed_slot()
        if self._on_run_postprocess is None:
            return
        self._on_run_postprocess(self._current_annotation_target())

    def _run_postprocess_all(self):
        # Keep "Post-Process All" consistent with the currently visible UI
        # values, even if the user has not left an edited field yet.
        self._commit_postprocess_editor_values()
        self._on_postprocess_params_changed_slot()
        if self._on_run_postprocess_all is None:
            return
        self._on_run_postprocess_all(self._current_annotation_target())

    def _commit_postprocess_editor_values(self):
        """Force-commit any in-progress text edits in postprocess spin boxes."""
        spinboxes = [
            "_pp_min_branch_length_spin",
            "_pp_max_branch_length_spin",
            "_pp_resampling_step_size_spin",
            "_pp_smoothing_window_spin",
            "_pp_overlap_dist_threshold_spin",
            "_pp_confidence_threshold_spin",
            "_pp_mask_smoothing_size_spin",
            "_pp_merge_timeout_seconds_spin",
        ]
        for name in spinboxes:
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "interpretText"):
                widget.interpretText()

    def _undo_postprocess(self):
        if self._on_undo_postprocess is None:
            return
        self._on_undo_postprocess(self._current_annotation_target())

    def _run_evaluation(self):
        if self._on_run_evaluation is None:
            return
        self._on_run_evaluation()

    def _run_evaluation_all(self):
        if self._on_run_evaluation_all is None:
            return
        self._on_run_evaluation_all()

    def _save_eval_report(self):
        if self._on_save_eval_report is None:
            return
        self._on_save_eval_report()

    def _select_gt_swc_path(self):
        if self._on_select_gt_swc_path is None:
            return
        result = self._on_select_gt_swc_path()
        if isinstance(result, tuple):
            selected, swc_rows = result
        else:
            selected, swc_rows = result, None
        if selected is not None:
            self._gt_swc_path = selected
            self._refresh_output_path_labels()
        if swc_rows is not None:
            self._set_reference_swc_rows(swc_rows)
            self._tree_swc_preview_source = np.empty((0, 7), dtype=np.float32)
            self._tree_swc_preview_filtered = np.empty((0, 7), dtype=np.float32)
            self._tree_preview_seed_points = []
            self._tree_preview_removed_ids = set()
            self._has_clip_preview = False
            self._invalidate_tree_overlay_cache()
            self._clear_transient_selection_state()
            self._refresh_annotation_target_options()
            self._redraw()

    def _clear_gt_swc_path(self):
        if self._on_clear_gt_swc_path is not None:
            self._gt_swc_path = self._on_clear_gt_swc_path()
        else:
            self._gt_swc_path = None
        self._refresh_output_path_labels()

    def _select_scales_path(self):
        if self._on_select_scales_path is None:
            return
        selected = self._on_select_scales_path()
        if selected is not None:
            self._scales_path = selected
            self._refresh_output_path_labels()

    def _clear_scales_path(self):
        if self._on_clear_scales_path is not None:
            self._scales_path = self._on_clear_scales_path()
        else:
            self._scales_path = None
        self._refresh_output_path_labels()

    def _has_image_dir(self) -> bool:
        return self._image_dir is not None and len(str(self._image_dir).strip()) > 0

    def _format_output_path(self, path_value: Optional[str]) -> str:
        if path_value is None or len(path_value) == 0:
            return "(not set)"
        # Add soft break points so long filesystem paths can wrap inside narrow tab layouts.
        if "/" in path_value:
            return path_value.replace("/", "/\u200b")
        return path_value

    def _refresh_output_path_labels(self):
        if self.mode != "seed":
            return
        self.image_dir_value_label.setText(self._format_output_path(self._image_dir))
        self.seeds_input_value_label.setText(self._format_output_path(self._seeds_input_path))
        self.seeds_output_value_label.setText(self._format_output_path(self._seeds_output_path))
        self.trace_output_value_label.setText(self._format_output_path(self._trace_output_path))
        self.model_weights_value_label.setText(self._format_output_path(self._model_weights_path))
        self.postprocess_output_value_label.setText(self._format_output_path(self._postprocess_output_dir))
        self.eval_output_value_label.setText(self._format_output_path(self._eval_output_dir))
        self.gt_swc_value_label.setText(self._format_output_path(self._gt_swc_path))
        self.scales_path_value_label.setText(self._format_output_path(self._scales_path))
        self.eval_scales_path_value_label.setText(self._format_output_path(self._scales_path))
        self.filtered_swc_output_value_label.setText(self._format_output_path(self._filtered_swc_output_dir))

    def _select_seeds_output_path(self):
        if self._on_select_seeds_output_path is None:
            return
        selected = self._on_select_seeds_output_path()
        if selected is not None:
            self._seeds_output_path = selected
            self._refresh_output_path_labels()

    def _clear_seeds_output_path(self):
        if self._on_clear_seeds_output_path is not None:
            self._seeds_output_path = self._on_clear_seeds_output_path()
        else:
            self._seeds_output_path = None
        self._refresh_output_path_labels()

    def _select_trace_output_path(self):
        if self._on_select_trace_output_path is None:
            return
        selected = self._on_select_trace_output_path()
        if selected is not None:
            self._trace_output_path = selected
            self._refresh_output_path_labels()

    def _clear_trace_output_path(self):
        if self._on_clear_trace_output_path is not None:
            self._trace_output_path = self._on_clear_trace_output_path()
        else:
            self._trace_output_path = None
        self._refresh_output_path_labels()

    def _select_model_weights_path(self):
        if self._on_select_model_weights_path is None:
            return
        selected = self._on_select_model_weights_path()
        if selected is not None:
            self._model_weights_path = selected
            self._refresh_output_path_labels()

    def _clear_model_weights_path(self):
        if self._on_clear_model_weights_path is not None:
            self._model_weights_path = self._on_clear_model_weights_path()
        else:
            self._model_weights_path = None
        self._refresh_output_path_labels()

    def _select_image_dir(self):
        if self._on_select_image_dir is None:
            return
        selected = self._on_select_image_dir()
        if selected is not None:
            self._image_dir = selected
            self._refresh_output_path_labels()
            self._redraw()

    def _clear_image_dir(self):
        if self._on_clear_image_dir is not None:
            self._image_dir = self._on_clear_image_dir()
        else:
            self._image_dir = None
        self._refresh_output_path_labels()
        self._redraw()

    def _select_seeds_input_path(self):
        if self._on_select_seeds_input_path is None:
            return
        result = self._on_select_seeds_input_path()
        # Callback may return a plain path string or a (path, seeds_array) tuple.
        if isinstance(result, tuple):
            selected, seeds_array = result
        else:
            selected, seeds_array = result, None
        if selected is not None:
            self._seeds_input_path = selected
            self._refresh_output_path_labels()
        if seeds_array is not None:
            self._set_seeds_from_array(seeds_array)
            self._redraw()

    def _clear_seeds_input_path(self):
        if self._on_clear_seeds_input_path is not None:
            self._seeds_input_path = self._on_clear_seeds_input_path()
        else:
            self._seeds_input_path = None
        self._refresh_output_path_labels()

    def _on_advanced_params_changed(self, *_args):
        if self._on_trace_params_changed is None:
            return
        self._pending_trace_param_overrides = self.get_trace_params_overrides()
        if self._trace_params_debounce_timer is None:
            self._flush_pending_trace_params()
            return
        self._trace_params_debounce_timer.start(250)

    def _flush_pending_trace_params(self):
        if self._on_trace_params_changed is None:
            return
        if self._pending_trace_param_overrides is None:
            return
        overrides = dict(self._pending_trace_param_overrides)
        self._pending_trace_param_overrides = None
        self._on_trace_params_changed(overrides)

    def get_trace_params_overrides(self) -> Dict[str, object]:
        """Return the current advanced trace parameter values from the config panel."""
        return {
            "max_len": int(self._trace_max_len_spin.value()),
            "max_paths": int(self._trace_max_paths_spin.value()),
            "branching": bool(self._trace_branching_check.isChecked()),
            "repeat_starts": bool(self._trace_repeat_starts_check.isChecked()),
            "seed_jitter_count": int(self._trace_seed_jitter_count_spin.value()),
            "seed_jitter_radius": float(self._trace_seed_jitter_radius_spin.value()),
            "seed_jitter_weight_strategy": self._trace_seed_jitter_weight_combo.currentText(),
        }

    def get_postprocess_params_overrides(self) -> Dict[str, object]:
        """Return the current post-processing parameter values from the config panel."""
        max_branch_length = float(self._pp_max_branch_length_spin.value())
        if max_branch_length >= 1e9 - 1.0:
            max_branch_length = float("inf")
        return {
            "enable_length_filter": bool(self._pp_enable_length_filter_check.isChecked()),
            "min_branch_length": float(self._pp_min_branch_length_spin.value()),
            "max_branch_length": max_branch_length,
            "enable_resample": bool(self._pp_enable_resample_check.isChecked()),
            "resampling_step_size": float(self._pp_resampling_step_size_spin.value()),
            "enable_smooth_paths": bool(self._pp_enable_smooth_paths_check.isChecked()),
            "smoothing_window": int(self._pp_smoothing_window_spin.value()),
            "enable_merge": bool(self._pp_enable_merge_check.isChecked()),
            "join_roots_to_common_center": bool(self._pp_join_roots_check.isChecked()),
            "merge_threshold": float(self._pp_overlap_dist_threshold_spin.value()),
            "confidence_threshold": int(self._pp_confidence_threshold_spin.value()),
            "mask_smoothing_size": int(self._pp_mask_smoothing_size_spin.value()),
            "merge_timeout_seconds": float(self._pp_merge_timeout_seconds_spin.value()),
        }

    def _update_postprocess_step_controls(self, *_args):
        if self.mode != "seed":
            return
        self._pp_min_branch_length_spin.setEnabled(bool(self._pp_enable_length_filter_check.isChecked()))
        self._pp_max_branch_length_spin.setEnabled(bool(self._pp_enable_length_filter_check.isChecked()))
        self._pp_resampling_step_size_spin.setEnabled(bool(self._pp_enable_resample_check.isChecked()))
        self._pp_smoothing_window_spin.setEnabled(bool(self._pp_enable_smooth_paths_check.isChecked()))
        merge_enabled = bool(self._pp_enable_merge_check.isChecked())
        self._pp_overlap_dist_threshold_spin.setEnabled(merge_enabled)
        self._pp_confidence_threshold_spin.setEnabled(merge_enabled)
        self._pp_mask_smoothing_size_spin.setEnabled(merge_enabled)
        self._pp_merge_timeout_seconds_spin.setEnabled(merge_enabled)

    def get_eval_params_overrides(self) -> Dict[str, object]:
        """Return the current evaluation parameter values from the config panel."""
        return {
            "distance_threshold": float(self._eval_distance_threshold_spin.value()),
        }

    def _on_postprocess_params_changed_slot(self, *_args):
        if self._on_postprocess_params_changed is None:
            return
        self._on_postprocess_params_changed(self.get_postprocess_params_overrides())

    def _on_eval_params_changed_slot(self, *_args):
        if self._on_eval_params_changed is None:
            return
        self._on_eval_params_changed(self.get_eval_params_overrides())

    def _select_postprocess_output_dir(self):
        if self._on_select_postprocess_output_dir is None:
            return
        selected = self._on_select_postprocess_output_dir()
        if selected is not None:
            self._postprocess_output_dir = selected
            self._refresh_output_path_labels()

    def _clear_postprocess_output_dir(self):
        if self._on_clear_postprocess_output_dir is not None:
            self._postprocess_output_dir = self._on_clear_postprocess_output_dir()
        else:
            self._postprocess_output_dir = None
        self._refresh_output_path_labels()

    def _select_eval_output_dir(self):
        if self._on_select_eval_output_dir is None:
            return
        selected = self._on_select_eval_output_dir()
        if selected is not None:
            self._eval_output_dir = selected
            self._refresh_output_path_labels()

    def _clear_eval_output_dir(self):
        if self._on_clear_eval_output_dir is not None:
            self._eval_output_dir = self._on_clear_eval_output_dir()
        else:
            self._eval_output_dir = None
        self._refresh_output_path_labels()

    def _set_reference_swc_rows(self, swc_rows: Optional[object]) -> None:
        normalized = _normalize_swc_rows(swc_rows)
        self._tree_swc_committed = normalized
        self._editor_state.set_reference_swc_rows(normalized)

    def _set_prediction_paths(self, finished_paths) -> None:
        had_prediction = bool(self._editor_state.prediction_annotation.nodes_by_id)
        normalized_paths: List[np.ndarray] = []
        if finished_paths is not None:
            for path in finished_paths:
                path_np = np.asarray(path, dtype=np.float32)
                if path_np.ndim == 2 and path_np.shape[0] >= 2 and path_np.shape[1] >= 3:
                    normalized_paths.append(path_np[:, :3].copy())
        self.finished_paths = normalized_paths
        self._editor_state.set_prediction_paths(normalized_paths)
        has_prediction = bool(self._editor_state.prediction_annotation.nodes_by_id)
        if (not had_prediction) and has_prediction:
            self._editor_state.active_annotation = AnnotationTarget.PREDICTION.value
        if hasattr(self, "annotation_target_combo"):
            self._refresh_annotation_target_options()

    def _push_annotation_undo_snapshot(self) -> None:
        if not hasattr(self, "_annotation_undo_stack"):
            self._annotation_undo_stack = []
        target = self._editor_state.active_annotation
        reference_rows = self._editor_state.reference_annotation.to_swc_rows().copy()
        prediction_paths = [path.tolist() for path in self._editor_state.prediction_annotation.to_paths()]
        self._annotation_undo_stack.append((target, reference_rows, prediction_paths))
        if hasattr(self, "btn_undo_annotation"):
            self.btn_undo_annotation.setEnabled(True)

    def _undo_annotation_edit(self) -> None:
        if not hasattr(self, "_annotation_undo_stack"):
            self._annotation_undo_stack = []
        if not self._annotation_undo_stack:
            return
        target, reference_rows, prediction_paths = self._annotation_undo_stack.pop()
        self._editor_state.set_reference_swc_rows(reference_rows)
        self._editor_state.set_prediction_paths(prediction_paths)
        self._editor_state.active_annotation = target
        self._sync_annotation_graph_to_view(AnnotationTarget.REFERENCE.value)
        self._sync_annotation_graph_to_view(AnnotationTarget.PREDICTION.value)
        self._clear_current_selection()
        self._refresh_annotation_target_options()
        if hasattr(self, "btn_undo_annotation"):
            self.btn_undo_annotation.setEnabled(bool(self._annotation_undo_stack))
        self._redraw()

    def _clear_transient_selection_state(self) -> None:
        self._editor_state.clear_transient_selection()

    def _clear_current_selection(self) -> None:
        self._clear_transient_selection_state()
        self._refresh_seed_order_controls()
        self._refresh_edit_action_controls()

    def _active_annotation_graph(self):
        return self._editor_state.active_annotation_graph()

    def _current_annotation_target(self) -> str:
        return self._editor_state.active_annotation

    def _has_prediction_annotation(self) -> bool:
        return bool(self._editor_state.prediction_annotation.nodes_by_id)

    def _refresh_annotation_target_options(self) -> None:
        if not hasattr(self, "annotation_target_combo"):
            return
        combo = self.annotation_target_combo
        current_target = self._editor_state.active_annotation
        if current_target == AnnotationTarget.PREDICTION.value and not self._has_prediction_annotation():
            current_target = AnnotationTarget.REFERENCE.value
            self._editor_state.active_annotation = current_target
            self._clear_current_selection()

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Reference", AnnotationTarget.REFERENCE.value)
        if self._has_prediction_annotation():
            combo.addItem("Prediction", AnnotationTarget.PREDICTION.value)

        selected_index = combo.findData(current_target)
        if selected_index < 0:
            selected_index = 0
            current_target = str(combo.itemData(selected_index))
            self._editor_state.active_annotation = current_target
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _on_annotation_target_changed(self, index: int) -> None:
        if index < 0:
            return
        target = self.annotation_target_combo.itemData(index)
        if not isinstance(target, str):
            return
        if self._editor_state.active_annotation == target:
            return
        self._editor_state.active_annotation = target
        self._clear_current_selection()
        self._redraw()

    def _sync_annotation_graph_to_view(self, target: str) -> None:
        if target == AnnotationTarget.PREDICTION.value:
            prediction_paths = self._editor_state.prediction_annotation.to_paths()
            self._set_prediction_paths(prediction_paths)
            if self._on_prediction_paths_changed is not None:
                self._on_prediction_paths_changed(
                    self._current_image_key,
                    [path.tolist() for path in prediction_paths],
                )
            return
        self._set_reference_swc_rows(self._editor_state.reference_annotation.to_swc_rows())
        if self._on_filtered_swc_changed is not None:
            self._on_filtered_swc_changed(self._current_image_key, self._tree_swc_committed.tolist())

    def _active_annotation_is_visible(self) -> bool:
        active_annotation = self._editor_state.active_annotation
        if active_annotation == "prediction":
            return bool(self.trace_overlay_visible)
        return bool(self.gt_overlay_visible)

    def _select_annotation_node_at_view_coords(
        self,
        view: str,
        xdata: float,
        ydata: float,
        tolerance: float = 4.0,
    ) -> Optional[int]:
        if not self._active_annotation_is_visible():
            self._editor_state.selection.selected_annotation_node_ids.clear()
            self._refresh_edit_action_controls()
            return None
        node_id = hit_test_annotation_node(
            annotation=self._active_annotation_graph(),
            view=view,
            xdata=xdata,
            ydata=ydata,
            projection_mode=self.projection_mode,
            cursor_zyx=(self.current_z, self.current_y, self.current_x),
            tolerance=tolerance,
        )
        if node_id is None:
            return None
        self._editor_state.selection.selected_annotation_node_ids.clear()
        self._editor_state.selection.selected_annotation_node_ids.add(int(node_id))
        self._editor_state.selection.selection_view = view
        self._refresh_edit_action_controls()
        return node_id

    def _poll_trace_status(self):
        if self._get_trace_status is None:
            return
        active_target = self._current_annotation_target()
        status = self._get_trace_status(active_target) or {}
        message = str(status.get("message", ""))
        if message != self._last_trace_status_message:
            self.trace_status_label.setText(message)
            self._last_trace_status_message = message
        running = bool(status.get("running", False))
        completed = status.get("progress_completed", None)
        total = status.get("progress_total", None)
        progress_text = ""
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            progress_text = f"Trace Progress: {completed}/{total}"
        if progress_text != self._last_trace_progress_text:
            self.trace_progress_label.setText(progress_text)
            self._last_trace_progress_text = progress_text

        self._can_undo_postprocess = bool(status.get("can_undo_postprocess", False))
        if self._show_postprocess_controls and hasattr(self, "btn_undo_postprocess"):
            self.btn_undo_postprocess.setEnabled((not running) and self._can_undo_postprocess)

        if self._trace_controls_running_state is None or running != self._trace_controls_running_state:
            self._set_trace_controls_busy(running)
            self._trace_controls_running_state = running

        overlay_token = status.get("overlay_token")
        if overlay_token is not None and overlay_token != self._trace_overlay_token:
            self._trace_overlay_token = overlay_token
            overlay_paths = status.get("overlay_paths", None)
            if overlay_paths is not None:
                self._set_prediction_paths(overlay_paths)
                self._clear_transient_selection_state()
                self._redraw(fast=True)

        postprocess_token = status.get("postprocess_token")
        if postprocess_token is not None and postprocess_token != self._trace_postprocess_token:
            self._trace_postprocess_token = postprocess_token
            if active_target == AnnotationTarget.PREDICTION.value:
                postprocess_paths = status.get("postprocess_paths", None)
                if postprocess_paths is not None:
                    self._set_prediction_paths(postprocess_paths)
                    self._clear_transient_selection_state()
                    self._redraw(fast=True)
            else:
                reference_rows = status.get("reference_swc_rows", None)
                if reference_rows is not None:
                    self._set_reference_swc_rows(reference_rows)
                    self._clear_transient_selection_state()
                    self._redraw(fast=True)

        token = status.get("token")
        if token is not None and token != self._trace_status_token:
            self._trace_status_token = token
            trace_output_dir = status.get("trace_output_dir", None)
            if trace_output_dir is None:
                self._trace_output_path = None
                self._refresh_output_path_labels()
            elif isinstance(trace_output_dir, str) and len(trace_output_dir) > 0:
                self._trace_output_path = trace_output_dir
                self._refresh_output_path_labels()
            model_weights_path = status.get("model_weights_path", None)
            if model_weights_path is None:
                self._model_weights_path = None
                self._refresh_output_path_labels()
            elif isinstance(model_weights_path, str) and len(model_weights_path) > 0:
                self._model_weights_path = model_weights_path
                self._refresh_output_path_labels()

            eval_report_text = status.get("eval_report_text", None)
            if eval_report_text is not None and self.eval_report_widget is not None:
                self.eval_report_widget.setPlainText(str(eval_report_text))

            gt_swc_path = status.get("gt_swc_path", None)
            if gt_swc_path is None:
                self._gt_swc_path = None
                self._refresh_output_path_labels()
            elif isinstance(gt_swc_path, str) and len(gt_swc_path) > 0:
                self._gt_swc_path = gt_swc_path
                self._refresh_output_path_labels()

    def _set_trace_controls_busy(self, running: bool):
        if not self._show_trace_controls:
            return
        self.btn_trace_neuron.setEnabled(not running)
        self.btn_trace_all.setEnabled(not running)
        self.btn_save_trace.setEnabled(not running)
        self.btn_save_all_traces.setEnabled(not running)
        self.btn_discard_trace.setEnabled(not running)
        self.chk_trace_overlay.setEnabled(not running)
        self.chk_gt_overlay.setEnabled(not running)
        self.btn_apply_component_filter.setEnabled(not running)
        self.btn_save_filtered_swc.setEnabled(not running)
        self.btn_prev_image.setEnabled(self._show_prev_button)
        self.btn_next_image.setEnabled(self._show_next_button)
        self.btn_cancel_trace.setEnabled(running)
        if running:
            self.btn_remove_selected.setEnabled(False)
            self.btn_clip_selected.setEnabled(False)
        else:
            self._refresh_edit_action_controls()
        if self._show_postprocess_controls:
            self.btn_run_postprocess.setEnabled(not running)
            self.btn_undo_postprocess.setEnabled((not running) and self._can_undo_postprocess)
            self.btn_run_evaluation.setEnabled(not running)

    def _invalidate_mip_cache(self):
        self._mip_cache_by_view = {"xy": None, "xz": None, "yz": None}

    def _get_plane(self, view: str) -> np.ndarray:
        if not self._has_image_dir():
            return np.zeros((2, 2), dtype=np.float32)

        if self.projection_mode == "mip":
            cached = self._mip_cache_by_view.get(view)
            if cached is not None:
                return cached
            if view == "xy":
                cached = np.max(self.img_np, axis=0)
            elif view == "xz":
                cached = np.max(self.img_np, axis=1)
            else:
                cached = np.max(self.img_np, axis=2)
            self._mip_cache_by_view[view] = cached
            return cached

        if view == "xy":
            return self.img_np[self.current_z, :, :]
        if view == "xz":
            return self.img_np[:, self.current_y, :]
        return self.img_np[:, :, self.current_x]

    def _iter_views(self):
        if self.maximized_view is not None:
            return [self.maximized_view]
        return ["xy", "xz", "yz"]

    def _get_full_limits(self, view: str):
        if not self._has_image_dir():
            return (-0.5, 1.5), (-0.5, 1.5)
        if view == "xy":
            return (-0.5, self.shape[2] - 0.5), (-0.5, self.shape[1] - 0.5)
        if view == "xz":
            return (-0.5, self.shape[2] - 0.5), (-0.5, self.shape[0] - 0.5)
        return (-0.5, self.shape[1] - 0.5), (-0.5, self.shape[0] - 0.5)

    def _set_crosshair_for_view(self, view: str):
        if view not in self.crosshair_artists:
            return
        vline, hline = self.crosshair_artists[view]
        visible = self.mode == "seed" and self._has_image_dir()
        vline.set_visible(visible)
        hline.set_visible(visible)
        if not visible:
            return

        if view == "xy":
            vline.set_xdata([self.current_x, self.current_x])
            hline.set_ydata([self.current_y, self.current_y])
        elif view == "xz":
            vline.set_xdata([self.current_x, self.current_x])
            hline.set_ydata([self.current_z, self.current_z])
        else:
            vline.set_xdata([self.current_y, self.current_y])
            hline.set_ydata([self.current_z, self.current_z])

    def _clear_overlay_artists(self, view: str):
        for artist in self.overlay_artists.get(view, []):
            try:
                artist.remove()
            except Exception:
                pass
        self.overlay_artists[view] = []

    def _clear_selection_artists(self, view: str):
        for artist in self.selection_artists.get(view, []):
            try:
                artist.remove()
            except Exception:
                pass
        self.selection_artists[view] = []

    def _build_layout(self, views):
        self.figure.clear()
        self._invalidate_blit_background()
        ncols = len(views)
        self.axis_view_map = {}
        self.axes_by_view = {}
        self.image_artists = {}
        self.crosshair_artists = {}
        self.overlay_artists = {}
        self.selection_artists = {}

        for i, view in enumerate(views):
            ax = self.figure.add_subplot(1, ncols, i + 1)
            plane = self._get_plane(view)
            image_artist = ax.imshow(plane, cmap="gray", origin="lower")
            image_artist.set_visible(self._has_image_dir())
            ax.set_title(view.upper())
            if not self._has_image_dir():
                ax.text(
                    0.5,
                    0.5,
                    "Image directory not set",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="0.6",
                    fontsize=10,
                )
            self.image_artists[view] = image_artist

            vline = ax.axvline(self.current_x, color="cyan", linewidth=0.8, alpha=0.8)
            hline = ax.axhline(self.current_y, color="cyan", linewidth=0.8, alpha=0.8)
            if self._supports_blit:
                vline.set_animated(True)
                hline.set_animated(True)
            self.crosshair_artists[view] = (vline, hline)
            self._set_crosshair_for_view(view)

            self.overlay_artists[view] = []
            self.selection_artists[view] = []
            if view in self.zoom_limits:
                xlim, ylim = self.zoom_limits[view]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            else:
                full_xlim, full_ylim = self._get_full_limits(view)
                ax.set_xlim(full_xlim)
                ax.set_ylim(full_ylim)
            self.axes_by_view[view] = ax
            self.axis_view_map[ax] = view

        self._current_views = list(views)
        self.figure.tight_layout()
        self._layout_dirty = False

    def _refresh_info_label(self) -> None:
        if self.mode != "seed":
            return
        self.info_label.setText(
            f"Tool: {self._active_tool} | Seeds: {len(self.seeds)}"
            f" | Effective: {len(self.effective_seed_overlay)} | "
            f"Cursor (z,y,x)=({self.current_z}, {self.current_y}, {self.current_x})"
        )

    def _invalidate_blit_background(self) -> None:
        self._blit_background_by_view = {}
        self._blit_background_valid = False

    def _on_canvas_draw(self, _event) -> None:
        if not self._supports_blit:
            return
        backgrounds: Dict[str, object] = {}
        try:
            for view in self._current_views:
                ax = self.axes_by_view.get(view)
                if ax is None:
                    continue
                backgrounds[view] = self.canvas.copy_from_bbox(ax.bbox)
        except Exception:
            self._invalidate_blit_background()
            return
        self._blit_background_by_view = backgrounds
        self._blit_background_valid = len(backgrounds) == len(self._current_views)

    def _ensure_blit_background(self) -> bool:
        if not self._supports_blit:
            return False
        if self._blit_background_valid:
            return True
        try:
            self.canvas.draw()
        except Exception:
            return False
        return self._blit_background_valid

    def _blit_views(
        self,
        views: List[str],
        extra_artists_by_view: Optional[Dict[str, List[object]]] = None,
    ) -> bool:
        if not self._ensure_blit_background():
            return False

        extra_artists_by_view = extra_artists_by_view or {}
        try:
            for view in views:
                ax = self.axes_by_view.get(view)
                background = self._blit_background_by_view.get(view)
                crosshair = self.crosshair_artists.get(view)
                if ax is None or background is None or crosshair is None:
                    return False

                self.canvas.restore_region(background)
                for artist in crosshair:
                    if artist is not None and artist.get_visible():
                        ax.draw_artist(artist)

                for artist in extra_artists_by_view.get(view, []):
                    if artist is None:
                        continue
                    if getattr(artist, "axes", None) is not ax:
                        continue
                    if not artist.get_visible():
                        continue
                    ax.draw_artist(artist)

                self.canvas.blit(ax.bbox)
        except Exception:
            self._invalidate_blit_background()
            return False
        return True

    def _try_blit_crosshair_update(self) -> bool:
        if self.projection_mode != "mip":
            return False
        views = self._iter_views()
        if self._layout_dirty or self._current_views != views or len(self.axes_by_view) == 0:
            return False
        for view in views:
            self._set_crosshair_for_view(view)
        return self._blit_views(list(views))

    def _redraw_selection_only(self) -> bool:
        """Refresh only selection-highlight artists and present via blitting when possible."""
        if not hasattr(self, "selection_artists") or not hasattr(self, "axes_by_view"):
            self._redraw()
            return False
        if not hasattr(self, "_layout_dirty") or not hasattr(self, "_current_views"):
            self._redraw()
            return False
        views = self._iter_views()
        if self._layout_dirty or self._current_views != views or len(self.axes_by_view) == 0:
            self._redraw()
            return False

        for view in views:
            ax = self.axes_by_view[view]
            self._clear_selection_artists(view)
            self.selection_artists[view] = self._draw_selection_overlay(ax=ax, view=view)

        if self._blit_views(list(views), extra_artists_by_view=self.selection_artists):
            self._refresh_info_label()
            return True

        self.canvas.draw_idle()
        self._refresh_info_label()
        return False

    def _redraw(self, fast: bool = False, skip_overlay_redraw: bool = False):
        views = self._iter_views()
        needs_layout = self._layout_dirty or (self._current_views != views) or (len(self.axes_by_view) == 0)
        if needs_layout:
            self._build_layout(views)

        for view in views:
            ax = self.axes_by_view[view]
            self.image_artists[view].set_data(self._get_plane(view))
            self.image_artists[view].set_visible(self._has_image_dir())
            self._set_crosshair_for_view(view)

            can_skip_overlay = (
                skip_overlay_redraw
                and self.projection_mode == "mip"
                and len(self.overlay_artists.get(view, [])) > 0
            )
            if not can_skip_overlay:
                self._clear_overlay_artists(view)
                self.overlay_artists[view] = self._draw_overlay(ax, view)

            self._clear_selection_artists(view)
            self.selection_artists[view] = self._draw_selection_overlay(ax=ax, view=view)

            # Re-enforce limits AFTER drawing overlays so that ax.plot() calls
            # inside overlay drawing cannot trigger matplotlib autoscale and
            # zoom out to world-coordinate extents (which would push seeds off-screen).
            if view in self.zoom_limits:
                xlim, ylim = self.zoom_limits[view]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            else:
                full_xlim, full_ylim = self._get_full_limits(view)
                ax.set_xlim(full_xlim)
                ax.set_ylim(full_ylim)

        self._invalidate_blit_background()
        if self._supports_blit:
            # Draw static artists once, capture per-axis backgrounds (via draw_event),
            # then draw animated crosshair/selection artists with blit.
            self.canvas.draw()
            if not self._blit_views(list(views), extra_artists_by_view=self.selection_artists):
                self.canvas.draw_idle()
        else:
            self.canvas.draw_idle()
        self._refresh_info_label()

    @staticmethod
    def _project_xyz_to_view(point_xyz: np.ndarray, view: str) -> Tuple[float, float]:
        if view == "xy":
            return float(point_xyz[0]), float(point_xyz[1])
        if view == "xz":
            return float(point_xyz[0]), float(point_xyz[2])
        return float(point_xyz[1]), float(point_xyz[2])

    def _draw_tree_overlay(self, ax, view: str):
        artists = []
        if self._has_clip_preview:
            swc_source = self._tree_swc_preview_source
        else:
            swc_source = self._tree_swc_committed
        if swc_source.size == 0:
            return artists

        self._ensure_tree_overlay_cache(swc_source)
        child_idx = self._tree_overlay_cache["child_idx"]
        parent_idx = self._tree_overlay_cache["parent_idx"]
        child_xyz = self._tree_overlay_cache["child_xyz"]
        parent_xyz = self._tree_overlay_cache["parent_xyz"]

        if child_xyz.shape[0] > 0:
            if self.projection_mode == "mip":
                visible = np.ones((child_xyz.shape[0],), dtype=bool)
            elif view == "xy":
                visible = (np.abs(child_xyz[:, 2] - self.current_z) <= 0.5) & (np.abs(parent_xyz[:, 2] - self.current_z) <= 0.5)
            elif view == "xz":
                visible = (np.abs(child_xyz[:, 1] - self.current_y) <= 0.5) & (np.abs(parent_xyz[:, 1] - self.current_y) <= 0.5)
            else:
                visible = (np.abs(child_xyz[:, 0] - self.current_x) <= 0.5) & (np.abs(parent_xyz[:, 0] - self.current_x) <= 0.5)

            if np.any(visible):
                if view == "xy":
                    start_points = child_xyz[:, [0, 1]]
                    end_points = parent_xyz[:, [0, 1]]
                elif view == "xz":
                    start_points = child_xyz[:, [0, 2]]
                    end_points = parent_xyz[:, [0, 2]]
                else:
                    start_points = child_xyz[:, [1, 2]]
                    end_points = parent_xyz[:, [1, 2]]

                segments = np.stack((start_points[visible], end_points[visible]), axis=1)

                collection = LineCollection(
                    segments,
                    colors="tomato",
                    linewidths=1.2,
                    alpha=0.85,
                )
                artists.append(ax.add_collection(collection))

        if self._has_clip_preview:
            roots_source = self._tree_swc_preview_filtered
        else:
            roots_source = self._tree_swc_committed
        if roots_source.size > 0:
            roots = roots_source[roots_source[:, 6] == -1]
            if roots.size > 0:
                if view == "xy":
                    artists.append(ax.scatter(roots[:, 2], roots[:, 3], c="darkorange", s=16, alpha=0.9))
                elif view == "xz":
                    artists.append(ax.scatter(roots[:, 2], roots[:, 4], c="darkorange", s=16, alpha=0.9))
                else:
                    artists.append(ax.scatter(roots[:, 3], roots[:, 4], c="darkorange", s=16, alpha=0.9))
        return artists

    def _seed_visible_in_view(self, seed: Tuple[int, int, int], view: str) -> bool:
        return seed_visible_in_view(
            seed=seed,
            view=view,
            projection_mode=self.projection_mode,
            cursor_zyx=(self.current_z, self.current_y, self.current_x),
        )

    def _seed_plot_coords(self, seed: Tuple[int, int, int], view: str) -> Tuple[float, float]:
        return seed_plot_coords(seed=seed, view=view)

    def _select_seed_at_view_coords(self, view: str, xdata: float, ydata: float, tolerance: float = 4.0) -> Optional[int]:
        best_idx = hit_test_seed(
            seeds=self.seeds,
            view=view,
            xdata=xdata,
            ydata=ydata,
            projection_mode=self.projection_mode,
            cursor_zyx=(self.current_z, self.current_y, self.current_x),
            tolerance=tolerance,
        )
        if best_idx is None:
            return None
        self.selected_seed_index = best_idx
        self._editor_state.selection.selected_annotation_node_ids.clear()
        self._editor_state.selection.clip_preview_node_ids.clear()
        self._editor_state.selection.selection_view = view
        self._refresh_seed_order_controls()
        self._refresh_edit_action_controls()
        return best_idx

    def _draw_overlay(self, ax, view: str):
        artists = []
        if not self._has_image_dir():
            return artists

        if self.gt_overlay_visible:
            artists.extend(self._draw_tree_overlay(ax=ax, view=view))

        if self.seeds:
            if self.effective_seed_overlay:
                effective_visible_indices = visible_seed_indices(
                    seeds=self.effective_seed_overlay,
                    view=view,
                    projection_mode=self.projection_mode,
                    cursor_zyx=(self.current_z, self.current_y, self.current_x),
                )
                if effective_visible_indices:
                    xs_eff = []
                    ys_eff = []
                    for idx in effective_visible_indices:
                        sx, sy = self._seed_plot_coords(self.effective_seed_overlay[idx], view)
                        xs_eff.append(sx)
                        ys_eff.append(sy)
                    artists.append(
                        ax.scatter(xs_eff, ys_eff, s=22, c="deepskyblue", alpha=0.45, edgecolors="none")
                    )

            visible_indices = visible_seed_indices(
                seeds=self.seeds,
                view=view,
                projection_mode=self.projection_mode,
                cursor_zyx=(self.current_z, self.current_y, self.current_x),
            )
            if visible_indices:
                xs = []
                ys = []
                for idx in visible_indices:
                    sx, sy = self._seed_plot_coords(self.seeds[idx], view)
                    xs.append(sx)
                    ys.append(sy)
                artists.append(ax.scatter(xs, ys, s=35, c="lime", edgecolors="black"))

        if self.trace_overlay_visible:
            artists.extend(self._draw_prediction_paths(ax=ax, view=view, color="deepskyblue"))

        return artists

    def _draw_selection_overlay(self, ax, view: str):
        artists = []
        if not self._has_image_dir():
            return artists

        clip_preview_node_ids = self._editor_state.selection.clip_preview_node_ids
        if clip_preview_node_ids:
            active_annotation = self._active_annotation_graph()
            xs_clip = []
            ys_clip = []
            for node_id in sorted(clip_preview_node_ids):
                node = active_annotation.nodes_by_id.get(int(node_id))
                if node is None:
                    continue
                if not annotation_node_visible_in_view(
                    xyz=node.xyz,
                    view=view,
                    projection_mode=self.projection_mode,
                    cursor_zyx=(self.current_z, self.current_y, self.current_x),
                ):
                    continue
                px, py = annotation_plot_coords(node.xyz, view=view)
                xs_clip.append(px)
                ys_clip.append(py)
            if xs_clip:
                artists.append(ax.scatter(xs_clip, ys_clip, s=86, c="crimson", edgecolors="black", zorder=5))

        selected_node_ids = self._editor_state.selection.selected_annotation_node_ids
        if selected_node_ids:
            active_annotation = self._active_annotation_graph()
            xs_nodes = []
            ys_nodes = []
            for node_id in sorted(selected_node_ids):
                node = active_annotation.nodes_by_id.get(int(node_id))
                if node is None:
                    continue
                if not annotation_node_visible_in_view(
                    xyz=node.xyz,
                    view=view,
                    projection_mode=self.projection_mode,
                    cursor_zyx=(self.current_z, self.current_y, self.current_x),
                ):
                    continue
                px, py = annotation_plot_coords(node.xyz, view=view)
                xs_nodes.append(px)
                ys_nodes.append(py)
            if xs_nodes:
                artists.append(ax.scatter(xs_nodes, ys_nodes, s=72, c="gold", edgecolors="black", zorder=6))

        selected_seed_idx = self.selected_seed_index
        if selected_seed_idx is not None and 0 <= int(selected_seed_idx) < len(self.seeds):
            seed = self.seeds[int(selected_seed_idx)]
            if self._seed_visible_in_view(seed=seed, view=view):
                sx, sy = self._seed_plot_coords(seed, view)
                artists.append(ax.scatter([sx], [sy], s=65, c="yellow", edgecolors="black", zorder=7))

        if self._supports_blit:
            for artist in artists:
                artist.set_animated(True)
        return artists

    def _draw_prediction_paths(self, ax, view: str, color: str = "deepskyblue"):
        """Draw every finished path for *view* using a single LineCollection.

        All in-slice (or MIP-projected) segments across all paths are collected and
        rendered as one LineCollection plus one scatter for isolated points. Keeping
        the artist count constant regardless of path count greatly reduces matplotlib
        draw overhead when scrubbing slices or selecting with a trace overlay visible.
        """
        artists = []
        line_segments: List[np.ndarray] = []
        point_xs: List[float] = []
        point_ys: List[float] = []

        mip = self.projection_mode == "mip"
        for path in self.finished_paths:
            if path.ndim != 2 or path.shape[1] < 3 or path.shape[0] == 0:
                continue

            if mip:
                if view == "xy":
                    xs, ys = path[:, 0], path[:, 1]
                elif view == "xz":
                    xs, ys = path[:, 0], path[:, 2]
                else:
                    xs, ys = path[:, 1], path[:, 2]
                if xs.shape[0] >= 2:
                    line_segments.append(np.column_stack((xs, ys)))
                elif xs.shape[0] == 1:
                    point_xs.append(float(xs[0]))
                    point_ys.append(float(ys[0]))
                continue

            if view == "xy":
                in_slice = np.isclose(path[:, 2], self.current_z, atol=0.5)
                x_coords = path[:, 0]  # X
                y_coords = path[:, 1]  # Y
            elif view == "xz":
                in_slice = np.isclose(path[:, 1], self.current_y, atol=0.5)
                x_coords = path[:, 0]  # X
                y_coords = path[:, 2]  # Z
            else:
                in_slice = np.isclose(path[:, 0], self.current_x, atol=0.5)
                x_coords = path[:, 1]  # Y
                y_coords = path[:, 2]  # Z

            indices = np.flatnonzero(in_slice)
            if indices.size == 0:
                continue

            split_points = np.where(np.diff(indices) > 1)[0] + 1
            for seg in np.split(indices, split_points):
                if seg.size >= 2:
                    line_segments.append(np.column_stack((x_coords[seg], y_coords[seg])))
                elif seg.size == 1:
                    point_xs.append(float(x_coords[seg[0]]))
                    point_ys.append(float(y_coords[seg[0]]))

        if line_segments:
            collection = LineCollection(line_segments, colors=color, linewidths=1.5)
            artists.append(ax.add_collection(collection, autolim=False))
        if point_xs:
            artists.append(ax.scatter(point_xs, point_ys, s=10, c=color))

        return artists

    def _set_active_view(self, view: Optional[str]):
        if view in ("xy", "xz", "yz"):
            self._active_view = view

    def _get_active_view(self) -> str:
        if self.maximized_view in ("xy", "xz", "yz"):
            return self.maximized_view
        if self._active_view in ("xy", "xz", "yz"):
            return self._active_view
        return "xy"

    def _push_current_zoom(self, view: str, ax):
        self.zoom_history.setdefault(view, []).append((ax.get_xlim(), ax.get_ylim()))

    def _zoom_back(self):
        view = self._get_active_view()
        history = self.zoom_history.get(view, [])
        if not history:
            return
        xlim, ylim = history.pop()
        self.zoom_limits[view] = (xlim, ylim)
        self._redraw()

    def _zoom_home(self):
        view = self._get_active_view()
        target_ax = None
        for ax, v in self.axis_view_map.items():
            if v == view:
                target_ax = ax
                break
        if target_ax is None:
            return
        self._push_current_zoom(view, target_ax)
        if view in self.zoom_limits:
            del self.zoom_limits[view]
        self._redraw()

    def _set_cursor_from_view_coords(self, view: str, xdata: float, ydata: float, redraw: bool = True) -> bool:
        prev_cursor = (self.current_z, self.current_y, self.current_x)
        if view == "xy":
            x = int(np.clip(np.round(xdata), 0, self.shape[2] - 1))
            y = int(np.clip(np.round(ydata), 0, self.shape[1] - 1))
            self.current_x = x
            self.current_y = y
            if self.projection_mode == "mip":
                self.current_z = int(np.argmax(self.img_np[:, y, x]))

        elif view == "xz":
            x = int(np.clip(np.round(xdata), 0, self.shape[2] - 1))
            z = int(np.clip(np.round(ydata), 0, self.shape[0] - 1))
            self.current_x = x
            self.current_z = z
            if self.projection_mode == "mip":
                self.current_y = int(np.argmax(self.img_np[z, :, x]))

        else:
            y = int(np.clip(np.round(xdata), 0, self.shape[1] - 1))
            z = int(np.clip(np.round(ydata), 0, self.shape[0] - 1))
            self.current_y = y
            self.current_z = z
            if self.projection_mode == "mip":
                self.current_x = int(np.argmax(self.img_np[z, y, :]))

        changed = (self.current_z, self.current_y, self.current_x) != prev_cursor
        self._sync_sliders_from_cursor()
        if redraw:
            self._redraw()
        return changed

    def _on_mouse_press(self, event):
        if not self._has_image_dir():
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        view = self.axis_view_map.get(event.inaxes)
        if view is None:
            return
        self._set_active_view(view)

        if int(getattr(event, "button", 0)) == 3:
            self._show_add_branch_menu_for_selected_node(event=event, view=view)
            return

        if int(getattr(event, "button", 0)) != 1:
            return

        self._drag_start = (float(event.xdata), float(event.ydata))
        self._drag_view = view
        if self._drag_rect is not None:
            try:
                self._drag_rect.remove()
            except Exception:
                pass
            self._drag_rect = None

        self._drag_rect = Rectangle(
            (self._drag_start[0], self._drag_start[1]),
            0,
            0,
            linewidth=1.0,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        event.inaxes.add_patch(self._drag_rect)
        if not self._blit_views([view], extra_artists_by_view={view: [self._drag_rect]}):
            self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if self._drag_start is None or self._drag_rect is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if self.axis_view_map.get(event.inaxes) != self._drag_view:
            return

        x0, y0 = self._drag_start
        x1 = float(event.xdata)
        y1 = float(event.ydata)
        self._drag_rect.set_x(min(x0, x1))
        self._drag_rect.set_y(min(y0, y1))
        self._drag_rect.set_width(abs(x1 - x0))
        self._drag_rect.set_height(abs(y1 - y0))
        if not self._blit_views([self._drag_view], extra_artists_by_view={self._drag_view: [self._drag_rect]}):
            self.canvas.draw_idle()

    def _handle_click_without_drag(self, view: str, xdata: float, ydata: float) -> None:
        if self.mode == "seed":
            selected_seed_idx = self._select_seed_at_view_coords(view, xdata, ydata)
            if selected_seed_idx is None:
                self._select_annotation_node_at_view_coords(view, xdata, ydata)
            cursor_changed = self._set_cursor_from_view_coords(view, xdata, ydata, redraw=False)
            if cursor_changed:
                self._redraw()
            else:
                self._redraw_selection_only()
            return
        self.canvas.draw_idle()

    def _select_annotation_nodes_in_view_rect(
        self,
        view: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> List[int]:
        if not self._active_annotation_is_visible():
            self._editor_state.selection.selected_annotation_node_ids.clear()
            self._editor_state.selection.clip_preview_node_ids.clear()
            self._refresh_edit_action_controls()
            return []

        min_x, max_x = sorted((float(x0), float(x1)))
        min_y, max_y = sorted((float(y0), float(y1)))
        selected_node_ids: List[int] = []
        graph = self._active_annotation_graph()
        cursor_zyx = (self.current_z, self.current_y, self.current_x)
        for node_id, node in graph.nodes_by_id.items():
            if not annotation_node_visible_in_view(node.xyz, view, self.projection_mode, cursor_zyx):
                continue
            px, py = annotation_plot_coords(node.xyz, view)
            if min_x <= px <= max_x and min_y <= py <= max_y:
                selected_node_ids.append(int(node_id))

        self.selected_seed_index = None
        self._editor_state.selection.clip_preview_node_ids.clear()
        self._editor_state.selection.selected_annotation_node_ids = set(selected_node_ids)
        self._editor_state.selection.selection_view = view if selected_node_ids else None
        self._refresh_seed_order_controls()
        self._refresh_edit_action_controls()
        return selected_node_ids

    def _handle_drag_release(self, event, view: str, start_x: float, start_y: float, end_x: float, end_y: float) -> None:
        if self._active_tool == "select":
            self._select_annotation_nodes_in_view_rect(
                view=view,
                x0=start_x,
                y0=start_y,
                x1=end_x,
                y1=end_y,
            )
            cursor_changed = self._set_cursor_from_view_coords(view, end_x, end_y, redraw=False)
            if cursor_changed:
                self._redraw()
            else:
                self._redraw_selection_only()
            return
        if self._active_tool != "zoom":
            if not self._blit_views([view]):
                self.canvas.draw_idle()
            return

        x0, x1 = sorted([start_x, end_x])
        y0, y1 = sorted([start_y, end_y])

        self._push_current_zoom(view, event.inaxes)
        event.inaxes.set_xlim(x0, x1)
        event.inaxes.set_ylim(y0, y1)
        self.zoom_limits[view] = ((x0, x1), (y0, y1))
        self._invalidate_blit_background()
        self.canvas.draw_idle()

    def _on_mouse_release(self, event):
        if self._drag_start is None:
            return

        start_x, start_y = self._drag_start
        view = self._drag_view

        if self._drag_rect is not None:
            try:
                self._drag_rect.remove()
            except Exception:
                pass
            self._drag_rect = None

        self._drag_start = None
        self._drag_view = None

        if event.inaxes is None or event.xdata is None or event.ydata is None or view is None:
            if view is None or (not self._blit_views([view])):
                self.canvas.draw_idle()
            return

        current_view = self.axis_view_map.get(event.inaxes)
        if current_view != view:
            if not self._blit_views([view]):
                self.canvas.draw_idle()
            return

        self._set_active_view(view)

        end_x = float(event.xdata)
        end_y = float(event.ydata)
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        drag_threshold = 1.0

        if dx <= drag_threshold and dy <= drag_threshold:
            self._handle_click_without_drag(view=view, xdata=end_x, ydata=end_y)
            return

        self._handle_drag_release(
            event=event,
            view=view,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
        )

    def _add_current_seed(self):
        if not self._has_image_dir():
            return
        if self._insert_child_node_at_crosshair():
            return
        self._editor_state.selection.pending_branch_seed_xyz = None
        self.seeds.append((self.current_z, self.current_y, self.current_x))
        self.selected_seed_index = len(self.seeds) - 1
        self._refresh_seed_order_controls()
        self._refresh_effective_seed_overlay()
        self._redraw()

    def _add_branch_seed_from_selected_node(self) -> bool:
        selected_node_ids = self._editor_state.selection.selected_annotation_node_ids
        if len(selected_node_ids) != 1:
            return False

        graph = self._active_annotation_graph()
        anchor_id = int(next(iter(selected_node_ids)))
        anchor_node = graph.nodes_by_id.get(anchor_id)
        if anchor_node is None:
            return False

        x, y, z = anchor_node.xyz
        seed = (
            int(np.clip(np.round(z), 0, self.shape[0] - 1)),
            int(np.clip(np.round(y), 0, self.shape[1] - 1)),
            int(np.clip(np.round(x), 0, self.shape[2] - 1)),
        )
        self._editor_state.selection.pending_branch_seed_xyz = (float(x), float(y), float(z))
        self.seeds.append(seed)
        self.selected_seed_index = len(self.seeds) - 1
        self._refresh_seed_order_controls()
        self._refresh_effective_seed_overlay()
        self._redraw()
        return True

    def _show_add_branch_menu_for_selected_node(self, event, view: str) -> None:
        if self.mode != "seed":
            return

        selected_node_ids = self._editor_state.selection.selected_annotation_node_ids
        if len(selected_node_ids) != 1:
            return

        clicked_node_id = hit_test_annotation_node(
            annotation=self._active_annotation_graph(),
            view=view,
            xdata=float(event.xdata),
            ydata=float(event.ydata),
            projection_mode=self.projection_mode,
            cursor_zyx=(self.current_z, self.current_y, self.current_x),
            tolerance=4.0,
        )
        if clicked_node_id is None or int(clicked_node_id) not in selected_node_ids:
            return

        qt_widgets = importlib.import_module("qtpy.QtWidgets")
        menu = qt_widgets.QMenu(self.dialog)
        add_branch_action = menu.addAction("Add Branch")

        gui_event = getattr(event, "guiEvent", None)
        global_pos = None
        if gui_event is not None:
            if hasattr(gui_event, "globalPosition"):
                gp = gui_event.globalPosition()
                global_pos = gp.toPoint() if hasattr(gp, "toPoint") else gp
            elif hasattr(gui_event, "globalPos"):
                global_pos = gui_event.globalPos()
        if global_pos is None:
            qt_gui = importlib.import_module("qtpy.QtGui")
            global_pos = qt_gui.QCursor.pos()

        exec_fn = getattr(menu, "exec", None)
        if exec_fn is None:
            exec_fn = getattr(menu, "exec_", None)
        if exec_fn is None:
            return

        selected_action = exec_fn(global_pos)
        if selected_action == add_branch_action:
            self._add_branch_seed_from_selected_node()

    def _insert_child_node_at_crosshair(self) -> bool:
        selected_node_ids = self._editor_state.selection.selected_annotation_node_ids
        if len(selected_node_ids) != 1:
            return False

        parent_id = int(next(iter(selected_node_ids)))
        graph = self._active_annotation_graph()
        if parent_id not in graph.nodes_by_id:
            return False

        target = self._editor_state.active_annotation
        selection_view = self._editor_state.selection.selection_view
        parent_radius = float(graph.nodes_by_id[parent_id].radius)
        self._push_annotation_undo_snapshot()
        new_node_id = graph.add_child(
            parent_id=parent_id,
            xyz=(float(self.current_x), float(self.current_y), float(self.current_z)),
            radius=parent_radius,
        )

        self._sync_annotation_graph_to_view(target)
        self.selected_seed_index = None
        self._editor_state.selection.pending_branch_seed_xyz = None
        self._editor_state.selection.clip_preview_node_ids.clear()
        self._editor_state.selection.selected_annotation_node_ids.clear()
        self._editor_state.selection.selected_annotation_node_ids.add(int(new_node_id))
        self._editor_state.selection.selection_view = selection_view
        self._refresh_seed_order_controls()
        self._refresh_edit_action_controls()
        self._redraw()
        return True

    def _on_mpl_keypress(self, event):
        if event.key == "shift":
            self._shift_held = True
            return
        if self.mode != "seed":
            return
        if event.key in (" ", "space"):
            self._add_current_seed()
        elif event.key in ("backspace", "delete"):
            self._remove_selected()
        elif event.key == "escape":
            self._clear_current_selection()
            self._redraw_selection_only()

    def _on_mpl_keyrelease(self, event):
        if event.key == "shift":
            self._shift_held = False

    def _sync_sliders_from_cursor(self):
        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.z_slider.blockSignals(True)
        self.x_slider.setValue(self.current_x)
        self.y_slider.setValue(self.current_y)
        self.z_slider.setValue(self.current_z)
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.z_slider.blockSignals(False)

    def _step_slice_for_view(self, view: str, delta: int):
        if view == "xy":
            self.current_z = int(np.clip(self.current_z + delta, 0, self.shape[0] - 1))
        elif view == "xz":
            self.current_y = int(np.clip(self.current_y + delta, 0, self.shape[1] - 1))
        else:
            self.current_x = int(np.clip(self.current_x + delta, 0, self.shape[2] - 1))

        self._sync_sliders_from_cursor()
        if self.projection_mode == "mip" and self._try_blit_crosshair_update():
            self._refresh_info_label()
            return
        # In MIP mode overlays do not depend on the slice position, so scrubbing the
        # slice only needs to move the crosshair; skip the overlay rebuild.
        self._redraw(skip_overlay_redraw=(self.projection_mode == "mip"))

    def _canvas_wheel_event(self, qt_event):
        if not self._has_image_dir():
            self._original_wheel_event(qt_event)
            return
        delta = 0
        if hasattr(qt_event, "angleDelta"):
            delta = qt_event.angleDelta().y()
        if delta == 0 and hasattr(qt_event, "pixelDelta"):
            delta = qt_event.pixelDelta().y()
        step = int(np.sign(delta))
        if step == 0:
            self._original_wheel_event(qt_event)
            return

        x = None
        y = None
        if hasattr(qt_event, "position"):
            pos = qt_event.position()
            x = float(pos.x())
            y = float(pos.y())
        elif hasattr(qt_event, "pos"):
            pos = qt_event.pos()
            x = float(pos.x())
            y = float(pos.y())

        if x is None or y is None:
            self._original_wheel_event(qt_event)
            return

        mpl_y = float(self.canvas.height()) - y
        active_view = None
        for ax, view in self.axis_view_map.items():
            if ax.bbox.contains(x, mpl_y):
                active_view = view
                break

        if active_view is None:
            self._original_wheel_event(qt_event)
            return

        self._set_active_view(active_view)
        self._step_slice_for_view(active_view, step)
        if hasattr(qt_event, "accept"):
            qt_event.accept()

    def exec(self) -> int:
        exec_fn = getattr(self.dialog, "exec", None)
        if callable(exec_fn):
            return exec_fn()
        return self.dialog.exec_()


def _run_ortho_dialog(dialog: _OrthoViewDialog) -> int:
    return dialog.exec()


def _interactive_seed_selection_orthoview(
    image_data: np.ndarray,
    initial_seeds: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Orthoview-based manual seed selector returning seeds in (z, y, x)."""
    dialog = _OrthoViewDialog(image_data=image_data, mode="seed", initial_seeds=initial_seeds)
    _run_ortho_dialog(dialog)

    seed_array = np.asarray(dialog.seeds, dtype=np.float32)
    if seed_array.size == 0:
        shape = dialog.shape
        center_seed = np.array([[shape[0] // 2, shape[1] // 2, shape[2] // 2]], dtype=np.float32)
        seed_array = center_seed
        print("No seeds selected in orthoview UI; using center seed.")

    seed_array[:, 0] = np.clip(seed_array[:, 0], 0, dialog.shape[0] - 1)
    seed_array[:, 1] = np.clip(seed_array[:, 1], 0, dialog.shape[1] - 1)
    seed_array[:, 2] = np.clip(seed_array[:, 2], 0, dialog.shape[2] - 1)

    print(f"\nSelected {len(seed_array)} seed point(s):")
    for i, seed in enumerate(seed_array):
        print(f"  Seed {i+1}: (z, y, x) = ({seed[0]:.1f}, {seed[1]:.1f}, {seed[2]:.1f})")

    return torch.tensor(seed_array, dtype=torch.float32)


def interactive_seed_selection_step(
    image_data: np.ndarray,
    neuron_name: str = "",
    initial_seeds: Optional[np.ndarray] = None,
    show_prev_button: bool = False,
    show_next_button: bool = False,
    show_save_buttons: bool = False,
    on_save_current: Optional[Callable[[np.ndarray], None]] = None,
    on_save_all: Optional[Callable[[], None]] = None,
    show_trace_controls: bool = False,
    on_trace_current: Optional[Callable[[np.ndarray], Optional[List[np.ndarray]]]] = None,
    on_trace_all: Optional[Callable[[], None]] = None,
    on_cancel_trace: Optional[Callable[[], None]] = None,
    get_trace_status: Optional[Callable[[str], Dict[str, object]]] = None,
    finished_paths: Optional[List[np.ndarray]] = None,
    on_save_trace: Optional[Callable[[], None]] = None,
    on_save_all_traces: Optional[Callable[[], None]] = None,
    on_discard_trace: Optional[Callable[[], None]] = None,
    seeds_output_path: Optional[str] = None,
    trace_output_path: Optional[str] = None,
    on_select_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
    model_weights_path: Optional[str] = None,
    on_select_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
    on_get_effective_seed_overlay: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    trace_seed_jitter_count: int = 0,
    trace_seed_jitter_radius: float = 0.0,
    trace_seed_jitter_weight_strategy: str = "uniform",
) -> Tuple[torch.Tensor, str]:
    """Single-image seed editor step that returns selected seeds and navigation action."""
    if _is_jupyter_notebook():
        raise RuntimeError("Interactive seed step is not supported in Jupyter notebooks.")

    if not _has_gui_display():
        raise RuntimeError("Interactive seed step requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")

    dialog = _OrthoViewDialog(
        image_data=image_data,
        mode="seed",
        finished_paths=finished_paths,
        neuron_name=neuron_name,
        initial_seeds=initial_seeds,
        show_prev_button=show_prev_button,
        show_next_button=show_next_button,
        show_save_buttons=show_save_buttons,
        on_save_current=on_save_current,
        on_save_all=on_save_all,
        show_trace_controls=show_trace_controls,
        on_trace_current=on_trace_current,
        on_trace_all=on_trace_all,
        on_cancel_trace=on_cancel_trace,
        get_trace_status=get_trace_status,
        on_save_trace=on_save_trace,
        on_save_all_traces=on_save_all_traces,
        on_discard_trace=on_discard_trace,
        seeds_output_path=seeds_output_path,
        trace_output_path=trace_output_path,
        on_select_seeds_output_path=on_select_seeds_output_path,
        on_select_trace_output_path=on_select_trace_output_path,
        model_weights_path=model_weights_path,
        on_select_model_weights_path=on_select_model_weights_path,
        on_get_effective_seed_overlay=on_get_effective_seed_overlay,
        trace_seed_jitter_count=trace_seed_jitter_count,
        trace_seed_jitter_radius=trace_seed_jitter_radius,
        trace_seed_jitter_weight_strategy=trace_seed_jitter_weight_strategy,
    )
    _run_ortho_dialog(dialog)

    seed_array = np.asarray(dialog.seeds, dtype=np.float32)
    if seed_array.size == 0:
        shape = dialog.shape
        seed_array = np.array([[shape[0] // 2, shape[1] // 2, shape[2] // 2]], dtype=np.float32)

    seed_array[:, 0] = np.clip(seed_array[:, 0], 0, dialog.shape[0] - 1)
    seed_array[:, 1] = np.clip(seed_array[:, 1], 0, dialog.shape[1] - 1)
    seed_array[:, 2] = np.clip(seed_array[:, 2], 0, dialog.shape[2] - 1)
    return torch.tensor(seed_array, dtype=torch.float32), dialog.session_action


def interactive_seed_selection_session(
    initial_context: Dict[str, object],
    on_prev_image: Callable[[np.ndarray], Optional[Dict[str, object]]],
    on_next_image: Callable[[np.ndarray], Optional[Dict[str, object]]],
    on_get_effective_seed_overlay: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    on_save_current: Optional[Callable[[np.ndarray], None]] = None,
    on_save_all: Optional[Callable[[], None]] = None,
    show_trace_controls: bool = False,
    on_trace_current: Optional[Callable[[np.ndarray], Optional[List[np.ndarray]]]] = None,
    on_trace_all: Optional[Callable[[], None]] = None,
    on_cancel_trace: Optional[Callable[[], None]] = None,
    get_trace_status: Optional[Callable[[str], Dict[str, object]]] = None,
    on_save_trace: Optional[Callable[[], None]] = None,
    on_save_all_traces: Optional[Callable[[], None]] = None,
    on_discard_trace: Optional[Callable[[], None]] = None,
    on_select_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_clear_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_clear_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
    on_clear_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_image_dir: Optional[Callable[[], Optional[str]]] = None,
    on_select_seeds_input_path: Optional[Callable[[], Optional[str]]] = None,
    on_clear_image_dir: Optional[Callable[[], Optional[str]]] = None,
    on_clear_seeds_input_path: Optional[Callable[[], Optional[str]]] = None,
    trace_max_len: int = 10000,
    trace_max_paths: int = 1000,
    trace_branching: bool = True,
    trace_repeat_starts: bool = False,
    trace_seed_jitter_count: int = 0,
    trace_seed_jitter_radius: float = 0.0,
    trace_seed_jitter_weight_strategy: str = "uniform",
    on_trace_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
    show_postprocess_controls: bool = False,
    on_run_postprocess: Optional[Callable[[str], None]] = None,
    on_run_postprocess_all: Optional[Callable[[str], None]] = None,
    on_undo_postprocess: Optional[Callable[[str], None]] = None,
    on_run_evaluation: Optional[Callable[[], None]] = None,
    on_run_evaluation_all: Optional[Callable[[], None]] = None,
    on_save_eval_report: Optional[Callable[[], None]] = None,
    on_select_gt_swc_path: Optional[Callable[[], object]] = None,
    on_clear_gt_swc_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_scales_path: Optional[Callable[[], Optional[str]]] = None,
    on_clear_scales_path: Optional[Callable[[], Optional[str]]] = None,
    postprocess_output_dir: Optional[str] = None,
    postprocess_enable_length_filter: bool = True,
    postprocess_min_branch_length: float = 5.0,
    postprocess_max_branch_length: float = 1e9,
    postprocess_enable_resample: bool = True,
    postprocess_resampling_step_size: float = 4.0,
    postprocess_enable_smooth_paths: bool = True,
    postprocess_smoothing_window: int = 5,
    postprocess_enable_merge: bool = True,
    postprocess_join_roots_to_common_center: bool = True,
    postprocess_merge_threshold: float = 1.0,
    postprocess_confidence_threshold: int = 0,
    postprocess_mask_smoothing_size: int = 0,
    postprocess_merge_timeout_seconds: float = 30.0,
    on_select_postprocess_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_clear_postprocess_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_postprocess_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
    eval_output_dir: Optional[str] = None,
    eval_distance_threshold: float = 1.0,
    on_select_eval_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_clear_eval_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_eval_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
    filtered_swc_output_dir: Optional[str] = None,
    on_select_filtered_swc_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_clear_filtered_swc_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_save_filtered_swc: Optional[Callable[[str, List[List[float]]], Optional[str]]] = None,
    on_filtered_swc_changed: Optional[Callable[[str, List[List[float]]], None]] = None,
    on_prediction_paths_changed: Optional[Callable[[str, List[List[List[float]]]], None]] = None,
) -> torch.Tensor:
    """Open a persistent seed-session dialog and update content in-place while navigating images."""
    if _is_jupyter_notebook():
        raise RuntimeError("Interactive seed session is not supported in Jupyter notebooks.")

    if not _has_gui_display():
        raise RuntimeError("Interactive seed session requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")

    image_data = initial_context.get("image_data")
    if image_data is None:
        raise ValueError("initial_context must include image_data.")

    dialog = _OrthoViewDialog(
        image_data=np.asarray(image_data),
        mode="seed",
        finished_paths=initial_context.get("finished_paths"),
        tree_swc_rows=initial_context.get("tree_swc_rows"),
        neuron_name=str(initial_context.get("neuron_name", "")),
        initial_seeds=initial_context.get("initial_seeds"),
        effective_seed_overlay=initial_context.get("effective_seed_overlay"),
        show_prev_button=bool(initial_context.get("show_prev_button", False)),
        show_next_button=bool(initial_context.get("show_next_button", False)),
        show_save_buttons=True,
        on_save_current=on_save_current,
        on_save_all=on_save_all,
        show_trace_controls=show_trace_controls,
        on_trace_current=on_trace_current,
        on_trace_all=on_trace_all,
        on_cancel_trace=on_cancel_trace,
        get_trace_status=get_trace_status,
        on_save_trace=on_save_trace,
        on_save_all_traces=on_save_all_traces,
        on_discard_trace=on_discard_trace,
        seeds_output_path=initial_context.get("seeds_output_path"),
        trace_output_path=initial_context.get("trace_output_path"),
        on_select_seeds_output_path=on_select_seeds_output_path,
        on_select_trace_output_path=on_select_trace_output_path,
        on_clear_seeds_output_path=on_clear_seeds_output_path,
        on_clear_trace_output_path=on_clear_trace_output_path,
        model_weights_path=initial_context.get("model_weights_path"),
        on_select_model_weights_path=on_select_model_weights_path,
        on_clear_model_weights_path=on_clear_model_weights_path,
        image_dir=initial_context.get("image_dir"),
        seeds_input_path=initial_context.get("seeds_input_path"),
        on_select_image_dir=on_select_image_dir,
        on_select_seeds_input_path=on_select_seeds_input_path,
        on_clear_image_dir=on_clear_image_dir,
        on_clear_seeds_input_path=on_clear_seeds_input_path,
        trace_max_len=trace_max_len,
        trace_max_paths=trace_max_paths,
        trace_branching=trace_branching,
        trace_repeat_starts=trace_repeat_starts,
        trace_seed_jitter_count=trace_seed_jitter_count,
        trace_seed_jitter_radius=trace_seed_jitter_radius,
        trace_seed_jitter_weight_strategy=trace_seed_jitter_weight_strategy,
        on_trace_params_changed=on_trace_params_changed,
        on_prev_image=on_prev_image,
        on_next_image=on_next_image,
        on_get_effective_seed_overlay=on_get_effective_seed_overlay,
        show_postprocess_controls=show_postprocess_controls,
        on_run_postprocess=on_run_postprocess,
        on_run_postprocess_all=on_run_postprocess_all,
        on_undo_postprocess=on_undo_postprocess,
        on_run_evaluation=on_run_evaluation,
        on_run_evaluation_all=on_run_evaluation_all,
        on_save_eval_report=on_save_eval_report,
        gt_swc_path=initial_context.get("gt_swc_path"),
        on_select_gt_swc_path=on_select_gt_swc_path,
        on_clear_gt_swc_path=on_clear_gt_swc_path,
        scales_path=initial_context.get("scales_path"),
        on_select_scales_path=on_select_scales_path,
        on_clear_scales_path=on_clear_scales_path,
        postprocess_output_dir=postprocess_output_dir,
        postprocess_enable_length_filter=postprocess_enable_length_filter,
        postprocess_min_branch_length=postprocess_min_branch_length,
        postprocess_max_branch_length=postprocess_max_branch_length,
        postprocess_enable_resample=postprocess_enable_resample,
        postprocess_resampling_step_size=postprocess_resampling_step_size,
        postprocess_enable_smooth_paths=postprocess_enable_smooth_paths,
        postprocess_smoothing_window=postprocess_smoothing_window,
        postprocess_enable_merge=postprocess_enable_merge,
        postprocess_join_roots_to_common_center=postprocess_join_roots_to_common_center,
        postprocess_merge_threshold=postprocess_merge_threshold,
        postprocess_confidence_threshold=postprocess_confidence_threshold,
        postprocess_mask_smoothing_size=postprocess_mask_smoothing_size,
        postprocess_merge_timeout_seconds=postprocess_merge_timeout_seconds,
        on_select_postprocess_output_dir=on_select_postprocess_output_dir,
        on_clear_postprocess_output_dir=on_clear_postprocess_output_dir,
        on_postprocess_params_changed=on_postprocess_params_changed,
        eval_output_dir=eval_output_dir,
        eval_distance_threshold=eval_distance_threshold,
        on_select_eval_output_dir=on_select_eval_output_dir,
        on_clear_eval_output_dir=on_clear_eval_output_dir,
        on_eval_params_changed=on_eval_params_changed,
        filtered_swc_output_dir=filtered_swc_output_dir,
        on_select_filtered_swc_output_dir=on_select_filtered_swc_output_dir,
        on_clear_filtered_swc_output_dir=on_clear_filtered_swc_output_dir,
        on_save_filtered_swc=on_save_filtered_swc,
        on_filtered_swc_changed=on_filtered_swc_changed,
        on_prediction_paths_changed=on_prediction_paths_changed,
    )
    _run_ortho_dialog(dialog)

    seed_array = np.asarray(dialog.seeds, dtype=np.float32)
    if seed_array.size == 0:
        shape = dialog.shape
        seed_array = np.array([[shape[0] // 2, shape[1] // 2, shape[2] // 2]], dtype=np.float32)

    seed_array[:, 0] = np.clip(seed_array[:, 0], 0, dialog.shape[0] - 1)
    seed_array[:, 1] = np.clip(seed_array[:, 1], 0, dialog.shape[1] - 1)
    seed_array[:, 2] = np.clip(seed_array[:, 2], 0, dialog.shape[2] - 1)
    return torch.tensor(seed_array, dtype=torch.float32)


def _show_inference_overlay_orthoview(image_data: np.ndarray, finished_paths, neuron_name: str = ""):
    """Orthoview-based inference overlay review viewer with finish button."""
    dialog = _OrthoViewDialog(
        image_data=image_data,
        mode="overlay",
        finished_paths=finished_paths,
        neuron_name=neuron_name,
    )
    _run_ortho_dialog(dialog)


def interactive_seed_selection(
    image_data: np.ndarray,
) -> torch.Tensor:
    """
    Manual seed selection entrypoint.

    Uses the orthoview Qt dialog for interactive selection.
    In Jupyter notebook frontends, returns None to trigger automatic seed fallback.
    """
    if _is_jupyter_notebook():
        print("Interactive seed selection not supported in Jupyter notebooks.")
        print("Falling back to automatic seed selection...")
        return None

    if not _has_gui_display():
        print("Interactive seed selection requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")
        print("Falling back to automatic seed selection...")
        return None

    return _interactive_seed_selection_orthoview(image_data=image_data, initial_seeds=None)


def prompt_seed_session_paths(
    image_dir: Optional[str] = None,
    seeds_input_path: Optional[str] = None,
    seeds_output_path: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Prompt for startup paths for seed-selection sessions using native file dialogs.

    Only the image directory is prompted interactively. Seed input/output paths are
    optional and may be set later from the in-session controls.
    """
    if _is_jupyter_notebook():
        raise RuntimeError("Path prompt dialog is not supported in Jupyter notebooks.")

    if not _has_gui_display():
        raise RuntimeError("Path prompt dialog requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")

    _ensure_qapplication()
    qt_widgets = importlib.import_module("qtpy.QtWidgets")

    chosen_image_dir = image_dir
    if not chosen_image_dir:
        chosen_image_dir = qt_widgets.QFileDialog.getExistingDirectory(
            None,
            "Select image directory",
            os.getcwd(),
        )
    if not chosen_image_dir:
        raise ValueError("Image directory is required to start the seed-selection session.")

    return chosen_image_dir, seeds_input_path, seeds_output_path


def prompt_save_json_path(default_path: Optional[str] = None) -> Optional[str]:
    """Prompt user for a JSON output path and return selected path or None."""
    if _is_jupyter_notebook():
        raise RuntimeError("Save-path dialog is not supported in Jupyter notebooks.")

    if not _has_gui_display():
        raise RuntimeError("Save-path dialog requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")

    _ensure_qapplication()
    qt_widgets = importlib.import_module("qtpy.QtWidgets")
    _opts = qt_widgets.QFileDialog.Options()
    _opts |= qt_widgets.QFileDialog.DontConfirmOverwrite
    selected_output, _ = qt_widgets.QFileDialog.getSaveFileName(
        None,
        "Select seeds output JSON",
        default_path or os.path.join(os.getcwd(), "seeds.json"),
        "JSON Files (*.json)",
        options=_opts,
    )
    return selected_output or None



def prompt_select_directory(default_path: Optional[str] = None) -> Optional[str]:
    """Prompt user for an output directory and return selected path or None."""
    if _is_jupyter_notebook():
        raise RuntimeError("Directory prompt is not supported in Jupyter notebooks.")

    if not _has_gui_display():
        raise RuntimeError("Directory prompt requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")

    _ensure_qapplication()
    qt_widgets = importlib.import_module("qtpy.QtWidgets")
    selected_dir = qt_widgets.QFileDialog.getExistingDirectory(
        None,
        "Select output directory",
        default_path or os.getcwd(),
    )
    return selected_dir or None


def prompt_select_model_weights(default_path: Optional[str] = None) -> Optional[str]:
    """Prompt user for model weights file and return selected path or None."""
    if _is_jupyter_notebook():
        raise RuntimeError("Model-weights prompt is not supported in Jupyter notebooks.")

    if not _has_gui_display():
        raise RuntimeError("Model-weights prompt requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")

    _ensure_qapplication()
    qt_widgets = importlib.import_module("qtpy.QtWidgets")
    selected_file, _ = qt_widgets.QFileDialog.getOpenFileName(
        None,
        "Select model weights",
        default_path or os.getcwd(),
        "Weights Files (*.pt *.pth);;All Files (*)",
    )
    return selected_file or None


def show_inference_overlay_and_wait(image_data: np.ndarray,
                                    finished_paths,
                                    neuron_name: str = ""):
    """
    Inference overlay review entrypoint.

    Uses the orthoview Qt dialog for interactive overlay review.
    In Jupyter notebook frontends, continues automatically.
    """
    if _is_jupyter_notebook():
        print("Interactive overlay review not supported in Jupyter notebooks. Continuing automatically.")
        return

    if not _has_gui_display():
        print("Interactive overlay review requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")
        print("Continuing automatically.")
        return

    _show_inference_overlay_orthoview(image_data=image_data, finished_paths=finished_paths, neuron_name=neuron_name)
