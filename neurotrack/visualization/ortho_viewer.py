#!/usr/bin/env python

"""
Interactive orthoview-based seed selection and inference overlay UI.

Provides a Qt dialog with synchronized XY / XZ / YZ orthoviews for manual
seed placement and inference overlay review.

Author: Bryson Gray
2024
"""

import importlib
import os
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

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


class _OrthoViewDialog:
    """Qt dialog with synchronized XY/XZ/YZ orthoviews and controls."""

    def __init__(
        self,
        image_data: np.ndarray,
        mode: str,
        finished_paths=None,
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
        on_trace_revision_select_point: Optional[Callable[[np.ndarray], Optional[Dict[str, object]]]] = None,
        on_trace_revision_preview: Optional[Callable[[], Optional[List[np.ndarray]]]] = None,
        on_trace_revision_launch: Optional[Callable[[], Optional[List[np.ndarray]]]] = None,
        get_trace_status: Optional[Callable[[], Dict[str, object]]] = None,
        on_save_trace: Optional[Callable[[], None]] = None,
        on_save_all_traces: Optional[Callable[[], None]] = None,
        seeds_output_path: Optional[str] = None,
        trace_output_path: Optional[str] = None,
        on_select_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
        on_select_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
        model_weights_path: Optional[str] = None,
        on_select_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
        on_prev_image: Optional[Callable[[np.ndarray], Optional[Dict[str, object]]]] = None,
        on_next_image: Optional[Callable[[np.ndarray], Optional[Dict[str, object]]]] = None,
        show_postprocess_controls: bool = False,
        on_run_postprocess: Optional[Callable[[], None]] = None,
        on_run_evaluation: Optional[Callable[[], None]] = None,
        on_save_postprocessed: Optional[Callable[[], None]] = None,
        on_save_eval_report: Optional[Callable[[], None]] = None,
        gt_swc_path: Optional[str] = None,
        on_select_gt_swc_path: Optional[Callable[[], Optional[str]]] = None,
        scales_path: Optional[str] = None,
        on_select_scales_path: Optional[Callable[[], Optional[str]]] = None,
        image_dir: Optional[str] = None,
        seeds_input_path: Optional[str] = None,
        on_select_image_dir: Optional[Callable[[], Optional[str]]] = None,
        on_select_seeds_input_path: Optional[Callable[[], Optional[str]]] = None,
        trace_step_width: float = 4.0,
        trace_n_trials: int = 1,
        trace_max_len: int = 10000,
        trace_max_paths: int = 1000,
        trace_branching: bool = True,
        trace_repeat_starts: bool = False,
        trace_stochastic_actions: bool = False,
        trace_auto_seed_mode: str = "remote_endnode",
        on_trace_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
        postprocess_output_dir: Optional[str] = None,
        postprocess_min_branch_length: float = 5.0,
        postprocess_resampling_step_size: float = 4.0,
        postprocess_smoothing_window: int = 5,
        postprocess_overlap_threshold: float = 0.5,
        postprocess_overlap_distance_threshold: float = 1.0,
        on_select_postprocess_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_postprocess_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
        eval_output_dir: Optional[str] = None,
        eval_distance_threshold: float = 1.0,
        on_select_eval_output_dir: Optional[Callable[[], Optional[str]]] = None,
        on_eval_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
    ):
        (
            QApplication,
            QWidget,
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QPushButton,
            QLabel,
            QSlider,
            QComboBox,
            Qt,
            FigureCanvas,
        ) = _try_import_ui_dependencies()

        _ensure_qapplication()

        self.QApplication = QApplication
        self.Qt = Qt
        self.mode = mode
        self.img_np = _extract_first_channel_numpy(image_data)
        self.shape = self.img_np.shape
        self.finished_paths = []
        if finished_paths is not None:
            for path in finished_paths:
                path_np = np.asarray(path, dtype=np.float32)
                if path_np.ndim == 2 and path_np.shape[0] >= 2 and path_np.shape[1] >= 3:
                    self.finished_paths.append(path_np[:, :3])

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
        self.session_action = "finish"
        self._on_save_current = on_save_current
        self._on_save_all = on_save_all
        self._show_trace_controls = bool(show_trace_controls and mode == "seed")
        self._on_trace_current = on_trace_current
        self._on_trace_all = on_trace_all
        self._on_cancel_trace = on_cancel_trace
        self._on_trace_revision_select_point = on_trace_revision_select_point
        self._on_trace_revision_preview = on_trace_revision_preview
        self._on_trace_revision_launch = on_trace_revision_launch
        self._get_trace_status = get_trace_status
        self._on_save_trace = on_save_trace
        self._on_save_all_traces = on_save_all_traces
        self._on_select_seeds_output_path = on_select_seeds_output_path
        self._on_select_trace_output_path = on_select_trace_output_path
        self._seeds_output_path = seeds_output_path
        self._trace_output_path = trace_output_path
        self._model_weights_path = model_weights_path
        self._on_select_model_weights_path = on_select_model_weights_path
        self._on_prev_image = on_prev_image
        self._on_next_image = on_next_image
        self._show_postprocess_controls = bool(show_postprocess_controls and mode == "seed")
        self._on_run_postprocess = on_run_postprocess
        self._on_run_evaluation = on_run_evaluation
        self._on_save_postprocessed = on_save_postprocessed
        self._on_save_eval_report = on_save_eval_report
        self._gt_swc_path = gt_swc_path
        self._on_select_gt_swc_path = on_select_gt_swc_path
        self._scales_path = scales_path
        self._on_select_scales_path = on_select_scales_path
        self._image_dir = image_dir
        self._seeds_input_path = seeds_input_path
        self._on_select_image_dir = on_select_image_dir
        self._on_select_seeds_input_path = on_select_seeds_input_path
        self._on_trace_params_changed = on_trace_params_changed
        self._postprocess_output_dir = postprocess_output_dir
        self._eval_output_dir = eval_output_dir
        self._on_select_postprocess_output_dir = on_select_postprocess_output_dir
        self._on_postprocess_params_changed = on_postprocess_params_changed
        self._on_select_eval_output_dir = on_select_eval_output_dir
        self._on_eval_params_changed = on_eval_params_changed
        self._trace_status_token = None
        self._trace_overlay_token = None
        self._trace_controls_running_state: Optional[bool] = None
        self._last_trace_status_message = ""
        self._last_trace_progress_text = ""
        self._pending_trace_param_overrides: Optional[Dict[str, object]] = None
        self._trace_params_debounce_timer = None
        self.trace_overlay_visible = True
        self.trace_revision_mode_enabled = False
        self.trace_revision_selected_node_xyz: Optional[np.ndarray] = None
        self.trace_revision_selected_point_xyz: Optional[np.ndarray] = None
        self.trace_revision_preview_paths: List[np.ndarray] = []
        self.trace_revision_preview_active = False
        self.post_paths: List[np.ndarray] = []
        self.post_overlay_visible = True
        self._qt_timer = None
        self._show_prev_button = bool(show_prev_button)
        self._show_next_button = bool(show_next_button)

        self.dialog = QDialog()
        title = "Manual Seed Selection" if mode == "seed" else "Inference Overlay"
        if neuron_name:
            title = f"{title}: {neuron_name}"
        self.dialog.setWindowTitle(title)
        self.dialog.resize(1400, 860)

        # Root layout: vertical (canvas row on top, footer on bottom)
        outer = QVBoxLayout(self.dialog)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Main horizontal row: left sidebar | center canvas | right sidebar
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
        self.btn_toggle_trace_overlay = QPushButton("Toggle Trace Overlay")
        self.btn_trace_revision_mode = QPushButton("Trace Revision Mode")
        self.btn_trace_revision_mode.setCheckable(True)
        self.btn_trace_revision_preview = QPushButton("Preview")
        self.btn_trace_revision_launch = QPushButton("Launch Retrace")
        self.trace_progress_label = QLabel("")
        self.btn_run_postprocess = QPushButton("Run Post-Processing")
        self.btn_run_evaluation = QPushButton("Run Evaluation")
        self.btn_save_postprocessed = QPushButton("Save Processed SWC")
        self.btn_save_eval_report = QPushButton("Save Eval Report")
        self.btn_toggle_post_overlay = QPushButton("Toggle Processed Overlay")
        if mode == "seed":
            self.btn_undo = QPushButton("Undo Last Seed")
            self.btn_clear = QPushButton("Clear Seeds")
            self.btn_save_seeds = QPushButton("Save Seeds")
            self.btn_set_seeds_output = QPushButton("Set Seeds Output")
            self.btn_set_trace_output = QPushButton("Set Trace Output")
            self.btn_set_model_weights = QPushButton("Set Model Weights")
            self.seeds_output_value_label = QLabel("")
            self.seeds_output_value_label.setWordWrap(True)
            self.trace_output_value_label = QLabel("")
            self.trace_output_value_label.setWordWrap(True)
            self.model_weights_value_label = QLabel("")
            self.model_weights_value_label.setWordWrap(True)
            self.btn_set_image_dir = QPushButton("Set Image Dir")
            self.image_dir_value_label = QLabel("")
            self.image_dir_value_label.setWordWrap(True)
            self.btn_set_seeds_input = QPushButton("Set Seeds Input")
            self.seeds_input_value_label = QLabel("")
            self.seeds_input_value_label.setWordWrap(True)
            # Advanced trace parameter widgets
            _qt_w = importlib.import_module("qtpy.QtWidgets")
            self._trace_step_width_spin = _qt_w.QDoubleSpinBox()
            self._trace_step_width_spin.setRange(0.1, 100.0)
            self._trace_step_width_spin.setSingleStep(0.5)
            self._trace_step_width_spin.setDecimals(2)
            self._trace_step_width_spin.setValue(trace_step_width)
            self._trace_n_trials_spin = _qt_w.QSpinBox()
            self._trace_n_trials_spin.setRange(1, 100)
            self._trace_n_trials_spin.setValue(trace_n_trials)
            self._trace_max_len_spin = _qt_w.QSpinBox()
            self._trace_max_len_spin.setRange(1, 1000000)
            self._trace_max_len_spin.setValue(trace_max_len)
            self._trace_max_paths_spin = _qt_w.QSpinBox()
            self._trace_max_paths_spin.setRange(1, 99999)
            self._trace_max_paths_spin.setValue(trace_max_paths)
            self._trace_branching_check = _qt_w.QCheckBox("Branching")
            self._trace_branching_check.setChecked(trace_branching)
            self._trace_repeat_starts_check = _qt_w.QCheckBox("Repeat Starts")
            self._trace_repeat_starts_check.setChecked(trace_repeat_starts)
            self._trace_stochastic_check = _qt_w.QCheckBox("Stochastic Actions")
            self._trace_stochastic_check.setChecked(trace_stochastic_actions)
            self._trace_auto_seed_combo = QComboBox()
            self._trace_auto_seed_combo.addItems(["remote_endnode", "root_nodes"])
            _auto_seed_idx = self._trace_auto_seed_combo.findText(trace_auto_seed_mode)
            if _auto_seed_idx >= 0:
                self._trace_auto_seed_combo.setCurrentIndex(_auto_seed_idx)
            # Post-processing parameter widgets
            self.btn_set_postprocess_output = QPushButton("Set Post-Process Output")
            self.postprocess_output_value_label = QLabel("")
            self.postprocess_output_value_label.setWordWrap(True)
            self._pp_min_branch_length_spin = _qt_w.QDoubleSpinBox()
            self._pp_min_branch_length_spin.setRange(0.0, 1000.0)
            self._pp_min_branch_length_spin.setSingleStep(0.5)
            self._pp_min_branch_length_spin.setDecimals(2)
            self._pp_min_branch_length_spin.setValue(postprocess_min_branch_length)
            self._pp_resampling_step_size_spin = _qt_w.QDoubleSpinBox()
            self._pp_resampling_step_size_spin.setRange(0.1, 100.0)
            self._pp_resampling_step_size_spin.setSingleStep(0.5)
            self._pp_resampling_step_size_spin.setDecimals(2)
            self._pp_resampling_step_size_spin.setValue(postprocess_resampling_step_size)
            self._pp_smoothing_window_spin = _qt_w.QSpinBox()
            self._pp_smoothing_window_spin.setRange(1, 100)
            self._pp_smoothing_window_spin.setValue(postprocess_smoothing_window)
            self._pp_overlap_threshold_spin = _qt_w.QDoubleSpinBox()
            self._pp_overlap_threshold_spin.setRange(0.0, 1.0)
            self._pp_overlap_threshold_spin.setSingleStep(0.05)
            self._pp_overlap_threshold_spin.setDecimals(3)
            self._pp_overlap_threshold_spin.setValue(postprocess_overlap_threshold)
            self._pp_overlap_dist_threshold_spin = _qt_w.QDoubleSpinBox()
            self._pp_overlap_dist_threshold_spin.setRange(0.0, 100.0)
            self._pp_overlap_dist_threshold_spin.setSingleStep(0.1)
            self._pp_overlap_dist_threshold_spin.setDecimals(2)
            self._pp_overlap_dist_threshold_spin.setValue(postprocess_overlap_distance_threshold)
            # Postprocess/eval path widgets (always created in seed mode)
            self.btn_set_gt_swc = QPushButton("Set GT SWC Dir")
            self.gt_swc_value_label = QLabel("")
            self.gt_swc_value_label.setWordWrap(True)
            self.btn_set_scales_path = QPushButton("Set Scales JSON")
            self.scales_path_value_label = QLabel("")
            self.scales_path_value_label.setWordWrap(True)
            # Evaluation parameter widgets
            self.btn_set_eval_output = QPushButton("Set Eval Output")
            self.eval_output_value_label = QLabel("")
            self.eval_output_value_label.setWordWrap(True)
            self._eval_distance_threshold_spin = _qt_w.QDoubleSpinBox()
            self._eval_distance_threshold_spin.setRange(0.0, 100.0)
            self._eval_distance_threshold_spin.setSingleStep(0.1)
            self._eval_distance_threshold_spin.setDecimals(2)
            self._eval_distance_threshold_spin.setValue(eval_distance_threshold)
            self.eval_scales_path_value_label = QLabel("")
            self.eval_scales_path_value_label.setWordWrap(True)
            self.btn_set_eval_scales_path = QPushButton("Set Scales JSON")

        # -----------------------------------------------------------------
        # LEFT SIDEBAR — view controls, sliders, seed & trace actions
        # -----------------------------------------------------------------
        left_sidebar = QWidget()
        left_sidebar.setMinimumWidth(130)
        left_sidebar.setObjectName("leftSidebar")
        left_layout = QVBoxLayout(left_sidebar)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(4)

        # Projection
        left_layout.addWidget(QLabel("Projection:"))
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["Slice", "MIP"])
        left_layout.addWidget(self.projection_combo)

        # Maximize view
        left_layout.addWidget(self._sidebar_separator())
        left_layout.addWidget(QLabel("Maximize View:"))
        _max_row1 = self._row_widget()
        _max_row1.layout().addWidget(self.btn_all)
        _max_row1.layout().addWidget(self.btn_xy)
        left_layout.addWidget(_max_row1)
        _max_row2 = self._row_widget()
        _max_row2.layout().addWidget(self.btn_xz)
        _max_row2.layout().addWidget(self.btn_yz)
        left_layout.addWidget(_max_row2)

        # Zoom
        _zoom_row = self._row_widget()
        _zoom_row.layout().addWidget(self.btn_back)
        _zoom_row.layout().addWidget(self.btn_home)
        left_layout.addWidget(_zoom_row)

        # Sliders
        left_layout.addWidget(self._sidebar_separator())
        left_layout.addWidget(QLabel("Slice Position:"))
        self.z_slider = self._make_slider(0, self.shape[0] - 1, self.current_z, "Z", left_layout)
        self.y_slider = self._make_slider(0, self.shape[1] - 1, self.current_y, "Y", left_layout)
        self.x_slider = self._make_slider(0, self.shape[2] - 1, self.current_x, "X", left_layout)

        if mode == "seed":
            # Image navigation
            left_layout.addWidget(self._sidebar_separator())
            _nav_row = self._row_widget()
            _nav_row.layout().addWidget(self.btn_prev_image)
            _nav_row.layout().addWidget(self.btn_next_image)
            left_layout.addWidget(_nav_row)
            self.btn_prev_image.setVisible(self._show_prev_button)
            self.btn_next_image.setVisible(self._show_next_button)

            # Seed controls
            left_layout.addWidget(self._sidebar_separator())
            left_layout.addWidget(QLabel("Seed Controls:"))
            left_layout.addWidget(self.btn_undo)
            left_layout.addWidget(self.btn_clear)
            if show_save_buttons:
                left_layout.addWidget(self.btn_save_seeds)
                left_layout.addWidget(self.btn_save_all_seeds)

            # Trace controls
            if self._show_trace_controls:
                left_layout.addWidget(self._sidebar_separator())
                left_layout.addWidget(QLabel("Tracing:"))
                left_layout.addWidget(self.btn_trace_neuron)
                left_layout.addWidget(self.btn_trace_all)
                left_layout.addWidget(self.btn_cancel_trace)
                left_layout.addWidget(self.btn_save_trace)
                left_layout.addWidget(self.btn_save_all_traces)
                left_layout.addWidget(self.btn_toggle_trace_overlay)
                left_layout.addWidget(self.btn_trace_revision_mode)
                left_layout.addWidget(self.btn_trace_revision_preview)
                left_layout.addWidget(self.btn_trace_revision_launch)
                left_layout.addWidget(self.trace_progress_label)

        left_layout.addStretch(1)

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
        self._original_wheel_event = self.canvas.wheelEvent
        self.canvas.wheelEvent = self._canvas_wheel_event
        center_layout.addWidget(self.canvas, stretch=1)

        main_splitter.addWidget(center_widget)

        # -----------------------------------------------------------------
        # RIGHT SIDEBAR — three-tab config panel
        # -----------------------------------------------------------------
        _qt_widgets_mod = importlib.import_module("qtpy.QtWidgets")
        _qt_core_mod = importlib.import_module("qtpy.QtCore")

        # Narrow toggle strip
        right_toggle_btn = QPushButton("⚙")
        right_toggle_btn.setFixedWidth(22)
        right_toggle_btn.setToolTip("Toggle Config Panel")
        right_toggle_btn.setCheckable(True)
        right_toggle_btn.setChecked(True)
        right_toggle_btn.setAutoDefault(False)
        right_toggle_btn.setDefault(False)
        right_toggle_btn.setFocusPolicy(self.Qt.NoFocus)

        right_sidebar = QWidget()
        right_sidebar.setMinimumWidth(200)
        right_sidebar.setObjectName("rightSidebar")
        right_layout = QVBoxLayout(right_sidebar)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(2)

        if mode == "seed":
            _tab_widget = _qt_widgets_mod.QTabWidget()
            _tab_widget.setDocumentMode(True)

            def _make_tab_scroll():
                _sa = _qt_widgets_mod.QScrollArea()
                _sa.setWidgetResizable(True)
                _sa.setHorizontalScrollBarPolicy(_qt_core_mod.Qt.ScrollBarAlwaysOff)
                _tw = QWidget()
                _tl = QVBoxLayout(_tw)
                _tl.setContentsMargins(6, 6, 6, 6)
                _tl.setSpacing(4)
                _sa.setWidget(_tw)
                return _sa, _tl

            # ---- Tab 1: Tracing ----
            _trace_sa, _trace_lay = _make_tab_scroll()
            _trace_lay.addWidget(QLabel("Image Directory:"))
            _trace_lay.addWidget(self.image_dir_value_label)
            _trace_lay.addWidget(self.btn_set_image_dir)
            _trace_lay.addWidget(QLabel("Model Weights:"))
            _trace_lay.addWidget(self.model_weights_value_label)
            _trace_lay.addWidget(self.btn_set_model_weights)
            _trace_lay.addWidget(QLabel("Trace Output:"))
            _trace_lay.addWidget(self.trace_output_value_label)
            _trace_lay.addWidget(self.btn_set_trace_output)
            _trace_lay.addWidget(QLabel("Seeds Output:"))
            _trace_lay.addWidget(self.seeds_output_value_label)
            _trace_lay.addWidget(self.btn_set_seeds_output)
            _trace_lay.addWidget(QLabel("Seeds Input (optional):"))
            _trace_lay.addWidget(self.seeds_input_value_label)
            _trace_lay.addWidget(self.btn_set_seeds_input)
            _trace_lay.addWidget(self._sidebar_separator())

            _adv_toggle = _qt_widgets_mod.QToolButton()
            _adv_toggle.setText("▶ Advanced")
            _adv_toggle.setCheckable(True)
            _adv_toggle.setChecked(False)
            _adv_toggle.setAutoRaise(True)
            _adv_toggle.setFocusPolicy(self.Qt.NoFocus)
            _adv_toggle.setSizePolicy(
                _qt_widgets_mod.QSizePolicy.Expanding,
                _qt_widgets_mod.QSizePolicy.Fixed,
            )
            _trace_lay.addWidget(_adv_toggle)

            _adv_panel = QWidget()
            _adv_panel.setVisible(False)
            _adv_layout = QVBoxLayout(_adv_panel)
            _adv_layout.setContentsMargins(4, 0, 4, 0)
            _adv_layout.setSpacing(4)
            _adv_layout.addWidget(QLabel("Step Width:"))
            _adv_layout.addWidget(self._trace_step_width_spin)
            _adv_layout.addWidget(QLabel("Num Trials:"))
            _adv_layout.addWidget(self._trace_n_trials_spin)
            _adv_layout.addWidget(QLabel("Max Length:"))
            _adv_layout.addWidget(self._trace_max_len_spin)
            _adv_layout.addWidget(QLabel("Max Paths:"))
            _adv_layout.addWidget(self._trace_max_paths_spin)
            _adv_layout.addWidget(self._trace_branching_check)
            _adv_layout.addWidget(self._trace_repeat_starts_check)
            _adv_layout.addWidget(self._trace_stochastic_check)
            _adv_layout.addWidget(QLabel("Auto Seed Mode:"))
            _adv_layout.addWidget(self._trace_auto_seed_combo)
            _trace_lay.addWidget(_adv_panel)

            def _toggle_adv_panel(checked, panel=_adv_panel, btn=_adv_toggle):
                panel.setVisible(checked)
                btn.setText("▼ Advanced" if checked else "▶ Advanced")

            _adv_toggle.toggled.connect(_toggle_adv_panel)
            _trace_lay.addStretch(1)
            _tab_widget.addTab(_trace_sa, "Tracing")

            # ---- Tab 2: Post-Processing ----
            _pp_sa, _pp_lay = _make_tab_scroll()
            _pp_lay.addWidget(QLabel("Output Directory:"))
            _pp_lay.addWidget(self.postprocess_output_value_label)
            _pp_lay.addWidget(self.btn_set_postprocess_output)
            _pp_lay.addWidget(self._sidebar_separator())
            _pp_lay.addWidget(QLabel("Min Branch Length:"))
            _pp_lay.addWidget(self._pp_min_branch_length_spin)
            _pp_lay.addWidget(QLabel("Resampling Step Size:"))
            _pp_lay.addWidget(self._pp_resampling_step_size_spin)
            _pp_lay.addWidget(QLabel("Smoothing Window:"))
            _pp_lay.addWidget(self._pp_smoothing_window_spin)
            _pp_lay.addWidget(QLabel("Overlap Threshold:"))
            _pp_lay.addWidget(self._pp_overlap_threshold_spin)
            _pp_lay.addWidget(QLabel("Overlap Distance Threshold:"))
            _pp_lay.addWidget(self._pp_overlap_dist_threshold_spin)
            _pp_lay.addWidget(self._sidebar_separator())
            _pp_lay.addWidget(QLabel("Scales JSON (optional):"))
            _pp_lay.addWidget(self.scales_path_value_label)
            _pp_lay.addWidget(self.btn_set_scales_path)
            if self._show_postprocess_controls:
                _pp_lay.addWidget(self._sidebar_separator())
                _pp_lay.addWidget(self.btn_run_postprocess)
                _pp_lay.addWidget(self.btn_save_postprocessed)
                _pp_lay.addWidget(self.btn_toggle_post_overlay)
            _pp_lay.addStretch(1)
            _tab_widget.addTab(_pp_sa, "Post-Processing")

            # ---- Tab 3: Evaluation ----
            _eval_sa, _eval_lay = _make_tab_scroll()
            _eval_lay.addWidget(QLabel("GT SWC Directory:"))
            _eval_lay.addWidget(self.gt_swc_value_label)
            _eval_lay.addWidget(self.btn_set_gt_swc)
            _eval_lay.addWidget(QLabel("Eval Output Directory:"))
            _eval_lay.addWidget(self.eval_output_value_label)
            _eval_lay.addWidget(self.btn_set_eval_output)
            _eval_lay.addWidget(self._sidebar_separator())
            _eval_lay.addWidget(QLabel("Distance Threshold:"))
            _eval_lay.addWidget(self._eval_distance_threshold_spin)
            _eval_lay.addWidget(QLabel("Scales JSON (optional):"))
            _eval_lay.addWidget(self.eval_scales_path_value_label)
            _eval_lay.addWidget(self.btn_set_eval_scales_path)
            if self._show_postprocess_controls:
                _eval_lay.addWidget(self._sidebar_separator())
                _eval_lay.addWidget(self.btn_run_evaluation)
                _eval_lay.addWidget(self.btn_save_eval_report)
                _eval_lay.addWidget(self._sidebar_separator())
                _eval_lay.addWidget(QLabel("Evaluation Report:"))
                self.eval_report_widget = _qt_widgets_mod.QTextEdit()
                self.eval_report_widget.setReadOnly(True)
                self.eval_report_widget.setPlaceholderText(
                    "Evaluation report will appear here after running evaluation."
                )
                self.eval_report_widget.setMinimumHeight(120)
                _eval_lay.addWidget(self.eval_report_widget)
            else:
                self.eval_report_widget = None
            _eval_lay.addStretch(1)
            _tab_widget.addTab(_eval_sa, "Evaluation")

            right_layout.addWidget(_tab_widget)
        else:
            self.eval_report_widget = None

        # Wrap toggle strip + sidebar panel side-by-side
        right_area = QWidget()
        right_area_layout = QHBoxLayout(right_area)
        right_area_layout.setContentsMargins(0, 0, 0, 0)
        right_area_layout.setSpacing(0)
        right_area_layout.addWidget(right_toggle_btn)
        right_area_layout.addWidget(right_sidebar)
        main_splitter.addWidget(right_area)

        # Set initial sizes [left, center, right] and lock only center to stretch
        main_splitter.setSizes([210, 930, 282])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)

        right_toggle_btn.toggled.connect(right_sidebar.setVisible)

        if mode == "seed":
            self._refresh_output_path_labels()

        # -----------------------------------------------------------------
        # FOOTER — info bar and status label
        # -----------------------------------------------------------------
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(6, 2, 6, 2)
        self.info_label = QLabel(
            "Space=Add seed, Backspace=Undo, Scroll=Slice, Drag=Zoom Box"
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
            button_list.extend([self.btn_set_seeds_output, self.btn_set_trace_output, self.btn_set_model_weights])
            button_list.extend([self.btn_set_image_dir, self.btn_set_seeds_input])
            button_list.extend([self.btn_prev_image, self.btn_next_image])
            button_list.extend([self.btn_set_postprocess_output, self.btn_set_gt_swc, self.btn_set_scales_path])
            button_list.extend([self.btn_set_eval_output, self.btn_set_eval_scales_path])
            if show_save_buttons:
                button_list.extend([self.btn_save_seeds, self.btn_save_all_seeds])
            if self._show_trace_controls:
                button_list.extend([
                    self.btn_trace_neuron,
                    self.btn_trace_all,
                    self.btn_cancel_trace,
                    self.btn_save_trace,
                    self.btn_save_all_traces,
                    self.btn_toggle_trace_overlay,
                    self.btn_trace_revision_mode,
                    self.btn_trace_revision_preview,
                    self.btn_trace_revision_launch,
                ])
            if self._show_postprocess_controls:
                button_list.extend([
                    self.btn_run_postprocess,
                    self.btn_run_evaluation,
                    self.btn_save_postprocessed,
                    self.btn_save_eval_report,
                    self.btn_toggle_post_overlay,
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
            self.btn_set_seeds_output.clicked.connect(self._select_seeds_output_path)
            self.btn_set_trace_output.clicked.connect(self._select_trace_output_path)
            self.btn_set_model_weights.clicked.connect(self._select_model_weights_path)
            self.btn_set_image_dir.clicked.connect(self._select_image_dir)
            self.btn_set_seeds_input.clicked.connect(self._select_seeds_input_path)
            self.btn_prev_image.clicked.connect(self._go_prev_image)
            self.btn_next_image.clicked.connect(self._go_next_image)
            self._trace_step_width_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_n_trials_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_max_len_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_max_paths_spin.valueChanged.connect(self._on_advanced_params_changed)
            self._trace_branching_check.toggled.connect(self._on_advanced_params_changed)
            self._trace_repeat_starts_check.toggled.connect(self._on_advanced_params_changed)
            self._trace_stochastic_check.toggled.connect(self._on_advanced_params_changed)
            self._trace_auto_seed_combo.currentIndexChanged.connect(self._on_advanced_params_changed)
            self.btn_set_postprocess_output.clicked.connect(self._select_postprocess_output_dir)
            self._pp_min_branch_length_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_resampling_step_size_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_smoothing_window_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_overlap_threshold_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self._pp_overlap_dist_threshold_spin.valueChanged.connect(self._on_postprocess_params_changed_slot)
            self.btn_set_eval_output.clicked.connect(self._select_eval_output_dir)
            self._eval_distance_threshold_spin.valueChanged.connect(self._on_eval_params_changed_slot)
            self.btn_set_eval_scales_path.clicked.connect(self._select_scales_path)
            if show_save_buttons:
                self.btn_save_seeds.clicked.connect(self._save_current_seeds)
                self.btn_save_all_seeds.clicked.connect(self._save_all_seeds)
            if self._show_trace_controls:
                self.btn_trace_neuron.clicked.connect(self._trace_current_neuron)
                self.btn_trace_all.clicked.connect(self._trace_all_neurons)
                self.btn_cancel_trace.clicked.connect(self._cancel_trace)
                self.btn_save_trace.clicked.connect(self._save_trace)
                self.btn_save_all_traces.clicked.connect(self._save_all_traces)
                self.btn_toggle_trace_overlay.clicked.connect(self._toggle_trace_overlay)
                self.btn_trace_revision_mode.toggled.connect(self._toggle_trace_revision_mode)
                self.btn_trace_revision_preview.clicked.connect(self._preview_trace_revision)
                self.btn_trace_revision_launch.clicked.connect(self._launch_trace_revision)
            self.btn_set_gt_swc.clicked.connect(self._select_gt_swc_path)
            self.btn_set_scales_path.clicked.connect(self._select_scales_path)
            if self._show_postprocess_controls:
                self.btn_run_postprocess.clicked.connect(self._run_postprocess)
                self.btn_run_evaluation.clicked.connect(self._run_evaluation)
                self.btn_save_postprocessed.clicked.connect(self._save_postprocessed)
                self.btn_save_eval_report.clicked.connect(self._save_eval_report)
                self.btn_toggle_post_overlay.clicked.connect(self._toggle_post_overlay)

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
        self._update_trace_revision_controls()

        self._mpl_press_cid = self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._mpl_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._mpl_release_cid = self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._mpl_key_cid = self.canvas.mpl_connect("key_press_event", self._on_mpl_keypress)
        self._mpl_key_release_cid = self.canvas.mpl_connect("key_release_event", self._on_mpl_keyrelease)
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
            if self.mode == "seed" and event.key() == self.Qt.Key_Space:
                self._add_current_seed()
                return
            if self.mode == "seed" and event.key() in (self.Qt.Key_Backspace, self.Qt.Key_Delete):
                self._undo_seed()
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
        self._redraw()

    def _undo_seed(self):
        if self.seeds:
            self.seeds.pop()
            self._redraw()

    def _clear_seeds(self):
        self.seeds.clear()
        self._redraw()

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
        if seed_array is None:
            return
        arr = np.asarray(seed_array, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return
        arr[:, 0] = np.clip(arr[:, 0], 0, self.shape[0] - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, self.shape[1] - 1)
        arr[:, 2] = np.clip(arr[:, 2], 0, self.shape[2] - 1)
        self.seeds = [(int(round(z)), int(round(y)), int(round(x))) for z, y, x in arr]

    def _set_finished_paths(self, finished_paths):
        self.finished_paths = []
        if finished_paths is None:
            return
        for path in finished_paths:
            path_np = np.asarray(path, dtype=np.float32)
            if path_np.ndim == 2 and path_np.shape[0] >= 2 and path_np.shape[1] >= 3:
                self.finished_paths.append(path_np[:, :3])

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
        self._set_finished_paths(context.get("finished_paths"))

        self._show_prev_button = bool(context.get("show_prev_button", False))
        self._show_next_button = bool(context.get("show_next_button", False))
        self.btn_prev_image.setVisible(self._show_prev_button)
        self.btn_next_image.setVisible(self._show_next_button)

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
        self._refresh_output_path_labels()

        neuron_name = context.get("neuron_name", "")
        title = "Manual Seed Selection"
        if isinstance(neuron_name, str) and len(neuron_name) > 0:
            title = f"{title}: {neuron_name}"
        self.dialog.setWindowTitle(title)

        # clear per-image post-processing and evaluation state
        self.post_paths = []
        self.trace_revision_selected_node_xyz = None
        self.trace_revision_selected_point_xyz = None
        self.trace_revision_preview_paths = []
        self.trace_revision_preview_active = False
        self.trace_revision_mode_enabled = False
        self._update_trace_revision_controls()
        if self.eval_report_widget is not None:
            self.eval_report_widget.setPlainText("")

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
            self.finished_paths = []
            for path in paths:
                path_np = np.asarray(path, dtype=np.float32)
                if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] >= 2:
                    self.finished_paths.append(path_np[:, :3])
            self.trace_revision_selected_node_xyz = None
            self.trace_revision_selected_point_xyz = None
            self.trace_revision_preview_paths = []
            self.trace_revision_preview_active = False
            self._update_trace_revision_controls()
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

    def _toggle_trace_overlay(self):
        self.trace_overlay_visible = not self.trace_overlay_visible
        self._redraw()

    def _has_trace_revision_callbacks(self) -> bool:
        return (
            self._on_trace_revision_select_point is not None
            and self._on_trace_revision_preview is not None
            and self._on_trace_revision_launch is not None
        )

    def _update_trace_revision_controls(self):
        if self.mode != "seed" or not self._show_trace_controls:
            return
        callbacks_ready = self._has_trace_revision_callbacks()

        if self.btn_trace_revision_mode.isChecked() != self.trace_revision_mode_enabled:
            self.btn_trace_revision_mode.blockSignals(True)
            self.btn_trace_revision_mode.setChecked(self.trace_revision_mode_enabled)
            self.btn_trace_revision_mode.blockSignals(False)

        self.btn_trace_revision_mode.setEnabled(callbacks_ready)
        self.btn_trace_revision_preview.setEnabled(
            callbacks_ready
            and self.trace_revision_mode_enabled
            and self.trace_revision_selected_node_xyz is not None
        )
        self.btn_trace_revision_launch.setEnabled(
            callbacks_ready and self.trace_revision_mode_enabled and self.trace_revision_preview_active
        )

    def _toggle_trace_revision_mode(self, checked: bool):
        self.trace_revision_mode_enabled = bool(checked)
        if not self.trace_revision_mode_enabled:
            self.trace_revision_selected_node_xyz = None
            self.trace_revision_selected_point_xyz = None
            self.trace_revision_preview_paths = []
            self.trace_revision_preview_active = False
            self._redraw()
        self._update_trace_revision_controls()

    def _set_trace_revision_selected_node(self, node_xyz: Optional[object]):
        if node_xyz is None:
            self.trace_revision_selected_node_xyz = None
            return
        arr = np.asarray(node_xyz, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 3:
            self.trace_revision_selected_node_xyz = None
            return
        self.trace_revision_selected_node_xyz = arr[:3].copy()

    def _set_trace_revision_selected_point(self, point_xyz: Optional[object]):
        if point_xyz is None:
            self.trace_revision_selected_point_xyz = None
            return
        arr = np.asarray(point_xyz, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 3:
            self.trace_revision_selected_point_xyz = None
            return
        self.trace_revision_selected_point_xyz = arr[:3].copy()

    def _set_trace_revision_preview_paths(self, preview_paths: Optional[List[np.ndarray]]):
        self.trace_revision_preview_paths = []
        if preview_paths is None:
            return
        for path in preview_paths:
            path_np = np.asarray(path, dtype=np.float32)
            if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] > 0:
                self.trace_revision_preview_paths.append(path_np[:, :3])

    def _select_trace_revision_point(self):
        if (
            not self.trace_revision_mode_enabled
            or self._on_trace_revision_select_point is None
            or not self._has_trace_revision_callbacks()
        ):
            return

        selected_point_zyx = np.asarray(
            [self.current_z, self.current_y, self.current_x],
            dtype=np.float32,
        )
        result = self._on_trace_revision_select_point(selected_point_zyx)

        selected_node_xyz = result.get("selected_node_xyz") if isinstance(result, dict) else None
        selected_point_xyz = result.get("selected_point_xyz") if isinstance(result, dict) else None
        if selected_point_xyz is None and selected_node_xyz is not None:
            selected_point_xyz = [
                float(selected_point_zyx[2]),
                float(selected_point_zyx[1]),
                float(selected_point_zyx[0]),
            ]
        self._set_trace_revision_selected_node(selected_node_xyz)
        self._set_trace_revision_selected_point(selected_point_xyz)
        self.trace_revision_preview_active = False
        self.trace_revision_preview_paths = []
        self._update_trace_revision_controls()
        self._redraw()

    def _preview_trace_revision(self):
        if (
            not self.trace_revision_mode_enabled
            or self._on_trace_revision_preview is None
            or not self._has_trace_revision_callbacks()
        ):
            return
        preview_paths = self._on_trace_revision_preview()
        self._set_trace_revision_preview_paths(preview_paths)
        self.trace_revision_preview_active = len(self.trace_revision_preview_paths) > 0
        self._update_trace_revision_controls()
        self._redraw()

    def _launch_trace_revision(self):
        if (
            not self.trace_revision_mode_enabled
            or self._on_trace_revision_launch is None
            or not self._has_trace_revision_callbacks()
        ):
            return
        self._flush_pending_trace_params()
        paths = self._on_trace_revision_launch()
        if paths is None:
            return

        self.finished_paths = []
        for path in paths:
            path_np = np.asarray(path, dtype=np.float32)
            if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] >= 2:
                self.finished_paths.append(path_np[:, :3])

        self.trace_revision_selected_node_xyz = None
        self.trace_revision_selected_point_xyz = None
        self.trace_revision_preview_paths = []
        self.trace_revision_preview_active = False
        self._update_trace_revision_controls()
        self._redraw()

    def _save_trace(self):
        if self._on_save_trace is None:
            return
        self._on_save_trace()

    def _save_all_traces(self):
        if self._on_save_all_traces is None:
            return
        self._on_save_all_traces()

    def _run_postprocess(self):
        if self._on_run_postprocess is None:
            return
        self._on_run_postprocess()

    def _run_evaluation(self):
        if self._on_run_evaluation is None:
            return
        self._on_run_evaluation()

    def _save_postprocessed(self):
        if self._on_save_postprocessed is None:
            return
        self._on_save_postprocessed()

    def _save_eval_report(self):
        if self._on_save_eval_report is None:
            return
        self._on_save_eval_report()

    def _toggle_post_overlay(self):
        self.post_overlay_visible = not self.post_overlay_visible
        self._redraw()

    def _select_gt_swc_path(self):
        if self._on_select_gt_swc_path is None:
            return
        selected = self._on_select_gt_swc_path()
        if selected:
            self._gt_swc_path = selected
            self._refresh_output_path_labels()

    def _select_scales_path(self):
        if self._on_select_scales_path is None:
            return
        selected = self._on_select_scales_path()
        if selected:
            self._scales_path = selected
            self._refresh_output_path_labels()

    def _format_output_path(self, path_value: Optional[str]) -> str:
        if path_value is None or len(path_value) == 0:
            return "(not set)"
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

    def _select_seeds_output_path(self):
        if self._on_select_seeds_output_path is None:
            return
        selected = self._on_select_seeds_output_path()
        if selected:
            self._seeds_output_path = selected
            self._refresh_output_path_labels()

    def _select_trace_output_path(self):
        if self._on_select_trace_output_path is None:
            return
        selected = self._on_select_trace_output_path()
        if selected:
            self._trace_output_path = selected
            self._refresh_output_path_labels()

    def _select_model_weights_path(self):
        if self._on_select_model_weights_path is None:
            return
        selected = self._on_select_model_weights_path()
        if selected:
            self._model_weights_path = selected
            self._refresh_output_path_labels()

    def _select_image_dir(self):
        if self._on_select_image_dir is None:
            return
        selected = self._on_select_image_dir()
        if selected:
            self._image_dir = selected
            self._refresh_output_path_labels()

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
            "step_width": float(self._trace_step_width_spin.value()),
            "n_trials": int(self._trace_n_trials_spin.value()),
            "max_len": int(self._trace_max_len_spin.value()),
            "max_paths": int(self._trace_max_paths_spin.value()),
            "branching": bool(self._trace_branching_check.isChecked()),
            "repeat_starts": bool(self._trace_repeat_starts_check.isChecked()),
            "stochastic_actions": bool(self._trace_stochastic_check.isChecked()),
            "auto_seed_selection_mode": self._trace_auto_seed_combo.currentText(),
        }

    def get_postprocess_params_overrides(self) -> Dict[str, object]:
        """Return the current post-processing parameter values from the config panel."""
        return {
            "min_branch_length": float(self._pp_min_branch_length_spin.value()),
            "resampling_step_size": float(self._pp_resampling_step_size_spin.value()),
            "smoothing_window": int(self._pp_smoothing_window_spin.value()),
            "overlap_threshold": float(self._pp_overlap_threshold_spin.value()),
            "overlap_distance_threshold": float(self._pp_overlap_dist_threshold_spin.value()),
        }

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
        if selected:
            self._postprocess_output_dir = selected
            self._refresh_output_path_labels()

    def _select_eval_output_dir(self):
        if self._on_select_eval_output_dir is None:
            return
        selected = self._on_select_eval_output_dir()
        if selected:
            self._eval_output_dir = selected
            self._refresh_output_path_labels()

    def _poll_trace_status(self):
        if self._get_trace_status is None:
            return
        status = self._get_trace_status() or {}
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
        if self._trace_controls_running_state is None or running != self._trace_controls_running_state:
            self._set_trace_controls_busy(running)
            self._trace_controls_running_state = running

        overlay_token = status.get("overlay_token")
        if overlay_token is not None and overlay_token != self._trace_overlay_token:
            self._trace_overlay_token = overlay_token
            overlay_paths = status.get("overlay_paths", None)
            if overlay_paths is not None:
                self.finished_paths = []
                for path in overlay_paths:
                    path_np = np.asarray(path, dtype=np.float32)
                    if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] >= 2:
                        self.finished_paths.append(path_np[:, :3])
                # A new overlay invalidates any pending revision point/preview from the old trace.
                self.trace_revision_selected_node_xyz = None
                self.trace_revision_selected_point_xyz = None
                self.trace_revision_preview_paths = []
                self.trace_revision_preview_active = False
                self._update_trace_revision_controls()
                self._redraw(fast=True)

        token = status.get("token")
        if token is not None and token != self._trace_status_token:
            self._trace_status_token = token
            trace_output_dir = status.get("trace_output_dir", None)
            if isinstance(trace_output_dir, str) and len(trace_output_dir) > 0:
                self._trace_output_path = trace_output_dir
                self._refresh_output_path_labels()
            model_weights_path = status.get("model_weights_path", None)
            if isinstance(model_weights_path, str) and len(model_weights_path) > 0:
                self._model_weights_path = model_weights_path
                self._refresh_output_path_labels()

            postprocess_paths = status.get("postprocess_paths", None)
            if postprocess_paths is not None:
                self.post_paths = []
                for path in postprocess_paths:
                    path_np = np.asarray(path, dtype=np.float32)
                    if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] >= 2:
                        self.post_paths.append(path_np[:, :3])
                self._redraw(fast=True)

            eval_report_text = status.get("eval_report_text", None)
            if eval_report_text is not None and self.eval_report_widget is not None:
                self.eval_report_widget.setPlainText(str(eval_report_text))

            gt_swc_path = status.get("gt_swc_path", None)
            if isinstance(gt_swc_path, str) and len(gt_swc_path) > 0:
                self._gt_swc_path = gt_swc_path
                self._refresh_output_path_labels()

    def _set_trace_controls_busy(self, running: bool):
        if not self._show_trace_controls:
            return
        self.btn_trace_neuron.setEnabled(not running)
        self.btn_trace_all.setEnabled(not running)
        self.btn_save_trace.setEnabled(not running)
        self.btn_save_all_traces.setEnabled(not running)
        self.btn_prev_image.setEnabled((not running) and self.btn_prev_image.isVisible())
        self.btn_next_image.setEnabled((not running) and self.btn_next_image.isVisible())
        self.btn_cancel_trace.setEnabled(running)
        self._update_trace_revision_controls()
        if running:
            self.btn_trace_revision_mode.setEnabled(False)
            self.btn_trace_revision_preview.setEnabled(False)
            self.btn_trace_revision_launch.setEnabled(False)
        if self._show_postprocess_controls:
            self.btn_run_postprocess.setEnabled(not running)
            self.btn_run_evaluation.setEnabled(not running)

    def _invalidate_mip_cache(self):
        self._mip_cache_by_view = {"xy": None, "xz": None, "yz": None}

    def _get_plane(self, view: str) -> np.ndarray:
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
        if view == "xy":
            return (-0.5, self.shape[2] - 0.5), (-0.5, self.shape[1] - 0.5)
        if view == "xz":
            return (-0.5, self.shape[2] - 0.5), (-0.5, self.shape[0] - 0.5)
        return (-0.5, self.shape[1] - 0.5), (-0.5, self.shape[0] - 0.5)

    def _set_crosshair_for_view(self, view: str):
        if view not in self.crosshair_artists:
            return
        vline, hline = self.crosshair_artists[view]
        visible = self.mode == "seed"
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

    def _build_layout(self, views):
        self.figure.clear()
        ncols = len(views)
        self.axis_view_map = {}
        self.axes_by_view = {}
        self.image_artists = {}
        self.crosshair_artists = {}
        self.overlay_artists = {}

        for i, view in enumerate(views):
            ax = self.figure.add_subplot(1, ncols, i + 1)
            plane = self._get_plane(view)
            image_artist = ax.imshow(plane, cmap="gray", origin="lower")
            ax.set_title(view.upper())
            self.image_artists[view] = image_artist

            vline = ax.axvline(self.current_x, color="cyan", linewidth=0.8, alpha=0.8)
            hline = ax.axhline(self.current_y, color="cyan", linewidth=0.8, alpha=0.8)
            self.crosshair_artists[view] = (vline, hline)
            self._set_crosshair_for_view(view)

            self.overlay_artists[view] = []
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

    def _redraw(self, fast: bool = False):
        views = self._iter_views()
        needs_layout = self._layout_dirty or (self._current_views != views) or (len(self.axes_by_view) == 0)
        if needs_layout:
            self._build_layout(views)

        for view in views:
            ax = self.axes_by_view[view]
            self.image_artists[view].set_data(self._get_plane(view))
            self._set_crosshair_for_view(view)

            self._clear_overlay_artists(view)
            self.overlay_artists[view] = self._draw_overlay(ax, view)

            # Re-enforce limits AFTER drawing overlays so that ax.plot() calls
            # inside _plot_path_in_view cannot trigger matplotlib autoscale and
            # zoom out to world-coordinate extents (which would push seeds off-screen).
            if view in self.zoom_limits:
                xlim, ylim = self.zoom_limits[view]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            else:
                full_xlim, full_ylim = self._get_full_limits(view)
                ax.set_xlim(full_xlim)
                ax.set_ylim(full_ylim)

        self.canvas.draw_idle()
        if self.mode == "seed":
            self.info_label.setText(
                f"Seeds: {len(self.seeds)} | Cursor (z,y,x)=({self.current_z}, {self.current_y}, {self.current_x})"
            )

    def _draw_overlay(self, ax, view: str):
        artists = []
        if self.seeds:
            seeds_np = np.asarray(self.seeds, dtype=np.float32)
            if view == "xy":
                artists.append(ax.scatter(seeds_np[:, 2], seeds_np[:, 1], s=35, c="lime", edgecolors="black"))
            elif view == "xz":
                artists.append(ax.scatter(seeds_np[:, 2], seeds_np[:, 0], s=35, c="lime", edgecolors="black"))
            else:
                artists.append(ax.scatter(seeds_np[:, 1], seeds_np[:, 0], s=35, c="lime", edgecolors="black"))

        if self.trace_overlay_visible:
            trace_paths = self.finished_paths
            trace_color = "lime"
            if self.trace_revision_preview_active and self.trace_revision_preview_paths:
                trace_paths = self.trace_revision_preview_paths
                trace_color = "gold"
            for path in trace_paths:
                artists.extend(self._plot_path_in_view(ax=ax, path=path, view=view, color=trace_color))

        def _revision_marker_visible(point_xyz: np.ndarray) -> bool:
            if self.projection_mode == "mip":
                return True
            if view == "xy":
                return bool(np.isclose(point_xyz[2], self.current_z, atol=0.5))
            if view == "xz":
                return bool(np.isclose(point_xyz[1], self.current_y, atol=0.5))
            return bool(np.isclose(point_xyz[0], self.current_x, atol=0.5))

        def _project_revision_marker(point_xyz: np.ndarray) -> Tuple[float, float]:
            if view == "xy":
                return float(point_xyz[0]), float(point_xyz[1])
            if view == "xz":
                return float(point_xyz[0]), float(point_xyz[2])
            return float(point_xyz[1]), float(point_xyz[2])

        if self.trace_revision_selected_point_xyz is not None:
            point = self.trace_revision_selected_point_xyz
            if _revision_marker_visible(point):
                px, py = _project_revision_marker(point)
                artists.append(ax.scatter([px], [py], s=50, c="green", edgecolors="black"))

        if self.trace_revision_selected_node_xyz is not None:
            node = self.trace_revision_selected_node_xyz
            if _revision_marker_visible(node):
                px, py = _project_revision_marker(node)
                artists.append(ax.scatter([px], [py], s=55, c="red", edgecolors="black"))

        if self.post_overlay_visible and self.post_paths:
            for path in self.post_paths:
                artists.extend(self._plot_path_in_view(ax=ax, path=path, view=view, color="cyan"))

        return artists

    def _plot_path_in_view(self, ax, path: np.ndarray, view: str, color: str = "lime"):
        artists = []
        if path.ndim != 2 or path.shape[1] < 3 or path.shape[0] == 0:
            return artists

        if self.projection_mode == "mip":
            if view == "xy":
                # paths in XYZ: horizontal=X(0), vertical=Y(1)
                artists.extend(ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5))
            elif view == "xz":
                # paths in XYZ: horizontal=X(0), vertical=Z(2)
                artists.extend(ax.plot(path[:, 0], path[:, 2], color=color, linewidth=1.5))
            else:
                # paths in XYZ: horizontal=Y(1), vertical=Z(2)
                artists.extend(ax.plot(path[:, 1], path[:, 2], color=color, linewidth=1.5))
            return artists

        if view == "xy":
            # paths in XYZ: slice by Z(2)
            in_slice = np.isclose(path[:, 2], self.current_z, atol=0.5)
            x_coords = path[:, 0]  # X
            y_coords = path[:, 1]  # Y
        elif view == "xz":
            # paths in XYZ: slice by Y(1)
            in_slice = np.isclose(path[:, 1], self.current_y, atol=0.5)
            x_coords = path[:, 0]  # X
            y_coords = path[:, 2]  # Z
        else:
            # paths in XYZ: slice by X(0)
            in_slice = np.isclose(path[:, 0], self.current_x, atol=0.5)
            x_coords = path[:, 1]  # Y
            y_coords = path[:, 2]  # Z

        indices = np.flatnonzero(in_slice)
        if indices.size == 0:
            return artists

        split_points = np.where(np.diff(indices) > 1)[0] + 1
        segments = np.split(indices, split_points)
        for seg in segments:
            if seg.size >= 2:
                artists.extend(ax.plot(x_coords[seg], y_coords[seg], color=color, linewidth=1.5))
            elif seg.size == 1:
                artists.append(ax.scatter(x_coords[seg], y_coords[seg], s=10, c=color))

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

    def _set_cursor_from_view_coords(self, view: str, xdata: float, ydata: float):
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

        self._sync_sliders_from_cursor()
        self._redraw()

    def _on_mouse_press(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        view = self.axis_view_map.get(event.inaxes)
        if view is None:
            return
        self._set_active_view(view)

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
            self.canvas.draw_idle()
            return

        current_view = self.axis_view_map.get(event.inaxes)
        if current_view != view:
            self.canvas.draw_idle()
            return

        self._set_active_view(view)

        end_x = float(event.xdata)
        end_y = float(event.ydata)
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        drag_threshold = 1.0

        if dx <= drag_threshold and dy <= drag_threshold:
            if self.mode == "seed":
                self._set_cursor_from_view_coords(view, end_x, end_y)
                self._select_trace_revision_point()
            else:
                self.canvas.draw_idle()
            return

        x0, x1 = sorted([start_x, end_x])
        y0, y1 = sorted([start_y, end_y])

        self._push_current_zoom(view, event.inaxes)
        event.inaxes.set_xlim(x0, x1)
        event.inaxes.set_ylim(y0, y1)
        self.zoom_limits[view] = ((x0, x1), (y0, y1))
        self.canvas.draw_idle()

    def _add_current_seed(self):
        self.seeds.append((self.current_z, self.current_y, self.current_x))
        self._redraw()

    def _on_mpl_keypress(self, event):
        if event.key == "shift":
            self._shift_held = True
            return
        if self.mode != "seed":
            return
        if event.key in (" ", "space"):
            self._add_current_seed()
        elif event.key in ("backspace", "delete"):
            self._undo_seed()

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
        self._redraw(fast=True)

    def _canvas_wheel_event(self, qt_event):
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
    on_trace_revision_select_point: Optional[Callable[[np.ndarray], Optional[Dict[str, object]]]] = None,
    on_trace_revision_preview: Optional[Callable[[], Optional[List[np.ndarray]]]] = None,
    on_trace_revision_launch: Optional[Callable[[], Optional[List[np.ndarray]]]] = None,
    get_trace_status: Optional[Callable[[], Dict[str, object]]] = None,
    finished_paths: Optional[List[np.ndarray]] = None,
    on_save_trace: Optional[Callable[[], None]] = None,
    on_save_all_traces: Optional[Callable[[], None]] = None,
    seeds_output_path: Optional[str] = None,
    trace_output_path: Optional[str] = None,
    on_select_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
    model_weights_path: Optional[str] = None,
    on_select_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
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
        on_trace_revision_select_point=on_trace_revision_select_point,
        on_trace_revision_preview=on_trace_revision_preview,
        on_trace_revision_launch=on_trace_revision_launch,
        get_trace_status=get_trace_status,
        on_save_trace=on_save_trace,
        on_save_all_traces=on_save_all_traces,
        seeds_output_path=seeds_output_path,
        trace_output_path=trace_output_path,
        on_select_seeds_output_path=on_select_seeds_output_path,
        on_select_trace_output_path=on_select_trace_output_path,
        model_weights_path=model_weights_path,
        on_select_model_weights_path=on_select_model_weights_path,
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
    on_save_current: Optional[Callable[[np.ndarray], None]] = None,
    on_save_all: Optional[Callable[[], None]] = None,
    show_trace_controls: bool = False,
    on_trace_current: Optional[Callable[[np.ndarray], Optional[List[np.ndarray]]]] = None,
    on_trace_all: Optional[Callable[[], None]] = None,
    on_cancel_trace: Optional[Callable[[], None]] = None,
    on_trace_revision_select_point: Optional[Callable[[np.ndarray], Optional[Dict[str, object]]]] = None,
    on_trace_revision_preview: Optional[Callable[[], Optional[List[np.ndarray]]]] = None,
    on_trace_revision_launch: Optional[Callable[[], Optional[List[np.ndarray]]]] = None,
    get_trace_status: Optional[Callable[[], Dict[str, object]]] = None,
    on_save_trace: Optional[Callable[[], None]] = None,
    on_save_all_traces: Optional[Callable[[], None]] = None,
    on_select_seeds_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_trace_output_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_model_weights_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_image_dir: Optional[Callable[[], Optional[str]]] = None,
    on_select_seeds_input_path: Optional[Callable[[], Optional[str]]] = None,
    trace_step_width: float = 4.0,
    trace_n_trials: int = 1,
    trace_max_len: int = 10000,
    trace_max_paths: int = 1000,
    trace_branching: bool = True,
    trace_repeat_starts: bool = False,
    trace_stochastic_actions: bool = False,
    trace_auto_seed_mode: str = "remote_endnode",
    on_trace_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
    show_postprocess_controls: bool = False,
    on_run_postprocess: Optional[Callable[[], None]] = None,
    on_run_evaluation: Optional[Callable[[], None]] = None,
    on_save_postprocessed: Optional[Callable[[], None]] = None,
    on_save_eval_report: Optional[Callable[[], None]] = None,
    on_select_gt_swc_path: Optional[Callable[[], Optional[str]]] = None,
    on_select_scales_path: Optional[Callable[[], Optional[str]]] = None,
    postprocess_output_dir: Optional[str] = None,
    postprocess_min_branch_length: float = 5.0,
    postprocess_resampling_step_size: float = 4.0,
    postprocess_smoothing_window: int = 5,
    postprocess_overlap_threshold: float = 0.5,
    postprocess_overlap_distance_threshold: float = 1.0,
    on_select_postprocess_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_postprocess_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
    eval_output_dir: Optional[str] = None,
    eval_distance_threshold: float = 1.0,
    on_select_eval_output_dir: Optional[Callable[[], Optional[str]]] = None,
    on_eval_params_changed: Optional[Callable[[Dict[str, object]], None]] = None,
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
        neuron_name=str(initial_context.get("neuron_name", "")),
        initial_seeds=initial_context.get("initial_seeds"),
        show_prev_button=bool(initial_context.get("show_prev_button", False)),
        show_next_button=bool(initial_context.get("show_next_button", False)),
        show_save_buttons=True,
        on_save_current=on_save_current,
        on_save_all=on_save_all,
        show_trace_controls=show_trace_controls,
        on_trace_current=on_trace_current,
        on_trace_all=on_trace_all,
        on_cancel_trace=on_cancel_trace,
        on_trace_revision_select_point=on_trace_revision_select_point,
        on_trace_revision_preview=on_trace_revision_preview,
        on_trace_revision_launch=on_trace_revision_launch,
        get_trace_status=get_trace_status,
        on_save_trace=on_save_trace,
        on_save_all_traces=on_save_all_traces,
        seeds_output_path=initial_context.get("seeds_output_path"),
        trace_output_path=initial_context.get("trace_output_path"),
        on_select_seeds_output_path=on_select_seeds_output_path,
        on_select_trace_output_path=on_select_trace_output_path,
        model_weights_path=initial_context.get("model_weights_path"),
        on_select_model_weights_path=on_select_model_weights_path,
        image_dir=initial_context.get("image_dir"),
        seeds_input_path=initial_context.get("seeds_input_path"),
        on_select_image_dir=on_select_image_dir,
        on_select_seeds_input_path=on_select_seeds_input_path,
        trace_step_width=trace_step_width,
        trace_n_trials=trace_n_trials,
        trace_max_len=trace_max_len,
        trace_max_paths=trace_max_paths,
        trace_branching=trace_branching,
        trace_repeat_starts=trace_repeat_starts,
        trace_stochastic_actions=trace_stochastic_actions,
        trace_auto_seed_mode=trace_auto_seed_mode,
        on_trace_params_changed=on_trace_params_changed,
        on_prev_image=on_prev_image,
        on_next_image=on_next_image,
        show_postprocess_controls=show_postprocess_controls,
        on_run_postprocess=on_run_postprocess,
        on_run_evaluation=on_run_evaluation,
        on_save_postprocessed=on_save_postprocessed,
        on_save_eval_report=on_save_eval_report,
        gt_swc_path=initial_context.get("gt_swc_path"),
        on_select_gt_swc_path=on_select_gt_swc_path,
        scales_path=initial_context.get("scales_path"),
        on_select_scales_path=on_select_scales_path,
        postprocess_output_dir=postprocess_output_dir,
        postprocess_min_branch_length=postprocess_min_branch_length,
        postprocess_resampling_step_size=postprocess_resampling_step_size,
        postprocess_smoothing_window=postprocess_smoothing_window,
        postprocess_overlap_threshold=postprocess_overlap_threshold,
        postprocess_overlap_distance_threshold=postprocess_overlap_distance_threshold,
        on_select_postprocess_output_dir=on_select_postprocess_output_dir,
        on_postprocess_params_changed=on_postprocess_params_changed,
        eval_output_dir=eval_output_dir,
        eval_distance_threshold=eval_distance_threshold,
        on_select_eval_output_dir=on_select_eval_output_dir,
        on_eval_params_changed=on_eval_params_changed,
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
    """Prompt for startup paths for seed-selection sessions using native file dialogs."""
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

    chosen_input = seeds_input_path
    if chosen_input is None:
        selected_input, _ = qt_widgets.QFileDialog.getOpenFileName(
            None,
            "Optional: Select existing seeds JSON",
            chosen_image_dir,
            "JSON Files (*.json)",
        )
        chosen_input = selected_input or None

    chosen_output = seeds_output_path
    if chosen_output is None:
        selected_output, _ = qt_widgets.QFileDialog.getSaveFileName(
            None,
            "Optional: Select seeds output JSON",
            os.path.join(chosen_image_dir, "seeds.json"),
            "JSON Files (*.json)",
        )
        chosen_output = selected_output or None

    return chosen_image_dir, chosen_input, chosen_output


def prompt_save_json_path(default_path: Optional[str] = None) -> Optional[str]:
    """Prompt user for a JSON output path and return selected path or None."""
    if _is_jupyter_notebook():
        raise RuntimeError("Save-path dialog is not supported in Jupyter notebooks.")

    if not _has_gui_display():
        raise RuntimeError("Save-path dialog requires a desktop display (DISPLAY/WAYLAND_DISPLAY not set).")

    _ensure_qapplication()
    qt_widgets = importlib.import_module("qtpy.QtWidgets")
    selected_output, _ = qt_widgets.QFileDialog.getSaveFileName(
        None,
        "Select seeds output JSON",
        default_path or os.path.join(os.getcwd(), "seeds.json"),
        "JSON Files (*.json)",
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
