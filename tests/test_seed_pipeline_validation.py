import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np
import tifffile as tf
import torch

from neurotrack.data import NeuronPatchDataset, adjacency_dict, save_seeds_json
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.inference.runtime import build_env
from neurotrack.pipelines import interactive_tracing_pipeline as interactive_pipeline
from neurotrack.visualization.editor_state import AnnotationGraph
from neurotrack.visualization import ortho_viewer
import neurotrack.data.datasets as datasets_module


def _write_volume(path: Path, shape=(40, 40, 40)) -> None:
    volume = np.zeros(shape, dtype=np.uint8)
    center = tuple(dim // 2 for dim in shape)
    volume[center] = 255
    tf.imwrite(path, volume)


def _write_swc(path: Path, rows, shape_zyx=None) -> None:
    bounds_xyz = None
    if shape_zyx is not None:
        if len(shape_zyx) != 3:
            raise ValueError(f"shape_zyx must have length 3, got {shape_zyx}")
        z_max = max(int(shape_zyx[0]) - 1, 0)
        y_max = max(int(shape_zyx[1]) - 1, 0)
        x_max = max(int(shape_zyx[2]) - 1, 0)
        bounds_xyz = (x_max, y_max, z_max)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            node_id, node_type, x, y, z, radius, parent_id = row
            if bounds_xyz is not None:
                x = float(np.clip(x, 0.0, bounds_xyz[0]))
                y = float(np.clip(y, 0.0, bounds_xyz[1]))
                z = float(np.clip(z, 0.0, bounds_xyz[2]))
            handle.write(
                "{} {} {:.1f} {:.1f} {:.1f} {:.1f} {}\n".format(
                    node_id,
                    node_type,
                    x,
                    y,
                    z,
                    radius,
                    parent_id,
                )
            )


def _make_chain_rows(num_nodes: int, start_xyz=(8.0, 20.0, 20.0), step_xyz=(1.0, 0.0, 0.0)):
    rows = []
    x0, y0, z0 = start_xyz
    dx, dy, dz = step_xyz
    for idx in range(num_nodes):
        node_id = idx + 1
        parent_id = -1 if idx == 0 else idx
        rows.append(
            (
                node_id,
                3,
                x0 + dx * idx,
                y0 + dy * idx,
                z0 + dz * idx,
                1.0,
                parent_id,
            )
        )
    return rows


def _make_viewer_stub(
    *,
    seeds=None,
    reference_rows=None,
    prediction_paths=None,
    active_annotation="reference",
    projection_mode="slice",
):
    viewer = ortho_viewer._OrthoViewDialog.__new__(ortho_viewer._OrthoViewDialog)
    viewer.Qt = ortho_viewer.importlib.import_module("qtpy.QtCore").Qt
    viewer._editor_state = ortho_viewer.ViewerSessionState()
    viewer._editor_state.active_annotation = active_annotation
    if reference_rows is None:
        reference_rows = np.asarray(_make_chain_rows(1, start_xyz=(1.0, 1.0, 1.0)), dtype=np.float32)
    viewer._editor_state.set_reference_swc_rows(reference_rows)
    viewer._editor_state.set_prediction_paths(prediction_paths)
    viewer.mode = "seed"
    viewer.projection_mode = projection_mode
    viewer.shape = (16, 16, 16)
    viewer.current_z = 0
    viewer.current_y = 0
    viewer.current_x = 0
    viewer.seeds = list(seeds or [])
    viewer.selected_seed_index = None
    viewer.effective_seed_overlay = []
    viewer.finished_paths = []
    viewer.trace_overlay_visible = False
    viewer.gt_overlay_visible = True
    viewer._shift_held = False
    viewer._active_tool = "zoom"
    viewer._drag_start = None
    viewer._drag_view = None
    viewer._drag_rect = None
    viewer._has_image_dir = lambda: True
    viewer._refresh_seed_order_controls = lambda: None
    viewer._refresh_edit_action_controls = lambda: None
    viewer._refresh_effective_seed_overlay = lambda: None
    viewer._sync_sliders_from_cursor = lambda: None
    viewer._redraw = lambda *args, **kwargs: None
    viewer.canvas = SimpleNamespace(draw_idle=lambda: None)
    viewer.dialog = SimpleNamespace(accept=lambda: None)
    viewer._sync_annotation_graph_to_view = lambda target: None
    viewer._clear_current_selection = ortho_viewer._OrthoViewDialog._clear_current_selection.__get__(viewer)
    viewer._add_current_seed = ortho_viewer._OrthoViewDialog._add_current_seed.__get__(viewer)
    viewer._add_branch_seed_from_selected_node = ortho_viewer._OrthoViewDialog._add_branch_seed_from_selected_node.__get__(viewer)
    viewer._remove_selected = ortho_viewer._OrthoViewDialog._remove_selected.__get__(viewer)
    viewer._set_cursor_from_view_coords = ortho_viewer._OrthoViewDialog._set_cursor_from_view_coords.__get__(viewer)
    viewer._select_seed_at_view_coords = ortho_viewer._OrthoViewDialog._select_seed_at_view_coords.__get__(viewer)
    viewer._select_annotation_node_at_view_coords = ortho_viewer._OrthoViewDialog._select_annotation_node_at_view_coords.__get__(viewer)
    viewer._select_annotation_nodes_in_view_rect = ortho_viewer._OrthoViewDialog._select_annotation_nodes_in_view_rect.__get__(viewer)
    viewer._handle_click_without_drag = ortho_viewer._OrthoViewDialog._handle_click_without_drag.__get__(viewer)
    viewer._handle_drag_release = ortho_viewer._OrthoViewDialog._handle_drag_release.__get__(viewer)
    viewer._make_keypress_handler = ortho_viewer._OrthoViewDialog._make_keypress_handler.__get__(viewer)
    viewer._active_annotation_graph = ortho_viewer._OrthoViewDialog._active_annotation_graph.__get__(viewer)
    return viewer


class SeedPipelineValidationTests(unittest.TestCase):
    def test_trace_busy_state_keeps_neuron_navigation_enabled(self):
        viewer = ortho_viewer._OrthoViewDialog.__new__(ortho_viewer._OrthoViewDialog)
        viewer._show_trace_controls = True
        viewer._show_postprocess_controls = False
        viewer._show_prev_button = True
        viewer._show_next_button = True
        viewer._refresh_edit_action_controls = mock.Mock()

        control_names = [
            "btn_trace_neuron",
            "btn_trace_all",
            "btn_save_trace",
            "btn_save_all_traces",
            "btn_discard_trace",
            "chk_trace_overlay",
            "chk_gt_overlay",
            "btn_apply_component_filter",
            "btn_save_filtered_swc",
            "btn_prev_image",
            "btn_next_image",
            "btn_cancel_trace",
            "btn_remove_selected",
            "btn_clip_selected",
        ]
        for name in control_names:
            setattr(viewer, name, SimpleNamespace(setEnabled=mock.Mock()))

        ortho_viewer._OrthoViewDialog._set_trace_controls_busy(viewer, True)

        viewer.btn_prev_image.setEnabled.assert_called_once_with(True)
        viewer.btn_next_image.setEnabled.assert_called_once_with(True)
        viewer.btn_trace_neuron.setEnabled.assert_called_once_with(False)
        viewer.btn_cancel_trace.setEnabled.assert_called_once_with(True)

    def test_start_trace_current_runs_in_background_and_seed_overlay_stays_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(12, 12, 12))
            image_key = image_path.relative_to(root).as_posix()
            trace_started = threading.Event()
            release_trace = threading.Event()

            class _RuntimeStub:
                effective_seed_calls = 0

                def trace_image(self, **kwargs):
                    del kwargs
                    trace_started.set()
                    release_trace.wait(timeout=5.0)
                    return {
                        "paths": [[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]],
                        "timing_ms": {"total": 1.0, "steps": 1},
                    }

                def get_effective_seed_points(self, **kwargs):
                    del kwargs
                    self.effective_seed_calls += 1
                    return np.asarray([[9.0, 9.0, 9.0]], dtype=np.float32)

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
            )
            manager.enabled = True
            runtime = _RuntimeStub()
            manager._runtime = runtime

            try:
                manager.start_trace_current(0, image_key, [[4.0, 5.0, 6.0]])
                self.assertTrue(trace_started.wait(timeout=1.0))
                self.assertIsNotNone(manager._thread)
                self.assertTrue(manager._thread.is_alive())

                overlay = manager.get_effective_seed_overlay(0, image_key, [[4.0, 5.0, 6.0]])
                np.testing.assert_array_equal(overlay, np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32))
                self.assertEqual(runtime.effective_seed_calls, 0)
            finally:
                release_trace.set()
                if manager._thread is not None:
                    manager._thread.join(timeout=5.0)
                manager.close()

            self.assertFalse(manager._running)
            self.assertEqual(len(manager.trace_results_by_key[image_key]), 1)

    def test_select_gt_swc_path_updates_current_reference_and_redraws(self):
        viewer = _make_viewer_stub()
        new_rows = np.asarray(_make_chain_rows(3), dtype=np.float32)
        viewer._on_select_gt_swc_path = lambda: ("/tmp/reference", new_rows)
        viewer._refresh_output_path_labels = mock.Mock()
        viewer._invalidate_tree_overlay_cache = mock.Mock()
        viewer._refresh_annotation_target_options = mock.Mock()
        viewer._redraw = mock.Mock()

        ortho_viewer._OrthoViewDialog._select_gt_swc_path(viewer)

        self.assertEqual(viewer._gt_swc_path, "/tmp/reference")
        np.testing.assert_array_equal(viewer._tree_swc_committed, new_rows)
        self.assertEqual(len(viewer._editor_state.reference_annotation.nodes_by_id), 3)
        viewer._invalidate_tree_overlay_cache.assert_called_once_with()
        viewer._refresh_annotation_target_options.assert_called_once_with()
        viewer._redraw.assert_called_once_with()

    def test_annotation_graph_roundtrip_and_mutation(self):
        rows = np.asarray(
            [
                [1, 3, 1.0, 1.0, 1.0, 1.0, -1],
                [2, 3, 2.0, 1.0, 1.0, 1.0, 1],
                [3, 3, 2.0, 2.0, 1.0, 1.0, 1],
            ],
            dtype=np.float32,
        )

        graph = AnnotationGraph.from_swc_rows(rows)
        self.assertEqual(graph.root_ids, [1])
        self.assertEqual(graph.nodes_by_id[1].child_ids, [2, 3])

        paths = graph.to_paths()
        self.assertEqual(len(paths), 2)

        roundtrip = AnnotationGraph.from_swc_rows(graph.to_swc_rows())
        self.assertEqual(roundtrip.root_ids, [1])
        self.assertEqual(roundtrip.nodes_by_id[1].child_ids, [2, 3])

        added_id = roundtrip.add_child(1, (3.0, 1.0, 1.0))
        self.assertIn(added_id, roundtrip.nodes_by_id)
        self.assertIn(added_id, roundtrip.nodes_by_id[1].child_ids)

        roundtrip.remove_nodes({2})
        self.assertNotIn(2, roundtrip.nodes_by_id)
        self.assertEqual(roundtrip.nodes_by_id[1].child_ids, [3, added_id])

    def test_interaction_controller_resolves_seed_first_and_preserves_selection_on_empty_click(self):
        viewer = _make_viewer_stub(
            seeds=[(0, 0, 0)],
            reference_rows=np.asarray([[1, 3, 0.0, 0.0, 0.0, 1.0, -1]], dtype=np.float32),
            active_annotation="reference",
        )
        viewer._editor_state.selection.selected_annotation_node_ids.add(1)

        viewer._handle_click_without_drag("xy", 0.0, 0.0)
        self.assertEqual(viewer.selected_seed_index, 0)
        self.assertEqual(viewer._editor_state.selection.selected_annotation_node_ids, set())

        viewer.selected_seed_index = 0
        viewer._editor_state.selection.selected_annotation_node_ids = {1}
        viewer._editor_state.selection.clip_preview_node_ids = {1}
        viewer._editor_state.selection.pending_branch_seed_xyz = (1.0, 1.0, 1.0)

        viewer._handle_click_without_drag("xy", 12.0, 12.0)
        self.assertEqual(viewer.selected_seed_index, 0)
        self.assertEqual(viewer._editor_state.selection.selected_annotation_node_ids, {1})
        self.assertEqual(viewer._editor_state.selection.clip_preview_node_ids, {1})
        self.assertEqual(viewer._editor_state.selection.pending_branch_seed_xyz, (1.0, 1.0, 1.0))

    def test_interaction_controller_remove_selected_handles_seed_clip_and_annotation_nodes(self):
        viewer = _make_viewer_stub(
            seeds=[(0, 0, 0), (1, 1, 1)],
            reference_rows=np.asarray(
                [
                    [1, 3, 1.0, 1.0, 1.0, 1.0, -1],
                    [2, 3, 2.0, 1.0, 1.0, 1.0, 1],
                    [3, 3, 3.0, 1.0, 1.0, 1.0, 2],
                ],
                dtype=np.float32,
            ),
        )

        viewer._editor_state.selection.clip_preview_node_ids = {1, 2}
        viewer._remove_selected()
        self.assertEqual(set(viewer._editor_state.reference_annotation.nodes_by_id.keys()), {3})
        self.assertEqual(viewer._editor_state.selection.clip_preview_node_ids, set())

        viewer = _make_viewer_stub(seeds=[(0, 0, 0), (1, 1, 1)])
        viewer.selected_seed_index = 0
        viewer._remove_selected()
        self.assertEqual(viewer.seeds, [(1, 1, 1)])
        self.assertIsNone(viewer.selected_seed_index)

        viewer = _make_viewer_stub(
            reference_rows=np.asarray(
                [
                    [1, 3, 1.0, 1.0, 1.0, 1.0, -1],
                    [2, 3, 2.0, 1.0, 1.0, 1.0, 1],
                ],
                dtype=np.float32,
            ),
        )
        viewer._editor_state.selection.selected_annotation_node_ids = {2}
        viewer._remove_selected()
        self.assertEqual(set(viewer._editor_state.reference_annotation.nodes_by_id.keys()), {1})
        self.assertEqual(viewer._editor_state.selection.selected_annotation_node_ids, set())

    def test_interaction_controller_space_and_branch_actions_create_expected_nodes_and_seeds(self):
        viewer = _make_viewer_stub(
            reference_rows=np.asarray(
                [
                    [1, 3, 4.0, 5.0, 6.0, 1.0, -1],
                ],
                dtype=np.float32,
            ),
        )
        viewer.current_x = 7
        viewer.current_y = 8
        viewer.current_z = 9
        viewer._editor_state.selection.selected_annotation_node_ids = {1}
        viewer._editor_state.selection.selection_view = "xy"
        viewer._editor_state.selection.pending_branch_seed_xyz = (4.0, 5.0, 6.0)

        viewer._add_current_seed()
        self.assertEqual(viewer.selected_seed_index, None)
        self.assertEqual(viewer._editor_state.selection.selected_annotation_node_ids, {2})
        self.assertEqual(viewer._editor_state.selection.selection_view, "xy")
        self.assertIsNone(viewer._editor_state.selection.pending_branch_seed_xyz)
        self.assertEqual(viewer._editor_state.reference_annotation.nodes_by_id[2].parent_id, 1)
        self.assertEqual(viewer._editor_state.reference_annotation.nodes_by_id[2].xyz, (7.0, 8.0, 9.0))

        viewer._editor_state.selection.selected_annotation_node_ids = {1}
        viewer.seeds = []
        self.assertTrue(viewer._add_branch_seed_from_selected_node())
        self.assertEqual(viewer.seeds[-1], (6, 5, 4))
        self.assertEqual(viewer._editor_state.selection.pending_branch_seed_xyz, (4.0, 5.0, 6.0))

    def test_interaction_controller_keypress_routes_space_delete_and_enter(self):
        viewer = _make_viewer_stub()
        calls = []
        viewer._add_current_seed = lambda: calls.append("add")
        viewer._remove_selected = lambda: calls.append("remove")
        viewer.dialog = SimpleNamespace(accept=lambda: calls.append("accept"))

        handler = viewer._make_keypress_handler(lambda event: calls.append(f"fallback:{event.key()}"))

        handler(SimpleNamespace(key=lambda: viewer.Qt.Key_Space))
        handler(SimpleNamespace(key=lambda: viewer.Qt.Key_Delete))
        handler(SimpleNamespace(key=lambda: viewer.Qt.Key_Return))

        self.assertEqual(calls, ["add", "remove", "accept"])

    def test_interaction_controller_escape_clears_selection(self):
        viewer = _make_viewer_stub(seeds=[(0, 0, 0)])
        viewer.selected_seed_index = 0
        viewer._editor_state.selection.selected_annotation_node_ids = {1}
        viewer._editor_state.selection.clip_preview_node_ids = {1}
        viewer._editor_state.selection.pending_branch_seed_xyz = (1.0, 1.0, 1.0)

        handler = viewer._make_keypress_handler(lambda event: None)
        handler(SimpleNamespace(key=lambda: viewer.Qt.Key_Escape))

        self.assertIsNone(viewer.selected_seed_index)
        self.assertEqual(viewer._editor_state.selection.selected_annotation_node_ids, set())
        self.assertEqual(viewer._editor_state.selection.clip_preview_node_ids, set())
        self.assertIsNone(viewer._editor_state.selection.pending_branch_seed_xyz)

    def test_interaction_controller_select_drag_box_selects_visible_annotation_nodes(self):
        viewer = _make_viewer_stub(
            reference_rows=np.asarray(
                [
                    [1, 3, 1.0, 1.0, 0.0, 1.0, -1],
                    [2, 3, 3.0, 3.0, 0.0, 1.0, 1],
                    [3, 3, 8.0, 8.0, 0.0, 1.0, 2],
                ],
                dtype=np.float32,
            ),
        )
        viewer._active_tool = "select"
        viewer.current_z = 0

        event = SimpleNamespace(inaxes=SimpleNamespace())
        viewer._handle_drag_release(
            event=event,
            view="xy",
            start_x=0.5,
            start_y=0.5,
            end_x=4.5,
            end_y=4.5,
        )

        self.assertEqual(viewer._editor_state.selection.selected_annotation_node_ids, {1, 2})
        self.assertIsNone(viewer.selected_seed_index)

    def test_format_eval_report_includes_special_node_metrics(self):
        report = interactive_pipeline._format_eval_report(
            "sample.tif",
            {
                "bidirectional_distance": 1.25,
                "directed_div_pred_to_gt": 0.5,
                "n_substantial_pred_to_gt": 8,
                "directed_div_gt_to_pred": 0.75,
                "n_substantial_gt_to_pred": 7,
                "precision": 0.8,
                "coverage": 0.9,
                "endpoint_localization_error": 0.25,
                "endpoint_count_error": 2,
                "branchpoint_localization_error": float("nan"),
                "branchpoint_count_error": 1,
                "n_points_pred": 42,
                "n_points_gt": 40,
            },
        )

        self.assertIn("Endpoint Loc Error", report)
        self.assertIn("Endpoint Count Error", report)
        self.assertIn("Branchpoint Loc Error", report)
        self.assertIn("Branchpoint Count Error", report)
        self.assertIn("0.2500", report)
        self.assertIn("N/A", report)
        self.assertIn("42", report)
        self.assertIn("40", report)

    def test_format_eval_report_omits_legacy_lines_for_new_default_schema(self):
        report = interactive_pipeline._format_eval_report(
            "sample.tif",
            {
                "bidirectional_distance": 1.25,
                "directed_div_pred_to_gt": 0.5,
                "directed_div_gt_to_pred": 0.75,
                "precision": 0.8,
                "coverage": 0.9,
                "endpoint_localization_error": 0.25,
                "endpoint_count_error": 2,
                "branchpoint_localization_error": 0.5,
                "branchpoint_count_error": 1,
            },
        )

        self.assertNotIn("Substantial pred→gt", report)
        self.assertNotIn("Substantial gt→pred", report)
        self.assertNotIn("Pred Nodes:", report)
        self.assertNotIn("N/A", report)

    def test_format_eval_report_includes_l_measures_when_present(self):
        report = interactive_pipeline._format_eval_report(
            "sample.tif",
            {
                "bidirectional_distance": 1.25,
                "directed_div_pred_to_gt": 0.5,
                "directed_div_gt_to_pred": 0.75,
                "precision": 0.8,
                "coverage": 0.9,
                "endpoint_localization_error": 0.25,
                "endpoint_count_error": 2,
                "branchpoint_localization_error": 0.5,
                "branchpoint_count_error": 1,
                "num_bifurcations_pred": 3,
                "num_bifurcations_gt": 2,
                "span_pred": (12.0, 10.0, 9.0),
                "span_gt": (11.0, 9.0, 8.0),
                "different_structure_average": 1.5,
                "percentage_different_structure_pred_to_gt": 0.2,
                "percentage_different_structure_gt_to_pred": 0.1,
                "percent_different_structure_average": 0.15,
            },
        )

        self.assertIn("L-Measures", report)
        self.assertIn("Num Bifurcations (pred/gt): 3 / 2", report)
        self.assertIn("Span (pred/gt): (12.0, 10.0, 9.0) / (11.0, 9.0, 8.0)", report)
        self.assertIn("Different Structure Avg: 1.5000", report)
        self.assertIn("% Different Structure pred→gt: 0.2000", report)
        self.assertIn("% Different Structure gt→pred: 0.1000", report)
        self.assertIn("% Different Structure Avg: 0.1500", report)

    def test_adjacency_dict_handles_empty_and_single_row_inputs(self):
        self.assertEqual(adjacency_dict([]), {})

        single_row = np.array([1, 3, 10.0, 11.0, 12.0, 1.0, -1.0], dtype=np.float32)
        self.assertEqual(adjacency_dict(single_row), {1: []})

    def test_crop_mode_is_deterministic_and_seed_matches_selected_node(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            swc_dir = root / "swc"
            img_dir.mkdir()
            swc_dir.mkdir()

            volume_shape = (40, 40, 40)
            _write_volume(img_dir / "sample.tif", shape=volume_shape)
            _write_swc(swc_dir / "sample.swc", _make_chain_rows(8), shape_zyx=volume_shape)

            dataset = NeuronPatchDataset(
                swc_dir=swc_dir,
                img_dir=img_dir,
                crop_size=16,
                patches_per_image=12,
                alpha=1.0,
                step_width=2.0,
                rng=np.random.default_rng(123),
                crop_patches=True,
                inference_mode=False,
            )

            sample_a = dataset[3]
            sample_b = dataset[3]

            self.assertTrue(torch.equal(sample_a["image"], sample_b["image"]))
            self.assertTrue(torch.equal(sample_a["seed_points"], sample_b["seed_points"]))
            self.assertEqual(sample_a["seed_node_id"], sample_b["seed_node_id"])
            self.assertEqual(sample_a["neuron_tree"], sample_b["neuron_tree"])
            self.assertEqual(sample_a["neuron_mask"].dtype, torch.uint8)
            self.assertGreater(int(sample_a["neuron_mask"].max().item()), 0)
            self.assertEqual(int(sample_a["neuron_mask"].max().item()), 255)

            seed_node = next(node for node in sample_a["neuron_tree"] if int(node[0]) == sample_a["seed_node_id"])
            expected_seed_zyx = torch.tensor([seed_node[4], seed_node[3], seed_node[2]], dtype=torch.float32)
            torch.testing.assert_close(sample_a["seed_points"][0], expected_seed_zyx)

            env = NeuronTrackingEnvironment(
                dataset=dataset,
                radius=5,
                target_step_len=1.0,
                step_width=2.0,
                max_len=20,
                branching=False,
                inference_mode=False,
            )
            env.reset(dataset_index=3)
            state = env.get_state()
            self.assertEqual(state.shape[1], 2)
            self.assertEqual(state.dtype, torch.uint8)

    def test_root_sampling_probability_biases_root_selection_frequency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            swc_dir = root / "swc"
            img_dir.mkdir()
            swc_dir.mkdir()

            volume_shape = (40, 40, 40)
            _write_volume(img_dir / "sample.tif", shape=volume_shape)
            rows = _make_chain_rows(40, start_xyz=(0.0, 20.0, 20.0))
            _write_swc(swc_dir / "sample.swc", rows, shape_zyx=volume_shape)
            parent_by_id = {int(node_id): int(parent_id) for node_id, _t, _x, _y, _z, _r, parent_id in rows}

            dataset = NeuronPatchDataset(
                swc_dir=swc_dir,
                img_dir=img_dir,
                crop_size=16,
                patches_per_image=400,
                alpha=1.0,
                step_width=2.0,
                rng=np.random.default_rng(7),
                crop_patches=True,
                inference_mode=True,
                root_sampling_probability=0.7,
            )

            root_count = 0
            total = len(dataset)
            for idx in range(total):
                sample = dataset[idx]
                seed_node_id = int(sample["seed_node_id"])
                if parent_by_id.get(seed_node_id) == -1:
                    root_count += 1

            observed = root_count / total
            self.assertLess(abs(observed - 0.7), 0.08)

    def test_root_seed_draws_marker_in_predicted_path_channel(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            swc_dir = root / "swc"
            img_dir.mkdir()
            swc_dir.mkdir()

            volume_shape = (24, 24, 24)
            _write_volume(img_dir / "sample.tif", shape=volume_shape)
            _write_swc(
                swc_dir / "sample.swc",
                [(1, 3, 12.0, 12.0, 12.0, 1.0, -1)],
                shape_zyx=volume_shape,
            )

            dataset = NeuronPatchDataset(
                swc_dir=swc_dir,
                img_dir=img_dir,
                crop_size=12,
                patches_per_image=1,
                alpha=1.0,
                step_width=2.0,
                rng=np.random.default_rng(0),
                crop_patches=True,
                inference_mode=True,
                root_sampling_probability=1.0,
            )

            sample = dataset[0]
            path_channel = sample["image"][-1]
            seed = sample["seed_points"][0].round().to(dtype=torch.long)

            self.assertGreater(int(path_channel.sum().item()), 0)
            self.assertGreater(int(path_channel[seed[0], seed[1], seed[2]].item()), 0)

    def test_predicted_path_channel_pruning_keeps_seed_cut_end(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            swc_dir = root / "swc"
            img_dir.mkdir()
            swc_dir.mkdir()

            volume_shape = (24, 24, 24)
            _write_volume(img_dir / "sample.tif", shape=volume_shape)
            _write_swc(
                swc_dir / "sample.swc",
                [
                    (1, 3, 8.0, 8.0, 8.0, 1.0, -1),
                    (2, 3, 9.0, 8.0, 8.0, 1.0, 1),
                    (3, 3, 10.0, 8.0, 8.0, 1.0, 2),
                ],
                shape_zyx=volume_shape,
            )

            dataset = NeuronPatchDataset(
                swc_dir=swc_dir,
                img_dir=img_dir,
                crop_size=16,
                patches_per_image=1,
                alpha=1.0,
                step_width=2.0,
                rng=np.random.default_rng(0),
                crop_patches=True,
                inference_mode=False,
            )

            subtree = [
                [1, 3, 8.0, 8.0, 8.0, 1.0, -1],
                [2, 3, 9.0, 8.0, 8.0, 1.0, 1],
                [3, 3, 10.0, 8.0, 8.0, 1.0, 2],
            ]
            path_channel, updated_subtree, cut_end_ids = dataset._build_predicted_path_channel(
                subtree=subtree,
                seed_node_id=3,
                seed_point_xyz=torch.tensor([10.0, 8.0, 8.0], dtype=torch.float32),
                spatial_shape_zyx=torch.Size([16, 16, 16]),
            )

            self.assertEqual(path_channel.dtype, torch.uint8)
            self.assertEqual(tuple(path_channel.shape), (16, 16, 16))
            self.assertEqual(len(updated_subtree), 0)
            self.assertEqual(cut_end_ids, [])

    def test_getitem_retries_when_pruning_empties_patch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            swc_dir = root / "swc"
            img_dir.mkdir()
            swc_dir.mkdir()

            volume_shape = (24, 24, 24)
            _write_volume(img_dir / "sample.tif", shape=volume_shape)
            _write_swc(
                swc_dir / "sample.swc",
                [
                    (1, 3, 8.0, 8.0, 8.0, 1.0, -1),
                    (2, 3, 9.0, 8.0, 8.0, 1.0, 1),
                    (3, 3, 10.0, 8.0, 8.0, 1.0, 2),
                ],
                shape_zyx=volume_shape,
            )

            dataset = NeuronPatchDataset(
                swc_dir=swc_dir,
                img_dir=img_dir,
                crop_size=16,
                patches_per_image=2,
                alpha=1.0,
                step_width=2.0,
                rng=np.random.default_rng(0),
                crop_patches=True,
                inference_mode=False,
            )

            valid_patch = {
                "image": torch.zeros((2, 16, 16, 16), dtype=torch.uint8),
                "neuron_tree": [[1, 3, 8.0, 8.0, 8.0, 1.0, -1]],
                "neuron_mask": torch.zeros((1, 16, 16, 16), dtype=torch.uint8),
                "seed_point_xyz": torch.tensor([8.0, 8.0, 8.0], dtype=torch.float32),
                "seed_node_id": 1,
            }

            with mock.patch.object(
                dataset,
                "_extract_random_patch",
                side_effect=[datasets_module._ResamplePatch("empty after prune"), valid_patch],
            ) as mocked_extract:
                sample = dataset[0]

            self.assertEqual(mocked_extract.call_count, 2)
            self.assertEqual(sample["seed_node_id"], 1)
            self.assertEqual(tuple(sample["seed_points"].shape), (1, 3))

    def test_environment_uses_dataset_seed_points_and_terminates_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            _write_volume(img_dir / "sample.tif")
            seed_rows = [[10.0, 11.0, 12.0]]
            dataset = NeuronPatchDataset(
                swc_dir=None,
                img_dir=img_dir,
                step_width=2.0,
                crop_patches=False,
                inference_mode=True,
                seed_points_by_image={"sample.tif": seed_rows},
            )

            env = NeuronTrackingEnvironment(
                dataset=dataset,
                radius=5,
                target_step_len=1.0,
                step_width=2.0,
                max_len=10,
                max_paths=1,
                branching=False,
                inference_mode=True,
            )
            env.reset(dataset_index=0)
            torch.testing.assert_close(env.paths[0][0], torch.tensor(seed_rows[0], dtype=torch.float32))

            observation, reward, terminated, truncated, info = env.step(torch.zeros(3, dtype=torch.float32))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["terminate_episode"])
            self.assertEqual(observation.shape[1], 2)
            self.assertEqual(len(env.finished_paths), 1)

    def test_empty_configured_seeds_fall_back_to_root_seed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            swc_dir = root / "swc"
            img_dir.mkdir()
            swc_dir.mkdir()

            volume_shape = (24, 24, 24)
            _write_volume(img_dir / "sample.tif", shape=volume_shape)
            _write_swc(
                swc_dir / "sample.swc",
                [
                    (1, 3, 4.0, 5.0, 6.0, 1.0, -1),
                    (2, 3, 8.0, 9.0, 10.0, 1.0, 1),
                ],
                shape_zyx=volume_shape,
            )

            dataset = NeuronPatchDataset(
                swc_dir=swc_dir,
                img_dir=img_dir,
                step_width=2.0,
                crop_patches=False,
                inference_mode=True,
                seed_points_by_image={"sample.tif": []},
            )

            sample = dataset[0]
            self.assertEqual(tuple(sample["seed_points"].shape), (1, 3))
            torch.testing.assert_close(
                sample["seed_points"][0],
                torch.tensor([6.0, 5.0, 4.0], dtype=torch.float32),
            )

    def test_seed_jitter_order_is_originals_then_round_robin_cycles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            _write_volume(img_dir / "sample.tif", shape=(40, 40, 40))
            seed_rows = [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
            jitter_count = 2
            jitter_radius = 5.0

            dataset = NeuronPatchDataset(
                swc_dir=None,
                img_dir=img_dir,
                step_width=2.0,
                crop_patches=False,
                inference_mode=True,
                seed_points_by_image={"sample.tif": seed_rows},
                seed_jitter_count=jitter_count,
                seed_jitter_radius=jitter_radius,
                rng=np.random.default_rng(0),
            )

            sample_a = dataset[0]
            sample_b = dataset[0]
            seeds = sample_a["seed_points"]

            expected_n = len(seed_rows) * (1 + jitter_count)
            self.assertEqual(tuple(seeds.shape), (expected_n, 3))
            self.assertTrue(torch.equal(sample_a["seed_points"], sample_b["seed_points"]))

            originals = torch.tensor(seed_rows, dtype=torch.float32)
            torch.testing.assert_close(seeds[: len(seed_rows)], originals)

            # Cycle-wise order: [orig1, orig2, j1(orig1), j1(orig2), j2(orig1), j2(orig2)]
            for cycle in range(jitter_count):
                cycle_start = len(seed_rows) + cycle * len(seed_rows)
                for seed_idx in range(len(seed_rows)):
                    jitter_seed = seeds[cycle_start + seed_idx]
                    base_seed = originals[seed_idx]
                    dist = torch.linalg.vector_norm(jitter_seed - base_seed)
                    self.assertLessEqual(float(dist.item()), jitter_radius + 1e-4)

    def test_seed_jitter_intensity_weighted_targets_bright_voxel(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            volume = np.zeros((40, 40, 40), dtype=np.uint8)
            bright_zyx = (22, 20, 20)
            volume[bright_zyx] = 255
            tf.imwrite(img_dir / "sample.tif", volume)

            seed_rows = [[20.0, 20.0, 20.0]]
            jitter_count = 5

            dataset = NeuronPatchDataset(
                swc_dir=None,
                img_dir=img_dir,
                step_width=2.0,
                crop_patches=False,
                inference_mode=True,
                seed_points_by_image={"sample.tif": seed_rows},
                seed_jitter_count=jitter_count,
                seed_jitter_radius=3.0,
                seed_jitter_weight_strategy="intensity_weighted",
                rng=np.random.default_rng(0),
            )

            sample = dataset[0]
            seeds = sample["seed_points"]
            jittered = seeds[1:]
            expected = torch.tensor(bright_zyx, dtype=torch.float32)
            self.assertEqual(tuple(jittered.shape), (jitter_count, 3))
            for row in jittered:
                torch.testing.assert_close(row, expected)

    def test_seed_jitter_boundary_weighted_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            volume = np.zeros((40, 40, 40), dtype=np.uint8)
            volume[20:, :, :] = 255
            tf.imwrite(img_dir / "sample.tif", volume)

            seed_rows = [[20.0, 20.0, 20.0]]

            dataset = NeuronPatchDataset(
                swc_dir=None,
                img_dir=img_dir,
                step_width=2.0,
                crop_patches=False,
                inference_mode=True,
                seed_points_by_image={"sample.tif": seed_rows},
                seed_jitter_count=6,
                seed_jitter_radius=4.0,
                seed_jitter_weight_strategy="boundary_weighted",
                rng=np.random.default_rng(0),
            )

            sample_a = dataset[0]
            sample_b = dataset[0]
            self.assertTrue(torch.equal(sample_a["seed_points"], sample_b["seed_points"]))

    def test_inference_runtime_uses_external_seeds_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            _write_volume(img_dir / "sample.tif")
            seeds_path = root / "seeds.json"
            seed_rows = [[6.0, 7.0, 8.0]]
            save_seeds_json(seeds_path, {"sample.tif": seed_rows})

            env = build_env(
                {
                    "img_dir": str(img_dir),
                    "seeds_path": str(seeds_path),
                    "crop_patches": False,
                    "patches_per_image": 1,
                    "step_width": 4.0,
                    "max_len": 10,
                    "max_paths": 1,
                    "branching": False,
                    "repeat_starts": False,
                }
            )
            env.reset(dataset_index=0)

            torch.testing.assert_close(env.paths[0][0], torch.tensor(seed_rows[0], dtype=torch.float32))
            self.assertEqual(env.img.data.shape[0], 2)

    def test_interactive_runtime_uses_ui_selected_seeds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            _write_volume(img_dir / "sample.tif")
            seed_rows = [[4.0, 5.0, 6.0]]
            captured = {}

            def fake_trace_image(env, actor, dataset_idx, **kwargs):
                env.reset(dataset_index=dataset_idx)
                captured["seed"] = env.paths[0][0].detach().cpu().clone()
                captured["channels"] = int(env.img.data.shape[0])
                captured["seed_map"] = dict(env.dataset.seed_points_by_image)
                return {
                    "paths": [[[1.0, 2.0, 3.0]]],
                    "labeled_neuron": np.zeros(tuple(int(v) for v in env.img.data.shape[-3:]), dtype=np.uint8),
                    "timing_ms": {"reset": 1.0},
                }

            with mock.patch.object(interactive_pipeline, "load_models", return_value=(object(), None)):
                with mock.patch.object(interactive_pipeline, "sac_trace_image", side_effect=fake_trace_image):
                    runtime = interactive_pipeline._TraceRuntime(
                        {
                            "img_dir": str(img_dir),
                            "step_width": 4.0,
                            "n_trials": 1,
                            "max_len": 10,
                            "max_paths": 1,
                            "branching": False,
                            "repeat_starts": False,
                        }
                    )
                    result = runtime.trace_image(0, "sample.tif", seed_rows)

            self.assertEqual(captured["seed_map"]["sample.tif"], seed_rows)
            torch.testing.assert_close(captured["seed"], torch.tensor(seed_rows[0], dtype=torch.float32))
            self.assertEqual(captured["channels"], 2)
            self.assertIn("paths", result)

    def test_interactive_runtime_resolves_dataset_index_from_image_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            _write_volume(img_dir / "a.tif")
            _write_volume(img_dir / "b.tif")
            seed_rows = [[11.0, 12.0, 13.0]]
            captured = {}

            def fake_trace_image(env, actor, dataset_idx, **kwargs):
                captured["dataset_idx"] = int(dataset_idx)
                env.reset(dataset_index=dataset_idx)
                captured["seed"] = env.paths[0][0].detach().cpu().clone()
                captured["seed_map"] = dict(env.dataset.seed_points_by_image)
                return {
                    "paths": [[[1.0, 2.0, 3.0]]],
                    "labeled_neuron": np.zeros(tuple(int(v) for v in env.img.data.shape[-3:]), dtype=np.uint8),
                    "timing_ms": {"reset": 1.0},
                }

            with mock.patch.object(interactive_pipeline, "load_models", return_value=(object(), None)):
                with mock.patch.object(interactive_pipeline, "sac_trace_image", side_effect=fake_trace_image):
                    runtime = interactive_pipeline._TraceRuntime(
                        {
                            "img_dir": str(img_dir),
                            "step_width": 4.0,
                            "n_trials": 1,
                            "max_len": 10,
                            "max_paths": 1,
                            "branching": False,
                            "repeat_starts": False,
                        }
                    )
                    # Intentionally pass the wrong index (0), but the right key (b.tif).
                    runtime.trace_image(0, "b.tif", seed_rows)

            self.assertEqual(captured["dataset_idx"], 1)
            self.assertEqual(captured["seed_map"]["b.tif"], seed_rows)
            torch.testing.assert_close(captured["seed"], torch.tensor(seed_rows[0], dtype=torch.float32))

    def test_interactive_trace_uses_current_prediction_graph_as_initial_path_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()

            _write_volume(img_dir / "sample.tif", shape=(16, 16, 16))
            captured = {}

            def fake_trace_image(env, actor, dataset_idx, **kwargs):
                del env, actor, dataset_idx
                captured["initial_path_mask"] = kwargs.get("initial_path_mask", None)
                return {
                    "paths": [[[1.0, 2.0, 3.0]]],
                    "labeled_neuron": np.zeros((16, 16, 16), dtype=np.uint8),
                    "timing_ms": {"reset": 1.0},
                }

            with mock.patch.object(interactive_pipeline, "load_models", return_value=(object(), None)):
                with mock.patch.object(interactive_pipeline, "sac_trace_image", side_effect=fake_trace_image):
                    runtime = interactive_pipeline._TraceRuntime(
                        {
                            "img_dir": str(img_dir),
                            "step_width": 4.0,
                            "n_trials": 1,
                            "max_len": 10,
                            "max_paths": 1,
                            "branching": False,
                            "repeat_starts": False,
                        }
                    )
                    runtime.trace_image(
                        0,
                        "sample.tif",
                        seed_rows=[],
                        prediction_paths=[[[2.0, 2.0, 2.0], [2.0, 2.0, 8.0]]],
                    )

            initial_path_mask = captured["initial_path_mask"]
            self.assertIsNotNone(initial_path_mask)
            self.assertEqual(tuple(initial_path_mask.shape), (16, 16, 16))
            self.assertGreater(int(np.count_nonzero(initial_path_mask)), 0)

    def test_interactive_session_accepts_inference_style_config_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images"
            img_dir.mkdir()
            _write_volume(img_dir / "sample.tif")

            seeds_path = root / "seeds.json"
            save_seeds_json(seeds_path, {"sample.tif": [[6.0, 7.0, 8.0]]})

            out_dir = root / "inference_output"
            config_path = root / "interactive_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "img_dir": str(img_dir),
                        "seeds_path": str(seeds_path),
                        "out_dir": str(out_dir),
                    }
                ),
                encoding="utf-8",
            )

            captured_prompt_args = {}

            def fake_prompt_seed_session_paths(image_dir=None, seeds_input_path=None, seeds_output_path=None):
                captured_prompt_args["image_dir"] = image_dir
                captured_prompt_args["seeds_input_path"] = seeds_input_path
                captured_prompt_args["seeds_output_path"] = seeds_output_path
                return image_dir, seeds_input_path, seeds_output_path

            with mock.patch.object(
                interactive_pipeline,
                "prompt_seed_session_paths",
                side_effect=fake_prompt_seed_session_paths,
            ):
                with mock.patch.object(
                    interactive_pipeline,
                    "interactive_seed_selection_session",
                    return_value=torch.zeros((0, 3), dtype=torch.float32),
                ):
                    interactive_pipeline.run_interactive_tracing_session(
                        config_path=str(config_path),
                    )

            self.assertEqual(captured_prompt_args["image_dir"], str(img_dir))
            self.assertEqual(captured_prompt_args["seeds_input_path"], str(seeds_path))
            self.assertIsNone(captured_prompt_args["seeds_output_path"])
            self.assertTrue(out_dir.exists())

    def test_interactive_gt_selector_returns_current_swc_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path)
            swc_dir = root / "swcs"
            swc_dir.mkdir()
            expected_rows = _make_chain_rows(3)
            _write_swc(swc_dir / "sample.swc", expected_rows)
            captured = {}

            def fake_interactive_session(**kwargs):
                captured["result"] = kwargs["on_select_gt_swc_path"]()
                return torch.zeros((0, 3), dtype=torch.float32)

            with mock.patch.object(
                interactive_pipeline,
                "prompt_seed_session_paths",
                return_value=(str(root), None, None),
            ), mock.patch.object(
                interactive_pipeline,
                "prompt_select_directory",
                return_value=str(swc_dir),
            ), mock.patch.object(
                interactive_pipeline,
                "interactive_seed_selection_session",
                side_effect=fake_interactive_session,
            ):
                interactive_pipeline.run_interactive_tracing_session(image_dir=str(root))

            selected_path, selected_rows = captured["result"]
            self.assertEqual(selected_path, str(swc_dir))
            np.testing.assert_allclose(np.asarray(selected_rows), np.asarray(expected_rows))

    def test_postprocess_prediction_target_updates_and_undo_restores_trace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(12, 12, 12))
            image_key = image_path.relative_to(root).as_posix()

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
                postprocess_config=interactive_pipeline.PostprocessConfig(),
            )

            original_paths = [
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
            ]
            processed_paths = [
                [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]],
            ]
            manager.trace_results_by_key[image_key] = original_paths

            with mock.patch.object(
                interactive_pipeline,
                "process_results",
                return_value=[
                    {
                        "neuron_name": image_key,
                        "processed_paths": processed_paths,
                        "n_processed_paths": 1,
                    }
                ],
            ):
                result = manager.run_postprocess(image_key=image_key, target="prediction")

            self.assertIsNotNone(result)
            self.assertEqual(len(manager.trace_results_by_key[image_key]), 1)
            np.testing.assert_allclose(
                np.asarray(manager.trace_results_by_key[image_key][0], dtype=np.float32),
                np.asarray(processed_paths[0], dtype=np.float32),
            )

            status_before_undo = manager.get_status(current_key=image_key, target="prediction")
            self.assertTrue(bool(status_before_undo.get("can_undo_postprocess", False)))

            restored = manager.undo_postprocess(image_key=image_key, target="prediction")
            self.assertIsNotNone(restored)
            np.testing.assert_allclose(
                np.asarray(manager.trace_results_by_key[image_key][0], dtype=np.float32),
                np.asarray(original_paths[0], dtype=np.float32),
            )

            status_after_undo = manager.get_status(current_key=image_key, target="prediction")
            self.assertFalse(bool(status_after_undo.get("can_undo_postprocess", False)))

    def test_postprocess_reference_target_updates_and_undo_restores_reference_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(16, 16, 16))
            image_key = image_path.relative_to(root).as_posix()

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
                postprocess_config=interactive_pipeline.PostprocessConfig(),
            )

            original_rows = [
                [1.0, 0.0, 2.0, 2.0, 2.0, 1.0, -1.0],
                [2.0, 0.0, 3.0, 2.0, 2.0, 1.0, 1.0],
                [3.0, 0.0, 4.0, 2.0, 2.0, 1.0, 2.0],
            ]
            manager.set_filtered_swc_rows(image_key=image_key, swc_rows=original_rows)

            processed_paths = [
                [[20.0, 20.0, 20.0], [21.0, 20.0, 20.0], [22.0, 20.0, 20.0]],
            ]
            with mock.patch.object(
                interactive_pipeline,
                "process_results",
                return_value=[
                    {
                        "neuron_name": image_key,
                        "processed_paths": processed_paths,
                        "n_processed_paths": 1,
                    }
                ],
            ):
                result = manager.run_postprocess(image_key=image_key, target="reference")

            self.assertIsNotNone(result)
            processed_rows = manager.filtered_swc_by_key[image_key]
            self.assertGreaterEqual(len(processed_rows), 2)
            np.testing.assert_allclose(
                np.asarray(processed_rows[0][2:5], dtype=np.float32),
                np.asarray(processed_paths[0][0], dtype=np.float32),
            )

            status_before_undo = manager.get_status(current_key=image_key, target="reference")
            self.assertTrue(bool(status_before_undo.get("can_undo_postprocess", False)))

            restored = manager.undo_postprocess(image_key=image_key, target="reference")
            self.assertIsNotNone(restored)
            np.testing.assert_allclose(
                np.asarray(manager.filtered_swc_by_key[image_key], dtype=np.float32),
                np.asarray(original_rows, dtype=np.float32),
            )

            status_after_undo = manager.get_status(current_key=image_key, target="reference")
            self.assertFalse(bool(status_after_undo.get("can_undo_postprocess", False)))

    def test_trace_current_appends_new_paths_instead_of_replacing_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(12, 12, 12))
            image_key = image_path.relative_to(root).as_posix()

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
                postprocess_config=interactive_pipeline.PostprocessConfig(),
            )
            manager.enabled = True

            class _RuntimeStub:
                def __init__(self):
                    self.calls = 0

                def trace_image(self, image_index, image_relative_key, seed_rows, prediction_paths=None, cancel_event=None):
                    del image_index, image_relative_key, seed_rows, prediction_paths, cancel_event
                    self.calls += 1
                    if self.calls == 1:
                        return {
                            "paths": [[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]]],
                            "timing_ms": {"total": 1.0, "steps": 1},
                        }
                    return {
                        "paths": [[[9.0, 9.0, 9.0], [10.0, 9.0, 9.0]]],
                        "timing_ms": {"total": 1.0, "steps": 1},
                    }

            manager._runtime = _RuntimeStub()
            manager.trace_results_by_key[image_key] = [
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            ]

            first = manager.trace_current(image_index=0, image_key=image_key, seed_rows=[])
            self.assertIsNotNone(first)
            self.assertEqual(len(manager.trace_results_by_key[image_key]), 2)
            np.testing.assert_allclose(
                np.asarray(manager.trace_results_by_key[image_key][0], dtype=np.float32),
                np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32),
            )
            np.testing.assert_allclose(
                np.asarray(manager.trace_results_by_key[image_key][1], dtype=np.float32),
                np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=np.float32),
            )

            second = manager.trace_current(image_index=0, image_key=image_key, seed_rows=[])
            self.assertIsNotNone(second)
            self.assertEqual(len(manager.trace_results_by_key[image_key]), 3)
            np.testing.assert_allclose(
                np.asarray(manager.trace_results_by_key[image_key][2], dtype=np.float32),
                np.asarray([[9.0, 9.0, 9.0], [10.0, 9.0, 9.0]], dtype=np.float32),
            )

    def test_trace_current_append_keeps_path_order_existing_then_new(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(12, 12, 12))
            image_key = image_path.relative_to(root).as_posix()

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
                postprocess_config=interactive_pipeline.PostprocessConfig(),
            )
            manager.enabled = True

            class _RuntimeStub:
                def trace_image(self, image_index, image_relative_key, seed_rows, prediction_paths=None, cancel_event=None):
                    del image_index, image_relative_key, seed_rows, prediction_paths, cancel_event
                    return {
                        "paths": [
                            [[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]],
                            [[30.0, 30.0, 30.0], [31.0, 30.0, 30.0]],
                        ],
                        "timing_ms": {"total": 1.0, "steps": 1},
                    }

            manager._runtime = _RuntimeStub()
            manager.trace_results_by_key[image_key] = [
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                [[3.0, 3.0, 3.0], [4.0, 3.0, 3.0]],
            ]

            merged = manager.trace_current(image_index=0, image_key=image_key, seed_rows=[])
            self.assertIsNotNone(merged)
            self.assertEqual(len(merged), 4)
            np.testing.assert_allclose(np.asarray(merged[0], dtype=np.float32), np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32))
            np.testing.assert_allclose(np.asarray(merged[1], dtype=np.float32), np.asarray([[3.0, 3.0, 3.0], [4.0, 3.0, 3.0]], dtype=np.float32))
            np.testing.assert_allclose(np.asarray(merged[2], dtype=np.float32), np.asarray([[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]], dtype=np.float32))
            np.testing.assert_allclose(np.asarray(merged[3], dtype=np.float32), np.asarray([[30.0, 30.0, 30.0], [31.0, 30.0, 30.0]], dtype=np.float32))

    def test_trace_current_rerun_uses_only_new_seed_rows_and_keeps_prediction_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(12, 12, 12))
            image_key = image_path.relative_to(root).as_posix()

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
                postprocess_config=interactive_pipeline.PostprocessConfig(),
            )
            manager.enabled = True
            manager.trace_results_by_key[image_key] = [
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            ]
            manager.traced_seed_rows_by_key[image_key] = [[5.0, 5.0, 5.0]]

            captured_seed_rows = []
            captured_prediction_paths = []

            class _RuntimeStub:
                def trace_image(self, image_index, image_relative_key, seed_rows, prediction_paths=None, cancel_event=None):
                    del image_index, image_relative_key, cancel_event
                    captured_seed_rows.append(seed_rows)
                    captured_prediction_paths.append(prediction_paths)
                    return {
                        "paths": [[[9.0, 9.0, 9.0], [10.0, 9.0, 9.0]]],
                        "timing_ms": {"total": 1.0, "steps": 1},
                    }

            manager._runtime = _RuntimeStub()

            merged = manager.trace_current(
                image_index=0,
                image_key=image_key,
                seed_rows=[[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]],
            )

            self.assertIsNotNone(merged)
            self.assertEqual(captured_seed_rows, [[[7.0, 7.0, 7.0]]])
            np.testing.assert_allclose(
                np.asarray(captured_prediction_paths[0], dtype=np.float32),
                np.asarray([[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]], dtype=np.float32),
            )
            self.assertEqual(manager.traced_seed_rows_by_key[image_key], [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]])

    def test_reference_postprocess_does_not_change_prediction_trace_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(12, 12, 12))
            image_key = image_path.relative_to(root).as_posix()

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
                postprocess_config=interactive_pipeline.PostprocessConfig(),
            )
            manager.enabled = True

            original_prediction = [
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            ]
            manager.trace_results_by_key[image_key] = [
                [list(point) for point in path] for path in original_prediction
            ]
            manager.filtered_swc_by_key[image_key] = _make_chain_rows(3)

            def fake_process_results(results, params):
                del params
                return [
                    {
                        "processed_paths": results[0]["paths"],
                        "n_processed_paths": len(results[0]["paths"]),
                    }
                ]

            class _RuntimeStub:
                def __init__(self):
                    self.captured_prediction_paths = None

                def trace_image(self, image_index, image_relative_key, seed_rows, prediction_paths=None, cancel_event=None):
                    del image_index, image_relative_key, seed_rows, cancel_event
                    self.captured_prediction_paths = prediction_paths
                    return {
                        "paths": [[[9.0, 9.0, 9.0], [10.0, 9.0, 9.0]]],
                        "timing_ms": {"total": 1.0, "steps": 1},
                    }

            runtime_stub = _RuntimeStub()
            manager._runtime = runtime_stub

            with mock.patch.object(interactive_pipeline, "process_results", side_effect=fake_process_results):
                manager.run_postprocess(image_key, target="reference")

            np.testing.assert_allclose(
                np.asarray(manager.trace_results_by_key[image_key], dtype=np.float32),
                np.asarray(original_prediction, dtype=np.float32),
            )

            manager.trace_current(image_index=0, image_key=image_key, seed_rows=[])

            np.testing.assert_allclose(
                np.asarray(runtime_stub.captured_prediction_paths, dtype=np.float32),
                np.asarray(original_prediction, dtype=np.float32),
            )

    def test_start_trace_all_uses_edited_prediction_state_and_appends_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "sample.tif"
            _write_volume(image_path, shape=(12, 12, 12))
            image_key = image_path.relative_to(root).as_posix()

            manager = interactive_pipeline._TraceSessionManager(
                image_paths=[image_path],
                image_root=root,
                trace_params={},
                postprocess_config=interactive_pipeline.PostprocessConfig(),
            )
            manager.enabled = True
            manager.trace_results_by_key[image_key] = [
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            ]

            captured_prediction_paths = []

            class _RuntimeStub:
                def trace_image(self, image_index, image_relative_key, seed_rows, prediction_paths=None, cancel_event=None):
                    del image_index, image_relative_key, seed_rows, cancel_event
                    captured_prediction_paths.append(prediction_paths)
                    return {
                        "paths": [[[9.0, 9.0, 9.0], [10.0, 9.0, 9.0]]],
                        "timing_ms": {"total": 1.0, "steps": 1},
                    }

            manager._runtime = _RuntimeStub()
            manager.start_trace_all(seeds_by_key={image_key: [[5.0, 5.0, 5.0]]})

            self.assertIsNotNone(manager._thread)
            manager._thread.join(timeout=5.0)
            self.assertFalse(manager._thread.is_alive())
            self.assertEqual(len(captured_prediction_paths), 1)
            np.testing.assert_allclose(
                np.asarray(captured_prediction_paths[0], dtype=np.float32),
                np.asarray(manager._normalize_paths_payload([[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]]), dtype=np.float32),
            )
            self.assertEqual(len(manager.trace_results_by_key[image_key]), 2)
            np.testing.assert_allclose(
                np.asarray(manager.trace_results_by_key[image_key][1], dtype=np.float32),
                np.asarray([[9.0, 9.0, 9.0], [10.0, 9.0, 9.0]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()