import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile as tf
import torch

from neurotrack.data import NeuronPatchDataset, adjacency_dict, save_seeds_json
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.inference.runtime import build_env
from neurotrack.pipelines import interactive_tracing_pipeline as interactive_pipeline
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


class SeedPipelineValidationTests(unittest.TestCase):
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
                step_width=4.0,
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
                step_width=4.0,
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
                step_width=4.0,
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
                step_width=4.0,
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

    def test_predicted_path_channel_pruning_can_empty_subtree(self):
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
                step_width=4.0,
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
                step_width=4.0,
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
                step_width=4.0,
                crop_patches=False,
                inference_mode=True,
                seed_points_by_image={"sample.tif": seed_rows},
            )

            env = NeuronTrackingEnvironment(
                dataset=dataset,
                radius=5,
                target_step_len=1.0,
                step_width=4.0,
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
                step_width=4.0,
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


if __name__ == "__main__":
    unittest.main()