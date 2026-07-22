import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile as tf
import torch

from neurotrack.data import NeuronPatchDataset
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.environments.tracking_reward import distance_reward
from neurotrack.training.memory import ReplayBuffer


def _write_volume(path: Path, shape=(40, 40, 40)) -> None:
    volume = np.zeros(shape, dtype=np.uint8)
    center = tuple(dim // 2 for dim in shape)
    volume[center] = 255
    tf.imwrite(path, volume)


def _write_swc(path: Path, rows, shape_zyx=None) -> None:
    bounds_xyz = None
    if shape_zyx is not None:
        z_max = max(int(shape_zyx[0]) - 1, 0)
        y_max = max(int(shape_zyx[1]) - 1, 0)
        x_max = max(int(shape_zyx[2]) - 1, 0)
        bounds_xyz = (x_max, y_max, z_max)

    with path.open("w", encoding="utf-8") as handle:
        for node_id, node_type, x, y, z, radius, parent_id in rows:
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


class MultiTargetRewardTests(unittest.TestCase):
    def test_distance_reward_uses_nearest_valid_candidate(self):
        reward = distance_reward(
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(reward, torch.tensor([0.0], dtype=torch.float32))

        batched_reward = distance_reward(
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [8.0, 8.0, 8.0]],
                ],
                dtype=torch.float32,
            ),
            valid_mask=torch.tensor([[True, True, False], [False, True, False]]),
        )
        torch.testing.assert_close(batched_reward, torch.tensor([0.0, -1.0], dtype=torch.float32))


class ReplayBufferTargetSetTests(unittest.TestCase):
    def test_replay_buffer_pads_variable_length_target_sets(self):
        buffer = ReplayBuffer(4, obs_shape=(1, 2, 2, 2), action_shape=(3,))
        obs = torch.zeros((1, 2, 2, 2), dtype=torch.uint8)
        next_obs = torch.full((1, 2, 2, 2), 255, dtype=torch.uint8)
        next_obs_2 = torch.full((1, 2, 2, 2), 200, dtype=torch.uint8)

        buffer.push(
            obs,
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            next_obs,
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            False,
        )
        buffer.push(
            obs + 1,
            torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
            next_obs_2,
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
            True,
        )

        (
            obs_batch,
            _actions,
            next_obs_batch,
            _rewards,
            current_target_vectors,
            current_target_mask,
            next_target_vectors,
            next_target_mask,
            _dones,
        ) = buffer.sample(2, transform=False)

        self.assertEqual(obs_batch.dtype, torch.uint8)
        self.assertEqual(next_obs_batch.dtype, torch.uint8)

        self.assertEqual(tuple(current_target_vectors.shape), (2, 2, 3))
        self.assertEqual(sorted(current_target_mask.sum(dim=1).tolist()), [1, 2])
        self.assertEqual(tuple(next_target_vectors.shape), (2, 3, 3))
        self.assertEqual(sorted(next_target_mask.sum(dim=1).tolist()), [1, 3])

        for row in range(current_target_vectors.shape[0]):
            valid_count = int(current_target_mask[row].sum().item())
            if valid_count < current_target_vectors.shape[1]:
                torch.testing.assert_close(
                    current_target_vectors[row, valid_count:],
                    torch.zeros_like(current_target_vectors[row, valid_count:]),
                )

    def test_replay_buffer_transform_updates_all_target_candidates(self):
        buffer = ReplayBuffer(1, obs_shape=(1, 2, 2, 2), action_shape=(3,))
        buffer.push(
            torch.zeros((1, 2, 2, 2), dtype=torch.uint8),
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.full((1, 2, 2, 2), 255, dtype=torch.uint8),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=torch.float32),
            torch.tensor([[7.0, 8.0, 9.0]], dtype=torch.float32),
            False,
        )

        with mock.patch(
            "neurotrack.training.memory.torch.randperm",
            side_effect=[torch.tensor([0]), torch.tensor([1, 0])],
        ):
            with mock.patch(
                "neurotrack.training.memory.torch.rand",
                side_effect=[torch.tensor([0.6]), torch.tensor([0.6])],
            ):
                (
                    _obs,
                    actions,
                    _next_obs,
                    _rewards,
                    current_target_vectors,
                    current_target_mask,
                    next_target_vectors,
                    next_target_mask,
                    _dones,
                ) = buffer.sample(1, transform=True)

        torch.testing.assert_close(
            actions,
            torch.tensor([[1.0, -3.0, -2.0]], dtype=torch.float32, device=actions.device),
        )
        torch.testing.assert_close(
            current_target_vectors,
            torch.tensor(
                [[[10.0, -30.0, -20.0], [40.0, -60.0, -50.0]]],
                dtype=torch.float32,
                device=current_target_vectors.device,
            ),
        )
        torch.testing.assert_close(
            next_target_vectors,
            torch.tensor(
                [[[7.0, -9.0, -8.0]]],
                dtype=torch.float32,
                device=next_target_vectors.device,
            ),
        )
        self.assertTrue(current_target_mask.all())
        self.assertTrue(next_target_mask.all())

    def test_replay_buffer_rejects_non_uint8_observations(self):
        buffer = ReplayBuffer(1, obs_shape=(1, 2, 2, 2), action_shape=(3,))
        with self.assertRaises(TypeError):
            buffer.push(
                torch.zeros((1, 2, 2, 2), dtype=torch.float32),
                torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                torch.zeros((1, 2, 2, 2), dtype=torch.uint8),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                False,
            )


class EnvironmentMultiTargetStepTests(unittest.TestCase):
    def test_init_path_handles_empty_unvisited_tree_for_seed_restart(self):
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
                patches_per_image=1,
                alpha=1.0,
                step_width=4.0,
                rng=np.random.default_rng(123),
                crop_patches=True,
                inference_mode=False,
            )

            env = NeuronTrackingEnvironment(
                dataset=dataset,
                radius=5,
                target_step_len=1.0,
                step_width=4.0,
                max_len=20,
                branching=False,
                inference_mode=False,
            )
            env.reset(dataset_index=0)

            restart_seed = env.paths[0][0].clone()
            env.paths = [[restart_seed]]
            env.unvisited_tree = torch.empty((0, 7), dtype=torch.float32)
            env.id_to_idx = {}
            env.section_nodes = None
            env.section_assigned = False
            env.cut_ends = []

            env._init_path()

            self.assertIsNone(env.section_nodes)
            self.assertFalse(env.section_assigned)
            torch.testing.assert_close(
                env.target_vectors,
                torch.zeros((1, 3), dtype=torch.float32, device=env.target_vectors.device),
            )

    def test_environment_step_keeps_all_next_target_vectors(self):
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
                patches_per_image=1,
                alpha=1.0,
                step_width=4.0,
                rng=np.random.default_rng(123),
                crop_patches=True,
                inference_mode=False,
            )

            env = NeuronTrackingEnvironment(
                dataset=dataset,
                radius=5,
                target_step_len=1.0,
                step_width=4.0,
                max_len=20,
                branching=False,
                inference_mode=False,
            )
            env.reset(dataset_index=0)
            env.target_vectors = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
            current_position = env.paths[0][-1].clone()
            action = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            next_position = current_position + action
            expected_target_vectors = torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            )

            with mock.patch(
                "neurotrack.environments.neuron_tracking_environment._compute_target_action",
                return_value=(expected_target_vectors, False),
            ):
                _observation, reward, terminated, truncated, info = env.step(action)

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            torch.testing.assert_close(reward, torch.tensor([0.0], dtype=torch.float32))
            torch.testing.assert_close(info["next_target_vectors"], expected_target_vectors)
            torch.testing.assert_close(env.target_vectors, expected_target_vectors)


if __name__ == "__main__":
    unittest.main()