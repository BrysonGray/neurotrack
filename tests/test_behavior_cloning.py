import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from neurotrack.inference.runtime import load_models
from neurotrack.inference.tracing import _select_action_from_actor_output, trace_image
from neurotrack.models import ConvNet
from neurotrack.training.behavior_cloning import (
    _build_supervision_loss_config,
    _compute_dagger_beta,
    _optimize_prepared_batch,
    multi_target_direction_loss,
    pad_target_candidate_batch,
    select_expert_action,
    train_dagger,
)
from neurotrack.training.memory import BehaviorCloningReplayBuffer


class _ConstantActor(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(action, dtype=torch.float32))

    def forward(self, x):
        return self.bias.unsqueeze(0).expand(x.shape[0], -1)


class _ToyEnv:
    def __init__(self, target_sequence):
        self.dataset = SimpleNamespace(alpha=0.0)
        self.current_neuron_info = {"neuron_name": "toy_neuron"}
        self._target_sequence = [torch.as_tensor(target, dtype=torch.float32).view(-1, 3) for target in target_sequence]
        self.recorded_actions = []
        self._step_index = 0
        self.target_vectors = self._target_sequence[0].clone()

    def reset(self, return_state=True):
        self._step_index = 0
        self.target_vectors = self._target_sequence[0].clone()
        self.recorded_actions.clear()
        return self.get_state()

    def get_state(self):
        return torch.full((1, 2, 35, 35, 35), fill_value=self._step_index, dtype=torch.uint8)

    def step(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
        self.recorded_actions.append(action_t)

        self._step_index += 1
        final_step = self._step_index >= len(self._target_sequence)
        if final_step:
            self.target_vectors = torch.zeros((1, 3), dtype=torch.float32)
            return (
                self.get_state(),
                torch.tensor(1.0, dtype=torch.float32),
                True,
                False,
                {
                    "terminate_episode": True,
                    "current_target_vectors": None,
                    "next_target_vectors": self.target_vectors,
                },
            )

        self.target_vectors = self._target_sequence[self._step_index].clone()
        return (
            self.get_state(),
            torch.tensor(1.0, dtype=torch.float32),
            False,
            False,
            {
                "terminate_episode": False,
                "current_target_vectors": self._target_sequence[self._step_index - 1].clone(),
                "next_target_vectors": self.target_vectors,
            },
        )


class _TraceEnv:
    def __init__(self, current_target_vectors):
        self.dataset = SimpleNamespace(img_files=["trace_image.tif"], inference_mode=True)
        self.current_neuron_info = {"neuron_name": "trace_image.tif"}
        self.img = SimpleNamespace(data=torch.zeros((2, 35, 35, 35), dtype=torch.uint8))
        self.finished_paths = [
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        ]
        self._current_target_vectors = torch.as_tensor(current_target_vectors, dtype=torch.float32).view(-1, 3)

    def reset(self, dataset_index=None, move_to_next=True):
        return self.get_state()

    def get_state(self):
        return torch.zeros((1, 2, 35, 35, 35), dtype=torch.uint8)

    def step(self, action):
        return (
            self.get_state(),
            torch.tensor(0.0, dtype=torch.float32),
            True,
            False,
            {
                "terminate_episode": True,
                "status": "choose_stop",
                "current_target_vectors": self._current_target_vectors.clone(),
                "next_target_vectors": torch.zeros((1, 3), dtype=torch.float32),
            },
        )


class BehaviorCloningLossTests(unittest.TestCase):
    def test_multi_target_direction_loss_uses_nearest_candidate(self):
        loss = multi_target_direction_loss(
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float32))

        batched_loss = multi_target_direction_loss(
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
        torch.testing.assert_close(batched_loss, torch.tensor(0.5, dtype=torch.float32))

    def test_pad_target_candidate_batch_pads_and_masks(self):
        padded, mask = pad_target_candidate_batch(
            [
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
            ]
        )
        self.assertEqual(tuple(padded.shape), (2, 2, 3))
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])


class BehaviorCloningReplayBufferTests(unittest.TestCase):
    def test_replay_buffer_is_fifo_and_bounded(self):
        buffer = BehaviorCloningReplayBuffer(capacity=3)
        obs = torch.zeros((2, 35, 35, 35), dtype=torch.uint8)
        for idx in range(4):
            buffer.push(obs + idx, torch.tensor([[float(idx), 0.0, 0.0]], dtype=torch.float32))

        self.assertEqual(len(buffer), 3)
        vectors = {
            tuple(target.view(-1).tolist())
            for target in buffer.target_vectors
            if target is not None
        }
        self.assertNotIn((0.0, 0.0, 0.0), vectors)
        self.assertIn((3.0, 0.0, 0.0), vectors)

    def test_replay_buffer_transform_sampling_keeps_shapes(self):
        buffer = BehaviorCloningReplayBuffer(capacity=4)
        for idx in range(4):
            obs = torch.full((2, 35, 35, 35), fill_value=idx, dtype=torch.uint8)
            targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
            buffer.push(obs, targets)

        sampled_obs, sampled_targets, sampled_mask = buffer.sample(batch_size=2, replacement=False, transform=True)
        self.assertEqual(tuple(sampled_obs.shape), (2, 2, 35, 35, 35))
        self.assertEqual(tuple(sampled_targets.shape), (2, 2, 3))
        self.assertEqual(tuple(sampled_mask.shape), (2, 2))
        self.assertEqual(sampled_obs.dtype, torch.uint8)
        self.assertEqual(sampled_targets.dtype, torch.float32)
        self.assertTrue(sampled_mask.all())


class BehaviorCloningPolicyTests(unittest.TestCase):
    def test_select_expert_action_prefers_previous_direction_alignment(self):
        action = select_expert_action(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            previous_action=torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32),
        )
        torch.testing.assert_close(action, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))

    def test_select_action_from_actor_output_supports_direct_vector_mode(self):
        action, variance = _select_action_from_actor_output(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            policy_output_mode="direct_vector",
            stochastic=False,
        )
        torch.testing.assert_close(action, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        self.assertIsNone(variance)

    def test_load_models_supports_direct_vector_checkpoints(self):
        actor = ConvNet(chin=2, chout=3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = Path(tmp_dir) / "bc_weights.pt"
            torch.save(
                {
                    "policy_state_dict": actor.state_dict(),
                    "policy_output_mode": "direct_vector",
                    "policy_output_dim": 3,
                },
                weights_path,
            )
            loaded_actor, q_net = load_models(
                {"sac_weights": str(weights_path), "n_trials": 1},
                in_channels=2,
                device=torch.device("cpu"),
            )

            self.assertEqual(getattr(loaded_actor, "policy_output_mode", None), "direct_vector")
            self.assertIsNone(q_net)
            output = loaded_actor(torch.zeros((1, 2, 35, 35, 35), dtype=torch.float32))
            self.assertEqual(tuple(output.shape), (1, 3))

    def test_load_models_rejects_trials_without_q_network(self):
        actor = ConvNet(chin=2, chout=3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = Path(tmp_dir) / "bc_weights.pt"
            torch.save(
                {
                    "policy_state_dict": actor.state_dict(),
                    "policy_output_mode": "direct_vector",
                    "policy_output_dim": 3,
                },
                weights_path,
            )
            with self.assertRaises(ValueError):
                load_models(
                    {"sac_weights": str(weights_path), "n_trials": 2},
                    in_channels=2,
                    device=torch.device("cpu"),
                )

    def test_compute_dagger_beta_interpolates_linearly(self):
        self.assertEqual(_compute_dagger_beta(0, dagger_rounds=3, beta_start=1.0, beta_end=0.0), 1.0)
        self.assertAlmostEqual(_compute_dagger_beta(1, dagger_rounds=3, beta_start=1.0, beta_end=0.0), 0.5)
        self.assertEqual(_compute_dagger_beta(2, dagger_rounds=3, beta_start=1.0, beta_end=0.0), 0.0)

    def test_optimize_prepared_batch_applies_continue_norm_floor(self):
        actor = _ConstantActor([0.1, 0.0, 0.0])
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)
        obs_tensor = torch.zeros((1, 2, 35, 35, 35), dtype=torch.uint8)
        target_tensor = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)
        target_mask = torch.tensor([[True]], dtype=torch.bool)
        loss_config = _build_supervision_loss_config(
            continue_target_norm_threshold=1.0,
            continue_weight=1.0,
            norm_floor=1.0,
            norm_floor_weight=0.5,
            stop_violation_weight=1.0,
        )

        batch_stats = _optimize_prepared_batch(
            actor=actor,
            actor_optimizer=optimizer,
            obs_tensor=obs_tensor,
            target_tensor=target_tensor,
            target_mask=target_mask,
            loss_config=loss_config,
        )

        self.assertAlmostEqual(batch_stats.loss, 4.015, places=3)
        self.assertAlmostEqual(batch_stats.step_mse, 3.61, places=3)
        self.assertEqual(batch_stats.true_continue_count, 1)
        self.assertEqual(batch_stats.false_choose_stop_count, 1)
        self.assertEqual(batch_stats.true_stop_count, 0)
        self.assertEqual(batch_stats.false_continue_count, 0)

    def test_optimize_prepared_batch_margin_stop_only_allows_zero_loss_batch(self):
        actor = _ConstantActor([0.2, 0.0, 0.0])
        initial_bias = actor.bias.detach().clone()
        optimizer = torch.optim.SGD(actor.parameters(), lr=1.0)
        obs_tensor = torch.zeros((1, 2, 35, 35, 35), dtype=torch.uint8)
        # Stop-only target with margin objective and zero auxiliary weights yields zero scalar loss.
        target_tensor = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        target_mask = torch.tensor([[True]], dtype=torch.bool)
        loss_config = _build_supervision_loss_config(
            continue_target_norm_threshold=0.5,
            continue_weight=1.0,
            norm_floor=0.0,
            norm_floor_weight=0.0,
            stop_violation_weight=1.0,
            objective_mode="norm_classifier_margin",
            continue_direction_weight=1.0,
            norm_cls_weight=0.0,
            norm_cls_temperature=0.2,
            norm_margin_weight=0.0,
            stop_margin=0.1,
            continue_margin=0.0,
        )

        batch_stats = _optimize_prepared_batch(
            actor=actor,
            actor_optimizer=optimizer,
            obs_tensor=obs_tensor,
            target_tensor=target_tensor,
            target_mask=target_mask,
            loss_config=loss_config,
        )

        self.assertAlmostEqual(batch_stats.loss, 0.0, places=6)
        self.assertAlmostEqual(batch_stats.step_mse, 0.04, places=6)
        self.assertEqual(batch_stats.true_continue_count, 0)
        self.assertEqual(batch_stats.false_choose_stop_count, 0)
        self.assertEqual(batch_stats.true_stop_count, 1)
        self.assertEqual(batch_stats.false_continue_count, 0)
        torch.testing.assert_close(actor.bias.detach(), initial_bias)

    def test_trace_image_reports_false_stop_diagnostics(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        actor = _ConstantActor([0.2, 0.0, 0.0]).to(device=device)
        actor.policy_output_mode = "direct_vector"
        env = _TraceEnv(current_target_vectors=[[3.0, 0.0, 0.0]])

        result = trace_image(
            env=env,
            actor=actor,
            dataset_idx=0,
            Q_net=None,
            n_trials=1,
            show=False,
            show_live=False,
            stochastic=False,
            return_stats=True,
            terminal_target_norm_threshold=1.0,
            false_stop_distance_threshold=1.5,
        )

        self.assertEqual(result["choose_stop_count"], 1)
        self.assertEqual(result["false_choose_stop_count"], 1)
        self.assertAlmostEqual(result["false_choose_stop_rate"], 1.0)
        self.assertIn("terminal_state_action_norm_histogram", result)
        self.assertIn("nonterminal_state_action_norm_histogram", result)

    def test_train_dagger_uses_policy_rollin_when_beta_zero(self):
        env = _ToyEnv([
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        ])
        actor = _ConstantActor([0.0, 0.0, 2.0])
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_dagger(
                env=env,
                actor=actor,
                actor_optimizer=optimizer,
                outdir=Path(tmp_dir),
                logdir=Path(tmp_dir),
                name="dagger_policy_rollin",
                batch_size=2,
                n_episodes=1,
                warmstart_episodes=0,
                update_after_steps=1,
                update_every=1,
                updates_per_step=1,
                beta_start=0.0,
                beta_end=0.0,
                save_every_steps=1,
                rng=np.random.default_rng(0),
            )

        torch.testing.assert_close(env.recorded_actions[0], torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32))

    def test_train_dagger_saves_dagger_metadata(self):
        env = _ToyEnv([
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        ])
        actor = _ConstantActor([0.0, 0.0, 2.0])
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            outdir = Path(tmp_dir) / "weights"
            logdir = Path(tmp_dir) / "logs"
            train_dagger(
                env=env,
                actor=actor,
                actor_optimizer=optimizer,
                outdir=outdir,
                logdir=logdir,
                name="dagger_metadata",
                batch_size=2,
                n_episodes=2,
                warmstart_episodes=1,
                update_after_steps=1,
                update_every=1,
                updates_per_step=1,
                beta_start=0.25,
                beta_end=0.25,
                save_every_steps=1,
            )

            checkpoints = list(outdir.glob("model_state_dicts_dagger_metadata_*.pt"))
            self.assertEqual(len(checkpoints), 1)
            checkpoint = torch.load(checkpoints[0], map_location="cpu")

        self.assertEqual(checkpoint["algorithm"], "multi_target_dagger_online")
        self.assertEqual(checkpoint["policy_output_mode"], "direct_vector")
        self.assertEqual(checkpoint["policy_output_dim"], 3)
        self.assertEqual(checkpoint["dagger_episode"], 1)
        self.assertEqual(checkpoint["beta"], 0.25)
        self.assertGreaterEqual(checkpoint["dataset_size"], 2)
        self.assertEqual(checkpoint["aggregate_memory_budget"], 10000)
        self.assertEqual(checkpoint["update_after_steps"], 1)
        self.assertEqual(checkpoint["update_every"], 1)
        self.assertEqual(checkpoint["updates_per_step"], 1)

    def test_train_dagger_respects_fifo_memory_budget(self):
        env = _ToyEnv([
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        ])
        actor = _ConstantActor([0.0, 0.0, 2.0])
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            outdir = Path(tmp_dir) / "weights"
            logdir = Path(tmp_dir) / "logs"
            train_dagger(
                env=env,
                actor=actor,
                actor_optimizer=optimizer,
                outdir=outdir,
                logdir=logdir,
                name="dagger_fifo_budget",
                batch_size=2,
                n_episodes=4,
                warmstart_episodes=2,
                update_after_steps=1,
                update_every=1,
                updates_per_step=1,
                beta_start=0.5,
                beta_end=0.5,
                save_every_steps=1,
                aggregate_memory_budget=3,
                rng=np.random.default_rng(0),
            )

            checkpoints = list(outdir.glob("model_state_dicts_dagger_fifo_budget_*.pt"))
            self.assertEqual(len(checkpoints), 1)
            checkpoint = torch.load(checkpoints[0], map_location="cpu")

        self.assertEqual(checkpoint["aggregate_memory_budget"], 3)
        self.assertEqual(checkpoint["dataset_size"], 3)


if __name__ == "__main__":
    unittest.main()