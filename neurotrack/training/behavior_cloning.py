"""Deterministic multi-target behavior cloning trainer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
import sys
import traceback
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from neurotrack.training.memory import BehaviorCloningReplayBuffer
from neurotrack.training.policy_utils import prepare_observation_for_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
date_time = datetime.now().strftime("'%Y-%m-%d_%H-%M-%S'")


@dataclass(frozen=True)
class SupervisionLossConfig:
    """Configuration for continue-state stabilization losses."""

    continue_target_norm_threshold: float = 1.0
    continue_weight: float = 1.0
    norm_floor: float = 0.0
    norm_floor_weight: float = 0.0


@dataclass(frozen=True)
class SupervisionBatchStats:
    """Per-optimization batch metrics for loss and stop/continue diagnostics."""

    loss: float
    step_mse: float
    true_continue_count: int
    false_choose_stop_count: int
    true_stop_count: int
    false_continue_count: int


def _build_supervision_loss_config(
    continue_target_norm_threshold: float,
    continue_weight: float,
    norm_floor: float,
    norm_floor_weight: float,
) -> SupervisionLossConfig:
    threshold = float(continue_target_norm_threshold)
    continue_weight_f = float(continue_weight)
    norm_floor_f = float(norm_floor)
    norm_floor_weight_f = float(norm_floor_weight)

    if threshold < 0.0:
        raise ValueError("continue_target_norm_threshold must be non-negative")
    if continue_weight_f <= 0.0:
        raise ValueError("continue_weight must be positive")
    if norm_floor_f < 0.0:
        raise ValueError("norm_floor must be non-negative")
    if norm_floor_weight_f < 0.0:
        raise ValueError("norm_floor_weight must be non-negative")

    return SupervisionLossConfig(
        continue_target_norm_threshold=threshold,
        continue_weight=continue_weight_f,
        norm_floor=norm_floor_f,
        norm_floor_weight=norm_floor_weight_f,
    )


def _compute_min_target_norm(
    target_tensor: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample minimum valid target norm and a valid-target mask."""
    target_norms = torch.linalg.norm(target_tensor, dim=-1)
    masked_norms = target_norms.masked_fill(~target_mask, torch.inf)
    has_valid_target = target_mask.any(dim=1)
    min_target_norm = masked_norms.min(dim=1).values
    min_target_norm = torch.where(has_valid_target, min_target_norm, torch.zeros_like(min_target_norm))
    return min_target_norm, has_valid_target


def _compute_policy_error_counts(
    pred_actions: torch.Tensor,
    min_target_norm: torch.Tensor,
    has_valid_target: torch.Tensor,
    continue_target_norm_threshold: float,
) -> Tuple[int, int, int, int]:
    """Compute stop/continue error counts from predicted action magnitudes."""
    pred_norms = torch.linalg.norm(pred_actions, dim=1)
    choose_stop = pred_norms < float(continue_target_norm_threshold)
    true_continue_mask = has_valid_target & (min_target_norm > float(continue_target_norm_threshold))
    true_stop_mask = has_valid_target & ~true_continue_mask

    true_continue_count = int(true_continue_mask.sum().item())
    false_choose_stop_count = int((true_continue_mask & choose_stop).sum().item())
    true_stop_count = int(true_stop_mask.sum().item())
    false_continue_count = int((true_stop_mask & ~choose_stop).sum().item())
    return (
        true_continue_count,
        false_choose_stop_count,
        true_stop_count,
        false_continue_count,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _aggregate_supervision_stats(
    batch_stats: Sequence[SupervisionBatchStats],
) -> Tuple[int, int, int, int]:
    true_continue_count = int(sum(stat.true_continue_count for stat in batch_stats))
    false_choose_stop_count = int(sum(stat.false_choose_stop_count for stat in batch_stats))
    true_stop_count = int(sum(stat.true_stop_count for stat in batch_stats))
    false_continue_count = int(sum(stat.false_continue_count for stat in batch_stats))
    return (
        true_continue_count,
        false_choose_stop_count,
        true_stop_count,
        false_continue_count,
    )


def _ensure_action_batch(actions: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
    action_t = torch.as_tensor(actions, dtype=torch.float32)
    if action_t.ndim == 1:
        if action_t.shape[0] != 3:
            raise ValueError(f"Actions must have 3 components, got shape {tuple(action_t.shape)}")
        action_t = action_t.unsqueeze(0)
    elif action_t.ndim != 2 or action_t.shape[1] != 3:
        raise ValueError(f"Actions must have shape (B, 3), got {tuple(action_t.shape)}")
    return action_t


def _prepare_target_candidates(
    actions: torch.Tensor | np.ndarray | Sequence[float],
    target_vectors: torch.Tensor | np.ndarray | Sequence[float],
    valid_mask: Optional[torch.Tensor | np.ndarray | Sequence[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize action and candidate tensors to ``(B, 3)`` and ``(B, K, 3)`` shapes."""
    action_t = _ensure_action_batch(actions)
    target_t = torch.as_tensor(target_vectors, dtype=torch.float32, device=action_t.device)

    if target_t.ndim == 1:
        if target_t.shape[0] != 3:
            raise ValueError(f"Target vectors must have 3 components, got shape {tuple(target_t.shape)}")
        target_t = target_t.view(1, 1, 3)
    elif target_t.ndim == 2:
        if target_t.shape[1] != 3:
            raise ValueError(f"Target vectors must have shape (K, 3) or (B, 3), got {tuple(target_t.shape)}")
        if action_t.shape[0] == 1:
            target_t = target_t.unsqueeze(0)
        elif target_t.shape[0] == action_t.shape[0]:
            target_t = target_t.unsqueeze(1)
        else:
            raise ValueError(
                f"Target vectors with shape {tuple(target_t.shape)} are incompatible with action batch {tuple(action_t.shape)}."
            )
    elif target_t.ndim == 3:
        if target_t.shape[-1] != 3:
            raise ValueError(f"Target vectors must end with shape 3, got {tuple(target_t.shape)}")
        if target_t.shape[0] != action_t.shape[0]:
            raise ValueError(
                f"Target vector batch {tuple(target_t.shape)} does not match action batch {tuple(action_t.shape)}."
            )
    else:
        raise ValueError(f"Target vectors must have 1, 2, or 3 dimensions, got {target_t.ndim}")

    if valid_mask is None:
        mask_t = torch.ones(target_t.shape[:2], dtype=torch.bool, device=action_t.device)
    else:
        mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=action_t.device)
        if mask_t.ndim == 1:
            if action_t.shape[0] == 1 and mask_t.shape[0] == target_t.shape[1]:
                mask_t = mask_t.unsqueeze(0)
            elif target_t.shape[1] == 1 and mask_t.shape[0] == action_t.shape[0]:
                mask_t = mask_t.view(action_t.shape[0], 1)
            else:
                raise ValueError(
                    f"Valid mask shape {tuple(mask_t.shape)} is incompatible with target candidates {tuple(target_t.shape[:2])}."
                )
        elif mask_t.ndim == 2:
            if tuple(mask_t.shape) != tuple(target_t.shape[:2]):
                raise ValueError(
                    f"Valid mask shape {tuple(mask_t.shape)} must match target candidates {tuple(target_t.shape[:2])}."
                )
        else:
            raise ValueError(f"Valid mask must have 1 or 2 dimensions, got {mask_t.ndim}")

    return action_t, target_t, mask_t


def pad_target_candidate_batch(target_batches: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a sequence of ``(K_i, 3)`` candidate tensors into a dense batch."""
    target_list: List[torch.Tensor] = []
    for target in target_batches:
        target_t = torch.as_tensor(target, dtype=torch.float32)
        if target_t.ndim == 1:
            target_t = target_t.view(1, 3)
        if target_t.ndim != 2 or target_t.shape[1] != 3:
            raise ValueError(f"Each target candidate tensor must have shape (K, 3), got {tuple(target_t.shape)}")
        target_list.append(target_t)

    if len(target_list) == 0:
        raise ValueError("At least one target candidate tensor is required.")

    max_candidates = max(target.shape[0] for target in target_list)
    padded = torch.zeros((len(target_list), max_candidates, 3), dtype=torch.float32)
    mask = torch.zeros((len(target_list), max_candidates), dtype=torch.bool)
    for row, target in enumerate(target_list):
        n_targets = target.shape[0]
        padded[row, :n_targets] = target
        mask[row, :n_targets] = True
    return padded, mask


def multi_target_direction_loss(
    actions: torch.Tensor | np.ndarray | Sequence[float],
    target_vectors: torch.Tensor | np.ndarray | Sequence[float],
    valid_mask: Optional[torch.Tensor | np.ndarray | Sequence[bool]] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Min-squared-distance loss against the nearest valid target candidate."""
    action_t, target_t, mask_t = _prepare_target_candidates(actions, target_vectors, valid_mask=valid_mask)
    sq_dist = torch.sum((target_t - action_t.unsqueeze(1)) ** 2, dim=-1)
    sq_dist = sq_dist.masked_fill(~mask_t, torch.inf)
    has_valid_target = mask_t.any(dim=1)
    min_sq_dist = sq_dist.min(dim=1).values
    min_sq_dist = torch.where(has_valid_target, min_sq_dist, torch.zeros_like(min_sq_dist))

    if reduction == "mean":
        return min_sq_dist.mean()
    if reduction == "sum":
        return min_sq_dist.sum()
    if reduction == "none":
        return min_sq_dist
    raise ValueError(f"Unsupported reduction '{reduction}'")


def select_expert_action(
    target_vectors: torch.Tensor | np.ndarray | Sequence[float],
    previous_action: Optional[torch.Tensor | np.ndarray | Sequence[float]] = None,
) -> torch.Tensor:
    """Choose a deterministic expert action from a set of valid target vectors."""
    targets = torch.as_tensor(target_vectors, dtype=torch.float32).view(-1, 3)
    if targets.shape[0] == 0:
        return torch.zeros((3,), dtype=torch.float32)
    if targets.shape[0] == 1:
        return targets[0]

    norms = torch.linalg.norm(targets, dim=1)
    if previous_action is not None:
        prev = torch.as_tensor(previous_action, dtype=torch.float32).view(3)
        prev_norm = torch.linalg.norm(prev)
        valid = norms > torch.finfo(torch.float32).eps
        if prev_norm > torch.finfo(torch.float32).eps and torch.any(valid):
            prev_dir = prev / prev_norm
            target_dirs = torch.zeros_like(targets)
            target_dirs[valid] = targets[valid] / norms[valid].unsqueeze(1)
            cos_sim = torch.mv(target_dirs, prev_dir)
            cos_sim = torch.where(valid, cos_sim, torch.full_like(cos_sim, -torch.inf))
            best_idx = int(torch.argmax(cos_sim))
            return targets[best_idx]

    best_idx = int(torch.argmax(norms))
    return targets[best_idx]


def _optimize_batch(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    obs_batch: Sequence[torch.Tensor],
    target_batches: Sequence[torch.Tensor],
    loss_config: SupervisionLossConfig,
) -> SupervisionBatchStats:
    """Run one optimizer step on a buffered supervision batch."""
    obs_tensor = torch.cat([obs.detach().cpu() for obs in obs_batch], dim=0)
    target_tensor, target_mask = pad_target_candidate_batch(target_batches)

    return _optimize_prepared_batch(
        actor=actor,
        actor_optimizer=actor_optimizer,
        obs_tensor=obs_tensor,
        target_tensor=target_tensor,
        target_mask=target_mask,
        loss_config=loss_config,
    )


def _optimize_prepared_batch(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    obs_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    loss_config: SupervisionLossConfig,
) -> SupervisionBatchStats:
    """Run one optimizer step from an already-collated batch."""
    model_device = next(actor.parameters()).device

    obs_model = prepare_observation_for_model(obs_tensor, device=model_device, model_dtype=dtype)
    pred_actions = actor(obs_model)
    target_tensor = target_tensor.to(device=model_device, dtype=dtype)
    target_mask = target_mask.to(device=model_device)

    direction_loss_per_sample = multi_target_direction_loss(
        pred_actions,
        target_tensor,
        valid_mask=target_mask,
        reduction="none",
    )
    min_target_norm, has_valid_target = _compute_min_target_norm(target_tensor, target_mask)
    continue_mask = has_valid_target & (min_target_norm > loss_config.continue_target_norm_threshold)

    sample_weights = torch.ones_like(direction_loss_per_sample)
    if loss_config.continue_weight != 1.0:
        sample_weights = torch.where(
            continue_mask,
            torch.full_like(sample_weights, fill_value=loss_config.continue_weight),
            sample_weights,
        )

    direction_loss = (direction_loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
    step_mse = float(direction_loss_per_sample.mean().item())

    norm_floor_loss = torch.zeros((), dtype=direction_loss.dtype, device=direction_loss.device)
    if loss_config.norm_floor > 0.0 and loss_config.norm_floor_weight > 0.0 and torch.any(continue_mask):
        pred_norms = torch.linalg.norm(pred_actions, dim=1)
        norm_gap = torch.relu(loss_config.norm_floor - pred_norms)
        norm_floor_loss = (norm_gap[continue_mask] ** 2).mean()

    loss = direction_loss + (loss_config.norm_floor_weight * norm_floor_loss)
    (
        true_continue_count,
        false_choose_stop_count,
        true_stop_count,
        false_continue_count,
    ) = _compute_policy_error_counts(
        pred_actions=pred_actions,
        min_target_norm=min_target_norm,
        has_valid_target=has_valid_target,
        continue_target_norm_threshold=loss_config.continue_target_norm_threshold,
    )

    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    return SupervisionBatchStats(
        loss=float(loss.item()),
        step_mse=step_mse,
        true_continue_count=true_continue_count,
        false_choose_stop_count=false_choose_stop_count,
        true_stop_count=true_stop_count,
        false_continue_count=false_continue_count,
    )


def _optimize_from_bc_replay_buffer(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    replay_buffer: BehaviorCloningReplayBuffer,
    batch_size: int,
    transform: bool,
    loss_config: SupervisionLossConfig,
) -> Optional[SupervisionBatchStats]:
    if len(replay_buffer) == 0:
        return None

    sample_size = min(batch_size, len(replay_buffer))
    obs_tensor, target_tensor, target_mask = replay_buffer.sample(
        batch_size=sample_size,
        replacement=False,
        transform=transform,
    )
    return _optimize_prepared_batch(
        actor=actor,
        actor_optimizer=actor_optimizer,
        obs_tensor=obs_tensor,
        target_tensor=target_tensor,
        target_mask=target_mask,
        loss_config=loss_config,
    )


def _current_target_vectors_from_env(env) -> torch.Tensor:
    current_target_vectors = env.target_vectors
    if current_target_vectors is None:
        current_target_vectors = torch.zeros((1, 3), dtype=torch.float32)
    return torch.as_tensor(current_target_vectors, dtype=torch.float32).view(-1, 3)


def _append_supervision_example(
    obs: torch.Tensor,
    target_vectors: torch.Tensor,
    obs_store: List[torch.Tensor],
    target_store: List[torch.Tensor],
) -> None:
    obs_cpu = obs.detach().cpu()
    targets_cpu = torch.as_tensor(target_vectors, dtype=torch.float32).detach().cpu().view(-1, 3)
    obs_store.append(obs_cpu)
    target_store.append(targets_cpu)


def _flush_supervision_batch(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    obs_store: List[torch.Tensor],
    target_store: List[torch.Tensor],
    losses: List[float],
    batch_stats_store: List[SupervisionBatchStats],
    loss_config: SupervisionLossConfig,
) -> None:
    if not obs_store:
        return
    batch_stats = _optimize_batch(actor, actor_optimizer, obs_store, target_store, loss_config=loss_config)
    losses.append(float(batch_stats.loss))
    batch_stats_store.append(batch_stats)
    obs_store.clear()
    target_store.clear()


def _predict_policy_action(actor: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    model_device = next(actor.parameters()).device
    obs_model = prepare_observation_for_model(obs.detach(), device=model_device, model_dtype=dtype)
    with torch.no_grad():
        pred_actions = actor(obs_model)
    pred_actions = _ensure_action_batch(pred_actions)
    return pred_actions[0].detach().to(dtype=torch.float32).cpu()


def _sample_rollin_source(
    beta: float,
    rng: np.random.Generator,
) -> str:
    beta = float(beta)
    if beta >= 1.0:
        return "expert"
    if beta <= 0.0:
        return "policy"
    if float(rng.random()) < beta:
        return "expert"
    return "policy"


def _run_supervision_epoch(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    aggregate_buffer: BehaviorCloningReplayBuffer,
    batch_size: int,
    transform: bool,
    loss_config: SupervisionLossConfig,
) -> List[SupervisionBatchStats]:
    if len(aggregate_buffer) == 0:
        return []

    n_batches = max(1, (len(aggregate_buffer) + batch_size - 1) // batch_size)
    batch_stats: List[SupervisionBatchStats] = []
    for _batch_idx in range(n_batches):
        maybe_loss = _optimize_from_bc_replay_buffer(
            actor=actor,
            actor_optimizer=actor_optimizer,
            replay_buffer=aggregate_buffer,
            batch_size=batch_size,
            transform=transform,
            loss_config=loss_config,
        )
        if maybe_loss is not None:
            batch_stats.append(maybe_loss)
    return batch_stats


def _reward_to_float(reward: torch.Tensor | float) -> float:
    return float(torch.as_tensor(reward, dtype=torch.float32).item())


def _min_target_norm(
    target_vectors: Optional[torch.Tensor | np.ndarray | Sequence[float]],
) -> Optional[float]:
    if target_vectors is None:
        return None
    targets = torch.as_tensor(target_vectors, dtype=torch.float32).view(-1, 3)
    if targets.numel() == 0:
        return None
    norms = torch.linalg.norm(targets, dim=1)
    if norms.numel() == 0:
        return None
    return float(torch.min(norms).item())


def _compute_dagger_beta(round_index: int, dagger_rounds: int, beta_start: float, beta_end: float) -> float:
    if dagger_rounds < 1:
        raise ValueError("dagger_rounds must be at least 1")
    if dagger_rounds == 1:
        return float(beta_start)
    fraction = float(round_index) / float(dagger_rounds - 1)
    return float(beta_start + fraction * (beta_end - beta_start))


def _log_episode_exception(
    phase: str,
    episode_index: int,
    exc: Exception,
    env,
) -> None:
    neuron_name = "unknown"
    if getattr(env, "current_neuron_info", None) is not None:
        neuron_name = str(env.current_neuron_info.get("neuron_name", "unknown"))
    print(
        (
            f"[WARN] {phase} episode {episode_index} failed for neuron '{neuron_name}': "
            f"{type(exc).__name__}: {exc}"
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc(file=sys.stderr)


def _run_expert_episode(
    env,
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    batch_size: int,
    loss_config: SupervisionLossConfig,
    aggregate_buffer: Optional[BehaviorCloningReplayBuffer] = None,
) -> dict[str, float | int]:
    obs = env.reset(return_state=True)
    prev_expert_action: Optional[torch.Tensor] = None
    supervision_obs: List[torch.Tensor] = []
    supervision_targets: List[torch.Tensor] = []
    episode_losses: List[float] = []
    episode_batch_stats: List[SupervisionBatchStats] = []
    episode_rewards: List[float] = []
    episode_step_norms: List[float] = []
    steps_done = 0

    for _step_idx in count():
        current_target_vectors = _current_target_vectors_from_env(env)
        _append_supervision_example(obs, current_target_vectors, supervision_obs, supervision_targets)
        if aggregate_buffer is not None:
            aggregate_buffer.push(obs, current_target_vectors)

        if len(supervision_obs) >= batch_size:
            _flush_supervision_batch(
                actor,
                actor_optimizer,
                supervision_obs,
                supervision_targets,
                episode_losses,
                episode_batch_stats,
                loss_config=loss_config,
            )

        expert_action = select_expert_action(current_target_vectors, previous_action=prev_expert_action)
        episode_step_norms.append(float(torch.linalg.norm(expert_action).item()))

        next_obs, reward, terminated, _truncated, info = env.step(expert_action)
        steps_done += 1
        episode_rewards.append(_reward_to_float(reward))

        if info["terminate_episode"]:
            _flush_supervision_batch(
                actor,
                actor_optimizer,
                supervision_obs,
                supervision_targets,
                episode_losses,
                episode_batch_stats,
                loss_config=loss_config,
            )
            break

        if terminated:
            obs = env.get_state()
            prev_expert_action = None
        else:
            obs = next_obs
            prev_expert_action = expert_action.detach().cpu()

    (
        true_continue_count,
        false_choose_stop_count,
        true_stop_count,
        false_continue_count,
    ) = _aggregate_supervision_stats(episode_batch_stats)

    return {
        "episode_avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episode_avg_loss": float(np.mean(episode_losses)) if episode_losses else 0.0,
        "episode_avg_step_mse": float(np.mean([stat.step_mse for stat in episode_batch_stats])) if episode_batch_stats else 0.0,
        "episode_avg_expert_step_norm": float(np.mean(episode_step_norms)) if episode_step_norms else 0.0,
        "false_stop_rate": _safe_rate(false_choose_stop_count, true_continue_count),
        "false_continue_rate": _safe_rate(false_continue_count, true_stop_count),
        "steps_done": int(steps_done),
    }


def _run_dagger_collection_episode(
    env,
    actor: torch.nn.Module,
    aggregate_buffer: BehaviorCloningReplayBuffer,
    beta: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    obs = env.reset(return_state=True)
    prev_rollin_action: Optional[torch.Tensor] = None
    episode_rewards: List[float] = []
    episode_step_norms: List[float] = []
    steps_done = 0
    policy_steps = 0

    for _step_idx in count():
        current_target_vectors = _current_target_vectors_from_env(env)
        aggregate_buffer.push(obs, current_target_vectors)

        expert_action = select_expert_action(current_target_vectors, previous_action=prev_rollin_action)
        policy_action = _predict_policy_action(actor, obs).detach().cpu()
        episode_step_norms.append(float(torch.linalg.norm(expert_action).item()))
        if beta >= 1.0:
            rollin_action = expert_action.detach().cpu()
        elif float(rng.random()) < beta:
            rollin_action = expert_action.detach().cpu()
        else:
            rollin_action = policy_action
            policy_steps += 1

        next_obs, reward, terminated, _truncated, info = env.step(rollin_action)
        steps_done += 1
        episode_rewards.append(_reward_to_float(reward))

        if info["terminate_episode"]:
            break

        if terminated:
            obs = env.get_state()
            prev_rollin_action = None
        else:
            obs = next_obs
            prev_rollin_action = rollin_action.detach().cpu()

    return {
        "episode_avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episode_avg_expert_step_norm": float(np.mean(episode_step_norms)) if episode_step_norms else 0.0,
        "policy_rollin_fraction": float(policy_steps / steps_done) if steps_done > 0 else 0.0,
        "steps_done": int(steps_done),
    }


def _write_bc_log_row(
    csv_file_path: Path,
    episode_index: int,
    image_file: str,
    avg_reward: float,
    avg_loss: float,
    avg_step_mse: float,
    avg_step_norm: float,
    false_stop_rate: float,
    false_continue_rate: float,
    steps_done: int,
    complexity: float,
) -> None:
    file_exists = csv_file_path.exists()
    with open(csv_file_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow([
                "episode",
                "image_file",
                "episode_avg_reward",
                "episode_avg_loss",
                "episode_avg_step_mse",
                "episode_avg_expert_step_norm",
                "false_stop_rate",
                "false_continue_rate",
                "steps_done",
                "complexity",
            ])
        writer.writerow([
            episode_index,
            image_file,
            avg_reward,
            avg_loss,
            avg_step_mse,
            avg_step_norm,
            false_stop_rate,
            false_continue_rate,
            int(steps_done),
            complexity,
        ])


def _write_dagger_log_row(
    csv_file_path: Path,
    phase: str,
    round_index: int,
    episode_index: int,
    image_file: str,
    avg_reward: float,
    avg_loss: float,
    avg_step_mse: float,
    avg_step_norm: float,
    policy_rollin_fraction: float,
    false_stop_rate: float,
    false_continue_rate: float,
    steps_done: int,
    dataset_size: int,
    complexity: float,
    beta: float,
) -> None:
    file_exists = csv_file_path.exists()
    with open(csv_file_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow([
                "phase",
                "round",
                "episode",
                "image_file",
                "episode_avg_reward",
                "episode_avg_loss",
                "episode_avg_step_mse",
                "episode_avg_expert_step_norm",
                "policy_rollin_fraction",
                "false_stop_rate",
                "false_continue_rate",
                "steps_done",
                "dataset_size",
                "complexity",
                "beta",
            ])
        writer.writerow([
            phase,
            int(round_index),
            int(episode_index),
            image_file,
            avg_reward,
            avg_loss,
            avg_step_mse,
            avg_step_norm,
            policy_rollin_fraction,
            false_stop_rate,
            false_continue_rate,
            int(steps_done),
            int(dataset_size),
            complexity,
            beta,
        ])


def _save_checkpoint(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    outdir: Path,
    name: str,
    steps_done: int,
    episodes_done: int,
    algorithm: str = "multi_target_behavior_cloning",
    extra_metadata: Optional[dict[str, float | int | str]] = None,
) -> None:
    checkpoint = {
        "policy_state_dict": actor.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "steps_done": int(steps_done),
        "episodes_done": int(episodes_done),
        "policy_output_mode": "direct_vector",
        "policy_output_dim": 3,
        "algorithm": algorithm,
    }
    if extra_metadata is not None:
        checkpoint.update(extra_metadata)
    final_path = outdir / f"model_state_dicts_{name}_{date_time}.pt"
    tmp_path = final_path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(final_path)


def _maybe_save_checkpoint(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    outdir: Path,
    name: str,
    steps_done: int,
    episodes_done: int,
    save_every_steps: int,
    last_save_bucket: int,
    algorithm: str,
    extra_metadata: Optional[dict[str, float | int | str]] = None,
) -> int:
    if save_every_steps <= 0:
        return last_save_bucket
    current_bucket = steps_done // save_every_steps
    if current_bucket > last_save_bucket:
        _save_checkpoint(
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=outdir,
            name=name,
            steps_done=steps_done,
            episodes_done=episodes_done,
            algorithm=algorithm,
            extra_metadata=extra_metadata,
        )
        return current_bucket
    return last_save_bucket


def train(
    env,
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    outdir: Path | str,
    logdir: Path | str,
    name: str,
    batch_size: int = 64,
    n_episodes: int = 1000,
    save_every_steps: int = 500,
    continue_target_norm_threshold: Optional[float] = None,
    continue_weight: float = 1.0,
    norm_floor: float = 0.0,
    norm_floor_weight: float = 0.0,
) -> None:
    """Train a deterministic actor with multi-target behavior cloning."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    outdir = Path(outdir)
    logdir = Path(logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    if continue_target_norm_threshold is None:
        continue_target_norm_threshold = float(getattr(env, "stall_threshold", 1.0))
    loss_config = _build_supervision_loss_config(
        continue_target_norm_threshold=continue_target_norm_threshold,
        continue_weight=continue_weight,
        norm_floor=norm_floor,
        norm_floor_weight=norm_floor_weight,
    )

    steps_done = 0
    last_save_bucket = -1
    use_progress_bar = sys.stdout.isatty()
    csv_file_path = logdir / f"{name}_{date_time}_log.csv"

    for ep in tqdm(
        range(n_episodes),
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=1.0,
        disable=not use_progress_bar,
    ):
        if not use_progress_bar:
            print(f"Starting BC episode {ep + 1}/{n_episodes}", flush=True)

        try:
            episode_metrics = _run_expert_episode(
                env=env,
                actor=actor,
                actor_optimizer=actor_optimizer,
                batch_size=batch_size,
                loss_config=loss_config,
            )
        except Exception as exc:
            _log_episode_exception("BC", ep + 1, exc, env)
            continue
        steps_done += int(episode_metrics["steps_done"])
        last_save_bucket = _maybe_save_checkpoint(
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=outdir,
            name=name,
            steps_done=steps_done,
            episodes_done=ep + 1,
            save_every_steps=save_every_steps,
            last_save_bucket=last_save_bucket,
            algorithm="multi_target_behavior_cloning",
            extra_metadata={
                "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                "continue_weight": float(loss_config.continue_weight),
                "norm_floor": float(loss_config.norm_floor),
                "norm_floor_weight": float(loss_config.norm_floor_weight),
            },
        )
        _write_bc_log_row(
            csv_file_path=csv_file_path,
            episode_index=ep,
            image_file=env.current_neuron_info["neuron_name"],
            avg_reward=float(episode_metrics["episode_avg_reward"]),
            avg_loss=float(episode_metrics["episode_avg_loss"]),
            avg_step_mse=float(episode_metrics["episode_avg_step_mse"]),
            avg_step_norm=float(episode_metrics["episode_avg_expert_step_norm"]),
            false_stop_rate=float(episode_metrics["false_stop_rate"]),
            false_continue_rate=float(episode_metrics["false_continue_rate"]),
            steps_done=steps_done,
            complexity=float(env.dataset.alpha),
        )

    _save_checkpoint(
        actor=actor,
        actor_optimizer=actor_optimizer,
        outdir=outdir,
        name=name,
        steps_done=steps_done,
        episodes_done=n_episodes,
        algorithm="multi_target_behavior_cloning",
        extra_metadata={
            "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
            "continue_weight": float(loss_config.continue_weight),
            "norm_floor": float(loss_config.norm_floor),
            "norm_floor_weight": float(loss_config.norm_floor_weight),
        },
    )


def train_dagger(
    env,
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    outdir: Path | str,
    logdir: Path | str,
    name: str,
    batch_size: int = 64,
    warmstart_episodes: int = 100,
    dagger_rounds: int = 5,
    rollout_episodes_per_round: int = 100,
    dataset_epochs_per_round: int = 1,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    save_every_steps: int = 500,
    dynamic_complexity: bool = True,
    aggregate_memory_budget: int = 10000,
    rng: Optional[np.random.Generator] = None,
    continue_target_norm_threshold: Optional[float] = None,
    continue_weight: float = 1.0,
    norm_floor: float = 0.0,
    norm_floor_weight: float = 0.0,
) -> None:
    """Train a deterministic actor with DAgger using aggregated multi-target supervision."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmstart_episodes < 0:
        raise ValueError("warmstart_episodes must be non-negative")
    if dagger_rounds < 1:
        raise ValueError("dagger_rounds must be at least 1")
    if rollout_episodes_per_round < 1:
        raise ValueError("rollout_episodes_per_round must be at least 1")
    if dataset_epochs_per_round < 1:
        raise ValueError("dataset_epochs_per_round must be at least 1")
    if aggregate_memory_budget < 1:
        raise ValueError("aggregate_memory_budget must be at least 1")

    outdir = Path(outdir)
    logdir = Path(logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    if rng is None:
        rng = np.random.default_rng()

    if continue_target_norm_threshold is None:
        continue_target_norm_threshold = float(getattr(env, "stall_threshold", 1.0))
    loss_config = _build_supervision_loss_config(
        continue_target_norm_threshold=continue_target_norm_threshold,
        continue_weight=continue_weight,
        norm_floor=norm_floor,
        norm_floor_weight=norm_floor_weight,
    )

    steps_done = 0
    episodes_done = 0
    last_save_bucket = -1
    use_progress_bar = sys.stdout.isatty()
    csv_file_path = logdir / f"{name}_{date_time}_log.csv"
    aggregate_buffer = BehaviorCloningReplayBuffer(capacity=aggregate_memory_budget)
    memory_budget_meta = int(aggregate_memory_budget)

    warmstart_iter = tqdm(
        range(warmstart_episodes),
        desc="DAgger warmstart",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=1.0,
        disable=not use_progress_bar,
    )
    for warmstart_episode in warmstart_iter:
        if not use_progress_bar:
            print(f"Starting DAgger warmstart episode {warmstart_episode + 1}/{warmstart_episodes}", flush=True)

        try:
            episode_metrics = _run_expert_episode(
                env=env,
                actor=actor,
                actor_optimizer=actor_optimizer,
                batch_size=batch_size,
                loss_config=loss_config,
                aggregate_buffer=aggregate_buffer,
            )
        except Exception as exc:
            _log_episode_exception("DAgger warmstart", warmstart_episode + 1, exc, env)
            continue
        steps_done += int(episode_metrics["steps_done"])
        episodes_done += 1
        last_save_bucket = _maybe_save_checkpoint(
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=outdir,
            name=name,
            steps_done=steps_done,
            episodes_done=episodes_done,
            save_every_steps=save_every_steps,
            last_save_bucket=last_save_bucket,
            algorithm="multi_target_dagger",
            extra_metadata={
                "dagger_round": 0,
                "beta": 1.0,
                "dataset_size": len(aggregate_buffer),
                "aggregate_memory_budget": memory_budget_meta,
                "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                "continue_weight": float(loss_config.continue_weight),
                "norm_floor": float(loss_config.norm_floor),
                "norm_floor_weight": float(loss_config.norm_floor_weight),
            },
        )
        _write_dagger_log_row(
            csv_file_path=csv_file_path,
            phase="warmstart",
            round_index=0,
            episode_index=episodes_done,
            image_file=env.current_neuron_info["neuron_name"],
            avg_reward=float(episode_metrics["episode_avg_reward"]),
            avg_loss=float(episode_metrics["episode_avg_loss"]),
            avg_step_mse=float(episode_metrics["episode_avg_step_mse"]),
            avg_step_norm=float(episode_metrics["episode_avg_expert_step_norm"]),
            policy_rollin_fraction=0.0,
            false_stop_rate=float(episode_metrics["false_stop_rate"]),
            false_continue_rate=float(episode_metrics["false_continue_rate"]),
            steps_done=steps_done,
            dataset_size=len(aggregate_buffer),
            complexity=float(env.dataset.alpha),
            beta=1.0,
        )

    for round_index in range(dagger_rounds):
        beta = _compute_dagger_beta(round_index, dagger_rounds=dagger_rounds, beta_start=beta_start, beta_end=beta_end)
        if not use_progress_bar:
            print(f"Starting DAgger round {round_index + 1}/{dagger_rounds} with beta={beta:.3f}", flush=True)

        rollout_rewards: List[float] = []
        rollout_step_norms: List[float] = []
        rollout_policy_fractions: List[float] = []
        round_collection_steps = 0
        round_iter = tqdm(
            range(rollout_episodes_per_round),
            desc=f"DAgger round {round_index + 1}/{dagger_rounds}",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
            mininterval=1.0,
            disable=not use_progress_bar,
        )
        for _round_episode in round_iter:
            try:
                episode_metrics = _run_dagger_collection_episode(
                    env=env,
                    actor=actor,
                    aggregate_buffer=aggregate_buffer,
                    beta=beta,
                    rng=rng,
                )
            except Exception as exc:
                _log_episode_exception(
                    f"DAgger rollout round {round_index + 1}",
                    episodes_done + 1,
                    exc,
                    env,
                )
                continue
            steps_done += int(episode_metrics["steps_done"])
            episodes_done += 1
            round_collection_steps += int(episode_metrics["steps_done"])
            rollout_rewards.append(float(episode_metrics["episode_avg_reward"]))
            rollout_step_norms.append(float(episode_metrics["episode_avg_expert_step_norm"]))
            rollout_policy_fractions.append(float(episode_metrics["policy_rollin_fraction"]))

            # When the buffer is full and we have collected one full-capacity turnover
            # of new samples without any intermediate policy updates, more collection is
            # typically redundant under a fixed policy.
            if len(aggregate_buffer) >= aggregate_memory_budget and round_collection_steps >= aggregate_memory_budget:
                break

        round_losses: List[float] = []
        round_batch_stats: List[SupervisionBatchStats] = []
        for _epoch in range(dataset_epochs_per_round):
            epoch_batch_stats = _run_supervision_epoch(
                actor=actor,
                actor_optimizer=actor_optimizer,
                aggregate_buffer=aggregate_buffer,
                batch_size=batch_size,
                transform=True,
                loss_config=loss_config,
            )
            round_batch_stats.extend(epoch_batch_stats)
            round_losses.extend(stat.loss for stat in epoch_batch_stats)
            last_save_bucket = _maybe_save_checkpoint(
                actor=actor,
                actor_optimizer=actor_optimizer,
                outdir=outdir,
                name=name,
                steps_done=steps_done,
                episodes_done=episodes_done,
                save_every_steps=save_every_steps,
                last_save_bucket=last_save_bucket,
                algorithm="multi_target_dagger",
                extra_metadata={
                    "dagger_round": int(round_index + 1),
                    "beta": float(beta),
                    "dataset_size": len(aggregate_buffer),
                    "aggregate_memory_budget": memory_budget_meta,
                    "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                    "continue_weight": float(loss_config.continue_weight),
                    "norm_floor": float(loss_config.norm_floor),
                    "norm_floor_weight": float(loss_config.norm_floor_weight),
                },
            )

        (
            round_true_continue_count,
            round_false_choose_stop_count,
            round_true_stop_count,
            round_false_continue_count,
        ) = _aggregate_supervision_stats(round_batch_stats)

        _write_dagger_log_row(
            csv_file_path=csv_file_path,
            phase="dagger",
            round_index=round_index + 1,
            episode_index=episodes_done,
            image_file="__round_summary__",
            avg_reward=float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
            avg_loss=float(np.mean(round_losses)) if round_losses else 0.0,
            avg_step_mse=float(np.mean([stat.step_mse for stat in round_batch_stats])) if round_batch_stats else 0.0,
            avg_step_norm=float(np.mean(rollout_step_norms)) if rollout_step_norms else 0.0,
            policy_rollin_fraction=float(np.mean(rollout_policy_fractions)) if rollout_policy_fractions else 0.0,
            false_stop_rate=_safe_rate(round_false_choose_stop_count, round_true_continue_count),
            false_continue_rate=_safe_rate(round_false_continue_count, round_true_stop_count),
            steps_done=steps_done,
            dataset_size=len(aggregate_buffer),
            complexity=float(env.dataset.alpha),
            beta=float(beta),
        )

    final_beta = _compute_dagger_beta(dagger_rounds - 1, dagger_rounds=dagger_rounds, beta_start=beta_start, beta_end=beta_end)
    _save_checkpoint(
        actor=actor,
        actor_optimizer=actor_optimizer,
        outdir=outdir,
        name=name,
        steps_done=steps_done,
        episodes_done=episodes_done,
        algorithm="multi_target_dagger",
        extra_metadata={
            "dagger_round": int(dagger_rounds),
            "beta": float(final_beta),
            "dataset_size": len(aggregate_buffer),
            "aggregate_memory_budget": memory_budget_meta,
            "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
            "continue_weight": float(loss_config.continue_weight),
            "norm_floor": float(loss_config.norm_floor),
            "norm_floor_weight": float(loss_config.norm_floor_weight),
        },
    )


def train_dagger_online(
    env,
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    outdir: Path | str,
    logdir: Path | str,
    name: str,
    batch_size: int = 64,
    n_episodes: int = 600,
    warmstart_episodes: int = 100,
    update_after_steps: Optional[int] = None,
    update_every: int = 64,
    updates_per_step: int = 1,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    save_every_steps: int = 500,
    aggregate_memory_budget: int = 10000,
    rng: Optional[np.random.Generator] = None,
    continue_target_norm_threshold: Optional[float] = None,
    continue_weight: float = 1.0,
    norm_floor: float = 0.0,
    norm_floor_weight: float = 0.0,
) -> None:
    """Train a deterministic actor with online DAgger and replay-buffer updates."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1")
    if warmstart_episodes < 0:
        raise ValueError("warmstart_episodes must be non-negative")
    if warmstart_episodes > n_episodes:
        raise ValueError("warmstart_episodes must be less than or equal to n_episodes")
    if update_every < 1:
        raise ValueError("update_every must be at least 1")
    if updates_per_step < 1:
        raise ValueError("updates_per_step must be at least 1")
    if aggregate_memory_budget < 1:
        raise ValueError("aggregate_memory_budget must be at least 1")

    outdir = Path(outdir)
    logdir = Path(logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    if rng is None:
        rng = np.random.default_rng()
    if update_after_steps is None:
        update_after_steps = batch_size
    if update_after_steps < 1:
        raise ValueError("update_after_steps must be at least 1")

    if continue_target_norm_threshold is None:
        continue_target_norm_threshold = float(getattr(env, "stall_threshold", 1.0))
    loss_config = _build_supervision_loss_config(
        continue_target_norm_threshold=continue_target_norm_threshold,
        continue_weight=continue_weight,
        norm_floor=norm_floor,
        norm_floor_weight=norm_floor_weight,
    )

    steps_done = 0
    episodes_done = 0
    last_save_bucket = -1
    use_progress_bar = sys.stdout.isatty()
    csv_file_path = logdir / f"{name}_{date_time}_log.csv"
    aggregate_buffer = BehaviorCloningReplayBuffer(capacity=aggregate_memory_budget, include_z_flip=True)
    memory_budget_meta = int(aggregate_memory_budget)
    n_dagger_episodes = int(n_episodes - warmstart_episodes)

    warmstart_iter = tqdm(
        range(warmstart_episodes),
        desc="DAgger warmstart",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=1.0,
        disable=not use_progress_bar,
    )
    for warmstart_episode in warmstart_iter:
        if not use_progress_bar:
            print(f"Starting DAgger warmstart episode {warmstart_episode + 1}/{warmstart_episodes}", flush=True)

        try:
            episode_metrics = _run_expert_episode(
                env=env,
                actor=actor,
                actor_optimizer=actor_optimizer,
                batch_size=batch_size,
                loss_config=loss_config,
                aggregate_buffer=aggregate_buffer,
            )
        except Exception as exc:
            _log_episode_exception("DAgger online warmstart", warmstart_episode + 1, exc, env)
            continue
        steps_done += int(episode_metrics["steps_done"])
        episodes_done += 1
        last_save_bucket = _maybe_save_checkpoint(
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=outdir,
            name=name,
            steps_done=steps_done,
            episodes_done=episodes_done,
            save_every_steps=save_every_steps,
            last_save_bucket=last_save_bucket,
            algorithm="multi_target_dagger_online",
            extra_metadata={
                "dagger_episode": 0,
                "beta": 1.0,
                "dataset_size": len(aggregate_buffer),
                "aggregate_memory_budget": memory_budget_meta,
                "update_after_steps": int(update_after_steps),
                "update_every": int(update_every),
                "updates_per_step": int(updates_per_step),
                "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                "continue_weight": float(loss_config.continue_weight),
                "norm_floor": float(loss_config.norm_floor),
                "norm_floor_weight": float(loss_config.norm_floor_weight),
            },
        )
        _write_dagger_log_row(
            csv_file_path=csv_file_path,
            phase="warmstart",
            round_index=0,
            episode_index=episodes_done,
            image_file=env.current_neuron_info["neuron_name"],
            avg_reward=float(episode_metrics["episode_avg_reward"]),
            avg_loss=float(episode_metrics["episode_avg_loss"]),
            avg_step_mse=float(episode_metrics["episode_avg_step_mse"]),
            avg_step_norm=float(episode_metrics["episode_avg_expert_step_norm"]),
            policy_rollin_fraction=0.0,
            false_stop_rate=float(episode_metrics["false_stop_rate"]),
            false_continue_rate=float(episode_metrics["false_continue_rate"]),
            steps_done=steps_done,
            dataset_size=len(aggregate_buffer),
            complexity=float(env.dataset.alpha),
            beta=1.0,
        )

    dagger_iter = tqdm(
        range(n_dagger_episodes),
        desc="DAgger online",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=1.0,
        disable=not use_progress_bar,
    )
    for dagger_episode_idx in dagger_iter:
        beta = _compute_dagger_beta(
            dagger_episode_idx,
            dagger_rounds=max(1, n_dagger_episodes),
            beta_start=beta_start,
            beta_end=beta_end,
        )
        if not use_progress_bar:
            print(
                f"Starting DAgger online episode {dagger_episode_idx + 1}/{n_dagger_episodes} with beta={beta:.3f}",
                flush=True,
            )

        try:
            obs = env.reset(return_state=True)
            prev_rollin_action: Optional[torch.Tensor] = None
            episode_rewards: List[float] = []
            episode_step_norms: List[float] = []
            episode_losses: List[float] = []
            episode_step_mse: List[float] = []
            policy_steps = 0
            episode_steps = 0
            correct_continue_count = 0
            false_choose_stop_count = 0
            correct_stop_count = 0
            false_continue_count = 0

            for _step_idx in count():
                current_target_vectors = _current_target_vectors_from_env(env)
                aggregate_buffer.push(obs, current_target_vectors)

                action_source = _sample_rollin_source(beta=beta, rng=rng)
                if action_source == "expert":
                    rollin_action = select_expert_action(current_target_vectors, previous_action=prev_rollin_action)
                else:
                    rollin_action = _predict_policy_action(actor, obs)
                    policy_steps += 1
                episode_step_norms.append(float(torch.linalg.norm(rollin_action).item()))

                next_obs, reward, terminated, _truncated, info = env.step(rollin_action)
                steps_done += 1
                episode_steps += 1
                episode_rewards.append(_reward_to_float(reward))

                current_target_distance = _min_target_norm(info.get("current_target_vectors"))
                if action_source == "policy" and current_target_distance is not None:
                    status = info.get("status")
                    if current_target_distance > float(loss_config.continue_target_norm_threshold):
                        correct_continue_count += 1
                        if status == "choose_stop":
                            false_choose_stop_count += 1
                    else:
                        correct_stop_count += 1
                        if status == "continue":
                            false_continue_count += 1

                learning_started = len(aggregate_buffer) >= update_after_steps
                if learning_started and steps_done % update_every == 0:
                    for _update_idx in range(updates_per_step):
                        maybe_loss = _optimize_from_bc_replay_buffer(
                            actor=actor,
                            actor_optimizer=actor_optimizer,
                            replay_buffer=aggregate_buffer,
                            batch_size=batch_size,
                            transform=True,
                            loss_config=loss_config,
                        )
                        if maybe_loss is not None:
                            episode_losses.append(float(maybe_loss.loss))
                            episode_step_mse.append(float(maybe_loss.step_mse))

                    last_save_bucket = _maybe_save_checkpoint(
                        actor=actor,
                        actor_optimizer=actor_optimizer,
                        outdir=outdir,
                        name=name,
                        steps_done=steps_done,
                        episodes_done=episodes_done + 1,
                        save_every_steps=save_every_steps,
                        last_save_bucket=last_save_bucket,
                        algorithm="multi_target_dagger_online",
                        extra_metadata={
                            "dagger_episode": int(dagger_episode_idx + 1),
                            "beta": float(beta),
                            "dataset_size": len(aggregate_buffer),
                            "aggregate_memory_budget": memory_budget_meta,
                            "update_after_steps": int(update_after_steps),
                            "update_every": int(update_every),
                            "updates_per_step": int(updates_per_step),
                            "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                            "continue_weight": float(loss_config.continue_weight),
                            "norm_floor": float(loss_config.norm_floor),
                            "norm_floor_weight": float(loss_config.norm_floor_weight),
                        },
                    )

                if info["terminate_episode"]:
                    break

                if terminated:
                    obs = env.get_state()
                    prev_rollin_action = None
                else:
                    obs = next_obs
                    prev_rollin_action = rollin_action.detach().cpu()

            episodes_done += 1
            false_stop_rate = (
                float(false_choose_stop_count / correct_continue_count)
                if correct_continue_count > 0
                else 0.0
            )
            false_continue_rate = (
                float(false_continue_count / correct_stop_count)
                if correct_stop_count > 0
                else 0.0
            )
            _write_dagger_log_row(
                csv_file_path=csv_file_path,
                phase="dagger_online",
                round_index=0,
                episode_index=episodes_done,
                image_file=env.current_neuron_info["neuron_name"],
                avg_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                avg_loss=float(np.mean(episode_losses)) if episode_losses else 0.0,
                avg_step_mse=float(np.mean(episode_step_mse)) if episode_step_mse else 0.0,
                avg_step_norm=float(np.mean(episode_step_norms)) if episode_step_norms else 0.0,
                policy_rollin_fraction=float(policy_steps / episode_steps) if episode_steps > 0 else 0.0,
                false_stop_rate=false_stop_rate,
                false_continue_rate=false_continue_rate,
                steps_done=steps_done,
                dataset_size=len(aggregate_buffer),
                complexity=float(env.dataset.alpha),
                beta=float(beta),
            )
        except Exception as exc:
            _log_episode_exception("DAgger online", dagger_episode_idx + 1, exc, env)
            continue

    final_beta = _compute_dagger_beta(
        max(0, n_dagger_episodes - 1),
        dagger_rounds=max(1, n_dagger_episodes),
        beta_start=beta_start,
        beta_end=beta_end,
    )
    _save_checkpoint(
        actor=actor,
        actor_optimizer=actor_optimizer,
        outdir=outdir,
        name=name,
        steps_done=steps_done,
        episodes_done=episodes_done,
        algorithm="multi_target_dagger_online",
        extra_metadata={
            "dagger_episode": int(n_dagger_episodes),
            "beta": float(final_beta),
            "dataset_size": len(aggregate_buffer),
            "aggregate_memory_budget": memory_budget_meta,
            "update_after_steps": int(update_after_steps),
            "update_every": int(update_every),
            "updates_per_step": int(updates_per_step),
            "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
            "continue_weight": float(loss_config.continue_weight),
            "norm_floor": float(loss_config.norm_floor),
            "norm_floor_weight": float(loss_config.norm_floor_weight),
        },
    )


__all__ = [
    "train_dagger",
    "multi_target_direction_loss",
    "pad_target_candidate_batch",
    "select_expert_action",
    "train",
]