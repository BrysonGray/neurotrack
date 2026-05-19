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
import torch.nn.functional as F
from tqdm import tqdm

from neurotrack.data import save as data_save
from neurotrack.evaluation.metrics import evaluate_reconstruction
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
    stop_violation_weight: float = 1.0
    objective_mode: str = "norm_floor"
    continue_direction_weight: float = 1.0
    norm_cls_weight: float = 1.0
    norm_cls_temperature: float = 0.25
    norm_margin_weight: float = 1.0
    stop_margin: float = 0.1
    continue_margin: float = 0.1


@dataclass(frozen=True)
class SupervisionBatchStats:
    """Per-optimization batch metrics for loss and stop/continue diagnostics."""

    loss: float
    step_mse: float
    true_continue_count: int
    false_choose_stop_count: int
    true_stop_count: int
    false_continue_count: int


@dataclass(frozen=True)
class SupervisionBatchContext:
    """Shared per-batch tensors used by both loss and diagnostics."""

    direction_loss_per_sample: torch.Tensor
    min_target_norm: torch.Tensor
    has_valid_target: torch.Tensor
    continue_mask: torch.Tensor
    stop_mask: torch.Tensor
    pred_norms: torch.Tensor


def _build_supervision_loss_config(
    continue_target_norm_threshold: float,
    continue_weight: float,
    norm_floor: float,
    norm_floor_weight: float,
    stop_violation_weight: float,
    objective_mode: str = "norm_floor",
    continue_direction_weight: float = 1.0,
    norm_cls_weight: float = 1.0,
    norm_cls_temperature: float = 0.25,
    norm_margin_weight: float = 1.0,
    stop_margin: float = 0.1,
    continue_margin: float = 0.1,
) -> SupervisionLossConfig:
    threshold = float(continue_target_norm_threshold)
    continue_weight_f = float(continue_weight)
    norm_floor_f = float(norm_floor)
    norm_floor_weight_f = float(norm_floor_weight)
    stop_violation_weight_f = float(stop_violation_weight)
    objective_mode_s = str(objective_mode).strip().lower()
    continue_direction_weight_f = float(continue_direction_weight)
    norm_cls_weight_f = float(norm_cls_weight)
    norm_cls_temperature_f = float(norm_cls_temperature)
    norm_margin_weight_f = float(norm_margin_weight)
    stop_margin_f = float(stop_margin)
    continue_margin_f = float(continue_margin)

    if threshold < 0.0:
        raise ValueError("continue_target_norm_threshold must be non-negative")
    if continue_weight_f <= 0.0:
        raise ValueError("continue_weight must be positive")
    if norm_floor_f < 0.0:
        raise ValueError("norm_floor must be non-negative")
    if norm_floor_weight_f < 0.0:
        raise ValueError("norm_floor_weight must be non-negative")
    if stop_violation_weight_f < 0.0:
        raise ValueError("stop_violation_weight must be non-negative")
    if objective_mode_s not in {"norm_floor", "norm_classifier_margin", "direction_sse"}:
        raise ValueError(
            "objective_mode must be one of: {'norm_floor', 'norm_classifier_margin', 'direction_sse'}"
        )
    if continue_direction_weight_f < 0.0:
        raise ValueError("continue_direction_weight must be non-negative")
    if norm_cls_weight_f < 0.0:
        raise ValueError("norm_cls_weight must be non-negative")
    if norm_cls_temperature_f <= 0.0:
        raise ValueError("norm_cls_temperature must be positive")
    if norm_margin_weight_f < 0.0:
        raise ValueError("norm_margin_weight must be non-negative")
    if stop_margin_f < 0.0:
        raise ValueError("stop_margin must be non-negative")
    if continue_margin_f < 0.0:
        raise ValueError("continue_margin must be non-negative")

    return SupervisionLossConfig(
        continue_target_norm_threshold=threshold,
        continue_weight=continue_weight_f,
        norm_floor=norm_floor_f,
        norm_floor_weight=norm_floor_weight_f,
        stop_violation_weight=stop_violation_weight_f,
        objective_mode=objective_mode_s,
        continue_direction_weight=continue_direction_weight_f,
        norm_cls_weight=norm_cls_weight_f,
        norm_cls_temperature=norm_cls_temperature_f,
        norm_margin_weight=norm_margin_weight_f,
        stop_margin=stop_margin_f,
        continue_margin=continue_margin_f,
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


def _safe_float(value: object) -> Optional[float]:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_f):
        return None
    return value_f


def _csv_optional_float(value: Optional[float]) -> str | float:
    if value is None:
        return ""
    return float(value)


def _summarize_round_metric(
    values: Sequence[Optional[float]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    finite_values = [v for v in (_safe_float(value) for value in values) if v is not None]
    if len(finite_values) == 0:
        return None, None, None
    return (
        float(np.mean(finite_values)),
        float(np.min(finite_values)),
        float(np.max(finite_values)),
    )


def _compute_episode_tracing_metrics(
    env,
) -> dict[str, Optional[float]]:
    """Compute tracing metrics for one completed episode against ground truth."""
    if not bool(getattr(env, "has_ground_truth", False)):
        return {
            "bidirectional_distance": None,
            "precision": None,
            "coverage": None,
        }

    finished_paths = getattr(env, "finished_paths", None)
    full_tree = getattr(env, "full_tree", None)
    if not isinstance(finished_paths, list) or full_tree is None:
        return {
            "bidirectional_distance": None,
            "precision": None,
            "coverage": None,
        }

    pred_paths = [
        path.detach().cpu()
        for path in finished_paths
        if isinstance(path, torch.Tensor) and path.ndim == 2 and path.shape[1] == 3 and path.shape[0] >= 2
    ]
    if len(pred_paths) == 0:
        return {
            "bidirectional_distance": None,
            "precision": None,
            "coverage": None,
        }

    try:
        pred_swc = data_save.paths_to_swc(pred_paths)
        if len(pred_swc) == 0:
            return {
                "bidirectional_distance": None,
                "precision": None,
                "coverage": None,
            }

        gt_swc = torch.as_tensor(full_tree, dtype=torch.float32).detach().cpu().numpy().tolist()
        metrics = evaluate_reconstruction(
            pred_swc,
            gt_swc,
            return_l_measures=False,
        )
        return {
            "bidirectional_distance": _safe_float(metrics.get("bidirectional_distance")),
            "precision": _safe_float(metrics.get("precision")),
            "coverage": _safe_float(metrics.get("coverage")),
        }
    except Exception:
        return {
            "bidirectional_distance": None,
            "precision": None,
            "coverage": None,
        }


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


def _build_supervision_batch_context(
    pred_actions: torch.Tensor,
    target_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    continue_target_norm_threshold: float,
) -> SupervisionBatchContext:
    """Compute reusable supervision tensors once for loss and metrics."""
    direction_loss_per_sample = multi_target_direction_loss(
        pred_actions,
        target_tensor,
        valid_mask=target_mask,
        reduction="none",
    )
    min_target_norm, has_valid_target = _compute_min_target_norm(target_tensor, target_mask)
    continue_mask = has_valid_target & (min_target_norm > continue_target_norm_threshold)
    stop_mask = has_valid_target & ~continue_mask
    pred_norms = torch.linalg.norm(pred_actions, dim=1)
    return SupervisionBatchContext(
        direction_loss_per_sample=direction_loss_per_sample,
        min_target_norm=min_target_norm,
        has_valid_target=has_valid_target,
        continue_mask=continue_mask,
        stop_mask=stop_mask,
        pred_norms=pred_norms,
    )


def _norm_classifier_margin_loss(
    loss_config: SupervisionLossConfig,
    context: SupervisionBatchContext,
) -> torch.Tensor:
    direction_loss_per_sample = context.direction_loss_per_sample
    continue_mask = context.continue_mask
    stop_mask = context.stop_mask
    pred_norms = context.pred_norms

    zero = torch.zeros((), dtype=direction_loss_per_sample.dtype, device=direction_loss_per_sample.device)

    continue_direction_loss = zero
    if loss_config.continue_direction_weight > 0.0 and torch.any(continue_mask):
        continue_direction_loss = direction_loss_per_sample[continue_mask].mean()

    norm_cls_loss = zero
    if loss_config.norm_cls_weight > 0.0:
        logits = (
            pred_norms - loss_config.continue_target_norm_threshold
        ) / max(loss_config.norm_cls_temperature, torch.finfo(pred_norms.dtype).eps)
        if torch.any(continue_mask) and torch.any(stop_mask):
            cls_continue = F.binary_cross_entropy_with_logits(
                logits[continue_mask],
                torch.ones_like(logits[continue_mask]),
            )
            cls_stop = F.binary_cross_entropy_with_logits(
                logits[stop_mask],
                torch.zeros_like(logits[stop_mask]),
            )
            norm_cls_loss = (
                (loss_config.continue_weight * cls_continue) + cls_stop
            ) / (loss_config.continue_weight + 1.0)
        elif torch.any(continue_mask):
            norm_cls_loss = F.binary_cross_entropy_with_logits(
                logits[continue_mask],
                torch.ones_like(logits[continue_mask]),
            )
        elif torch.any(stop_mask):
            norm_cls_loss = F.binary_cross_entropy_with_logits(
                logits[stop_mask],
                torch.zeros_like(logits[stop_mask]),
            )

    norm_margin_loss = zero
    if loss_config.norm_margin_weight > 0.0:
        continue_margin_loss = zero
        stop_margin_loss = zero

        if torch.any(continue_mask):
            continue_floor = (
                loss_config.continue_target_norm_threshold + loss_config.continue_margin
            )
            continue_shortfall = torch.relu(continue_floor - pred_norms[continue_mask])
            continue_margin_loss = (continue_shortfall ** 2).mean()

        if torch.any(stop_mask):
            stop_ceiling = max(
                0.0,
                loss_config.continue_target_norm_threshold - loss_config.stop_margin,
            )
            stop_overshoot = torch.relu(pred_norms[stop_mask] - stop_ceiling)
            stop_margin_loss = (stop_overshoot ** 2).mean()

        if torch.any(continue_mask) and torch.any(stop_mask):
            norm_margin_loss = (
                (loss_config.continue_weight * continue_margin_loss) + stop_margin_loss
            ) / (loss_config.continue_weight + 1.0)
        elif torch.any(continue_mask):
            norm_margin_loss = continue_margin_loss
        elif torch.any(stop_mask):
            norm_margin_loss = stop_margin_loss

    # norm_floor_loss = zero
    # if loss_config.norm_floor > 0.0 and loss_config.norm_floor_weight > 0.0 and torch.any(continue_mask):
    #     norm_gap = torch.relu(loss_config.norm_floor - pred_norms[continue_mask])
    #     norm_floor_loss = (norm_gap ** 2).mean()

    loss = (
        (loss_config.continue_direction_weight * continue_direction_loss)
        + (loss_config.norm_cls_weight * norm_cls_loss)
        + (loss_config.norm_margin_weight * norm_margin_loss)
        # + (loss_config.norm_floor_weight * norm_floor_loss)
    )

    return loss

def _norm_floor_loss(
    loss_config: SupervisionLossConfig,
    context: SupervisionBatchContext,
) -> torch.Tensor:
    direction_loss_per_sample = context.direction_loss_per_sample
    continue_mask = context.continue_mask
    stop_mask = context.stop_mask
    pred_norms = context.pred_norms

    if torch.any(continue_mask) and torch.any(stop_mask):
        continue_direction_loss = direction_loss_per_sample[continue_mask].mean()
        stop_direction_loss = direction_loss_per_sample[stop_mask].mean()
        direction_loss = (
            (loss_config.continue_weight * continue_direction_loss) + stop_direction_loss
        ) / (loss_config.continue_weight + 1.0)
    elif torch.any(continue_mask):
        direction_loss = direction_loss_per_sample[continue_mask].mean()
    elif torch.any(stop_mask):
        direction_loss = direction_loss_per_sample[stop_mask].mean()
    else:
        direction_loss = direction_loss_per_sample.mean()

    norm_floor_loss = torch.zeros((), dtype=direction_loss.dtype, device=direction_loss.device)
    norm_stop_loss = torch.zeros((), dtype=direction_loss.dtype, device=direction_loss.device)

    if loss_config.norm_floor > 0.0 and loss_config.norm_floor_weight > 0.0:
        if torch.any(continue_mask):
            norm_gap = torch.relu(loss_config.norm_floor - pred_norms)
            norm_floor_loss = (norm_gap[continue_mask] ** 2).mean()

        if torch.any(stop_mask):
            stop_violation = torch.relu(
                pred_norms[stop_mask] - loss_config.continue_target_norm_threshold
            )
            norm_stop_loss = (stop_violation ** 2).mean()

    stop_regularization_weight = loss_config.norm_floor_weight * max(1.0, loss_config.stop_violation_weight)
    loss = (
        direction_loss
        + (loss_config.norm_floor_weight * norm_floor_loss)
        + (stop_regularization_weight * norm_stop_loss)
    )
    return loss


def _direction_sse_loss(context: SupervisionBatchContext) -> torch.Tensor:
    """Pure min-SSE supervision objective without auxiliary norm terms."""
    return context.direction_loss_per_sample.mean()


def _compute_supervision_loss(
    pred_actions: torch.Tensor,
    target_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    loss_config: SupervisionLossConfig,
) -> Tuple[torch.Tensor, SupervisionBatchContext]:
    """Pure loss path used by both optimization and loss-only testing."""
    context = _build_supervision_batch_context(
        pred_actions=pred_actions,
        target_tensor=target_tensor,
        target_mask=target_mask,
        continue_target_norm_threshold=loss_config.continue_target_norm_threshold,
    )

    if loss_config.objective_mode == "norm_classifier_margin":
        loss = _norm_classifier_margin_loss(loss_config=loss_config, context=context)
    elif loss_config.objective_mode == "norm_floor":
        loss = _norm_floor_loss(loss_config=loss_config, context=context)
    elif loss_config.objective_mode == "direction_sse":
        loss = _direction_sse_loss(context=context)
    else:
        raise ValueError(
            f"Unsupported objective_mode '{loss_config.objective_mode}'. "
            "Expected one of: {'norm_floor', 'norm_classifier_margin', 'direction_sse'}."
        )

    return loss, context


def _validate_tensors_before_backward(
    pred_actions: torch.Tensor,
    target_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    loss: torch.Tensor,
    context: SupervisionBatchContext,
) -> bool:
    """
    Validate all tensors involved in backward pass for corruption.
    
    Returns:
        True if all tensors are valid, False if any corruption detected.
        Logs detailed error info but does NOT raise exceptions.
    """
    errors = []
    
    # Check pred_actions
    if torch.isnan(pred_actions).any():
        nan_count = torch.isnan(pred_actions).sum().item()
        errors.append(f"pred_actions: {nan_count} NaN values out of {pred_actions.numel()}")
    if torch.isinf(pred_actions).any():
        inf_count = torch.isinf(pred_actions).sum().item()
        errors.append(f"pred_actions: {inf_count} Inf values out of {pred_actions.numel()}")
    if not pred_actions.requires_grad:
        errors.append(f"pred_actions: requires_grad=False (shape={pred_actions.shape})")
    
    # Check target_tensor
    if torch.isnan(target_tensor).any():
        nan_count = torch.isnan(target_tensor).sum().item()
        errors.append(f"target_tensor: {nan_count} NaN values out of {target_tensor.numel()}")
    if torch.isinf(target_tensor).any():
        inf_count = torch.isinf(target_tensor).sum().item()
        errors.append(f"target_tensor: {inf_count} Inf values out of {target_tensor.numel()}")
    
    # Check target_mask
    if target_mask.dtype != torch.bool:
        errors.append(f"target_mask: dtype={target_mask.dtype}, expected torch.bool")
    if not target_mask.any():
        errors.append(f"target_mask: all False (no valid targets in batch)")
    
    # Check loss
    if torch.isnan(loss):
        errors.append(f"loss: NaN (dtype={loss.dtype}, requires_grad={loss.requires_grad})")
    if torch.isinf(loss):
        errors.append(f"loss: Inf (dtype={loss.dtype}, requires_grad={loss.requires_grad})")
    if not loss.requires_grad:
        errors.append(f"loss: requires_grad=False (value={loss.item():.6f})")
    
    # Check context tensors
    if torch.isnan(context.direction_loss_per_sample).any():
        nan_count = torch.isnan(context.direction_loss_per_sample).sum().item()
        errors.append(f"direction_loss_per_sample: {nan_count} NaN values")
    if torch.isnan(context.min_target_norm).any():
        nan_count = torch.isnan(context.min_target_norm).sum().item()
        errors.append(f"min_target_norm: {nan_count} NaN values")
    if torch.isnan(context.pred_norms).any():
        nan_count = torch.isnan(context.pred_norms).sum().item()
        errors.append(f"pred_norms: {nan_count} NaN values")
    
    if errors:
        print("\n" + "="*70)
        print("TENSOR CORRUPTION DETECTED - SKIPPING BACKWARD PASS")
        print("="*70)
        for error in errors:
            print(f"  • {error}")
        loss_is_nan = bool(torch.isnan(loss).item())
        loss_value_str = "NaN" if loss_is_nan else f"{float(loss.item()):.8f}"
        print(f"  loss_value={loss_value_str}")
        print(f"  pred_actions.shape={pred_actions.shape}, device={pred_actions.device}")
        print(f"  target_tensor.shape={target_tensor.shape}, device={target_tensor.device}")
        print("="*70 + "\n")
        return False
    
    return True


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

    loss, context = _compute_supervision_loss(
        pred_actions=pred_actions,
        target_tensor=target_tensor,
        target_mask=target_mask,
        loss_config=loss_config,
    )

    (
        true_continue_count,
        false_choose_stop_count,
        true_stop_count,
        false_continue_count,
    ) = _compute_policy_error_counts(
        pred_actions=pred_actions,
        min_target_norm=context.min_target_norm,
        has_valid_target=context.has_valid_target,
        continue_target_norm_threshold=loss_config.continue_target_norm_threshold,
    )

    if loss.requires_grad:
        # === TENSOR VALIDATION BEFORE BACKWARD ===
        tensors_valid = _validate_tensors_before_backward(pred_actions, target_tensor, target_mask, loss, context)
        
        if tensors_valid:
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
        else:
            print("Skipping optimizer step for this batch due to tensor corruption")
            # Still compute metrics without updating weights
            pass

    step_mse = float(context.direction_loss_per_sample.mean().item())
    return SupervisionBatchStats(
        loss=float(loss.item()),
        step_mse=step_mse,
        true_continue_count=true_continue_count,
        false_choose_stop_count=false_choose_stop_count,
        true_stop_count=true_stop_count,
        false_continue_count=false_continue_count,
    )


def _current_target_vectors_from_env(env) -> torch.Tensor:
    current_target_vectors = env.target_vectors
    if current_target_vectors is None:
        current_target_vectors = torch.zeros((1, 3), dtype=torch.float32)
    return torch.as_tensor(current_target_vectors, dtype=torch.float32).view(-1, 3)


def _predict_policy_action(actor: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    model_device = next(actor.parameters()).device
    obs_model = prepare_observation_for_model(obs.detach(), device=model_device, model_dtype=dtype)
    with torch.no_grad():
        pred_actions = actor(obs_model)
    pred_actions = _ensure_action_batch(pred_actions)
    return pred_actions[0].detach().to(dtype=torch.float32).cpu()


def _run_supervision_epoch(
    actor: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    aggregate_buffer: BehaviorCloningReplayBuffer,
    batch_size: int,
    transform: bool,
    loss_config: SupervisionLossConfig,
) -> List[SupervisionBatchStats]:
    """
    Run one deterministic (no-replacement) pass over a snapshot of the replay buffer.

    This function samples the current contents of `aggregate_buffer` exactly once
    (in a random permutation) and runs optimization on disjoint minibatches so
    that no sample is reused within the epoch and no sample is overwritten
    before it has been used.
    """
    real_size = len(aggregate_buffer)
    if real_size == 0:
        return []

    perm = torch.randperm(real_size).tolist()

    batch_stats: List[SupervisionBatchStats] = []
    for start in range(0, real_size, batch_size):
        chunk = perm[start : start + batch_size]
        try:
            obs_batch, target_tensor, target_mask = aggregate_buffer.sample_indices(
                logical_indices=chunk,
                transform=transform,
            )
        except Exception:
            continue

        maybe_stats = _optimize_prepared_batch(
            actor=actor,
            actor_optimizer=actor_optimizer,
            obs_tensor=obs_batch,
            target_tensor=target_tensor,
            target_mask=target_mask,
            loss_config=loss_config,
        )
        batch_stats.append(maybe_stats)

    return batch_stats


def _reward_to_float(reward: torch.Tensor | float) -> float:
    return float(torch.as_tensor(reward, dtype=torch.float32).item())


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
    aggregate_buffer: BehaviorCloningReplayBuffer,
    steps_budget: Optional[int] = None,
    start_obs: Optional[torch.Tensor] = None,
    prev_expert_action: Optional[torch.Tensor] = None,
) -> Tuple[dict[str, float | int], dict[str, object]]:
    if steps_budget is not None and int(steps_budget) <= 0:
        return {
            "episode_avg_reward": 0.0,
            "episode_avg_expert_step_norm": 0.0,
            "steps_done": 0,
        }

    if start_obs is None:
        obs = env.reset(return_state=True)
        prev_expert_action = None
    else:
        obs = start_obs
    episode_rewards: List[float] = []
    episode_step_norms: List[float] = []
    steps_done = 0

    for _step_idx in count():
        current_target_vectors = _current_target_vectors_from_env(env)
        aggregate_buffer.push(obs, current_target_vectors)

        expert_action = select_expert_action(current_target_vectors, previous_action=prev_expert_action)
        episode_step_norms.append(float(torch.linalg.norm(expert_action).item()))

        next_obs, reward, terminated, _truncated, info = env.step(expert_action)
        steps_done += 1
        episode_rewards.append(_reward_to_float(reward))

        if steps_budget is not None and steps_done >= int(steps_budget):
            # Pause collection mid-episode; return resume state so trainer
            # can continue from this exact environment state after optimization.
            return (
                {
                    "episode_avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                    "episode_avg_expert_step_norm": float(np.mean(episode_step_norms)) if episode_step_norms else 0.0,
                    "steps_done": int(steps_done),
                },
                {
                    "obs": obs if (terminated := False) else obs,
                    "prev_action": prev_expert_action,
                    "terminated": False,
                },
            )

        if info["terminate_episode"]:
            break

        if terminated:
            obs = env.get_state()
            prev_expert_action = None
        else:
            obs = next_obs
            prev_expert_action = expert_action.detach().cpu()

    # Episode finished normally or hit terminal condition.
    return (
        {
            "episode_avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episode_avg_expert_step_norm": float(np.mean(episode_step_norms)) if episode_step_norms else 0.0,
            "steps_done": int(steps_done),
        },
        {
            "obs": env.get_state(),
            "prev_action": None,
            "terminated": True,
        },
    )


def _run_dagger_collection_episode(
    env,
    actor: torch.nn.Module,
    aggregate_buffer: BehaviorCloningReplayBuffer,
    beta: float,
    rng: np.random.Generator,
    steps_budget: Optional[int] = None,
    start_obs: Optional[torch.Tensor] = None,
    prev_rollin_action: Optional[torch.Tensor] = None,
) -> Tuple[dict[str, float | int], dict[str, object]]:
    if steps_budget is not None and int(steps_budget) <= 0:
        return {
            "episode_avg_reward": 0.0,
            "episode_avg_expert_step_norm": 0.0,
            "policy_rollin_fraction": 0.0,
            "steps_done": 0,
            "episode_bidirectional_distance": None,
            "episode_precision": None,
            "episode_coverage": None,
        }

    if start_obs is None:
        obs = env.reset(return_state=True)
        prev_rollin_action = None
    else:
        obs = start_obs
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

        if steps_budget is not None and steps_done >= int(steps_budget):
            # Pause mid-episode; return resume info so trainer can continue
            return (
                {
                    "episode_avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                    "episode_avg_expert_step_norm": float(np.mean(episode_step_norms)) if episode_step_norms else 0.0,
                    "policy_rollin_fraction": float(policy_steps / steps_done) if steps_done > 0 else 0.0,
                    "steps_done": int(steps_done),
                    "episode_bidirectional_distance": None,
                    "episode_precision": None,
                    "episode_coverage": None,
                },
                {
                    "obs": obs,
                    "prev_action": prev_rollin_action,
                    "terminated": False,
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

    tracing_metrics = _compute_episode_tracing_metrics(env)

    return (
        {
            "episode_avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episode_avg_expert_step_norm": float(np.mean(episode_step_norms)) if episode_step_norms else 0.0,
            "policy_rollin_fraction": float(policy_steps / steps_done) if steps_done > 0 else 0.0,
            "steps_done": int(steps_done),
            "episode_bidirectional_distance": tracing_metrics["bidirectional_distance"],
            "episode_precision": tracing_metrics["precision"],
            "episode_coverage": tracing_metrics["coverage"],
        },
        {
            "obs": env.get_state(),
            "prev_action": None,
            "terminated": True,
        },
    )


def _write_bc_log_row(
    csv_file_path: Path,
    buffer_epoch_index: int,
    avg_reward: float,
    avg_loss: float,
    avg_step_mse: float,
    avg_step_norm: float,
    false_stop_rate: float,
    false_continue_rate: float,
    steps_done: int,
    episodes_done: int,
    complexity: float,
) -> None:
    file_exists = csv_file_path.exists()
    with open(csv_file_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow([
                "buffer_epoch_index",
                "episode_avg_reward",
                "episode_avg_loss",
                "episode_avg_step_mse",
                "episode_avg_expert_step_norm",
                "false_stop_rate",
                "false_continue_rate",
                "steps_done",
                "episodes_done",
                "complexity",
            ])
        writer.writerow([
            int(buffer_epoch_index),
            avg_reward,
            avg_loss,
            avg_step_mse,
            avg_step_norm,
            false_stop_rate,
            false_continue_rate,
            int(steps_done),
            int(episodes_done),
            complexity,
        ])


def _write_dagger_log_row(
    csv_file_path: Path,
    phase: str,
    round_index: int,
    buffer_epoch_index: int,
    avg_reward: float,
    avg_loss: float,
    avg_step_mse: float,
    avg_step_norm: float,
    policy_rollin_fraction: float,
    false_stop_rate: float,
    false_continue_rate: float,
    steps_done: int,
    episodes_done: int,
    dataset_size: int,
    complexity: float,
    beta: float,
    round_bidirectional_distance_avg: Optional[float] = None,
    round_bidirectional_distance_min: Optional[float] = None,
    round_bidirectional_distance_max: Optional[float] = None,
    round_precision_avg: Optional[float] = None,
    round_precision_min: Optional[float] = None,
    round_precision_max: Optional[float] = None,
    round_coverage_avg: Optional[float] = None,
    round_coverage_min: Optional[float] = None,
    round_coverage_max: Optional[float] = None,
) -> None:
    file_exists = csv_file_path.exists()
    with open(csv_file_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow([
                "phase",
                "round",
                "buffer_epoch_index",
                "episode_avg_reward",
                "episode_avg_loss",
                "episode_avg_step_mse",
                "episode_avg_expert_step_norm",
                "policy_rollin_fraction",
                "false_stop_rate",
                "false_continue_rate",
                "steps_done",
                "episodes_done",
                "dataset_size",
                "complexity",
                "beta",
                "round_bidirectional_distance_avg",
                "round_bidirectional_distance_min",
                "round_bidirectional_distance_max",
                "round_precision_avg",
                "round_precision_min",
                "round_precision_max",
                "round_coverage_avg",
                "round_coverage_min",
                "round_coverage_max",
            ])
        writer.writerow([
            phase,
            int(round_index),
            int(buffer_epoch_index),
            avg_reward,
            avg_loss,
            avg_step_mse,
            avg_step_norm,
            policy_rollin_fraction,
            false_stop_rate,
            false_continue_rate,
            int(steps_done),
            int(episodes_done),
            int(dataset_size),
            complexity,
            beta,
            _csv_optional_float(round_bidirectional_distance_avg),
            _csv_optional_float(round_bidirectional_distance_min),
            _csv_optional_float(round_bidirectional_distance_max),
            _csv_optional_float(round_precision_avg),
            _csv_optional_float(round_precision_min),
            _csv_optional_float(round_precision_max),
            _csv_optional_float(round_coverage_avg),
            _csv_optional_float(round_coverage_min),
            _csv_optional_float(round_coverage_max),
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
    save_every_buffer_fills: int,
    buffer_fills_done: int,
    last_save_bucket: int,
    algorithm: str,
    extra_metadata: Optional[dict[str, float | int | str]] = None,
) -> int:
    if save_every_buffer_fills <= 0:
        return last_save_bucket
    current_bucket = int(buffer_fills_done) // int(save_every_buffer_fills)
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
    total_steps: int = 100000,
    buffer_capacity: int = 100000,
    epochs_per_buffer_fill: int = 1,
    save_every_buffer_fills: int = 1,
    continue_target_norm_threshold: Optional[float] = None,
    continue_weight: float = 1.0,
    norm_floor: float = 0.0,
    norm_floor_weight: float = 0.0,
    stop_violation_weight: float = 1.0,
    objective_mode: str = "norm_floor",
    continue_direction_weight: float = 1.0,
    norm_cls_weight: float = 1.0,
    norm_cls_temperature: float = 0.25,
    norm_margin_weight: float = 1.0,
    stop_margin: float = 0.1,
    continue_margin: float = 0.1,
) -> None:
    """Train a deterministic actor with multi-target behavior cloning."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if buffer_capacity < 1:
        raise ValueError("buffer_capacity must be at least 1")
    if epochs_per_buffer_fill < 1:
        raise ValueError("epochs_per_buffer_fill must be at least 1")

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
        stop_violation_weight=stop_violation_weight,
        objective_mode=objective_mode,
        continue_direction_weight=continue_direction_weight,
        norm_cls_weight=norm_cls_weight,
        norm_cls_temperature=norm_cls_temperature,
        norm_margin_weight=norm_margin_weight,
        stop_margin=stop_margin,
        continue_margin=continue_margin,
    )

    steps_done = 0
    episodes_done = 0
    buffer_fills_done = 0
    buffer_epoch_index = 0
    last_save_bucket = -1
    use_progress_bar = sys.stdout.isatty()
    csv_file_path = logdir / f"{name}_{date_time}_log.csv"

    # Persistent FIFO memory buffer used for the entire BC run.
    memory_buffer = BehaviorCloningReplayBuffer(capacity=buffer_capacity, include_z_flip=True)
    rewards: List[float] = []
    step_norms: List[float] = []
    steps_since_last_optimization = 0
    fill_rewards: List[float] = []
    fill_step_norms: List[float] = []

    resume_state: Optional[dict[str, object]] = None

    while steps_done < total_steps:
        # Limit collection so we pause exactly when enough new samples
        # have been accumulated for an optimization pass.
        try:
            episode_metrics, resume_state = _run_expert_episode(
                env=env,
                aggregate_buffer=memory_buffer,
                steps_budget=min(total_steps - steps_done, buffer_capacity - steps_since_last_optimization),
                start_obs=(None if resume_state is None or resume_state.get("terminated", True) else resume_state.get("obs")),
                prev_expert_action=(None if resume_state is None else resume_state.get("prev_action")),
            )
        except Exception as exc:
            _log_episode_exception("BC", episodes_done + 1, exc, env)
            continue

        episode_steps = int(episode_metrics["steps_done"])
        if episode_steps <= 0:
            break

        steps_done += episode_steps
        episodes_done += 1
        rewards.append(float(episode_metrics["episode_avg_reward"]))
        step_norms.append(float(episode_metrics["episode_avg_expert_step_norm"]))
        fill_rewards.append(float(episode_metrics["episode_avg_reward"]))
        fill_step_norms.append(float(episode_metrics["episode_avg_expert_step_norm"]))
        steps_since_last_optimization += episode_steps

        # When we've collected enough new samples, pause collection
        # and run supervised optimization epochs on the persistent buffer.
        if steps_since_last_optimization >= buffer_capacity:
            for _epoch_idx in range(epochs_per_buffer_fill):
                epoch_batch_stats = _run_supervision_epoch(
                    actor=actor,
                    actor_optimizer=actor_optimizer,
                    aggregate_buffer=memory_buffer,
                    batch_size=batch_size,
                    transform=True,
                    loss_config=loss_config,
                )

                (
                    true_continue_count,
                    false_choose_stop_count,
                    true_stop_count,
                    false_continue_count,
                ) = _aggregate_supervision_stats(epoch_batch_stats)

                buffer_epoch_index += 1
                _write_bc_log_row(
                    csv_file_path=csv_file_path,
                    buffer_epoch_index=buffer_epoch_index,
                    avg_reward=float(np.mean(fill_rewards)) if fill_rewards else 0.0,
                    avg_loss=float(np.mean([stat.loss for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                    avg_step_mse=float(np.mean([stat.step_mse for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                    avg_step_norm=float(np.mean(fill_step_norms)) if fill_step_norms else 0.0,
                    false_stop_rate=_safe_rate(false_choose_stop_count, true_continue_count),
                    false_continue_rate=_safe_rate(false_continue_count, true_stop_count),
                    steps_done=steps_done,
                    episodes_done=episodes_done,
                    complexity=float(env.dataset.alpha),
                )

            buffer_fills_done += 1
            last_save_bucket = _maybe_save_checkpoint(
                actor=actor,
                actor_optimizer=actor_optimizer,
                outdir=outdir,
                name=name,
                steps_done=steps_done,
                episodes_done=episodes_done,
                save_every_buffer_fills=save_every_buffer_fills,
                buffer_fills_done=buffer_fills_done,
                last_save_bucket=last_save_bucket,
                algorithm="multi_target_behavior_cloning",
                extra_metadata={
                    "buffer_fills_done": int(buffer_fills_done),
                    "buffer_capacity": int(buffer_capacity),
                    "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                    "continue_weight": float(loss_config.continue_weight),
                    "norm_floor": float(loss_config.norm_floor),
                    "norm_floor_weight": float(loss_config.norm_floor_weight),
                },
            )

            # Reset fill counters and resume collection on the same buffer and env
            steps_since_last_optimization = 0
            fill_rewards.clear()
            fill_step_norms.clear()

            # If the previous fill paused mid-episode, keep resume_state so
            # collection continues without resetting the environment. If the
            # episode terminated, clear resume_state so the next episode
            # starts fresh.
            if resume_state is not None and resume_state.get("terminated", False):
                resume_state = None

    # Final optimization epoch(s) at end of run (if there are unoptimized samples)
    if steps_since_last_optimization > 0:
        for _epoch_idx in range(epochs_per_buffer_fill):
            epoch_batch_stats = _run_supervision_epoch(
                actor=actor,
                actor_optimizer=actor_optimizer,
                aggregate_buffer=memory_buffer,
                batch_size=batch_size,
                transform=True,
                loss_config=loss_config,
            )

            (
                true_continue_count,
                false_choose_stop_count,
                true_stop_count,
                false_continue_count,
            ) = _aggregate_supervision_stats(epoch_batch_stats)

            buffer_epoch_index += 1
            _write_bc_log_row(
                csv_file_path=csv_file_path,
                buffer_epoch_index=buffer_epoch_index,
                avg_reward=float(np.mean(fill_rewards)) if fill_rewards else 0.0,
                avg_loss=float(np.mean([stat.loss for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                avg_step_mse=float(np.mean([stat.step_mse for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                avg_step_norm=float(np.mean(fill_step_norms)) if fill_step_norms else 0.0,
                false_stop_rate=_safe_rate(false_choose_stop_count, true_continue_count),
                false_continue_rate=_safe_rate(false_continue_count, true_stop_count),
                steps_done=steps_done,
                episodes_done=episodes_done,
                complexity=float(env.dataset.alpha),
            )

        buffer_fills_done += 1
        last_save_bucket = _maybe_save_checkpoint(
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=outdir,
            name=name,
            steps_done=steps_done,
            episodes_done=episodes_done,
            save_every_buffer_fills=save_every_buffer_fills,
            buffer_fills_done=buffer_fills_done,
            last_save_bucket=last_save_bucket,
            algorithm="multi_target_behavior_cloning",
            extra_metadata={
                "buffer_fills_done": int(buffer_fills_done),
                "buffer_capacity": int(buffer_capacity),
                "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                "continue_weight": float(loss_config.continue_weight),
                "norm_floor": float(loss_config.norm_floor),
                "norm_floor_weight": float(loss_config.norm_floor_weight),
            },
        )

    _save_checkpoint(
        actor=actor,
        actor_optimizer=actor_optimizer,
        outdir=outdir,
        name=name,
        steps_done=steps_done,
        episodes_done=episodes_done,
        algorithm="multi_target_behavior_cloning",
        extra_metadata={
            "buffer_fills_done": int(buffer_fills_done),
            "buffer_capacity": int(buffer_capacity),
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
    warmstart_steps: int = 100000,
    dagger_rounds: int = 5,
    steps_per_round: int = 100000,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    save_every_buffer_fills: int = 1,
    dynamic_complexity: bool = True,
    buffer_capacity: int = 100000,
    epochs_per_buffer_fill: int = 1,
    rng: Optional[np.random.Generator] = None,
    continue_target_norm_threshold: Optional[float] = None,
    continue_weight: float = 1.0,
    norm_floor: float = 0.0,
    norm_floor_weight: float = 0.0,
    stop_violation_weight: float = 1.0,
    objective_mode: str = "norm_floor",
    continue_direction_weight: float = 1.0,
    norm_cls_weight: float = 1.0,
    norm_cls_temperature: float = 0.25,
    norm_margin_weight: float = 1.0,
    stop_margin: float = 0.1,
    continue_margin: float = 0.1,
) -> None:
    """Train a deterministic actor with DAgger using aggregated multi-target supervision."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmstart_steps < 0:
        raise ValueError("warmstart_steps must be non-negative")
    if dagger_rounds < 1:
        raise ValueError("dagger_rounds must be at least 1")
    if steps_per_round < 1:
        raise ValueError("steps_per_round must be at least 1")
    if buffer_capacity < 1:
        raise ValueError("buffer_capacity must be at least 1")
    if epochs_per_buffer_fill < 1:
        raise ValueError("epochs_per_buffer_fill must be at least 1")
    if warmstart_steps > 0 and warmstart_steps < buffer_capacity:
        raise ValueError("warmstart_steps must be >= buffer_capacity to ensure full-buffer training")
    if steps_per_round < buffer_capacity:
        raise ValueError("steps_per_round must be >= buffer_capacity to ensure full-buffer training")

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
        stop_violation_weight=stop_violation_weight,
        objective_mode=objective_mode,
        continue_direction_weight=continue_direction_weight,
        norm_cls_weight=norm_cls_weight,
        norm_cls_temperature=norm_cls_temperature,
        norm_margin_weight=norm_margin_weight,
        stop_margin=stop_margin,
        continue_margin=continue_margin,
    )

    steps_done = 0
    episodes_done = 0
    buffer_fills_done = 0
    buffer_epoch_index = 0
    last_save_bucket = -1
    use_progress_bar = sys.stdout.isatty()
    csv_file_path = logdir / f"{name}_{date_time}_log.csv"
    memory_budget_meta = int(buffer_capacity)

    # Persistent FIFO memory buffer used across warmstart and all DAgger rounds.
    memory_buffer = BehaviorCloningReplayBuffer(capacity=buffer_capacity, include_z_flip=True)
    steps_since_last_optimization = 0
    # Round-level accumulators (not cleared between fills)
    # We'll also maintain per-fill accumulators used for logging when an optimization runs.

    if warmstart_steps > 0:
        warmstart_steps_done = 0
        warm_rewards: List[float] = []
        warm_step_norms: List[float] = []
        warm_fill_rewards: List[float] = []
        warm_fill_step_norms: List[float] = []

        warm_resume_state: Optional[dict[str, object]] = None

        while warmstart_steps_done < warmstart_steps:
            try:
                episode_metrics, warm_resume_state = _run_expert_episode(
                    env=env,
                    aggregate_buffer=memory_buffer,
                    steps_budget=min(warmstart_steps - warmstart_steps_done, buffer_capacity - steps_since_last_optimization),
                    start_obs=(None if warm_resume_state is None or warm_resume_state.get("terminated", True) else warm_resume_state.get("obs")),
                    prev_expert_action=(None if warm_resume_state is None else warm_resume_state.get("prev_action")),
                )
            except Exception as exc:
                _log_episode_exception("DAgger warmstart", episodes_done + 1, exc, env)
                continue

            episode_steps = int(episode_metrics["steps_done"])
            if episode_steps <= 0:
                break
            warmstart_steps_done += episode_steps
            steps_done += episode_steps
            episodes_done += 1
            warm_rewards.append(float(episode_metrics["episode_avg_reward"]))
            warm_step_norms.append(float(episode_metrics["episode_avg_expert_step_norm"]))
            # Clear warm_resume_state if episode terminated
            if warm_resume_state is not None and warm_resume_state.get("terminated", False):
                warm_resume_state = None
            warm_fill_rewards.append(float(episode_metrics["episode_avg_reward"]))
            warm_fill_step_norms.append(float(episode_metrics["episode_avg_expert_step_norm"]))
            steps_since_last_optimization += episode_steps

            if steps_since_last_optimization >= buffer_capacity:
                for _epoch_idx in range(epochs_per_buffer_fill):
                    epoch_batch_stats = _run_supervision_epoch(
                        actor=actor,
                        actor_optimizer=actor_optimizer,
                        aggregate_buffer=memory_buffer,
                        batch_size=batch_size,
                        transform=True,
                        loss_config=loss_config,
                    )
                    (
                        true_continue_count,
                        false_choose_stop_count,
                        true_stop_count,
                        false_continue_count,
                    ) = _aggregate_supervision_stats(epoch_batch_stats)

                    buffer_epoch_index += 1
                    _write_dagger_log_row(
                        csv_file_path=csv_file_path,
                        phase="warmstart",
                        round_index=0,
                        buffer_epoch_index=buffer_epoch_index,
                        avg_reward=float(np.mean(warm_fill_rewards)) if warm_fill_rewards else 0.0,
                        avg_loss=float(np.mean([stat.loss for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                        avg_step_mse=float(np.mean([stat.step_mse for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                        avg_step_norm=float(np.mean(warm_fill_step_norms)) if warm_fill_step_norms else 0.0,
                        policy_rollin_fraction=0.0,
                        false_stop_rate=_safe_rate(false_choose_stop_count, true_continue_count),
                        false_continue_rate=_safe_rate(false_continue_count, true_stop_count),
                        steps_done=steps_done,
                        episodes_done=episodes_done,
                        dataset_size=len(memory_buffer),
                        complexity=float(env.dataset.alpha),
                        beta=1.0,
                    )

                buffer_fills_done += 1
                last_save_bucket = _maybe_save_checkpoint(
                    actor=actor,
                    actor_optimizer=actor_optimizer,
                    outdir=outdir,
                    name=name,
                    steps_done=steps_done,
                    episodes_done=episodes_done,
                    save_every_buffer_fills=save_every_buffer_fills,
                    buffer_fills_done=buffer_fills_done,
                    last_save_bucket=last_save_bucket,
                    algorithm="multi_target_dagger",
                    extra_metadata={
                        "dagger_round": 0,
                        "beta": 1.0,
                        "dataset_size": len(memory_buffer),
                        "buffer_capacity": memory_budget_meta,
                        "buffer_fills_done": int(buffer_fills_done),
                        "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                        "continue_weight": float(loss_config.continue_weight),
                        "norm_floor": float(loss_config.norm_floor),
                        "norm_floor_weight": float(loss_config.norm_floor_weight),
                    },
                )

                steps_since_last_optimization = 0
                warm_fill_rewards.clear()
                warm_fill_step_norms.clear()

        # Final warmstart optimization if there are leftovers
        if steps_since_last_optimization > 0:
            for _epoch_idx in range(epochs_per_buffer_fill):
                epoch_batch_stats = _run_supervision_epoch(
                    actor=actor,
                    actor_optimizer=actor_optimizer,
                    aggregate_buffer=memory_buffer,
                    batch_size=batch_size,
                    transform=True,
                    loss_config=loss_config,
                )
                (
                    true_continue_count,
                    false_choose_stop_count,
                    true_stop_count,
                    false_continue_count,
                ) = _aggregate_supervision_stats(epoch_batch_stats)

                buffer_epoch_index += 1
                _write_dagger_log_row(
                    csv_file_path=csv_file_path,
                    phase="warmstart",
                    round_index=0,
                    buffer_epoch_index=buffer_epoch_index,
                    avg_reward=float(np.mean(warm_fill_rewards)) if warm_fill_rewards else 0.0,
                    avg_loss=float(np.mean([stat.loss for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                    avg_step_mse=float(np.mean([stat.step_mse for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                    avg_step_norm=float(np.mean(warm_fill_step_norms)) if warm_fill_step_norms else 0.0,
                    policy_rollin_fraction=0.0,
                    false_stop_rate=_safe_rate(false_choose_stop_count, true_continue_count),
                    false_continue_rate=_safe_rate(false_continue_count, true_stop_count),
                    steps_done=steps_done,
                    episodes_done=episodes_done,
                    dataset_size=len(memory_buffer),
                    complexity=float(env.dataset.alpha),
                    beta=1.0,
                )

            buffer_fills_done += 1
            last_save_bucket = _maybe_save_checkpoint(
                actor=actor,
                actor_optimizer=actor_optimizer,
                outdir=outdir,
                name=name,
                steps_done=steps_done,
                episodes_done=episodes_done,
                save_every_buffer_fills=save_every_buffer_fills,
                buffer_fills_done=buffer_fills_done,
                last_save_bucket=last_save_bucket,
                algorithm="multi_target_dagger",
                extra_metadata={
                    "dagger_round": 0,
                    "beta": 1.0,
                    "dataset_size": len(memory_buffer),
                    "buffer_capacity": memory_budget_meta,
                    "buffer_fills_done": int(buffer_fills_done),
                    "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                    "continue_weight": float(loss_config.continue_weight),
                    "norm_floor": float(loss_config.norm_floor),
                    "norm_floor_weight": float(loss_config.norm_floor_weight),
                },
            )
            steps_since_last_optimization = 0
            warm_fill_rewards.clear()
            warm_fill_step_norms.clear()
            if warm_resume_state is not None and warm_resume_state.get("terminated", False):
                warm_resume_state = None

    for round_index in range(dagger_rounds):
        beta = _compute_dagger_beta(round_index, dagger_rounds=dagger_rounds, beta_start=beta_start, beta_end=beta_end)
        if not use_progress_bar:
            print(f"Starting DAgger round {round_index + 1}/{dagger_rounds} with beta={beta:.3f}", flush=True)

        # Per-round aggregators (persist across fills within the round)
        round_rewards: List[float] = []
        round_step_norms: List[float] = []
        round_policy_fractions: List[float] = []
        round_bidirectional_distances: List[Optional[float]] = []
        round_precisions: List[Optional[float]] = []
        round_coverages: List[Optional[float]] = []

        # Per-fill accumulators used when we trigger optimization
        fill_rewards: List[float] = []
        fill_step_norms: List[float] = []
        fill_policy_fractions: List[float] = []
        fill_bidirectional_distances: List[Optional[float]] = []
        fill_precisions: List[Optional[float]] = []
        fill_coverages: List[Optional[float]] = []

        round_collection_steps = 0
        round_resume_state: Optional[dict[str, object]] = None

        while round_collection_steps < steps_per_round:
            try:
                episode_metrics, round_resume_state = _run_dagger_collection_episode(
                    env=env,
                    actor=actor,
                    aggregate_buffer=memory_buffer,
                    beta=beta,
                    rng=rng,
                    steps_budget=min(steps_per_round - round_collection_steps, buffer_capacity - steps_since_last_optimization),
                    start_obs=(None if round_resume_state is None or round_resume_state.get("terminated", True) else round_resume_state.get("obs")),
                    prev_rollin_action=(None if round_resume_state is None else round_resume_state.get("prev_action")),
                )
            except Exception as exc:
                _log_episode_exception(
                    f"DAgger rollout round {round_index + 1}",
                    episodes_done + 1,
                    exc,
                    env,
                )
                continue

            episode_steps = int(episode_metrics["steps_done"])
            if episode_steps <= 0:
                break
            steps_done += episode_steps
            episodes_done += 1
            round_collection_steps += episode_steps

            r_reward = float(episode_metrics["episode_avg_reward"])
            # Clear resume_state if episode terminated so next episode starts fresh
            if round_resume_state is not None and round_resume_state.get("terminated", False):
                round_resume_state = None
            r_step_norm = float(episode_metrics["episode_avg_expert_step_norm"])
            r_policy_frac = float(episode_metrics["policy_rollin_fraction"])
            r_bidist = _safe_float(episode_metrics.get("episode_bidirectional_distance"))
            r_prec = _safe_float(episode_metrics.get("episode_precision"))
            r_cov = _safe_float(episode_metrics.get("episode_coverage"))

            round_rewards.append(r_reward)
            round_step_norms.append(r_step_norm)
            round_policy_fractions.append(r_policy_frac)
            round_bidirectional_distances.append(r_bidist)
            round_precisions.append(r_prec)
            round_coverages.append(r_cov)

            fill_rewards.append(r_reward)
            fill_step_norms.append(r_step_norm)
            fill_policy_fractions.append(r_policy_frac)
            fill_bidirectional_distances.append(r_bidist)
            fill_precisions.append(r_prec)
            fill_coverages.append(r_cov)

            steps_since_last_optimization += episode_steps

            # If buffer reached the threshold of new samples, run optimization
            if steps_since_last_optimization >= buffer_capacity:
                for _epoch_idx in range(epochs_per_buffer_fill):
                    epoch_batch_stats = _run_supervision_epoch(
                        actor=actor,
                        actor_optimizer=actor_optimizer,
                        aggregate_buffer=memory_buffer,
                        batch_size=batch_size,
                        transform=True,
                        loss_config=loss_config,
                    )
                    (
                        round_true_continue_count,
                        round_false_choose_stop_count,
                        round_true_stop_count,
                        round_false_continue_count,
                    ) = _aggregate_supervision_stats(epoch_batch_stats)

                    buffer_epoch_index += 1
                    _write_dagger_log_row(
                        csv_file_path=csv_file_path,
                        phase="dagger",
                        round_index=round_index + 1,
                        buffer_epoch_index=buffer_epoch_index,
                        avg_reward=float(np.mean(fill_rewards)) if fill_rewards else 0.0,
                        avg_loss=float(np.mean([stat.loss for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                        avg_step_mse=float(np.mean([stat.step_mse for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                        avg_step_norm=float(np.mean(fill_step_norms)) if fill_step_norms else 0.0,
                        policy_rollin_fraction=float(np.mean(fill_policy_fractions)) if fill_policy_fractions else 0.0,
                        false_stop_rate=_safe_rate(round_false_choose_stop_count, round_true_continue_count),
                        false_continue_rate=_safe_rate(round_false_continue_count, round_true_stop_count),
                        steps_done=steps_done,
                        episodes_done=episodes_done,
                        dataset_size=len(memory_buffer),
                        complexity=float(env.dataset.alpha),
                        beta=float(beta),
                        round_bidirectional_distance_avg=_safe_float(np.mean([v for v in fill_bidirectional_distances if v is not None])) if any(v is not None for v in fill_bidirectional_distances) else None,
                        round_bidirectional_distance_min=_safe_float(np.min([v for v in fill_bidirectional_distances if v is not None])) if any(v is not None for v in fill_bidirectional_distances) else None,
                        round_bidirectional_distance_max=_safe_float(np.max([v for v in fill_bidirectional_distances if v is not None])) if any(v is not None for v in fill_bidirectional_distances) else None,
                        round_precision_avg=_safe_float(np.mean([v for v in fill_precisions if v is not None])) if any(v is not None for v in fill_precisions) else None,
                        round_precision_min=_safe_float(np.min([v for v in fill_precisions if v is not None])) if any(v is not None for v in fill_precisions) else None,
                        round_precision_max=_safe_float(np.max([v for v in fill_precisions if v is not None])) if any(v is not None for v in fill_precisions) else None,
                        round_coverage_avg=_safe_float(np.mean([v for v in fill_coverages if v is not None])) if any(v is not None for v in fill_coverages) else None,
                        round_coverage_min=_safe_float(np.min([v for v in fill_coverages if v is not None])) if any(v is not None for v in fill_coverages) else None,
                        round_coverage_max=_safe_float(np.max([v for v in fill_coverages if v is not None])) if any(v is not None for v in fill_coverages) else None,
                    )

                buffer_fills_done += 1
                last_save_bucket = _maybe_save_checkpoint(
                    actor=actor,
                    actor_optimizer=actor_optimizer,
                    outdir=outdir,
                    name=name,
                    steps_done=steps_done,
                    episodes_done=episodes_done,
                    save_every_buffer_fills=save_every_buffer_fills,
                    buffer_fills_done=buffer_fills_done,
                    last_save_bucket=last_save_bucket,
                    algorithm="multi_target_dagger",
                    extra_metadata={
                        "dagger_round": int(round_index + 1),
                        "beta": float(beta),
                        "dataset_size": len(memory_buffer,
                        ),
                        "buffer_capacity": memory_budget_meta,
                        "buffer_fills_done": int(buffer_fills_done),
                        "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                        "continue_weight": float(loss_config.continue_weight),
                        "norm_floor": float(loss_config.norm_floor),
                        "norm_floor_weight": float(loss_config.norm_floor_weight),
                    },
                )

                # Reset fill counters and continue collection on same buffer
                steps_since_last_optimization = 0
                fill_rewards.clear()
                fill_step_norms.clear()
                fill_policy_fractions.clear()
                fill_bidirectional_distances.clear()
                fill_precisions.clear()
                fill_coverages.clear()

        # End of round collection; run final optimization on any remaining new samples
        if steps_since_last_optimization > 0:
            for _epoch_idx in range(epochs_per_buffer_fill):
                epoch_batch_stats = _run_supervision_epoch(
                    actor=actor,
                    actor_optimizer=actor_optimizer,
                    aggregate_buffer=memory_buffer,
                    batch_size=batch_size,
                    transform=True,
                    loss_config=loss_config,
                )
                (
                    round_true_continue_count,
                    round_false_choose_stop_count,
                    round_true_stop_count,
                    round_false_continue_count,
                ) = _aggregate_supervision_stats(epoch_batch_stats)

                buffer_epoch_index += 1
                _write_dagger_log_row(
                    csv_file_path=csv_file_path,
                    phase="dagger",
                    round_index=round_index + 1,
                    buffer_epoch_index=buffer_epoch_index,
                    avg_reward=float(np.mean(fill_rewards)) if fill_rewards else 0.0,
                    avg_loss=float(np.mean([stat.loss for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                    avg_step_mse=float(np.mean([stat.step_mse for stat in epoch_batch_stats])) if epoch_batch_stats else 0.0,
                    avg_step_norm=float(np.mean(fill_step_norms)) if fill_step_norms else 0.0,
                    policy_rollin_fraction=float(np.mean(fill_policy_fractions)) if fill_policy_fractions else 0.0,
                    false_stop_rate=_safe_rate(round_false_choose_stop_count, round_true_continue_count),
                    false_continue_rate=_safe_rate(round_false_continue_count, round_true_stop_count),
                    steps_done=steps_done,
                    episodes_done=episodes_done,
                    dataset_size=len(memory_buffer),
                    complexity=float(env.dataset.alpha),
                    beta=float(beta),
                    round_bidirectional_distance_avg=_safe_float(np.mean([v for v in fill_bidirectional_distances if v is not None])) if any(v is not None for v in fill_bidirectional_distances) else None,
                    round_bidirectional_distance_min=_safe_float(np.min([v for v in fill_bidirectional_distances if v is not None])) if any(v is not None for v in fill_bidirectional_distances) else None,
                    round_bidirectional_distance_max=_safe_float(np.max([v for v in fill_bidirectional_distances if v is not None])) if any(v is not None for v in fill_bidirectional_distances) else None,
                    round_precision_avg=_safe_float(np.mean([v for v in fill_precisions if v is not None])) if any(v is not None for v in fill_precisions) else None,
                    round_precision_min=_safe_float(np.min([v for v in fill_precisions if v is not None])) if any(v is not None for v in fill_precisions) else None,
                    round_precision_max=_safe_float(np.max([v for v in fill_precisions if v is not None])) if any(v is not None for v in fill_precisions) else None,
                    round_coverage_avg=_safe_float(np.mean([v for v in fill_coverages if v is not None])) if any(v is not None for v in fill_coverages) else None,
                    round_coverage_min=_safe_float(np.min([v for v in fill_coverages if v is not None])) if any(v is not None for v in fill_coverages) else None,
                    round_coverage_max=_safe_float(np.max([v for v in fill_coverages if v is not None])) if any(v is not None for v in fill_coverages) else None,
                )

            buffer_fills_done += 1
            last_save_bucket = _maybe_save_checkpoint(
                actor=actor,
                actor_optimizer=actor_optimizer,
                outdir=outdir,
                name=name,
                steps_done=steps_done,
                episodes_done=episodes_done,
                save_every_buffer_fills=save_every_buffer_fills,
                buffer_fills_done=buffer_fills_done,
                last_save_bucket=last_save_bucket,
                algorithm="multi_target_dagger",
                extra_metadata={
                    "dagger_round": int(round_index + 1),
                    "beta": float(beta),
                    "dataset_size": len(memory_buffer),
                    "buffer_capacity": memory_budget_meta,
                    "buffer_fills_done": int(buffer_fills_done),
                    "continue_target_norm_threshold": float(loss_config.continue_target_norm_threshold),
                    "continue_weight": float(loss_config.continue_weight),
                    "norm_floor": float(loss_config.norm_floor),
                    "norm_floor_weight": float(loss_config.norm_floor_weight),
                },
            )

            steps_since_last_optimization = 0
            fill_rewards.clear()
            fill_step_norms.clear()
            fill_policy_fractions.clear()
            fill_bidirectional_distances.clear()
            fill_precisions.clear()
            fill_coverages.clear()

        # reset env between rounds per spec, but keep memory_buffer contents
        try:
            env.reset(return_state=True)
        except Exception:
            # best-effort reset; continue regardless
            pass

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
            "dataset_size": int(buffer_capacity),
            "buffer_capacity": memory_budget_meta,
            "buffer_fills_done": int(buffer_fills_done),
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