"""Shared policy utilities used by training and inference paths."""

from __future__ import annotations

import torch


def bound_direction_magnitude(direction: torch.Tensor, max_norm: float = 10.0) -> torch.Tensor:
    """Smoothly bound direction magnitude while preserving direction."""
    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    direction_scale = torch.tanh(direction_norm) * float(max_norm)
    return direction * direction_scale / (direction_norm + torch.finfo(direction.dtype).eps)

def sample_from_output(out: torch.Tensor) -> torch.distributions.MultivariateNormal:
    """Build a bounded Gaussian action distribution from actor outputs."""
    mean = bound_direction_magnitude(out[:, :3], max_norm=10.0)
    logvar = out[:, 3:]

    logvar = torch.tanh(logvar) * 3 - 1
    return torch.distributions.MultivariateNormal(
        mean[:, :3],
        torch.exp(logvar)[:, None] * torch.eye(3, device=out.device)[None],
    )


def prepare_observation_for_model(
    obs: torch.Tensor,
    device: torch.device | None = None,
    model_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert uint8 observations to normalized float tensors at model boundaries."""
    if device is None:
        device = obs.device
    if obs.dtype == torch.uint8:
        return obs.to(device=device, dtype=model_dtype) * (1.0 / 255.0)
    return obs.to(device=device, dtype=model_dtype)
