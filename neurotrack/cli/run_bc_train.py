"""Train a deterministic multi-target behavior cloning or DAgger policy from a JSON config."""

import argparse
from datetime import datetime
import json
import os
import random
from pathlib import Path
from typing import Dict, List
import warnings

# Must be set before any CUDA context initialization for deterministic cuBLAS behavior.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.optim.adamw import AdamW

from neurotrack.data import NeuronPatchDataset
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.models import ConvNet
from neurotrack.training import behavior_cloning
from neurotrack.training.bc_config import BCTrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
dtype = torch.float32
date_time = datetime.now().strftime("'%Y-%m-%d_%H-%M-%S'")


def _configure_reproducibility(seed: int, allow_tf32: bool = True) -> None:
    """Configure process-wide deterministic behavior for reproducible training."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32

    # Keep deterministic mode enabled, but fall back to warnings for ops that
    # do not currently have deterministic CUDA implementations.
    warnings.filterwarnings(
        "ignore",
        message=r".*adaptive_avg_pool3d_backward_cuda does not have a deterministic implementation.*",
        category=UserWarning,
    )

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to enable deterministic PyTorch algorithms. "
            "Ensure your CUDA/PyTorch stack supports deterministic execution."
        ) from exc


def _get_param(params: dict, *names: str, default=None):
    for name in names:
        if name in params:
            return params[name]
    return default


def _resolve_experiment_configs(base_params: dict) -> List[dict]:
    """Expand optional ablation overrides into concrete experiment configs.
    
    Uses only canonical parameter names; deprecated parameter aliases not supported.
    """
    ablations = _get_param(base_params, "ablations", "ablation_overrides", default=None)
    if ablations is None:
        return [base_params]
    if not isinstance(ablations, list) or len(ablations) == 0:
        raise ValueError("'ablations' must be a non-empty list when provided.")

    base_name = _get_param(base_params, "name")
    base_outdir_raw = _get_param(base_params, "outdir")
    if base_name is None or base_outdir_raw is None:
        raise ValueError("Base config must define name and outdir when using ablations.")
    base_outdir = Path(str(base_outdir_raw))
    experiments: List[dict] = []
    for idx, override in enumerate(ablations):
        if not isinstance(override, dict):
            raise ValueError(f"Each ablation override must be an object, got {type(override)!r} at index {idx}.")

        variant = dict(base_params)
        variant.update(override)
        if "name" not in override:
            suffix = str(override.get("name_suffix", override.get("label", f"ablation_{idx + 1}")))
            variant["name"] = f"{base_name}__{suffix}"
        if "outdir" not in override and "out_dir" not in override:
            variant["outdir"] = str(base_outdir / str(variant["name"]))
        experiments.append(variant)

    return experiments


def _run_single_experiment(params: Dict, config_path: Path) -> None:
    config = BCTrainConfig.from_params(params)
    _configure_reproducibility(config.rng_seed, allow_tf32=False) # Disable TF32 for better determinism. Enable for faster training if exact reproducibility is not required.

    rng = np.random.default_rng(config.rng_seed)
    dataset = NeuronPatchDataset(
        swc_dir=config.swc_dir,
        img_dir=config.img_dir,
        crop_size=config.crop_size,
        patches_per_image=config.patches_per_image,
        alpha=config.start_complexity,
        step_width=config.step_width,
        rng=rng,
        crop_patches=config.crop_patches,
        inference_mode=False,
        seeds_path=config.seeds_path,
        root_sampling_probability=config.root_sampling_probability,
        soma_sample_radius=config.soma_sample_radius,
        random_offset=config.random_offset,
    )

    env = NeuronTrackingEnvironment(
        dataset=dataset,
        radius=17,
        target_step_len=config.target_step_len,
        step_width=config.step_width,
        stall_threshold=config.stall_threshold,
        max_len=config.max_len,
        max_paths=config.max_paths,
        gamma=config.gamma,
        branching=config.branching,
        repeat_starts=config.repeat_starts,
        start_idx=config.start_idx,
        inference_mode=False,
    )

    actor = ConvNet(chin=2, chout=3, rng_seed=config.rng_seed).to(device=DEVICE, dtype=dtype)
    actor.policy_output_mode = "direct_vector"
    actor_optimizer = AdamW(actor.parameters(), lr=config.lr)

    if config.policy_weights is not None:
        print("Loading policy weights from:", config.policy_weights)
        state_dicts = torch.load(config.policy_weights, map_location=DEVICE)
        actor.load_state_dict(state_dicts["policy_state_dict"])
        if "actor_optimizer_state_dict" in state_dicts:
            actor_optimizer.load_state_dict(state_dicts["actor_optimizer_state_dict"])

    script_path = Path(__file__).resolve()
    logdir = script_path.parent.parent / "logs" / config.name
    os.makedirs(logdir, exist_ok=True)

    params_to_save = config.to_log_dict()
    params_to_save["resolved_from_config"] = str(config_path)
    with open(logdir / f"training_params_{date_time}.json", "w", encoding="utf-8") as handle:
        json.dump(params_to_save, handle, indent=4)

    warmstart_steps = config.warmstart_steps if config.dagger_rounds > 0 else config.total_steps
    behavior_cloning.train(
        env=env,
        actor=actor,
        actor_optimizer=actor_optimizer,
        outdir=config.outdir,
        logdir=logdir,
        name=config.name,
        batch_size=config.batch_size,
        warmstart_steps=warmstart_steps,
        dagger_rounds=config.dagger_rounds,
        steps_per_round=config.steps_per_round,
        steps_per_update=config.steps_per_update,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        beta_decay=config.beta_decay,
        beta_step=config.beta_step,
        save_every_updates=config.save_every_updates,
        buffer_capacity=config.buffer_capacity,
        epochs_per_update=config.epochs_per_update,
        rng=rng if config.dagger_rounds > 0 else None,
        continue_target_norm_threshold=config.continue_target_norm_threshold,
        continue_weight=config.continue_weight,
        norm_floor=config.norm_floor,
        norm_floor_weight=config.norm_floor_weight,
        stop_violation_weight=config.stop_violation_weight,
        objective_mode=config.objective_mode,
        continue_direction_weight=config.continue_direction_weight,
        norm_cls_weight=config.norm_cls_weight,
        norm_cls_temperature=config.norm_cls_temperature,
        norm_margin_weight=config.norm_margin_weight,
        stop_margin=config.stop_margin,
        continue_margin=config.continue_margin,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a deterministic behavior cloning or DAgger policy from a JSON config.")
    parser.add_argument("-i", "--json", type=str, required=True, help="Path to input parameters json file.")
    args = parser.parse_args()

    config_path = Path(args.json).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        params = json.load(handle)

    experiments = _resolve_experiment_configs(params)
    total = len(experiments)
    for idx, experiment_params in enumerate(experiments, start=1):
        exp_name = _get_param(experiment_params, "name")
        print(f"[{idx}/{total}] Starting BC/DAgger run: {exp_name}", flush=True)
        _run_single_experiment(experiment_params, config_path=config_path)

    print("Done!")


if __name__ == "__main__":
    main()