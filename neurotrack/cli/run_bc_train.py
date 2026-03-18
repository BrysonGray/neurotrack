"""Train a deterministic multi-target behavior cloning or DAgger policy from a JSON config."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim.adamw import AdamW

from neurotrack.data import NeuronPatchDataset
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.models import ConvNet
from neurotrack.training import behavior_cloning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
date_time = datetime.now().strftime("'%Y-%m-%d_%H-%M-%S'")


def _get_param(params: dict, *names: str, default=None):
    for name in names:
        if name in params:
            return params[name]
    return default


def _resolve_experiment_configs(base_params: dict) -> List[dict]:
    """Expand optional ablation overrides into concrete experiment configs."""
    ablations = _get_param(base_params, "ablations", "ablation_overrides", default=None)
    if ablations is None:
        return [base_params]
    if not isinstance(ablations, list) or len(ablations) == 0:
        raise ValueError("'ablations' must be a non-empty list when provided.")

    base_name = _get_param(base_params, "name")
    base_outdir_raw = _get_param(base_params, "outdir", "out_dir")
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
    img_dir = _get_param(params, "img_dir", "img_path")
    swc_dir = _get_param(params, "swc_dir")
    outdir = _get_param(params, "outdir", "out_dir")
    name = _get_param(params, "name")
    if img_dir is None or swc_dir is None or outdir is None or name is None:
        raise ValueError("Config must define img_dir, swc_dir, outdir, and name.")

    target_step_len = float(_get_param(params, "target_step_len", "step_size", default=1.0))
    step_width = float(_get_param(params, "step_width", default=1.0))
    batch_size = int(_get_param(params, "batch_size", "batchsize", default=64))
    lr = float(_get_param(params, "lr", "learning_rate", default=0.001))
    n_episodes = int(_get_param(params, "n_episodes", "epochs", default=1000))
    repeat_starts = bool(_get_param(params, "repeat_starts", default=True))
    branching = bool(_get_param(params, "branching", default=False))
    rng_seed = int(_get_param(params, "rng_seed", default=1))
    start_complexity = float(_get_param(params, "start_complexity", default=0.0))
    start_idx = int(_get_param(params, "start_idx", default=0))
    crop_size = int(_get_param(params, "crop_size", default=128))
    patches_per_image = int(_get_param(params, "patches_per_image", default=10))
    save_every_steps = int(_get_param(params, "save_every_steps", default=500))
    policy_weights = _get_param(params, "policy_weights", "bc_weights")
    seeds_path = _get_param(params, "seeds_path")
    root_sampling_probability = _get_param(params, "root_sampling_probability")
    dagger_rounds = int(_get_param(params, "dagger_rounds", default=0))
    warmstart_episodes = int(_get_param(params, "warmstart_episodes", default=n_episodes))
    rollout_episodes_per_round = int(_get_param(params, "rollout_episodes_per_round", "episodes_per_round", default=n_episodes))
    dataset_epochs_per_round = int(_get_param(params, "dataset_epochs_per_round", "epochs_per_round", default=1))
    beta_start = float(_get_param(params, "beta_start", default=1.0))
    beta_end = float(_get_param(params, "beta_end", default=0.0))
    aggregate_memory_budget = int(_get_param(params, "aggregate_memory_budget", "dagger_memory_budget", default=10000))
    stall_threshold = float(_get_param(params, "stall_threshold", default=1.0))
    continue_target_norm_threshold_raw = _get_param(params, "continue_target_norm_threshold", default=None)
    continue_target_norm_threshold = None if continue_target_norm_threshold_raw is None else float(continue_target_norm_threshold_raw)
    continue_weight = float(_get_param(params, "continue_weight", default=1.0))
    norm_floor = float(_get_param(params, "norm_floor", default=0.0))
    norm_floor_weight = float(_get_param(params, "norm_floor_weight", default=0.0))

    rng = np.random.default_rng(rng_seed)
    dataset = NeuronPatchDataset(
        swc_dir=swc_dir,
        img_dir=img_dir,
        crop_size=crop_size,
        patches_per_image=patches_per_image,
        alpha=start_complexity,
        step_width=step_width,
        rng=rng,
        crop_patches=True,
        inference_mode=False,
        seeds_path=seeds_path,
        root_sampling_probability=root_sampling_probability,
    )

    env = NeuronTrackingEnvironment(
        dataset=dataset,
        radius=17,
        target_step_len=target_step_len,
        step_width=step_width,
        stall_threshold=stall_threshold,
        max_len=int(_get_param(params, "max_len", default=1000)),
        max_paths=int(_get_param(params, "max_paths", default=1000)),
        gamma=float(_get_param(params, "gamma", default=0.99)),
        branching=branching,
        repeat_starts=repeat_starts,
        start_idx=start_idx,
        inference_mode=False,
    )

    actor = ConvNet(chin=2, chout=3).to(device=DEVICE, dtype=dtype)
    actor.policy_output_mode = "direct_vector"
    actor_optimizer = AdamW(actor.parameters(), lr=lr)

    if policy_weights is not None:
        state_dicts = torch.load(policy_weights, map_location=DEVICE)
        actor.load_state_dict(state_dicts["policy_state_dict"])
        if "actor_optimizer_state_dict" in state_dicts:
            actor_optimizer.load_state_dict(state_dicts["actor_optimizer_state_dict"])

    script_path = Path(__file__).resolve()
    logdir = script_path.parent.parent / "logs" / name
    os.makedirs(logdir, exist_ok=True)

    params_to_save = dict(params)
    params_to_save["resolved_from_config"] = str(config_path)
    with open(logdir / f"training_params_{date_time}.json", "w", encoding="utf-8") as handle:
        json.dump(params_to_save, handle, indent=4)

    dynamic_complexity = bool(_get_param(params, "dynamic_complexity", default=True))
    if dagger_rounds > 0:
        behavior_cloning.train_dagger(
            env=env,
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=outdir,
            logdir=logdir,
            name=name,
            batch_size=batch_size,
            warmstart_episodes=warmstart_episodes,
            dagger_rounds=dagger_rounds,
            rollout_episodes_per_round=rollout_episodes_per_round,
            dataset_epochs_per_round=dataset_epochs_per_round,
            beta_start=beta_start,
            beta_end=beta_end,
            save_every_steps=save_every_steps,
            dynamic_complexity=dynamic_complexity,
            aggregate_memory_budget=aggregate_memory_budget,
            rng=rng,
            continue_target_norm_threshold=continue_target_norm_threshold,
            continue_weight=continue_weight,
            norm_floor=norm_floor,
            norm_floor_weight=norm_floor_weight,
        )
    else:
        behavior_cloning.train(
            env=env,
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=outdir,
            logdir=logdir,
            name=name,
            batch_size=batch_size,
            n_episodes=n_episodes,
            save_every_steps=save_every_steps,
            dynamic_complexity=dynamic_complexity,
            continue_target_norm_threshold=continue_target_norm_threshold,
            continue_weight=continue_weight,
            norm_floor=norm_floor,
            norm_floor_weight=norm_floor_weight,
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