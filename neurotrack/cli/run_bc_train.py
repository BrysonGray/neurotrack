"""Train a deterministic multi-target behavior cloning or DAgger policy from a JSON config."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

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


@dataclass
class BCTrainConfig:
    img_dir: str
    swc_dir: str
    outdir: str
    name: str
    target_step_len: float = 1.0
    step_width: float = 1.0
    batch_size: int = 64
    lr: float = 0.001
    n_episodes: int = 1000
    repeat_starts: bool = True
    branching: bool = False
    rng_seed: int = 1
    start_complexity: float = 0.0
    start_idx: int = 0
    crop_size: int = 128
    patches_per_image: int = 10
    save_every_steps: int = 500
    policy_weights: Optional[str] = None
    seeds_path: Optional[str] = None
    root_sampling_probability: Optional[float] = None
    soma_sample_radius: float = 0.0
    random_offset: float = 0.0
    crop_patches: bool = True
    dagger_rounds: int = 0
    rollout_episodes_per_round: int = 1000
    dataset_epochs_per_round: int = 1
    update_after_steps: int = 500
    update_every: Optional[int] = None
    updates_per_step: int = 1
    warmstart_episodes: int = 200
    beta_start: float = 1.0
    beta_end: float = 0.0
    aggregate_memory_budget: int = 10000
    continue_target_norm_threshold: Optional[float] = None
    continue_weight: float = 1.0
    norm_floor: float = 0.0
    norm_floor_weight: float = 0.0
    stop_violation_weight: float = 1.0
    objective_mode: str = "legacy"
    continue_direction_weight: float = 1.0
    norm_cls_weight: float = 1.0
    norm_cls_temperature: float = 0.25
    norm_margin_weight: float = 1.0
    stop_margin: float = 0.1
    continue_margin: float = 0.1
    stall_threshold: float = 1.0
    max_len: int = 1000
    max_paths: int = 1000
    gamma: float = 0.0

    @classmethod
    def from_params(cls, params: Dict) -> "BCTrainConfig":
        img_dir = _get_param(params, "img_dir", "img_path")
        swc_dir = _get_param(params, "swc_dir")
        outdir = _get_param(params, "outdir", "out_dir")
        name = _get_param(params, "name")
        if img_dir is None or swc_dir is None or outdir is None or name is None:
            raise ValueError("Config must define img_dir, swc_dir, outdir, and name.")

        n_episodes = int(_get_param(params, "n_episodes", "epochs", default=1000))
        rollout_episodes_per_round = int(
            _get_param(params, "rollout_episodes_per_round", "episodes_per_round", default=n_episodes)
        )

        update_every_raw = _get_param(params, "update_every", default=None)
        continue_target_norm_threshold_raw = _get_param(params, "continue_target_norm_threshold", default=None)

        config = cls(
            img_dir=str(img_dir),
            swc_dir=str(swc_dir),
            outdir=str(outdir),
            name=str(name),
            target_step_len=float(_get_param(params, "target_step_len", "step_size", default=1.0)),
            step_width=float(_get_param(params, "step_width", default=1.0)),
            batch_size=int(_get_param(params, "batch_size", "batchsize", default=64)),
            lr=float(_get_param(params, "lr", "learning_rate", default=0.001)),
            n_episodes=n_episodes,
            repeat_starts=bool(_get_param(params, "repeat_starts", default=True)),
            branching=bool(_get_param(params, "branching", default=False)),
            rng_seed=int(_get_param(params, "rng_seed", default=1)),
            start_complexity=float(_get_param(params, "start_complexity", default=0.0)),
            start_idx=int(_get_param(params, "start_idx", default=0)),
            crop_size=int(_get_param(params, "crop_size", default=128)),
            patches_per_image=int(_get_param(params, "patches_per_image", default=10)),
            save_every_steps=int(_get_param(params, "save_every_steps", default=500)),
            policy_weights=_get_param(params, "policy_weights", "bc_weights"),
            seeds_path=_get_param(params, "seeds_path"),
            root_sampling_probability=_get_param(params, "root_sampling_probability"),
            soma_sample_radius=float(_get_param(params, "soma_sample_radius", default=0.0)),
            random_offset=float(_get_param(params, "random_offset", default=0.0)),
            crop_patches=bool(_get_param(params, "crop_patches", default=True)),
            dagger_rounds=int(_get_param(params, "dagger_rounds", default=0)),
            rollout_episodes_per_round=rollout_episodes_per_round,
            dataset_epochs_per_round=int(_get_param(params, "dataset_epochs_per_round", "epochs_per_round", default=1)),
            update_after_steps=int(_get_param(params, "update_after_steps", default=500)),
            update_every=None if update_every_raw is None else int(update_every_raw),
            updates_per_step=int(_get_param(params, "updates_per_step", default=1)),
            warmstart_episodes=int(_get_param(params, "warmstart_episodes", default=n_episodes // 5)),
            beta_start=float(_get_param(params, "beta_start", default=1.0)),
            beta_end=float(_get_param(params, "beta_end", default=0.0)),
            aggregate_memory_budget=int(_get_param(params, "aggregate_memory_budget", "dagger_memory_budget", default=10000)),
            continue_target_norm_threshold=None if continue_target_norm_threshold_raw is None else float(continue_target_norm_threshold_raw),
            continue_weight=float(_get_param(params, "continue_weight", default=1.0)),
            norm_floor=float(_get_param(params, "norm_floor", default=0.0)),
            norm_floor_weight=float(_get_param(params, "norm_floor_weight", default=0.0)),
            stop_violation_weight=float(_get_param(params, "stop_violation_weight", default=1.0)),
            objective_mode=str(_get_param(params, "objective_mode", default="legacy")),
            continue_direction_weight=float(_get_param(params, "continue_direction_weight", default=1.0)),
            norm_cls_weight=float(_get_param(params, "norm_cls_weight", default=1.0)),
            norm_cls_temperature=float(_get_param(params, "norm_cls_temperature", default=0.25)),
            norm_margin_weight=float(_get_param(params, "norm_margin_weight", default=1.0)),
            stop_margin=float(_get_param(params, "stop_margin", default=0.1)),
            continue_margin=float(_get_param(params, "continue_margin", default=0.1)),
            stall_threshold=float(_get_param(params, "stall_threshold", default=1.0)),
            max_len=int(_get_param(params, "max_len", default=1000)),
            max_paths=int(_get_param(params, "max_paths", default=1000)),
            gamma=float(_get_param(params, "gamma", default=0.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.n_episodes <= 0:
            raise ValueError("n_episodes must be > 0.")
        if self.lr <= 0:
            raise ValueError("lr must be > 0.")
        if self.crop_size <= 0:
            raise ValueError("crop_size must be > 0.")
        if self.patches_per_image <= 0:
            raise ValueError("patches_per_image must be > 0.")
        if self.save_every_steps <= 0:
            raise ValueError("save_every_steps must be > 0.")
        if self.rollout_episodes_per_round <= 0:
            raise ValueError("rollout_episodes_per_round must be > 0.")
        if self.dataset_epochs_per_round <= 0:
            raise ValueError("dataset_epochs_per_round must be > 0.")
        if self.update_after_steps <= 0:
            raise ValueError("update_after_steps must be > 0.")
        if self.updates_per_step <= 0:
            raise ValueError("updates_per_step must be > 0.")
        if self.warmstart_episodes < 0:
            raise ValueError("warmstart_episodes must be >= 0.")
        if self.aggregate_memory_budget <= 0:
            raise ValueError("aggregate_memory_budget must be > 0.")
        if self.max_len <= 0 or self.max_paths <= 0:
            raise ValueError("max_len and max_paths must be > 0.")
        if self.start_idx < 0:
            raise ValueError("start_idx must be >= 0.")
        if self.update_every is not None and self.update_every <= 0:
            raise ValueError("update_every must be > 0 when provided.")
        if self.stop_violation_weight < 0:
            raise ValueError("stop_violation_weight must be >= 0.")
        if self.objective_mode not in {"legacy", "norm_classifier_margin"}:
            raise ValueError("objective_mode must be one of: {'legacy', 'norm_classifier_margin'}")
        if self.continue_direction_weight < 0:
            raise ValueError("continue_direction_weight must be >= 0.")
        if self.norm_cls_weight < 0:
            raise ValueError("norm_cls_weight must be >= 0.")
        if self.norm_cls_temperature <= 0:
            raise ValueError("norm_cls_temperature must be > 0.")
        if self.norm_margin_weight < 0:
            raise ValueError("norm_margin_weight must be >= 0.")
        if self.stop_margin < 0:
            raise ValueError("stop_margin must be >= 0.")
        if self.continue_margin < 0:
            raise ValueError("continue_margin must be >= 0.")

    def to_log_dict(self) -> Dict:
        return asdict(self)


def _run_single_experiment(params: Dict, config_path: Path) -> None:
    config = BCTrainConfig.from_params(params)

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

    actor = ConvNet(chin=2, chout=3).to(device=DEVICE, dtype=dtype)
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

    # check for offline params first, if zero dagger rounds
    # then check for online params, if update_every is set, otherwise default to regular BC training
    if config.dagger_rounds > 0:
        behavior_cloning.train_dagger(
            env=env,
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=config.outdir,
            logdir=logdir,
            name=config.name,
            batch_size=config.batch_size,
            warmstart_episodes=config.warmstart_episodes,
            dagger_rounds=config.dagger_rounds,
            rollout_episodes_per_round=config.rollout_episodes_per_round,
            dataset_epochs_per_round=config.dataset_epochs_per_round,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            save_every_steps=config.save_every_steps,
            aggregate_memory_budget=config.aggregate_memory_budget,
            rng=rng,
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
    elif config.update_every is not None:
        behavior_cloning.train_dagger_online(
            env=env,
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=config.outdir,
            logdir=logdir,
            name=config.name,
            batch_size=config.batch_size,
            n_episodes=config.n_episodes,
            warmstart_episodes=config.warmstart_episodes,
            update_after_steps=config.update_after_steps,
            update_every=config.update_every,
            updates_per_step=config.updates_per_step,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            save_every_steps=config.save_every_steps,
            aggregate_memory_budget=config.aggregate_memory_budget,
            rng=rng,
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
    else:
        behavior_cloning.train(
            env=env,
            actor=actor,
            actor_optimizer=actor_optimizer,
            outdir=config.outdir,
            logdir=logdir,
            name=config.name,
            batch_size=config.batch_size,
            n_episodes=config.n_episodes,
            save_every_steps=config.save_every_steps,
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