"""Shared inference runtime utilities for SAC and deterministic BC policies."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from neurotrack.data import NeuronPatchDataset
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.models import ConvNet
from .tracing import trace_image


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def build_env(params: Dict[str, Any]) -> NeuronTrackingEnvironment:
    rng = np.random.default_rng(params.get("rng_seed", 0))

    dataset = NeuronPatchDataset(
        img_dir=params["img_dir"],
        swc_dir=params.get("swc_dir", None),
        alpha=1.0,
        step_width=float(params.get("step_width", 2.0)),
        rng=rng,
        crop_patches=params.get("crop_patches", False),
        patches_per_image=int(params.get("patches_per_image", 1)),
        seeds_path=params.get("seeds_path", None),
        root_sampling_probability=params.get("root_sampling_probability", None),
        inference_mode=True,
    )

    env = NeuronTrackingEnvironment(
        dataset=dataset,
        radius=17,
        step_width=params.get("step_width", 2.0),
        stall_threshold=float(params.get("stall_threshold", 1.0)),
        max_len=params.get("max_len", 10000),
        max_paths=params.get("max_paths", 1000),
        branching=params.get("branching", True),
        repeat_starts=params.get("repeat_starts", False),
        start_idx=0,
        inference_mode=True,
    )

    return env


def load_models(
    params: Dict[str, Any],
    in_channels: int = 2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dicts = torch.load(params["sac_weights"], map_location=device)
    policy_output_mode = str(state_dicts.get("policy_output_mode", "gaussian"))
    if policy_output_mode == "direct_vector":
        policy_output_dim = int(state_dicts.get("policy_output_dim", 3))
    else:
        policy_output_dim = int(state_dicts.get("policy_output_dim", 4))

    actor = ConvNet(chin=in_channels, chout=policy_output_dim).to(device=device, dtype=dtype)
    actor.load_state_dict(state_dicts["policy_state_dict"])
    actor.eval()
    actor.policy_output_mode = policy_output_mode

    q_net = None
    if int(params.get("n_trials", 1)) > 1:
        if "Q1_state_dict" not in state_dicts:
            raise ValueError(
                "n_trials > 1 requires a checkpoint with Q1_state_dict. "
                "Deterministic behavior-cloning checkpoints should use n_trials=1."
            )
        q_net = ConvNet(chin=in_channels + 3, chout=1).to(device=device, dtype=dtype)
        q_net.load_state_dict(state_dicts["Q1_state_dict"])
        q_net.eval()

    return actor, q_net


def run_inference(params: Dict[str, Any], out_dir: Path | str) -> Dict[str, Any]:
    run_out_dir = Path(out_dir)
    inference_out_dir = run_out_dir / "tracing_results"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    inference_out_dir.mkdir(parents=True, exist_ok=True)

    actor, q_net = load_models(params)
    env = build_env(params)

    n_trials = int(params.get("n_trials", 1))
    if n_trials == 1:
        q_net = None
    show = bool(params.get("show", False))
    show_live = bool(params.get("show_live", False))
    stochastic = bool(params.get("stochastic_actions", False))
    return_stats = bool(params.get("return_stats", False))
    sync = bool(params.get("sync", False))
    terminal_target_norm_threshold = float(params.get("terminal_target_norm_threshold", params.get("stall_threshold", 1.0)))
    false_stop_distance_threshold = float(params.get("false_stop_distance_threshold", terminal_target_norm_threshold))

    img_indices = list(range(len(env.dataset.img_files)))
    if sync:
        processed_stems = {f.stem for f in inference_out_dir.glob("*_trace.json")}
        img_indices = [
            i for i in img_indices
            if Path(env.dataset.img_files[i]).stem + "_trace" not in processed_stems
        ]

    results = []
    progress = tqdm(img_indices, desc="Tracing", unit="img", dynamic_ncols=True)
    for idx in progress:
        img_name = Path(env.dataset.img_files[idx]).stem
        progress.set_postfix_str(img_name)
        result = trace_image(
            env=env,
            actor=actor,
            dataset_idx=idx,
            Q_net=q_net,
            n_trials=n_trials,
            show=show,
            show_live=show_live,
            stochastic=stochastic,
            return_stats=return_stats,
            terminal_target_norm_threshold=terminal_target_norm_threshold,
            false_stop_distance_threshold=false_stop_distance_threshold,
        )
        results.append(result)

        # Save per-image result (exclude labeled_neuron — too large for JSON)
        img_stem = Path(result["neuron_name"]).stem
        per_image_data = {k: _to_serializable(v) for k, v in result.items() if k != "labeled_neuron"}
        with open(inference_out_dir / f"{img_stem}_trace.json", "w") as handle:
            json.dump(per_image_data, handle, indent=2)

    return {
        "results": results,
        "run_out_dir": run_out_dir,
        "tracing_results_dir": inference_out_dir,
    }
