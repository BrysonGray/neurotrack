"""Forward-pass tracing loop for Gaussian SAC and deterministic BC policies."""

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import count
from typing import Optional

from neurotrack.training.policy_utils import (
    prepare_observation_for_model,
    sample_from_output,
)
from neurotrack.training.env_inspector import show_state

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _select_action_from_actor_output(
    actor_out: torch.Tensor,
    policy_output_mode: str,
    stochastic: bool = False,
):
    """Convert actor outputs into an action tensor for tracing."""
    if policy_output_mode == "direct_vector":
        action = actor_out[0, :3].detach().cpu()
        return action, None

    direction_dist = sample_from_output(actor_out)
    if stochastic:
        action = direction_dist.sample()[0]
    else:
        action = direction_dist.mean[0]
    variance = float(direction_dist.variance[0].mean().detach().cpu())
    return action, variance


def _min_target_norm(target_vectors) -> Optional[float]:
    if target_vectors is None:
        return None
    target_t = torch.as_tensor(target_vectors, dtype=torch.float32).view(-1, 3)
    if target_t.numel() == 0:
        return None
    return float(torch.linalg.norm(target_t, dim=1).min().item())


def _action_norm_histogram(action_norms, bins):
    counts, _ = np.histogram(np.asarray(action_norms, dtype=np.float32), bins=np.asarray(bins, dtype=np.float32))
    return {
        "bin_edges": [float(v) for v in bins],
        "counts": [int(v) for v in counts.tolist()],
    }


def trace_image(
    env,
    actor,
    dataset_idx,
    Q_net=None,
    n_trials=1,
    show=True,
    show_live=False,
    stochastic=False,
    return_stats=False,
    cancel_event=None,
    initial_path_mask=None,
    terminal_target_norm_threshold: float = 1.0,
    false_stop_distance_threshold: Optional[float] = None,
):
    """
    Trace a single neuron image using the given actor.

    Parameters
    ----------
    env : object
        The environment object which provides the state and handles actions.
    actor : object
        The actor model used to perform actions in the environment.
    dataset_idx : int
        Index into the dataset for the image to run inference on.
    Q_net : object, optional
        Q-network for estimating returns (required if n_trials > 1).
    n_trials : int, optional
        The number of trials to perform (default is 1). If > 1, the trial with
        the highest estimated return is selected; Q_net must be provided.
    show : bool, optional
        If True, display the environment state at the end of each trial (default is True).
    show_live : bool, optional
        If True, show state after every step (default is False).
    stochastic : bool, optional
        If True, sample actions stochastically instead of using the mean (default is False).
    return_stats : bool, optional
        If True, include step statistics and estimated return info in the result
        (default is False).
    cancel_event : threading.Event, optional
        If provided, the trace will abort and raise RuntimeError when the event is set.
    initial_path_mask : torch.Tensor or np.ndarray, optional
        Optional starting path mask in (Z, Y, X) to load into ``env.img.data[-1]``
        immediately after each environment reset (used for revision retracing).
    terminal_target_norm_threshold : float, optional
        Target-distance threshold used to bucket states as terminal-like vs non-terminal.
    false_stop_distance_threshold : float, optional
        A choose_stop event is counted as false-stop when current target distance exceeds this value.
        Defaults to ``terminal_target_norm_threshold`` when not provided.

    Returns
    -------
    dict
        A dictionary containing:
        - 'neuron_name': Name/path of the neuron.
        - 'labeled_neuron': Image volume (torch.Tensor) with traced paths drawn.
        - 'paths': List of reconstructed paths from the best trial.
        If return_stats is True, also includes:
        - 'estimated_returns': Estimated returns for all trials.
        - 'mean_step_variance', 'max_step_variance', 'min_step_variance'
        - 'mean_step_magnitude', 'max_step_magnitude', 'min_step_magnitude'
    """

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials > 1 and Q_net is None:
        raise ValueError("Q_net must be provided if n_trials > 1")
    if terminal_target_norm_threshold < 0.0:
        raise ValueError("terminal_target_norm_threshold must be non-negative")
    if false_stop_distance_threshold is None:
        false_stop_distance_threshold = terminal_target_norm_threshold
    if false_stop_distance_threshold < 0.0:
        raise ValueError("false_stop_distance_threshold must be non-negative")

    if show:
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        plt.ion()

    actor.eval()
    policy_output_mode = getattr(actor, "policy_output_mode", "gaussian")
    trace_start_time = time.perf_counter()
    timing_ms = {
        "reset_and_mask": 0.0,
        "get_state": 0.0,
        "actor_forward": 0.0,
        "sample_action": 0.0,
        "env_step": 0.0,
        "q_eval": 0.0,
        "postprocess": 0.0,
        "steps": 0,
        "trials": int(n_trials),
    }
    env.dataset.inference_mode = True  # Disable mask drawing during inference.

    img_idx = dataset_idx % len(env.dataset.img_files)
    reset_start = time.perf_counter()
    env.reset(dataset_index=img_idx)

    def _apply_initial_path_mask_if_provided():
        if initial_path_mask is None:
            return
        if env.img is None:
            return

        mask_tensor = torch.as_tensor(initial_path_mask, device=env.img.data.device)
        if mask_tensor.ndim == 4 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor[0]
        if mask_tensor.ndim != 3:
            raise ValueError("initial_path_mask must be a 3D tensor/array in (Z, Y, X) order.")

        expected_shape = tuple(int(v) for v in env.img.data.shape[-3:])
        if tuple(int(v) for v in mask_tensor.shape) != expected_shape:
            raise ValueError(
                f"initial_path_mask shape {tuple(mask_tensor.shape)} does not match image shape {expected_shape}."
            )

        if env.img.data.shape[0] == 1:
            env.img.data = torch.cat(
                (
                    env.img.data,
                    torch.zeros((1,) + env.img.data.shape[1:], dtype=env.img.data.dtype, device=env.img.data.device),
                ),
                dim=0,
            )

        mask_tensor = mask_tensor.to(dtype=env.img.data.dtype, device=env.img.data.device)
        env.img.data[-1] = mask_tensor
        if env.paths:
            env.img.draw_point(
                env.paths[0][-1],
                radius=(env.step_width / 2.35),
                channel=-1,
                mode="gaussian",
                binary=False,
            )

    _apply_initial_path_mask_if_provided()
    timing_ms["reset_and_mask"] = (time.perf_counter() - reset_start) * 1000.0

    estimated_returns = []
    labeled_neurons = []
    trial_paths = []
    variance = []
    step_magnitudes = []
    terminal_state_action_norms = []
    nonterminal_state_action_norms = []
    choose_stop_count = 0
    choose_stop_with_target_count = 0
    false_choose_stop_count = 0
    choose_stop_target_distances = []

    # Run n_trials.
    for trial in range(n_trials):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Trace cancelled.")

        estimated_return = 0
        state_start = time.perf_counter()
        obs = env.get_state()
        timing_ms["get_state"] += (time.perf_counter() - state_start) * 1000.0

        for t in count():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Trace cancelled.")
            with torch.no_grad():
                actor_start = time.perf_counter()
                obs_on_device = prepare_observation_for_model(obs, device=DEVICE, model_dtype=torch.float32)
                actor_out = actor(obs_on_device)
                timing_ms["actor_forward"] += (time.perf_counter() - actor_start) * 1000.0

                sample_start = time.perf_counter()
                action, action_variance = _select_action_from_actor_output(
                    actor_out,
                    policy_output_mode=policy_output_mode,
                    stochastic=stochastic,
                )
                if action_variance is not None:
                    variance.append(action_variance)
                action_norm = float(action.norm().detach().cpu())
                step_magnitudes.append(action_norm)
                timing_ms["sample_action"] += (time.perf_counter() - sample_start) * 1000.0

            step_start = time.perf_counter()
            action_cpu = action.detach().cpu()
            next_obs, reward, terminated, truncated, info = env.step(action_cpu)
            timing_ms["env_step"] += (time.perf_counter() - step_start) * 1000.0
            timing_ms["steps"] += 1

            current_target_distance = _min_target_norm(info.get("current_target_vectors"))
            if current_target_distance is not None:
                if current_target_distance <= terminal_target_norm_threshold:
                    terminal_state_action_norms.append(action_norm)
                else:
                    nonterminal_state_action_norms.append(action_norm)

            if info.get("status") == "choose_stop":
                choose_stop_count += 1
                if current_target_distance is not None:
                    choose_stop_with_target_count += 1
                    choose_stop_target_distances.append(current_target_distance)
                    if current_target_distance > false_stop_distance_threshold:
                        false_choose_stop_count += 1

            if Q_net is not None:
                q_start = time.perf_counter()
                action_on_device = action if action.device == DEVICE else action.to(device=DEVICE, non_blocking=True)
                action_channels = action_on_device.view(1, 3, 1, 1, 1).expand(
                    obs_on_device.shape[0],
                    3,
                    obs_on_device.shape[2],
                    obs_on_device.shape[3],
                    obs_on_device.shape[4],
                )
                current_state = torch.cat((obs_on_device, action_channels), dim=1)
                q_val = Q_net(current_state)[:, 0]
                estimated_return += q_val.cpu().item()
                timing_ms["q_eval"] += (time.perf_counter() - q_start) * 1000.0

            # Show state after every step
            if show_live and show:
                try:
                    shell = get_ipython().__class__.__name__  # type: ignore  # noqa: F821
                    if shell:
                        show_state(env, fig, live=True, reward=reward.cpu().item())
                except NameError:
                    pass

            if info["terminate_episode"]:
                estimated_returns.append(estimated_return)
                labeled_neurons.append(env.img.data[-1].detach().clone().cpu())
                trial_paths.append([path.detach().cpu().numpy().tolist() for path in env.finished_paths if isinstance(path, torch.Tensor) and len(path) > 3])
                if show:
                    try:
                        shell = get_ipython().__class__.__name__  # type: ignore  # noqa: F821
                        if shell:
                            show_state(env, fig)
                            print(f"num branches: {len(env.finished_paths)}")
                            print(f"num long paths: {len(trial_paths[-1])}")
                            if Q_net is not None:
                                print(f"estimated return: {estimated_return:.2f}")
                            try:
                                print("Press Enter to continue to the next trial (or 'q' to quit)...", flush=True)
                                user_input = input().strip()
                                if user_input.lower() == 'q':
                                    print("Quitting inference.", flush=True)
                                    break
                            except EOFError:
                                pass
                    except NameError:
                        pass
                if trial < n_trials - 1:
                    reset_start = time.perf_counter()
                    env.reset(move_to_next=False)
                    _apply_initial_path_mask_if_provided()
                    timing_ms["reset_and_mask"] += (time.perf_counter() - reset_start) * 1000.0
                break  # Move to next trial.

            state_start = time.perf_counter()
            obs = env.get_state()
            timing_ms["get_state"] += (time.perf_counter() - state_start) * 1000.0

    # Select best trial based on estimated return.
    post_start = time.perf_counter()
    if n_trials > 1 and len(estimated_returns) > 0:
        index = int(np.argmax(estimated_returns))
    else:
        index = 0

    labeled_neuron = labeled_neurons[index]
    paths = trial_paths[index]
    # Return from voxel (z-y-x) to world (x-y-z) coordinates.
    paths = [np.array(p)[:, ::-1].tolist() for p in paths]
    name = env.current_neuron_info["neuron_name"]

    result = {
        'neuron_name': name,
        'labeled_neuron': labeled_neuron,
        'paths': paths,
        'timing_ms': {
            **{k: round(float(v), 3) for k, v in timing_ms.items() if k not in {"steps", "trials"}},
            'steps': int(timing_ms["steps"]),
            'trials': int(timing_ms["trials"]),
            'total': round((time.perf_counter() - trace_start_time) * 1000.0, 3),
        },
    }
    result['timing_ms']['postprocess'] = round(
        float(result['timing_ms'].get('postprocess', 0.0)) + (time.perf_counter() - post_start) * 1000.0,
        3,
    )

    if return_stats:
        if len(variance) > 0:
            mean_step_variance = float(np.mean(variance))
            max_step_variance = float(np.max(variance))
            min_step_variance = float(np.min(variance))
        else:
            mean_step_variance = max_step_variance = min_step_variance = None

        if len(step_magnitudes) > 0:
            mean_step_magnitude = float(np.mean(step_magnitudes))
            max_step_magnitude = float(np.max(step_magnitudes))
            min_step_magnitude = float(np.min(step_magnitudes))
        else:
            mean_step_magnitude = max_step_magnitude = min_step_magnitude = None

        if len(terminal_state_action_norms) > 0:
            terminal_mean_step_magnitude = float(np.mean(terminal_state_action_norms))
        else:
            terminal_mean_step_magnitude = None

        if len(nonterminal_state_action_norms) > 0:
            nonterminal_mean_step_magnitude = float(np.mean(nonterminal_state_action_norms))
        else:
            nonterminal_mean_step_magnitude = None

        if terminal_mean_step_magnitude is None or nonterminal_mean_step_magnitude is None:
            action_norm_separation = None
        else:
            action_norm_separation = float(nonterminal_mean_step_magnitude - terminal_mean_step_magnitude)

        false_choose_stop_rate = None
        if choose_stop_with_target_count > 0:
            false_choose_stop_rate = float(false_choose_stop_count / choose_stop_with_target_count)

        hist_bins = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
        terminal_hist = _action_norm_histogram(terminal_state_action_norms, bins=hist_bins)
        nonterminal_hist = _action_norm_histogram(nonterminal_state_action_norms, bins=hist_bins)

        result.update({
            'estimated_returns': estimated_returns,
            'mean_step_variance': mean_step_variance,
            'max_step_variance': max_step_variance,
            'min_step_variance': min_step_variance,
            'mean_step_magnitude': mean_step_magnitude,
            'max_step_magnitude': max_step_magnitude,
            'min_step_magnitude': min_step_magnitude,
            'choose_stop_count': int(choose_stop_count),
            'choose_stop_with_target_count': int(choose_stop_with_target_count),
            'false_choose_stop_count': int(false_choose_stop_count),
            'false_choose_stop_rate': false_choose_stop_rate,
            'choose_stop_target_distance_mean': (
                float(np.mean(choose_stop_target_distances)) if len(choose_stop_target_distances) > 0 else None
            ),
            'terminal_state_mean_step_magnitude': terminal_mean_step_magnitude,
            'nonterminal_state_mean_step_magnitude': nonterminal_mean_step_magnitude,
            'action_norm_separation': action_norm_separation,
            'terminal_state_action_norm_histogram': terminal_hist,
            'nonterminal_state_action_norm_histogram': nonterminal_hist,
            'terminal_target_norm_threshold': float(terminal_target_norm_threshold),
            'false_stop_distance_threshold': float(false_stop_distance_threshold),
        })

    return result
