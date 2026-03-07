"""SAC-based forward-pass tracing loop for inference."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import count

from neurotrack.training.sac import sample_from_output
from neurotrack.training.env_inspector import show_state

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def trace_image(env, actor, dataset_idx, Q_net=None, n_trials=1, show=True, show_live=False,
                stochastic=False, return_stats=False, cancel_event=None, initial_path_mask=None):
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

    if show:
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        plt.ion()

    actor.eval()
    env.dataset.inference_mode = True  # Disable mask drawing during inference.

    img_idx = dataset_idx % len(env.dataset.img_files)
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

    estimated_returns = []
    labeled_neurons = []
    trial_paths = []
    variance = []
    step_magnitudes = []

    # Run n_trials.
    for trial in range(n_trials):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Trace cancelled.")

        estimated_return = 0
        obs = env.get_state()

        for t in count():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Trace cancelled.")
            with torch.no_grad():
                actor_out = actor(obs.to(DEVICE))
                direction_dist = sample_from_output(actor_out.detach().cpu())
                if stochastic:
                    action = direction_dist.sample()[0]
                else:
                    action = direction_dist.mean[0]
                variance.append(float(direction_dist.variance[0].mean()))
                step_magnitudes.append(float(action.norm()))

            next_obs, reward, terminated, truncated, info = env.step(action, training=False)

            if Q_net is not None:
                current_state = torch.cat((obs.to(DEVICE), torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]), device=DEVICE) * action[None, :, None, None, None].to(DEVICE)), dim=1)
                q_val = Q_net(current_state)[:, 0]
                estimated_return += q_val.cpu().item()

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
                    env.reset(move_to_next=False)
                    _apply_initial_path_mask_if_provided()
                break  # Move to next trial.

            obs = env.get_state()

    # Select best trial based on estimated return.
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
    }

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

        result.update({
            'estimated_returns': estimated_returns,
            'mean_step_variance': mean_step_variance,
            'max_step_variance': max_step_variance,
            'min_step_variance': min_step_variance,
            'mean_step_magnitude': mean_step_magnitude,
            'max_step_magnitude': max_step_magnitude,
            'min_step_magnitude': min_step_magnitude,
        })

    return result
