#!/usr/bin/env python

""" Interface for interactively evaluating the tracking environment """
from itertools import count
from IPython import display
import torch
from neurotrack.data import loading as load
from neurotrack.environments import tracking_reward
from neurotrack.training.policy_utils import (
    prepare_observation_for_model,
    sample_from_output,
)
from neurotrack.training import behavior_cloning as bc

def get_unvisited_sections(env):
    """Safely parse the current unvisited tree into drawable sections."""
    tree = getattr(env, 'unvisited_tree', None)
    if tree is None:
        return {}
    if getattr(tree, 'ndim', 0) < 2 or tree.shape[0] == 0:
        return {}
    try:
        sections, _ = load.parse_swc(tree, verbose=False, transpose=False)
    except Exception:
        return {}
    return sections


def draw_2d_panel(ax, environment, cropped=False, sections=None,
                  skeleton_color='lightgray', path_color='red', target_color='blue', step_size=4.0, dim=0,
                  size_scale=1.0):
    """
    Draw a 2D projection of skeleton, traced path, and target points.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    cropped : bool
        Whether to crop around the current head position.
    environment : NeuronTrackingEnvironment
        The neuron tracking environment containing the state.
    sections : dict
        The sections of the neuron skeleton to draw. If None, they are derived
        from the current unvisited tree.
    skeleton_color : str
        Color for the skeleton lines.
    path_color : str
        Color for the traced path.
    target_color : str
        Color for the target points.
    step_size : float
        Step size for target point computation.
    dim : int
        Dimension to drop for 2D projection (0=z, 1=y, 2=x).
    size_scale : float
        Scales line widths and marker sizes for visibility.
    """
    env = environment
    size_scale = max(float(size_scale), 0.1)
    line_zorder = 2
    marker_zorder = 4
    if sections is None:
        sections = get_unvisited_sections(env)

    ax.clear()

    center_y = None
    center_x = None
    i = 1 if dim == 0 else 0
    j = 1 if dim == 2 else 2
    y_min = x_min = y_max = x_max = None
    if cropped and len(env.paths) > 0 and len(env.paths[0]) > 0:
        head = env.paths[0][-1]
        center_y = -float(head[i])
        center_x = float(head[j])
        half = float(getattr(env, 'radius', 40))
        y_min, y_max = center_y - half, center_y + half
        x_min, x_max = center_x - half, center_x + half

    def in_crop(y, x):
        if y_min is None:
            return True
        return (y_min <= y <= y_max) and (x_min <= x <= x_max)

    for _, section_data in sections.items():
        for seg in section_data:
            p0, p1 = seg[0], seg[1]
            y0, x0 = -float(p0[i]), float(p0[j])
            y1, x1 = -float(p1[i]), float(p1[j])
            if not cropped or (in_crop(y0, x0) or in_crop(y1, x1)):
                ax.plot([x0, x1], [y0, y1], color=skeleton_color, linewidth=2.0 * size_scale, alpha=1.0, zorder=line_zorder)

    if len(env.paths) > 0 and len(env.paths[0]) > 0:
        ys = [-float(pt[i]) for pt in env.paths[0]]
        xs = [float(pt[j]) for pt in env.paths[0]]
        if not cropped:
            ax.plot(xs, ys, color=path_color, linewidth=2.0 * size_scale, zorder=line_zorder)
        else:
            filt = [in_crop(y, x) for y, x in zip(ys, xs)]
            for k in range(1, len(xs)):
                if filt[k - 1] or filt[k]:
                    ax.plot([xs[k - 1], xs[k]], [ys[k - 1], ys[k]], color=path_color, linewidth=1.0 * size_scale, zorder=line_zorder)

    try:
        if len(env.paths) > 0 and len(env.paths[0]) > 0:
            target_vectors = getattr(env, 'target_vectors', None)
            if target_vectors is not None and target_vectors.numel() > 0:
                target_points = (env.paths[0][-1].unsqueeze(0) + target_vectors).detach().cpu().numpy()
            else:
                target_points = None

            if target_points is not None and len(target_points) > 0:
                tys = [-float(p[i]) for p in target_points]
                txs = [float(p[j]) for p in target_points]
                if not cropped:
                    ax.scatter(txs, tys, color=target_color, marker='x', s=100 * size_scale, linewidths=1.0 * size_scale, zorder=marker_zorder)
                else:
                    txs_c = []
                    tys_c = []
                    for ty, tx in zip(tys, txs):
                        if in_crop(ty, tx):
                            tys_c.append(ty)
                            txs_c.append(tx)
                    if len(txs_c) > 0:
                        ax.scatter(txs_c, tys_c, color=target_color, marker='x', s=100 * size_scale, linewidths=1.0 * size_scale, zorder=marker_zorder)
    except Exception as e:
        print(f"Warning: target point computation failed: {e}")

    tree = getattr(env, 'unvisited_tree', None)
    id_to_idx = getattr(env, 'id_to_idx', {})
    terminal_nodes = getattr(env, 'terminal_nodes', None)
    if terminal_nodes is not None and len(terminal_nodes) > 0:
        terminal_xs = []
        terminal_ys = []
        for node_id in terminal_nodes:
            idx = id_to_idx.get(int(node_id))
            if idx is None or tree is None or idx >= tree.shape[0]:
                continue
            node = tree[idx]
            ty = -float(node[i + 2])
            tx = float(node[j + 2])
            if not cropped or in_crop(ty, tx):
                terminal_ys.append(ty)
                terminal_xs.append(tx)
        if len(terminal_xs) > 0:
            ax.scatter(terminal_xs, terminal_ys, color='red', s=100 * size_scale, marker='o', facecolors='none', linewidths=1.0 * size_scale, zorder=marker_zorder)

    if env.section_nodes is not None and tree is not None and getattr(tree, 'ndim', 0) >= 2 and tree.shape[0] > 0:
        valid_ys = []
        valid_xs = []
        for node_id in env.section_nodes:
            idx = id_to_idx.get(int(node_id))
            if idx is None or idx >= tree.shape[0]:
                continue
            node = tree[idx]
            vy = -float(node[i + 2])
            vx = float(node[j + 2])
            if not cropped or in_crop(vy, vx):
                valid_ys.append(vy)
                valid_xs.append(vx)
        if len(valid_xs) > 0:
            ax.scatter(valid_xs, valid_ys, color='blue', s=80 * size_scale, marker='o', alpha=0.5, zorder=marker_zorder)

        if env.paths:
            # plot the nearest point on the unvisited tree to the current head position.
            head = env.paths[0][-1] if len(env.paths) > 0 and len(env.paths[0]) > 0 else None
            nearest_point, _ = tracking_reward._get_nearest_point(head, tree, id_to_idx, env.section_nodes, adj_dict=getattr(env, 'adj_dict', None))
            if nearest_point is not None:
                vy = -float(nearest_point[i])
                vx = float(nearest_point[j])
                if not cropped or in_crop(vy, vx):
                    ax.scatter([vx], [vy], color='cyan', s=70 * size_scale, marker='*', linewidths=1.0 * size_scale, alpha=1.0, zorder=marker_zorder+1)

    # Plot branch roots as yellow dots
    branch_roots = getattr(env, 'branch_roots', None)
    if branch_roots is not None and len(branch_roots) > 0:
        branch_ys = [-float(br[i]) for br in branch_roots]
        branch_xs = [float(br[j]) for br in branch_roots]
        if not cropped:
            ax.scatter(branch_xs, branch_ys, color='lime', s=100 * size_scale, marker='o', facecolors='none', linewidths=1.0 * size_scale, zorder=marker_zorder)
        else:
            branch_xs_c = []
            branch_ys_c = []
            for by, bx in zip(branch_ys, branch_xs):
                if in_crop(by, bx):
                    branch_ys_c.append(by)
                    branch_xs_c.append(bx)
            if len(branch_xs_c) > 0:
                ax.scatter(branch_xs_c, branch_ys_c, color='lime', s=100 * size_scale, marker='o', facecolors='none', linewidths=1.0 * size_scale, zorder=marker_zorder)

    # Plot cut ends as purple dots
    cut_ends = getattr(env, 'cut_ends', None)
    if cut_ends is not None and len(cut_ends) > 0:
        cut_ys = []
        cut_xs = []
        id_to_idx = getattr(env, 'id_to_idx', {})
        tree = getattr(env, 'unvisited_tree', None)
        if tree is not None and getattr(tree, 'ndim', 0) >= 2 and tree.shape[0] > 0:
            for cut_id in cut_ends:
                idx = id_to_idx.get(int(cut_id))
                if idx is not None and idx < tree.shape[0]:
                    node = tree[idx]
                    cy = -float(node[i + 2])
                    cx = float(node[j + 2])
                    if not cropped or in_crop(cy, cx):
                        cut_ys.append(cy)
                        cut_xs.append(cx)
            if len(cut_xs) > 0:
                ax.scatter(cut_xs, cut_ys, color='purple', s=130 * size_scale, marker='D', facecolors='none', linewidths=1.0 * size_scale, zorder=marker_zorder)

    if cropped and y_min is not None:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title('2D skeleton/path/targets (cropped)')
    elif len(sections) == 0:
        ax.set_title('2D path/targets (no unvisited tree)')
    else:
        ax.set_title('2D skeleton/path/targets')

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def run_expert_episode(env):
    """Run one expert-only episode and visualize the final traced result.

    Mirrors behavior_cloning expert stepping logic without storing transitions
    or running optimization.
    """
    import matplotlib.pyplot as plt
    from neurotrack.training.behavior_cloning import select_expert_action

    def _current_target_action_from_env(environment):
        target_vectors = getattr(environment, 'target_vectors', None)
        if target_vectors is None:
            target_vectors = torch.zeros((1, 3), dtype=torch.float32)
        return torch.as_tensor(target_vectors, dtype=torch.float32).view(-1, 3)

    obs = env.reset(move_to_next=False, return_state=True)
    prev_expert_action = None
    total_reward = 0.0
    steps_done = 0
    stop_steps = 0
    continue_steps = 0
    all_step_dists = []
    stop_step_dists = []
    continue_step_dists = []
    max_steps_limit = 50000  # Safety limit to prevent infinite loops

    for _step_idx in count():
        if steps_done >= max_steps_limit:
            print(f"\nWARNING: Reached max step limit ({max_steps_limit}). Breaking to prevent infinite loop.")
            print(f"  Final state: {steps_done} steps, {len(env.finished_paths)} paths finished")
            print(f"  section_nodes: {env.section_nodes is not None}")
            print("  target vectors available")
            print(f"  target_vectors: {env.target_vectors}")
            break

        current_target_vectors = _current_target_action_from_env(env)
        expert_action = select_expert_action(current_target_vectors, previous_action=prev_expert_action)

        next_obs, reward, terminated, _truncated, info = env.step(expert_action, step_count=steps_done)
        steps_done += 1
        total_reward += float(torch.as_tensor(reward, dtype=torch.float32).item())
        step_len = float(torch.linalg.norm(expert_action).item())
        all_step_dists.append(step_len)
        status = info.get('status', 'unknown')
        if status == 'choose_stop':
            stop_steps += 1
            stop_step_dists.append(step_len)
        else:
            continue_steps += 1
            continue_step_dists.append(step_len)

        if info.get('terminate_episode', False):
            break

        if terminated:
            obs = env.get_state()
            prev_expert_action = None
        else:
            obs = next_obs
            prev_expert_action = expert_action.detach().cpu()

    fig = plt.figure(figsize=(12, 8))
    show_state(env, fig, live=False)

    num_paths = len(getattr(env, 'finished_paths', []))
    long_paths = 0
    no_start_paths = 0
    for path in getattr(env, 'finished_paths', []):
        path_len = len(path) if path is not None else 0
        if path_len > 1:
            long_paths += 1
        elif path_len == 1:
            no_start_paths += 1

    if steps_done > 0:
        stop_fraction = float(stop_steps) / float(steps_done)
        continue_fraction = float(continue_steps) / float(steps_done)
    else:
        stop_fraction = 0.0
        continue_fraction = 0.0

    print(f"steps_done: {steps_done}")
    print(f"stop_steps: {stop_steps}")
    print(f"continue_steps: {continue_steps}")
    print(f"stop_fraction: {stop_fraction:.6f}")
    print(f"continue_fraction: {continue_fraction:.6f}")
    print(f"mean step length for all steps: {float(torch.as_tensor(all_step_dists).mean().item()):.6f}" if all_step_dists else "mean step length for all steps: N/A")
    print(f"mean step length for stop steps: {float(torch.as_tensor(stop_step_dists).mean().item()):.6f}" if stop_step_dists else "mean step length for stop steps: N/A")
    print(f"mean step length for continue steps: {float(torch.as_tensor(continue_step_dists).mean().item()):.6f}" if continue_step_dists else "mean step length for continue steps: N/A")
    print(f"finished_paths: {num_paths}")
    print(f"long_paths: {long_paths}")
    print(f"no_start_paths: {no_start_paths}")
    print(f"total_reward: {total_reward:.6f}")

    return {
        'steps_done': int(steps_done),
        'stop_steps': int(stop_steps),
        'continue_steps': int(continue_steps),
        'mean_step_length_all': float(torch.as_tensor(all_step_dists).mean().item()) if all_step_dists else None,
        'mean_step_length_stop': float(torch.as_tensor(stop_step_dists).mean().item()) if stop_step_dists else None,
        'mean_step_length_continue': float(torch.as_tensor(continue_step_dists).mean().item()) if continue_step_dists else None,
        'num_paths': num_paths,
        'long_paths': int(long_paths),
        'no_start_paths': int(no_start_paths),
        'total_reward': float(total_reward),
    }


def _policy_action_step(actor, obs, stochastic=False):
    """Compute one policy action from the current observation.

    Returns
    -------
    direction : torch.Tensor
        Decoded 3D direction vector to pass to env.step.
    choose_stop : bool
        Whether the action norm is below the environment stall threshold.
    auxiliary_metric : Optional[float]
        Not used in threshold-based mode.
    """
    model_device = next(actor.parameters()).device
    obs_model = prepare_observation_for_model(obs.detach(), device=model_device, model_dtype=torch.float32)

    with torch.no_grad():
        actor_out = actor(obs_model)

    policy_output_mode = getattr(actor, 'policy_output_mode', 'direct_vector')
    if policy_output_mode == 'direct_vector':
        direction = actor_out[0, :3]
        stop_threshold = float(getattr(getattr(actor, 'env', None), 'stall_threshold', 1.0))
        choose_stop = bool(torch.linalg.norm(direction).item() < stop_threshold)
        return direction.view(3).detach().cpu(), choose_stop, None

    direction_dist = sample_from_output(actor_out)
    if stochastic:
        action = direction_dist.sample()[0]
    else:
        action = direction_dist.mean[0]
    return action.detach().cpu(), False, None


def run_policy(env, actor, stochastic=False):
    """
    Step through environment dynamics one policy action at a time.

    Controls:
      - Enter: take one policy step
        - e: run policy action burst for N steps, then pause for input
        - f: run policy actions until the episode terminates
        - t: issue a zero-vector step (treated as stop by stall threshold)
      - r: reset environment
      - b: branch at current point
      - q: quit
    """
    from IPython.display import display as ipy_display
    from IPython.display import HTML
    import matplotlib.pyplot as plt

    plt.ioff()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, wspace=0.05, hspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    axes = [ax0, ax1, ax2, ax3, ax4, ax5]

    skeleton_color = 'tab:gray'
    path_color = 'tab:orange'
    target_color = 'tab:red'
    sections = get_unvisited_sections(env)

    total_steps = 0
    total_policy_steps = 0
    frame_handle = None
    stats_handle = None
    last_stats_text = None
    last_info_snapshot = None

    def _compute_step_metrics(direction):
        direction_t = torch.as_tensor(direction, dtype=torch.float32)
        target_vectors = env.target_vectors
        target_norms = torch.linalg.norm(target_vectors, dim=1) if target_vectors is not None else None
        true_stop = bool(
            target_norms is not None
            and target_norms.numel() > 0
            and target_norms.min().item() < getattr(env, 'stall_threshold', 1.0)
        )
        pred_stop = bool(torch.linalg.norm(direction_t).item() < getattr(env, 'stall_threshold', 1.0))

        if target_vectors is None or torch.as_tensor(target_vectors).numel() == 0:
            loss_value = 0.0
            step_mse_value = 0.0
        else:
            action_batch, target_tensor, target_mask = bc._prepare_target_candidates(
                direction_t.unsqueeze(0),
                target_vectors,
            )
            loss_config = bc._build_supervision_loss_config(
                continue_target_norm_threshold=float(getattr(env, 'stall_threshold', 1.0)),
                continue_weight=1.0,
                norm_floor=0.0,
                norm_floor_weight=0.0,
                stop_violation_weight=1.0,
            )
            loss_tensor, context = bc._compute_supervision_loss(
                pred_actions=action_batch,
                target_tensor=target_tensor,
                target_mask=target_mask,
                loss_config=loss_config,
            )
            loss_value = float(loss_tensor.item())
            step_mse_value = float(context.step_mse)
        step_len_value = float(torch.linalg.norm(direction_t).item())

        return {
            'avg_loss': loss_value,
            'step_mse': step_mse_value,
            'false_stop_rate': float(pred_stop and not true_stop),
            'false_continue_rate': float((not pred_stop) and true_stop),
            'avg_step_length': step_len_value,
        }

    def _average_metrics(metrics_list):
        if not metrics_list:
            return None
        n = float(len(metrics_list))
        return {
            'avg_loss': sum(m['avg_loss'] for m in metrics_list) / n,
            'step_mse': sum(m['step_mse'] for m in metrics_list) / n,
            'false_stop_rate': sum(m['false_stop_rate'] for m in metrics_list) / n,
            'false_continue_rate': sum(m['false_continue_rate'] for m in metrics_list) / n,
            'avg_step_length': sum(m['avg_step_length'] for m in metrics_list) / n,
        }

    def _render(observation, reward=None, info=None, direction=None, step_metrics=None):
        nonlocal last_stats_text, last_info_snapshot
        for a in axes:
            a.clear()

        img = env.img.data[0].amax(dim=0)
        path_im = env.img.data[-1].amax(dim=0)
        ax0.imshow(img, cmap='gray')
        ax0.imshow(path_im, cmap='gray', alpha=0.5)
        ax0.set_title('Full image + path')
        ax0.axis('off')

        patch = observation[0]
        z_index = getattr(env, 'radius', patch.shape[1] // 2)
        z_index = int(max(0, min(int(z_index), patch.shape[1] - 1)))
        slice_ = patch[:, z_index]
        ax1.imshow(slice_[0], cmap='gray')
        ax1.imshow(slice_[-1], cmap='gray', alpha=0.5)
        ax1.set_title('Cropped patch + path')
        ax1.axis('off')

        current_sections = get_unvisited_sections(env)

        draw_2d_panel(ax2, env, cropped=False, sections=current_sections, dim=0,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        draw_2d_panel(ax3, env, cropped=True, sections=current_sections, dim=0,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        draw_2d_panel(ax4, env, cropped=True, sections=current_sections, dim=1,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        draw_2d_panel(ax5, env, cropped=True, sections=current_sections, dim=2,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)

        nonlocal frame_handle, stats_handle
        if frame_handle is None:
            frame_handle = ipy_display(fig, display_id=True)
            stats_handle = ipy_display(HTML('<pre></pre>'), display_id=True)
        else:
            frame_handle.update(fig)

        if direction is None:
            direction_norm = 0.0
            direction_list = None
        else:
            direction_tensor = torch.as_tensor(direction, dtype=torch.float32)
            direction_norm = float(torch.linalg.norm(direction_tensor).item())
            direction_list = direction_tensor.tolist()

        lines = [
            f'step: {total_steps} (policy steps: {total_policy_steps})',
            f'direction: {direction_list}',
            f'direction_norm: {direction_norm:.4f}',
            f'reward: {reward}',
        ]
        if step_metrics is None:
            lines.extend([
                'avg_loss: None',
                'avg_step_mse: None',
                'false_stop_rate: None',
                'false_continue_rate: None',
                'avg_step_length: None',
            ])
        else:
            lines.extend([
                f"avg_loss: {step_metrics['avg_loss']:.6f}",
                f"avg_step_mse: {step_metrics['step_mse']:.6f}",
                f"false_stop_rate: {step_metrics['false_stop_rate']:.6f}",
                f"false_continue_rate: {step_metrics['false_continue_rate']:.6f}",
                f"avg_step_length: {step_metrics['avg_step_length']:.6f}",
            ])
        if info is None:
            lines.extend([
                'status: None',
                'terminated: None',
                'diagnostics: threshold-stop mode',
            ])
        else:
            lines.extend([
                f"status: {info.get('status')}",
                f"terminated: {info.get('terminate_episode')}",
                'diagnostics: threshold-stop mode',
            ])

        if stats_handle is not None:
            rendered_text = '\n'.join(lines)
            stats_handle.update(HTML('<pre>' + rendered_text + '</pre>'))
            last_stats_text = rendered_text
            last_info_snapshot = info

    # Initial render before any stepping so the user sees the starting state and stop label.
    _render(env.get_state(), reward=None, info=None, direction=None)

    while True:
        action_key = input('Press Enter for policy step, or [e burst, f full episode, t stop, r reset, b branch, q quit]: ').strip().lower()
        reward = None

        if action_key == 'q':
            break
        elif action_key == 'r':
            observation = env.reset(return_state=True)
            sections = get_unvisited_sections(env)
            print('Environment reset.')
            _render(observation, reward=None, info=None, direction=None)
            continue
        elif action_key == 'b':
            if len(env.paths) > 0 and len(env.paths[0]) > 0:
                point = env.paths[0][-1]
                env.paths.append([point])
                env._append_branch_root(point)
                env.img.draw_point(point, radius=(env.step_width / 2.35), channel=-1, mode='gaussian', binary=False)
            sections = get_unvisited_sections(env)
            print('Added branch at current point.')
            _render(env.get_state(), reward=None, info=None, direction=None)
            continue
        elif action_key == 'e':
            try:
                burst_steps = int(input('Number of policy steps to run [default=1]: ').strip() or '1')
            except ValueError:
                burst_steps = 1
                print('Invalid step count; defaulting to 1')

            if burst_steps <= 0:
                print('Step count must be >= 1')
                continue

            last_observation = env.get_state()
            last_reward = None
            last_info = None
            last_direction = None
            episode_terminated = False

            burst_metrics = []
            for _ in range(burst_steps):
                obs = env.get_state()
                direction, _choose_stop, _unused_metric = _policy_action_step(actor=actor, obs=obs, stochastic=stochastic)
                burst_metrics.append(_compute_step_metrics(direction))
                last_observation, last_reward, terminated, truncated, last_info = env.step(direction, verbose=True)
                total_steps += 1
                total_policy_steps += 1
                last_direction = direction

                if last_info.get('terminate_episode'):
                    print('All paths finished during policy burst. Final state is shown; press r to reset.')
                    episode_terminated = True
                    break

            if episode_terminated:
                _render(
                    last_observation,
                    reward=last_reward,
                    info=last_info,
                    direction=last_direction,
                    step_metrics=_average_metrics(burst_metrics),
                )
                if last_stats_text is not None:
                    print('Final episode stats:')
                    print(last_stats_text)
            else:
                _render(
                    last_observation,
                    reward=last_reward,
                    info=last_info,
                    direction=last_direction,
                    step_metrics=_average_metrics(burst_metrics),
                )
            continue
        elif action_key == 'f':
            last_observation = env.get_state()
            last_reward = None
            last_info = None
            last_direction = None
            episode_terminated = False
            max_episode_steps = 50000

            episode_metrics = []
            for _ in range(max_episode_steps):
                obs = env.get_state()
                direction, _choose_stop, _unused_metric = _policy_action_step(actor=actor, obs=obs, stochastic=stochastic)
                episode_metrics.append(_compute_step_metrics(direction))
                last_observation, last_reward, terminated, truncated, last_info = env.step(direction, verbose=True)
                total_steps += 1
                total_policy_steps += 1
                last_direction = direction

                if last_info.get('terminate_episode'):
                    episode_terminated = True
                    break

            averaged_metrics = _average_metrics(episode_metrics)
            if episode_terminated:
                print('All paths finished during full policy episode run. Final state is shown; press r to reset.')
                _render(
                    last_observation,
                    reward=last_reward,
                    info=last_info,
                    direction=last_direction,
                    step_metrics=averaged_metrics,
                )
                if last_stats_text is not None:
                    print('Final episode stats:')
                    print(last_stats_text)
            else:
                print(f'WARNING: Reached max full-episode step limit ({max_episode_steps}).')
                _render(
                    last_observation,
                    reward=last_reward,
                    info=last_info,
                    direction=last_direction,
                    step_metrics=averaged_metrics,
                )
            continue

        if action_key == 't':
            direction = torch.zeros((3,), dtype=torch.float32)
            step_metrics = _compute_step_metrics(direction)
            observation, reward, terminated, truncated, info = env.step(direction, verbose=True)
        else:
            obs = env.get_state()
            direction, _choose_stop, _unused_metric = _policy_action_step(actor=actor, obs=obs, stochastic=stochastic)
            step_metrics = _compute_step_metrics(direction)

            observation, reward, terminated, truncated, info = env.step(direction, verbose=True)
            total_policy_steps += 1

        total_steps += 1
        _render(observation, reward=reward, info=info, direction=direction, step_metrics=step_metrics)

        if info.get('terminate_episode'):
            print('All paths finished. Final state is shown; press r to reset.')
            if last_stats_text is not None:
                print('Final episode stats:')
                print(last_stats_text)

    try:
        sections = get_unvisited_sections(env)
        draw_2d_panel(ax2, env, cropped=False, sections=sections, dim=0,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        draw_2d_panel(ax3, env, cropped=True, sections=sections, dim=0,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        draw_2d_panel(ax4, env, cropped=True, sections=sections, dim=1,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        draw_2d_panel(ax5, env, cropped=True, sections=sections, dim=2,
                      skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        ipy_display(plt.gcf())
    except Exception:
        pass


# Backward-compatible alias for existing imports/call sites.
policy_step = run_policy


def manual_step(env, step_size=4.0, display_mode='all'):
    """
    Interactive manual stepping helper.

        Display modes:
            - all: full six-panel layout (image, crop, and 2D projections)
            - tree: only XY projection of the full unvisited skeleton/path view
            - image: only XY projection of the full image with path overlay

    Controls:
      - w/a/s/d: move in-plane (y/x)
      - p/l: move along z (positive/negative)
        - t: zero direction (treated as stop by stall threshold)
        - z: zero direction (debug no-op)
        - x: use expert direction
                - e: run expert action burst for N steps, then pause for input
                - g: enter an action scale hint for manual probing
      - r: reset environment
      - b: branch at current point
      - q: quit
    """
    from IPython.display import display as ipy_display
    from IPython.display import HTML
    import matplotlib.pyplot as plt

    plt.ioff()
    valid_display_modes = {'all', 'tree', 'image'}
    if display_mode not in valid_display_modes:
        raise ValueError(f"Invalid display_mode '{display_mode}'. Expected one of {sorted(valid_display_modes)}")

    device = env.img.data.device
    user_input_dict = {
        'a': torch.tensor([0.0, 0.0, -1.0]),
        'w': torch.tensor([0.0, -1.0, 0.0]),
        'd': torch.tensor([0.0, 0.0, 1.0]),
        's': torch.tensor([0.0, 1.0, 0.0]),
        'p': torch.tensor([1.0, 0.0, 0.0]),
        'l': torch.tensor([-1.0, 0.0, 0.0]),
        'z': torch.tensor([0.0, 0.0, 0.0]),
    }

    fig = plt.figure(figsize=(16, 12))
    if display_mode == 'all':
        gs = fig.add_gridspec(
            3,
            2,
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.99,
            wspace=0.03,
            hspace=0.08,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        axes = [ax0, ax1, ax2, ax3, ax4, ax5]
        ax_main = None
    else:
        ax_main = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        axes = [ax_main]
        ax0 = None
        ax1 = None
        ax2 = None
        ax3 = None
        ax4 = None
        ax5 = None

    skeleton_color = 'tab:gray'
    path_color = 'tab:orange'
    target_color = 'tab:red'
    total_steps = 0
    frame_handle = None
    stats_handle = None
    last_stats_text = None
    last_info_snapshot = None

    def _get_expert_action_and_stop():
        from neurotrack.training.behavior_cloning import select_expert_action

        prev_action = env.paths[0][-1] - env.paths[0][-2] if len(env.paths[0]) > 1 else None
        expert_action = select_expert_action(target_vectors=getattr(env, 'target_vectors', None), previous_action=prev_action)
        return expert_action, False

    def _render(observation, reward=None, info=None, action=None):
        nonlocal last_stats_text, last_info_snapshot
        for a in axes:
            a.clear()

        if display_mode == 'all':
            img = env.img.data[0].amax(dim=0)
            path_im = env.img.data[-1].amax(dim=0)
            ax0.imshow(img, cmap='gray')
            ax0.imshow(path_im, cmap='gray', alpha=0.5)
            ax0.set_title('Full image + path')
            ax0.axis('off')

            patch = observation[0]
            z_index = getattr(env, 'radius', patch.shape[1] // 2)
            z_index = int(max(0, min(int(z_index), patch.shape[1] - 1)))
            slice_ = patch[:, z_index]
            ax1.imshow(slice_[0], cmap='gray')
            ax1.imshow(slice_[-1], cmap='gray', alpha=0.5)
            ax1.set_title('Cropped patch + path')
            ax1.axis('off')

            current_sections = get_unvisited_sections(env)

            draw_2d_panel(ax2, env, cropped=False, sections=current_sections, dim=0,
                          skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
            draw_2d_panel(ax3, env, cropped=True, sections=current_sections, dim=0,
                          skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
            draw_2d_panel(ax4, env, cropped=True, sections=current_sections, dim=1,
                          skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
            draw_2d_panel(ax5, env, cropped=True, sections=current_sections, dim=2,
                          skeleton_color=skeleton_color, path_color=path_color, target_color=target_color)
        elif display_mode == 'tree':
            from matplotlib.lines import Line2D

            current_sections = get_unvisited_sections(env)
            draw_2d_panel(
                ax_main,
                env,
                cropped=False,
                sections=current_sections,
                dim=0,
                skeleton_color=skeleton_color,
                path_color=path_color,
                target_color=target_color,
                size_scale=3.0,
            )
            ax_main.set_title('')
            legend_handles = [
                Line2D([0], [0], marker='x', color='none', markeredgecolor=target_color, markersize=10, markeredgewidth=4, label='Target points'),
                Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markerfacecolor='none', markersize=9, markeredgewidth=4, label='Terminal points'),
                Line2D([0], [0], marker='o', color='none', markeredgecolor='blue', markerfacecolor='blue', alpha=0.5, markersize=8, label='Section nodes'),
                Line2D([0], [0], marker='*', color='none', markeredgecolor='cyan', markerfacecolor='cyan', markersize=10, label='Nearest point'),
                Line2D([0], [0], marker='o', color='none', markeredgecolor='lime', markerfacecolor='none', markersize=9, markeredgewidth=4, label='Branch roots'),
                Line2D([0], [0], marker='D', color='none', markeredgecolor='purple', markerfacecolor='none', markersize=8, markeredgewidth=4, label='Cut ends'),
            ]
            # Add vertical breathing room without hard-coding absolute limits.
            ax_main.margins(y=0.2)
            # ax_main.legend(
            # handles=legend_handles,
            # loc='center right',
            # framealpha=0.85,
            # fontsize=24,
            # markerscale=4.0,
            # borderpad=0.8,
            # labelspacing=1.0,
            # handletextpad=0.7,
            # )
            ax_main.set_position([0.0, 0.0, 1.0, 1.0])
        else:
            img = env.img.data[0].amax(dim=0)
            path_im = env.img.data[-1].amax(dim=0)
            ax_main.imshow(img, cmap='gray')
            ax_main.imshow(path_im, cmap='gray', alpha=0.5)
            ax_main.axis('off')
            ax_main.set_position([0.0, 0.0, 1.0, 1.0])

        nonlocal frame_handle, stats_handle
        if frame_handle is None:
            frame_handle = ipy_display(fig, display_id=True)
            stats_handle = ipy_display(HTML('<pre></pre>'), display_id=True)
        else:
            frame_handle.update(fig)

        lines = [
            f'step: {total_steps}',
            f'action: {None if action is None else torch.as_tensor(action, dtype=torch.float32).tolist()}',
            f'reward: {reward}',
        ]
        if info is None:
            lines.extend([
                'status: None',
                'terminated: None',
                'diagnostics: threshold-stop mode',
            ])
        else:
            lines.extend([
                f"status: {info.get('status')}",
                f"terminated: {info.get('terminate_episode')}",
                'diagnostics: threshold-stop mode',
            ])

        if stats_handle is not None:
            rendered_text = '\n'.join(lines)
            stats_handle.update(HTML('<pre>' + rendered_text + '</pre>'))
            last_stats_text = rendered_text
            last_info_snapshot = info

    # Initial render before any stepping so the user sees the starting state and stop label.
    _render(env.get_state(), reward=None, info=None, action=None)

    while True:
        action_key = input('Choose an action [w/a/s/d, p/l, x, e, t, z, g, r, b, q]: ').strip().lower()
        reward = None

        if action_key == 'q':
            break
        elif action_key == 'r':
            observation = env.reset(return_state=True)
            _render(observation, reward=None, info=None, action=None)
        elif action_key == 'b':
            point = env.paths[0][-1]
            env.paths.append([point])
            env._append_branch_root(point)
            env.img.draw_point(point, radius=(env.step_width / 2.35), channel=-1, mode='gaussian', binary=False)
            _render(env.get_state(), reward=None, info=None, action=None)

        else:

            if action_key not in user_input_dict and action_key not in {'x', 'e', 't', 'g'}:
                print(f"Unrecognized action '{action_key}'. Valid: w/a/s/d, p/l, x, e, t, z, g, r, b, q")
                continue

            if action_key == 'x': # select expert action
                action, choose_stop = _get_expert_action_and_stop()
                use_explicit_stop = True
            elif action_key == 'e':
                try:
                    burst_steps = int(input('Number of expert steps to run [default=1]: ').strip() or '1')
                except ValueError:
                    burst_steps = 1
                    print('Invalid step count; defaulting to 1')
                if burst_steps <= 0:
                    print('Step count must be >= 1')
                    continue

                last_observation = env.get_state()
                last_reward = None
                last_info = None
                last_action = None
                episode_terminated = False

                for _ in range(burst_steps):
                    action, choose_stop = _get_expert_action_and_stop()
                    last_observation, last_reward, terminated, truncated, last_info = env.step(action, verbose=True, step_count=total_steps)
                    total_steps += 1
                    last_action = action

                    if last_info.get('terminate_episode'):
                        print('All paths finished during expert burst. Final state is shown; press r to reset.')
                        episode_terminated = True
                        break

                if episode_terminated:
                    _render(last_observation, reward=last_reward, info=last_info, action=last_action)
                    if last_stats_text is not None:
                        print('Final episode stats:')
                        print(last_stats_text)
                else:
                    _render(last_observation, reward=last_reward, info=last_info, action=last_action)
                continue
            elif action_key == 't':
                action = torch.zeros((3,), dtype=torch.float32, device=device)
                choose_stop = True
                use_explicit_stop = True
            elif action_key == 'g':
                from neurotrack.training.behavior_cloning import select_expert_action

                prev_action = env.paths[0][-1] - env.paths[0][-2] if len(env.paths[0]) > 1 else None
                direction = select_expert_action(target_vectors=getattr(env, 'target_vectors', None), previous_action=prev_action)
                try:
                    stop_logit = float(input('stop logit for action[3] (e.g. 4.0 stop, -4.0 continue) [default=4.0]: ').strip() or '4.0')
                except ValueError:
                    stop_logit = 4.0
                    print('Invalid stop logit; defaulting to 4.0')
                action = direction.to(device=device, dtype=torch.float32)
                choose_stop = bool(torch.linalg.norm(action).item() < float(getattr(env, 'stall_threshold', 1.0)))
                use_explicit_stop = True
            else:
                action = user_input_dict[action_key].to(device=device)
                action = action * getattr(env, 'target_step_len', step_size)
                choose_stop = False
                use_explicit_stop = True

            observation, reward, terminated, truncated, info = env.step(action, verbose=True, step_count=total_steps)
            total_steps += 1

            _render(observation, reward=reward, info=info, action=action)
            # fig.canvas.draw()  # ensure the figure is rendered before saving
            # fig.savefig(f'step{step_count}_snapshot.png')

            if info.get('terminate_episode'):
                print('All paths finished. Final state is shown; press r to reset.')
                if last_stats_text is not None:
                    print('Final episode stats:')
                    print(last_stats_text)

    try:
        _render(env.get_state(), reward=None, info=last_info_snapshot, action=None)
        ipy_display(plt.gcf())
        # save the current state of the figure to a file for reference
        # fig.canvas.draw()  # ensure the figure is rendered before saving
        # fig.savefig(f'step{step_count}_snapshot.png')
    except Exception:
        pass


def show_state(env, fig, live=False, ep_return=None, reward=None, policy_loss=None):
    """Show a single max-intensity projection (over first spatial axis) of the whole neuron.

    - Base image: input neuron image (grayscale)
    - Overlay: current path (plasma colormap, semi-transparent)
    Also draws a red rectangle centered at the current position with
    width and height equal to env.radius (in pixels).
    If live=True, also show a second subplot to the right with the current
    cropped state (env.get_state()) as an overlay: image (grayscale) + path (plasma).
    """
    from matplotlib import patches  # local import to avoid global dependency

    print(f"image: {env.current_neuron_info['neuron_name']}")
    display.clear_output(wait=True)

    # Reset view; one or more axes depending on `live`
    fig.clf()
    if live:
        # GridSpec: Left spans both rows, right has two stacked axes.
        # Left is wider than each right axis; the midline aligns with the split between right-top and right-bottom.
        gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], wspace=0.05, hspace=0.05)
        ax_left = fig.add_subplot(gs[:, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_right_bottom = fig.add_subplot(gs[1, 1])
    else:
        ax_left = fig.add_subplot(1, 1, 1)

    # Prepare volumes
    env_img = env.img.data.clone().detach().cpu()
    if env_img.dtype == torch.uint8:
        env_img = env_img.float() / 255.0

    # Separate channels: img (all but last) and path (last)
    img = env_img[:-1]                  # (C_img, H, W, D) or (1, H, W, D)
    path = env_img[-1]                  # (H, W, D)

    # Max-intensity projection of input image (img) and path
    if img.ndim == 4:
        img_vol = img.amax(dim=0)       # (H, W, D)
    else:
        img_vol = img                   # already 3D

    img_proj = img_vol.amax(dim=0)      # (W, D)
    path_proj = path.amax(dim=0)        # (W, D)

    # Show RGB MIP (left axis)
    ax_left.imshow(img_proj, cmap='gray', vmax=1.0, vmin=0.0)

    # Overlay path as a transparent colored mask
    ax_left.imshow(path_proj, cmap='plasma', alpha=0.5)

    # Overlay seed points as red dots (projected using y=seed[1], x=seed[2])
    h, w = img_proj.shape[0], img_proj.shape[1]
    if hasattr(env, 'seeds') and len(env.seeds) > 0:
        xs, ys = [], []
        for s in env.seeds:
            try:
                y_s = int(round(s[1]))
                x_s = int(round(s[2]))
            except Exception:
                continue
            if 0 <= y_s < h and 0 <= x_s < w:
                ys.append(y_s)
                xs.append(x_s)
        if xs:
            ax_left.scatter(xs, ys, s=25, c='red', marker='o', linewidths=0)

    if env.paths:
        # Draw red box around current position with size env.radius x env.radius
        pos = env.paths[0][-1]
        y = int(round(pos[1].item()))  # axis-1 (rows)
        x = int(round(pos[2].item()))  # axis-2 (cols)
        r = int(env.radius)
        h, w = img_proj.shape[0], img_proj.shape[1]

        # Compute box extents; ensure width and height equal to r, clipped to bounds
        half_down = r // 2
        half_up = r - half_down
        y0 = max(0, y - half_down)
        x0 = max(0, x - half_down)
        # Clip width/height so rectangle stays within image
        width = min(r, w - x0)
        height = min(r, h - y0)

        rect = patches.Rectangle((x0, y0), width, height, linewidth=1.5, edgecolor='red', facecolor='none')
        ax_left.add_patch(rect)

    # Optional right subplot: current cropped state overlay
    if live:
        try:
            obs = env.get_state()[0].detach().cpu()  # (C, H, W, D)
            if obs.dtype == torch.uint8:
                obs = obs.to(dtype=torch.float32) * (1.0 / 255.0)
            img_obs = obs[:-1]
            path_obs = obs[-1]
            if img_obs.ndim == 4:
                img_obs_vol = img_obs.amax(dim=0)  # (H, W, D)
            else:
                img_obs_vol = img_obs
            img_obs_proj = img_obs_vol.amax(dim=0)
            path_obs_proj = path_obs.amax(dim=0)
            ax_right_bottom.imshow(img_obs_proj, cmap='gray', vmax=1.0, vmin=0.0)
            ax_right_bottom.imshow(path_obs_proj, cmap='plasma', vmax=1.0, vmin=0.0, alpha=0.5)
            ax_right_bottom.axis('off')

            # Bottom-right: overlay path_obs on cropped true_density MIP
            center = env.paths[0][-1]
            density_patch = env.true_density.crop(center, env.radius, interp=False)[0]
            if density_patch.dtype == torch.uint8:
                density_patch = density_patch.float() / 255.0
            # Use first channel of density if multi-channel
            if density_patch.ndim == 4:
                density_ch = density_patch[0]
            else:
                density_ch = density_patch

            density_proj = density_ch[env.radius]  # single slice through center
            ax_right.imshow(density_proj, cmap='Reds', vmax=1.0, vmin=0.0)
            ax_right.imshow(path_obs_proj, cmap='Greens', vmax=1.0, vmin=0.0, alpha=0.5)
            ax_right.axis('off')
            if reward is not None:
                ax_right.set_title(f'Reward: {reward:.4f}')
        except Exception:
            # Fail quietly if state isn't available
            ax_right_bottom.axis('off')
            try:
                ax_right.axis('off')
            except Exception:
                pass

    ax_left.axis('off')
    display.display(fig)

    return

if __name__ == "__main__":
    pass
