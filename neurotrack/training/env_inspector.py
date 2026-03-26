#!/usr/bin/env python

""" Interface for interactively evaluating the tracking environment """
from itertools import count
from IPython import display
import torch
from neurotrack.data import loading as load
from neurotrack.environments import tracking_reward
from neurotrack.training.behavior_cloning import select_expert_action
from neurotrack.training.policy_utils import (
    decode_direct_vector_output,
    prepare_observation_for_model,
    sample_from_output,
)

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
                  skeleton_color='lightgray', path_color='red', target_color='blue', step_size=4.0, dim=0):
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
    """
    env = environment
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
                ax.plot([x0, x1], [y0, y1], color=skeleton_color, linewidth=3.0, alpha=0.8)

    if len(env.paths) > 0 and len(env.paths[0]) > 0:
        ys = [-float(pt[i]) for pt in env.paths[0]]
        xs = [float(pt[j]) for pt in env.paths[0]]
        if not cropped:
            ax.plot(xs, ys, color=path_color, linewidth=2.0)
        else:
            filt = [in_crop(y, x) for y, x in zip(ys, xs)]
            for k in range(1, len(xs)):
                if filt[k - 1] or filt[k]:
                    ax.plot([xs[k - 1], xs[k]], [ys[k - 1], ys[k]], color=path_color, linewidth=3.0)

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
                    ax.scatter(txs, tys, color=target_color, marker='x', s=50)
                else:
                    txs_c = []
                    tys_c = []
                    for ty, tx in zip(tys, txs):
                        if in_crop(ty, tx):
                            tys_c.append(ty)
                            txs_c.append(tx)
                    if len(txs_c) > 0:
                        ax.scatter(txs_c, tys_c, color=target_color, marker='x',s=50)
    except Exception as e:
        print(f"Warning: target point computation failed: {e}")

    if env.terminal_points is not None and len(env.terminal_points) > 0:
        term_ys = [-pt[i].item() for pt in env.terminal_points]
        term_xs = [pt[j].item() for pt in env.terminal_points]
        if not cropped:
            ax.scatter(term_xs, term_ys, color='purple', s=50, marker='o', facecolors='none', linewidths=1.5)
        else:
            term_xs_c = []
            term_ys_c = []
            for ty, tx in zip(term_ys, term_xs):
                if in_crop(ty, tx):
                    term_ys_c.append(ty)
                    term_xs_c.append(tx)
            if len(term_xs_c) > 0:
                ax.scatter(term_xs_c, term_ys_c, color='purple', s=50, marker='o', facecolors='none', linewidths=1.5)
    

    tree = getattr(env, 'unvisited_tree', None)
    id_to_idx = getattr(env, 'id_to_idx', {})
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
            ax.scatter(valid_xs, valid_ys, color='blue', s=20, marker='o', alpha=0.5)

        if env.paths:
            # plot the nearest point on the unvisited tree to the current head position.
            head = env.paths[0][-1] if len(env.paths) > 0 and len(env.paths[0]) > 0 else None
            nearest_point, _ = tracking_reward._get_nearest_point(head, tree, id_to_idx, env.section_nodes, adj_dict=getattr(env, 'adj_dict', None))
            if nearest_point is not None:
                vy = -float(nearest_point[i])
                vx = float(nearest_point[j])
                if not cropped or in_crop(vy, vx):
                    ax.scatter([vx], [vy], color='cyan', s=60, marker='*', alpha=0.8)

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

    def _current_target_action_from_env(environment):
        target_vectors = getattr(environment, 'target_vectors', None)
        if target_vectors is None:
            target_vectors = torch.zeros((1, 3), dtype=torch.float32)
        stop_label = bool(getattr(environment, 'target_stop_label', False))
        return torch.as_tensor(target_vectors, dtype=torch.float32).view(-1, 3), stop_label

    obs = env.reset(return_state=True)
    prev_expert_action = None
    total_reward = 0.0
    steps_done = 0

    for _step_idx in count():
        current_target_vectors, current_target_stop = _current_target_action_from_env(env)
        expert_action = select_expert_action(current_target_vectors, previous_action=prev_expert_action)
        expert_choose_stop = bool(current_target_stop)

        next_obs, reward, terminated, _truncated, info = env.step(expert_action, stop=expert_choose_stop)
        steps_done += 1
        total_reward += float(torch.as_tensor(reward, dtype=torch.float32).item())

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
    print(f"steps_done: {steps_done}")
    print(f"finished_paths: {num_paths}")
    print(f"total_reward: {total_reward:.6f}")

    return {
        'steps_done': int(steps_done),
        'num_paths': num_paths,
        'total_reward': float(total_reward),
    }


def _policy_action_step(actor, obs, stochastic=False):
    """Compute one policy action from the current observation.

    Returns
    -------
    direction : torch.Tensor
        Decoded 3D direction vector to pass to env.step.
    choose_stop : bool
        Explicit stop decision decoded from policy output when applicable.
    stop_probability : Optional[float]
        Decoded stop probability for logging (None for non-direct-vector policies).
    """
    model_device = next(actor.parameters()).device
    obs_model = prepare_observation_for_model(obs.detach(), device=model_device, model_dtype=torch.float32)

    with torch.no_grad():
        actor_out = actor(obs_model)

    policy_output_mode = getattr(actor, 'policy_output_mode', 'direct_vector')
    if policy_output_mode == 'direct_vector':
        direction, stop_prob, _choose_stop = decode_direct_vector_output(
            actor_out,
            stop_action_threshold=float(getattr(getattr(actor, 'env', None), 'stop_action_threshold', 0.5)),
        )
        if direction.ndim == 2:
            direction = direction[0]
        if stop_prob.ndim > 0:
            stop_prob = stop_prob[0]
        choose_stop = bool(stop_prob.item() > float(getattr(getattr(actor, 'env', None), 'stop_action_threshold', 0.5)))
        return direction.view(3).detach().cpu(), choose_stop, float(stop_prob.item())

    direction_dist = sample_from_output(actor_out)
    if stochastic:
        action = direction_dist.sample()[0]
    else:
        action = direction_dist.mean[0]
    return action.detach().cpu(), False, None


def policy_step(env, actor, stochastic=False):
    """
    Step through environment dynamics one policy action at a time.

    Controls:
      - Enter: take one policy step
      - t: force explicit stop on this step
      - r: reset environment
      - b: branch at current point
      - q: quit
    """
    from IPython.display import clear_output, display as ipy_display
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

    while True:
        action_key = input('Press Enter for policy step, or [t stop, r reset, b branch, q quit]: ').strip().lower()
        reward = None

        if action_key == 'q':
            break
        elif action_key == 'r':
            env.reset()
            sections = get_unvisited_sections(env)
            print('Environment reset.')
            continue
        elif action_key == 'b':
            if len(env.paths) > 0 and len(env.paths[0]) > 0:
                point = env.paths[0][-1]
                env.paths.append([point])
                env._append_branch_root(point)
                env.img.draw_point(point, radius=(env.step_width / 2.35), channel=-1, mode='gaussian', binary=False)
            sections = get_unvisited_sections(env)
            print('Added branch at current point.')
            continue

        clear_output(wait=True)

        if action_key == 't':
            direction = torch.zeros((3,), dtype=torch.float32)
            observation, reward, terminated, truncated, info = env.step(direction, verbose=True, stop=True)
        else:
            obs = env.get_state()
            direction, choose_stop, stop_probability = _policy_action_step(actor=actor, obs=obs, stochastic=stochastic)
            observation, reward, terminated, truncated, info = env.step(direction, verbose=True, stop=choose_stop)
            if stop_probability is not None:
                info['stop_probability'] = stop_probability
            total_policy_steps += 1

        total_steps += 1

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

        direction_norm = float(torch.linalg.norm(torch.as_tensor(direction, dtype=torch.float32)).item())
        print(f'step: {total_steps} (policy steps: {total_policy_steps})')
        print(f'direction: {torch.as_tensor(direction, dtype=torch.float32).tolist()}')
        print(f'direction_norm: {direction_norm:.4f}')
        print(f'reward: {reward}')
        print(f"status: {info.get('status')}")
        print(f"terminated: {info.get('terminate_episode')}")
        print(f"stop_probability: {info.get('stop_probability')}")
        print(f"current_target_stop_label: {info.get('current_target_stop_label')}")

        if info.get('terminate_episode'):
            print('All paths finished. Resetting environment.')
            env.reset()
            sections = get_unvisited_sections(env)

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


def manual_step(env, step_size=4.0):
    """
    Interactive manual stepping helper.

    Controls:
      - w/a/s/d: move in-plane (y/x)
      - p/l: move along z (positive/negative)
        - t: explicit stop (calls env.step(..., stop=True))
        - z: zero direction without stop (debug no-op)
        - x: use expert direction + expert stop label
                - g: enter stop logit; decoded with policy_utils before env.step
      - r: reset environment
      - b: branch at current point
      - q: quit
    """
    from IPython.display import clear_output, display as ipy_display
    import matplotlib.pyplot as plt

    plt.ioff()
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

    while True:
        action_key = input('Choose an action [w/a/s/d, p/l, x, t, z, g, r, b, q]: ').strip().lower()
        reward = None

        if action_key == 'q':
            break
        elif action_key == 'r':
            env.reset()
            sections = get_unvisited_sections(env)
        elif action_key == 'b':
            point = env.paths[0][-1]
            env.paths.append([point])
            env._append_branch_root(point)
            env.img.draw_point(point, radius=(env.step_width / 2.35), channel=-1, mode='gaussian', binary=False)

        else:
            if action_key not in user_input_dict and action_key not in {'x', 't', 'g'}:
                print(f"Unrecognized action '{action_key}'. Valid: w/a/s/d, p/l, x, t, z, g, r, b, q")
                continue

            if action_key == 'x': # select expert action
                prev_action = env.paths[0][-1] - env.paths[0][-2] if len(env.paths[0]) > 1 else None
                action = select_expert_action(target_vectors=getattr(env, 'target_vectors', None), previous_action=prev_action)
                choose_stop = bool(getattr(env, 'target_stop_label', False))
                use_explicit_stop = True
            elif action_key == 't':
                action = torch.zeros((3,), dtype=torch.float32, device=device)
                choose_stop = True
                use_explicit_stop = True
            elif action_key == 'g':
                prev_action = env.paths[0][-1] - env.paths[0][-2] if len(env.paths[0]) > 1 else None
                direction = select_expert_action(target_vectors=getattr(env, 'target_vectors', None), previous_action=prev_action)
                try:
                    stop_logit = float(input('stop logit for action[3] (e.g. 4.0 stop, -4.0 continue) [default=4.0]: ').strip() or '4.0')
                except ValueError:
                    stop_logit = 4.0
                    print('Invalid stop logit; defaulting to 4.0')
                action4 = torch.cat((direction.to(device=device, dtype=torch.float32), torch.tensor([stop_logit], device=device)))
                direction_decoded, _stop_prob, choose_stop = decode_direct_vector_output(
                    action4,
                    stop_action_threshold=float(getattr(env, 'stop_action_threshold', 0.5)),
                )
                action = direction_decoded.to(device=device, dtype=torch.float32)
                use_explicit_stop = True
            else:
                action = user_input_dict[action_key].to(device=device)
                action = action * getattr(env, 'target_step_len', step_size)
                choose_stop = False
                use_explicit_stop = True

            clear_output(wait=True)
            observation, reward, terminated, truncated, info = env.step(action, verbose=True, stop=choose_stop)

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

            if reward is not None:
                print(f'reward: {reward}')
                print(f"status: {info.get('status')}")
                print(f"terminated: {info.get('terminate_episode')}")
                print(f"stop_probability: {info.get('stop_probability')}")
                print(f"current_target_stop_label: {info.get('current_target_stop_label')}")

            if info.get('terminate_episode'):
                print('All paths finished. Resetting environment.')
                env.reset()
                sections = get_unvisited_sections(env)

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
    if hasattr(env, 'seeds') and env.seeds:
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
