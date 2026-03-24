#!/usr/bin/env python

""" Interface for interactively evaluating the tracking environment """

from IPython import display
from neurotrack.data.image import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from neurotrack.data import loading as load
from neurotrack.environments import tracking_reward
from neurotrack.training.behavior_cloning import select_expert_action

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


def manual_step(env, step_size=4.0):
    """
    Interactive manual stepping helper.

    Controls:
      - w/a/s/d: move in-plane (y/x)
      - p/l: move along z (positive/negative)
        - t: explicit stop (calls env.step(..., stop=True))
        - z: zero direction without stop (debug no-op)
        - x: use expert direction + expert stop label
        - g: raw 4D action mode; prompts for stop logit and uses env action[3] decoding
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
            env._append_root(point)
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
                action = torch.cat((direction.to(device=device, dtype=torch.float32), torch.tensor([stop_logit], device=device)))
                choose_stop = False
                use_explicit_stop = False
            else:
                action = user_input_dict[action_key].to(device=device)
                action = action * getattr(env, 'target_step_len', step_size)
                choose_stop = False
                use_explicit_stop = True

            clear_output(wait=True)
            if use_explicit_stop:
                observation, reward, terminated, truncated, info = env.step(action, verbose=True, stop=choose_stop)
            else:
                observation, reward, terminated, truncated, info = env.step(action, verbose=True)

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

if __name__ == "__main__":
    pass
