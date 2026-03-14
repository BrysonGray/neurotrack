#!/usr/bin/env python

""" Interface for interactively evaluating the tracking environment """

from IPython import display
from neurotrack.data.image import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

def manual_step(env, step_size=2.0):
    plt.ioff()
    device = env.img.data.device
    user_input_dict = {'a': torch.tensor([0.0, 0.0, -1.0]),
                    'w': torch.tensor([0.0, -1.0, 0.0]),
                    'd': torch.tensor([0.0, 0.0, 1.0]),
                    's': torch.tensor([0.0, 1.0, 0.0]),
                    'p': torch.tensor([1.0, 0.0, 0.0]),
                    'l': torch.tensor([-1.0, 0.0, 0.0])}
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    
    # Top row spans the full width
    ax0 = fig.add_subplot(gs[0, :])
    
    # Bottom row has two plots side by side
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    ax = [ax0, ax1, ax2]
    while True:
        action = input("Choose an action: ")
        if action == 'q':
            break
        elif action == 'r':
            env.reset()
        elif action == 'b':
            point = env.paths[env.head_id][-1]
            env.paths.append(point[None])
            # env.path_labels.append(0)
            env.prev_children.append(env.prev_children[env.head_id])
            env.roots.append(point)
            env.img.draw_point(point, radius=env.step_width, channel=-1)
        else:
            action = user_input_dict[action]
            action = action.to(device=device)
            action = action * getattr(env, 'step_size', step_size)
            display.clear_output(wait=True)
            observation, reward, terminated, truncated, info = env.step(action, verbose=True, training=False)

            # Show:
            # 1) Whole image with path overlayed,
            # 2) Cropped image with path overlayed,
            # 3) Cropped mask, true density, and path overlayed
            img = env.img.data[:-1].amax(dim=1).permute(1,2,0)
            img = img.squeeze()
            path = env.img.data[-1].amax(dim=0)#.permute(1,0)
            ax[0].imshow(img)
            ax[0].imshow(path, cmap='plasma', alpha=0.5)
            # if len(env.path_labels) > 0:
            #     label = env.path_labels[env.head_id]
            # else:
            #     label = None
            # ax[0].set_title(f'path label: {label}')
            # patch, _ = env.img.crop(env.paths[env.head_id][-1], env.radius, interp=False)
            patch = observation[0]
            patch = patch[:, env.radius]
            ax[1].imshow(patch[:-1].permute(1,2,0).squeeze())
            ax[1].imshow(patch[-1], cmap='plasma', alpha=0.5)

            if not info["terminate_episode"]:
                center = env.paths[env.head_id][-1]

                density_patch = env.true_density.crop(center, env.radius, interp=False)[0]

                density_patch_masked = density_patch        
                density_patch_masked = density_patch_masked[0,env.radius]
                ax[2].imshow(density_patch_masked, cmap='Reds', alpha=0.5)
                ax[2].imshow(patch[-1], cmap='Greens', alpha=0.5)
            else:
                ax[2].imshow(torch.zeros_like(patch[-1]))
                env.reset()
        # Turn off axes for all subplots
        for a in ax:
            a.axis('off')
        print(f"reward: {reward}")
        print(f"terminated: {info['terminate_episode']}")

        display.display(plt.gcf())


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
