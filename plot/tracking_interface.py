#!/usr/bin/env python

""" Interface for interactively evaluating the tracking environment """

from IPython import display
from data_prep.image import Image
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
        if action == 'r':
            env.reset()
        if action == 'b':
            point = env.paths[env.head_id][-1]
            env.paths.append(point[None])
            env.path_labels.append(0)
            env.prev_children.append(env.prev_children[env.head_id])
            env.roots.append(point)
            env.img.draw_point(point, radius=env.step_width, channel=-1)
        else:
            action = user_input_dict[action]
            action = action.to(device=device)
            action = action * step_size
            display.clear_output(wait=True)
            observation, reward, terminated = env.step(action, verbose=True)

            # Show:
            # 1) Whole image with path overlayed,
            # 2) Cropped image with path overlayed,
            # 3) Cropped mask, true density, and path overlayed
            img = env.img.data[:3].amax(dim=1).permute(1,2,0)
            path = env.img.data[3].amax(dim=0)#.permute(1,0)
            ax[0].imshow(img)
            ax[0].imshow(path, cmap='plasma', alpha=0.5)
            if len(env.path_labels) > 0:
                label = env.path_labels[env.head_id]
            else:
                label = None
            ax[0].set_title(f'path label: {label}')
            # patch, _ = env.img.crop(env.paths[env.head_id][-1], env.radius, interp=False)
            patch = observation[0]
            patch = patch[:, env.radius]
            ax[1].imshow(patch[:3].permute(1,2,0))
            # ax[1].imshow(patch[3].permute(1,0), cmap='plasma', alpha=0.5)

            if not terminated:
                center = env.paths[env.head_id][-1]

                density_patch = env.true_density.crop(center, env.radius, interp=False)[0]

                # labels_patch, _ = env.section_labels.crop(center, env.radius, interp=False, pad=False)
                # new_label = int(labels_patch[0, env.radius, env.radius, env.radius].item())
                # current_label = env.path_labels[env.head_id]
                # # Here mask out any sections that are not the current section or its children. 
                # if current_label != 0:
                #     # The graph is necessarily an undirected graph. Here "children" means connected sections
                #     # that have not been previously available to the path.
                #     children = [x for x in env.graph[current_label] if x not in env.prev_children[env.head_id]]
                #     section_ids = [current_label] + children
                #     section_mask = torch.zeros_like(density_patch)
                #     for id in section_ids:
                #         section_mask += torch.where(labels_patch == id, 1, 0)
                #     density_patch_masked = density_patch * section_mask
                #     if new_label != current_label and new_label in section_ids: # only change label if the new label is a child section
                #         env.path_labels[env.head_id] = new_label
                #         env.prev_children[env.head_id] = env.graph[current_label]
                # else:
                #     density_patch_masked = density_patch
                #     if new_label != 0:
                #         env.path_labels[env.head_id] = new_label

                density_patch_masked = density_patch        
                density_patch_masked = density_patch_masked[0,env.radius]
                # mask = mask[0,env.radius]
                # ax[2].imshow(mask, cmap='Blues')
                ax[2].imshow(density_patch_masked, cmap='Reds', alpha=0.5)#.permute(1,0), cmap='Reds', alpha=0.5)
                ax[2].imshow(patch[3], cmap='Greens', alpha=0.5)#.permute(1,0), cmap='Greens', alpha=0.5)
            else:
                ax[2].imshow(torch.zeros_like(patch[3]))
                env.reset()
        # Turn off axes for all subplots
        for a in ax:
            a.axis('off')
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")

        display.display(plt.gcf())


def show_state(env, z=None, finished=False, path_id=0, t=-1):
    
    if finished:
        paths = env.finished_paths
    else:
        paths = env.paths
        path_id = env.head_id

    state = env.get_state()[0]
    true_density_patch, _ = env.true_density.crop(paths[path_id][t], env.radius, pad=True)
    mask = Image(env.mask)
    mask_patch, _ = mask.crop(paths[path_id][t], env.radius, interp=False, pad=True)
    I = np.array(env.img.data.to('cpu'))
    O = np.array(state.to('cpu'))
    T = np.array(true_density_patch.to('cpu'))
    M = np.array(mask_patch.to('cpu'))
    if z is not None:
        z_ = state.shape[2]//2
        I = I[:, z]
        O = O[:, z_]
        T = T[0,z_]
        M = M[0,z_]
    else: # display a maximum intensity projection along z
        I = I.max(axis=1)
        O = O.max(axis=1)
        T = T[0].max(axis=0)
        M = M[0].max(axis=0)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(I[3], cmap='hot', alpha=0.5) #, int(paths[env.head_id][-1, 0])])
    ax[0].imshow(I[:3].transpose(1,2,0), alpha=0.5) #, int(paths[env.head_id][-1, 0])])
    ax[0].axis('off')
    ax[1].imshow(O[:3].transpose(1,2,0), alpha=0.75)
    ax[1].imshow(O[3], alpha=0.25, cmap='hot') #, env.radius//2])
    ax[1].axis('off')
    toshow = np.stack((O[3], T, M), axis=-1)
    ax[2].imshow(toshow)
    ax[2].axis('off')

    display.display(plt.gcf())
    plt.close()
    
if __name__ == "__main__":
    pass