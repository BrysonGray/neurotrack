#!/usr/bin/env python

""" Interface for interactively evaluating the tracking environment """

from IPython import display
from image import Image
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
                    'l': torch.tensor([1.0, 0.0, 0.0])}
    
    # img = env.img.data[:3].amax(dim=1).permute(1,2,0).cpu()
    # ax[0].imshow(img)
    fig, ax = plt.subplots(1,3)
    while True:
        action = input("Choose an action: ")
        if action == 'q':
            break
        if action == 't':
            env.reset()
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
            img = env.img.data[:3].amax(dim=1).permute(1,2,0).cpu()
            path = env.img.data[3].amax(dim=0).cpu()
            ax[0].imshow(img)
            ax[0].imshow(path, cmap='plasma', alpha=0.5)
            # patch, _ = env.img.crop(env.paths[env.head_id][-1], env.radius, interp=True)
            patch = observation[0]
            patch = patch[:, env.radius].cpu()
            ax[1].imshow(patch[:3].permute(1,2,0))
            ax[1].imshow(patch[3], cmap='plasma', alpha=0.5)

            if len(env.paths) > 0:
                mask = Image(env.mask)
                mask, _ = mask.crop(env.paths[env.head_id][-1], env.radius, interp=True)
                mask = mask[0,env.radius].cpu()
                true_density, _ = env.true_density.crop(env.paths[env.head_id][-1], env.radius, interp=True)
                true_density = true_density[0,env.radius].cpu()
                ax[2].imshow(mask, cmap='Blues')
                ax[2].imshow(true_density, cmap='Reds', alpha=0.5)
                ax[2].imshow(patch[3], cmap='Greens', alpha=0.5)
            else:
                ax[2].imshow(torch.zeros_like(patch[3]))
                env.reset()
        
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