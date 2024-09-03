#!/usr/bin/env python

""" Interface for interactively evaluating the tracking environment """

import torch
import matplotlib.pyplot as plt
from IPython import display
from image import Image

def manual_step(env, step_size=2.0):
    plt.ioff()
    device = env.img.data.device
    user_input_dict = {'a': torch.tensor([0.0, 0.0, -1.0, 0.0]),
                    'w': torch.tensor([0.0, -1.0, 0.0, 0.0]),
                    'd': torch.tensor([0.0, 0.0, 1.0, 0.0]),
                    's': torch.tensor([0.0, 1.0, 0.0, 0.0]),
                    'p': torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    'l': torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    't': torch.tensor([0.0, 0.0, 0.0, 1.0]),
                    'b': torch.tensor([0.0, 0.0, 0.0, 2.0])}
    
    # img = env.img.data[:3].amax(dim=1).permute(1,2,0).cpu()
    # ax[0].imshow(img)
    while True:
        fig, ax = plt.subplots(1,3)
        action = input("Choose an action: ")
        if action == 'q':
            break
        action = user_input_dict[action]
        action = action.to(device=device)
        action[:3] = action[:3] * step_size
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
        patch = observation[0][0]
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


