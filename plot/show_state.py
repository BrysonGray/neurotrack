from IPython.display import display, clear_output
import os
from pathlib import Path
import torch
import sys
script_path = Path(os.path.abspath(__file__))
parent_dir = script_path.parent.parent
sys.path.append(str(parent_dir))
from data_prep.image import Image

def show_state(env, fig, returns=None, rewards=None, policy_loss=None, accuracies=None, detections=None):
    print(f"image: {env.img_files[env.img_idx].split('/')[-1]}")
    ax = fig.axes
    clear_output(wait=True)
    for x in ax:
        x.cla()

    img = env.img.data.clone().detach().cpu()

    if env.branching:
        img = Image(img)
        for point in env.roots:
            img.draw_point(point, radius=3.0, channel=3)
        img = img.data

    path = img[3]
    img = img[:3]

    mask = torch.where(env.section_labels.data[0].clone().detach().cpu() > 0, 1.0, 0.0)
    true_density = env.true_density.data[0].detach().clone().cpu()
    # for j in range(3):
    #     ax[j].imshow(img.permute(1,2,3,0).amax(j))
    #     ax[j].imshow(path.amax(j), cmap='gray', alpha=0.8)

    for j in range(3):
        toshow = torch.stack((true_density.amax(j), path.amax(j), mask.amax(j)), dim=-1)
        # ax[j+3].imshow(toshow)
        ax[j].imshow(toshow)
    if rewards is not None:
        ax[4].plot(rewards)
        ax[4].set_title("ep rewards")
    if returns is not None:
        ax[5].plot(returns)
        ax[5].set_title("ep returns")
    if accuracies is not None:
        ax[6].plot(accuracies, color='orange')
        ax[6].set_title("ep accuracies")
    if policy_loss is not None:
        ax[7].plot(policy_loss)
        ax[7].set_title("policy loss")
    if detections is not None:
        print(f"TP: {detections['TP']}, FP: {detections['FP']}, FN: {detections['FN']}, TN: {detections['TN']}")

    display(fig)

    
    return

if __name__ == "__main__":
    pass