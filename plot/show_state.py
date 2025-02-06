from IPython.display import display, clear_output
import torch

def show_state(env, fig, returns=None, rewards=None, policy_loss=None):
    print(f"image: {env.img_files[env.img_idx].split('/')[-1]}")
    ax = fig.axes
    clear_output(wait=True)
    for x in ax:
        x.cla()

    img = env.img.data[:3].clone().detach().cpu()
    path = env.img.data[3].clone().detach().cpu()
    mask = torch.where(env.section_labels.data[0].clone().detach().cpu() > 0, 1.0, 0.0)
    true_density = env.true_density.data[0].detach().clone().cpu()
    for j in range(3):
        ax[j].imshow(img.permute(1,2,3,0).amax(j))
        ax[j].imshow(path.amax(j), cmap='gray', alpha=0.8)

    for j in range(3):
        toshow = torch.stack((true_density.amax(j), path.amax(j), mask.amax(j)), dim=-1)
        ax[j+3].imshow(toshow)
    if rewards is not None:
        ax[6].plot(rewards)
        ax[6].set_title("ep rewards")
    if returns is not None:
        ax[7].plot(returns)
        ax[7].set_title("ep returns")
    if policy_loss is not None:
        ax[8].plot(policy_loss)
        ax[8].set_title("policy loss")

    display(fig)

    
    return