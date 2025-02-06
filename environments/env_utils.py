#!/usr/bin/env python

"""
Environment helper functions

Author: Bryson Gray
2024
"""
import matplotlib.pyplot as plt
from IPython import display
import torch


def binary_matching_error(new_patch: torch.Tensor, old_patch: torch.Tensor, true_patch: torch.Tensor) -> float:
    """
    Calculate the precision of the new estimate.
    
    Parameters
    ----------
    new_patch : torch.Tensor
        The new patch tensor to be evaluated.
    old_patch : torch.Tensor
        The old patch tensor for comparison.
    true_patch : torch.Tensor
        The ground truth patch tensor.
        
    Returns
    -------
    float
        The precision of the new patch, calculated as TP / (TP + FP), where TP is the number of true positive pixels 
        and FP is the number of false positive pixels.
    """
    
    # TODO: Make it possible to give a penalty for self-overlap. Ordinarily it will be considered FP.

    # binarize patches
    new_patch = new_patch > 0.0
    old_patch = old_patch > 0.0
    true_patch = true_patch > 0.0
    # get the intersection with old_patch
    new_and_old = torch.logical_and(new_patch, old_patch)
    # get the new segment minus its intersection with old_patch
    new_and_not_old = torch.logical_and(new_patch, ~new_and_old)

    # First find TP. This is the sum of the intersection between true_density and new_density.
    TP = torch.sum(torch.logical_and(true_patch, new_and_not_old))
    
    tot = torch.sum(new_patch)
    precision = (TP / tot) # precision = TP/(TP + FP)

    return precision.item()


def density_error_change(true_density, old_density, new_density):
    """ 
    Change in the error between true and estimated density maps. Error is the sum of square difference between density maps.

    Parameters
    ----------
    true_density : torch.Tensor
    old_density: torch.Tensor
    new_density: torch.Tensor
    
    Returns
    -------
    float
        The change in sum of square errors between the old and new density maps.
    """
    old = torch.sum((old_density - true_density)**2)
    new = torch.sum((new_density - true_density)**2)
    deltaSSE = (new - old).item()
    
    return deltaSSE


# plotting functions
def plot_durations(episode_durations, show_result=False):
    """
    Plots the durations of episodes and optionally shows the result.
    
    Parameters
    ----------
    episode_durations : list of int or float
        A list containing the durations of each episode.
    show_result : bool, optional
        If True, the plot will display the final result. If False, the plot will display the training progress (default is False).
        
    Notes
    -----
    - The function will plot the episode durations and, if there are at least 100 episodes, it will also plot the 100-episode moving average.
    - The plot will be updated in real-time during training if `show_result` is False.
    - The `display` module from IPython is used to update the plot in real-time.
    """
    
    plt.figure()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if not show_result:
        display.display(plt.figure(),display_id=1)
        display.clear_output(wait=True)
    else:
        display.display(plt.figure(), display_id=1)
    
    return


def plot_returns(episode_returns, show_result=False):
    """
    Plots the returns of episodes and optionally shows the result.
    Parameters
    ----------
    episode_returns : list or array-like
        A list or array of episode returns to be plotted.
    show_result : bool, optional
        If True, the plot will display the final result. If False, the plot will display the training progress (default is False).
    Notes
    -----
    - The function plots the episode returns and, if the length of `episode_returns` is 100 or more, it also plots the 100-episode moving average.
    - The plot is updated in real-time with a short pause to ensure the plot is rendered.
    - When `show_result` is False, the plot is cleared and updated continuously to show training progress.
    - When `show_result` is True, the plot displays the final result.
    """
    
    plt.figure()
    return_t = torch.tensor(episode_returns, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(return_t.numpy())
    # Take 100 episode averages and plot them too
    if len(return_t) >= 100:
        means = return_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if not show_result:
        display.display(plt.figure(), display_id=2)
        display.clear_output(wait=True)
    else:
        display.display(plt.figure(), display_id=2)