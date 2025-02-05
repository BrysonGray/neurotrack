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
    # Calculate the precision of the new estimate: TP/(TP+FP), where TP and FP are true positive and false positive
    # pixels and false positive pixels. I want to make it possible to give a penalty for self-overlap. Ordinarily it will be considered FP.

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
    """ Change in the error between true and estimated density maps. Error is mean absolute difference between density maps
        after normalizing by their sums.

        Parameters
        ----------
        true_density : torch.Tensor
        old_density: torch.Tensor
        new_density: torch.Tensor
        
        Returns
        -------
        error_change : scalar
    
    """
    # true_density = true_density / (true_density.sum() + np.finfo(float).eps)
    # old_density = old_density / (old_density.sum() + np.finfo(float).eps)
    # new_density = new_density / (new_density.sum() + np.finfo(float).eps)

    # old_mad = torch.mean(torch.abs(true_density - old_density))
    # new_mad = torch.mean(torch.abs(true_density - new_density))

    # TODO: test using sum of square error instead of mean absolute difference
    old = torch.sum((old_density - true_density)**2)
    new = torch.sum((new_density - true_density)**2)
    deltaSSE = (new - old).item()
    
    return deltaSSE


# plotting functions
def plot_durations(episode_durations, show_result=False):
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
        # display.display(plt.gcf())
        display.display(plt.figure(),display_id=1)
        display.clear_output(wait=True)
    else:
        # display.display(plt.gcf(), display_id=True)
        display.display(plt.figure(), display_id=1)


def plot_returns(episode_returns, show_result=False):
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
        # display.display(plt.gcf())
        display.display(plt.figure(), display_id=2)
        display.clear_output(wait=True)
    else:
        # display.display(plt.gcf())
        display.display(plt.figure(), display_id=2)