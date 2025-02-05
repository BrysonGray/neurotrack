#!/usr/bin/env python

"""
Deep Q-network reinforcement learning tractography environment

Author: Bryson Gray
2024
"""

import os
import numpy as np
import torch
from data_prep.image import Image, make_line_segment


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def binary_matching_error(new_patch, old_patch, true_patch, step_width=1.0):
    # Calculate the precision of the new estimate: TP/(TP+FP), where TP and FP are true positive
    # pixels and false positive pixels. I want to make it possible to give a penalty for self-overlap. Ordinarily it will be considered FP.

    # binarize patches
    new_patch = new_patch > 0.0
    old_patch = old_patch > 0.0
    true_patch = true_patch > 0.0
    # get the new segment minus its intersection with old_patch
    new_and_old = torch.logical_and(new_patch, old_patch)
    new_and_not_old = torch.logical_and(new_patch, ~new_and_old)

    # get the intersection with old_patch
    # First find TP. This is the sum of the intersection between true_density and new_density.
    TP = torch.sum(torch.logical_and(true_patch, new_and_not_old))

    # since the new segment begins at the endpoint of the last segment, there is always some overlap.
    # The minimum overlap is equal to the side length of the path cross-section cubed. So I will subtract it from the total and the intersection.
    correction = step_width**3 # TODO: this is not the true minimum, diagonal steps overlap less.
    
    tot = torch.sum(new_patch) - correction
    precision = (TP / tot) # precision = TP/(TP + FP)
    self_overlap = (torch.sum(new_and_old) - correction) / tot # fraction of positives that were already labeled positive.

    return precision, self_overlap


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

    return new - old


class Environment():
    """ Loads a volume. Takes a list of seeds which
    which will be cycled through on every call of reset.

    Parameters
    ----------
    img_path : str
        Path to image file.
    radius : int
        The radius of a state input image patch along each axis (total width is 2*radius + 1).
    seeds : ndarray
        N x 3 array of starting streamline coordinates
    mask : torch.Tensor
        Mask of the true neuron. Tensor  with shape 1 x h x w x d.
    true_denstiy : torch.Tensor
        True neuron position as a density. Tensor with shape 1 x h x w x d
    actions : ndarray
        N x 3 array of possible step directions.
    step_size : float
        Distance taken in each step in units of pixels.
    max_len : int
        Maximum number of steps allowed per streamline
    alpha : float
        Density matching weight
    beta : float
        Smoothness weight
    
    Attributes
    ----------
    head_id : int
        Index of the current streamline head out of the whole tracking bundle.
    img : torch.Tensor
        Tensor with shape c x h x w x d where the first channels are the input image and the last channel is the streamline density.
    radius : int
        The radius of a state input image patch along each axis (total width is 2*radius + 1).
    paths : torch.Tensor
        Tensor with shape t x N x 3, where t is the length of the streamlines.
    mask : torch.Tensor
        Mask of the true neuron. Tensor  with shape 1 x h x w x d.
    true_denstiy : torch.Tensor
        True neuron position as a density. Tensor with shape 1 x h x w x d
    action_space : torch.Tensor
        N x 3 tensor of possible step directions.
    step_size : float
        Distance taken in each step in units of pixels.
    max_len : int
        Maximum number of steps allowed per streamline
    n_resets : int
        Count of the number of episode resets during training.

    """

    def __init__(
            self,
            img_dir,
            radius,
            seeds,
            actions,
            n_seeds=1,
            step_size=1.0,
            step_width=1.0,
            max_len=10000,
            alpha=1.0,
            beta=1e-3,
            friction=1e-4,
            branching=True):
        
        self.head_id = 0
        self.n_resets = 0 # count number of resets
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        img, true_density, mask = list(torch.load(self.img_files[0]).values())
        self.img = Image(img)
        self.radius = radius
        self.seeds = seeds        
        self.mask = mask
        self.true_density = Image(true_density)
        # make copies of the branch and terminal points so these can be changed while saving the originals
        self.n_seeds = n_seeds # the number of paths per bundle
        self.step_size = step_size
        self.step_width = step_width
        self.max_len = max_len
        self.action_space = actions
        self.branching = branching
        if not isinstance(self.action_space, torch.Tensor):
            self.action_space = torch.tensor(self.action_space)

        seed_id = self.n_resets % len(self.seeds)
        center = torch.Tensor(self.seeds[seed_id])
        self.r = 2.0 # radius around center to randomly place starting points
        offsets = torch.randn((self.n_seeds, 3))
        offsets /= torch.sum(offsets**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        bundle_seeds = center[None] + r * offsets

        self.paths = [*torch.Tensor(bundle_seeds)[:,None]] # a list of N paths. each path is a 1 x 3 tensor
        self.path_labels = [1.0]
        self.alpha = alpha
        self.beta = beta
        self.friction = friction

        last_step = torch.randn((len(self.paths),) + (1,3))
        last_step /= torch.norm(last_step)
        self.paths = [torch.cat((point - step*self.step_size, point)) for point, step in zip(self.paths, last_step)]

        # we will want to save completed paths
        self.finished_paths = []

        # self.img.data = torch.cat((self.img.data, torch.zeros((3,)+self.img.data.shape[1:])), dim=0) # add 3 channels for path, bifurcation points, and terminal points
        if self.branching:
            self.img.data = torch.cat((self.img.data, torch.zeros((2,)+self.img.data.shape[1:])), dim=0) # add 2 channels for path, and bifurcation points.
        else:
            self.img.data = torch.cat((self.img.data, torch.zeros((1,)+self.img.data.shape[1:])), dim=0) # add 1 channel for path.
        for i in range(len(self.paths)):
            segment = torch.stack((self.paths[i][0], self.paths[i][1]), dim=0)
            # self.img.draw_line_segment(segment, width=self.step_width, channel=3, value=self.step_size/100.0, binary=True) # TODO: test without terminate distance reward
            self.img.draw_line_segment(segment, width=self.step_width, channel=3, binary=True)

    def get_state(self, terminate=False):
        """ Get the state for the current step at streamline 'head_id'. The state consists of an image patch and
        streamline density patch centered on the streamline head plus the last three streamline positions.

        Returns
        -------
        patch : torch.Tensor
            Tensor with shape (n x c x h x w x d) where the first channels are the input image and the last channel is the streamline density.
        last_steps : torch.Tensor
         Tensor with shape 3 x 3 (3 step positions by 3 euclidean coordinates)
        """
        if terminate:
            patch = torch.zeros((self.img.data.shape[0],)+(2*self.radius + 1,)*3)
        else:
            patch, _ = self.img.crop(self.paths[self.head_id][-1], self.radius, pad=True, value=0.0)
            patch = patch.detach().clone()
        
        last_steps = self.paths[self.head_id][-2:].detach().clone() # 2 x 3 tensor of last three streamline positions

        return patch[None], last_steps[None]


    def get_reward(self, matching_precision=None, self_overlap=None, bifurcate=False, terminate=False, verbose=False, sigmaf=1.0, sigmab=0.4):
        """ Get the reward for the current state. The reward depends on the streamline smoothness and
        the change in distance between the streamline density and true denisty maps.

        Parameters
        ----------
        terminated : bool
        delta_density_diff : float
            Change in median absolute difference between bundle density and true density patch before and after taking a step.
        verbose : bool
            If true, print the values of each term of the reward. 
            
        Returns
        -------
        reward : torch.Tensor
            Tensor with shape 1
        """

        if terminate:
            travel_dist = np.linalg.norm(self.paths[self.head_id][-1] - self.paths[self.head_id][0])
            # reward = -5*np.exp(-travel_dist/15) # penalty for stopping early
            reward = 0.0 # TODO: test without terminate distance reward
            if verbose:
                print('terminate \n',
                      f'travel distance: {travel_dist}\n',
                      f'reward: {reward}\n')
                
        elif bifurcate: # constant penalty for bifurcation
            reward = 0.0
            if verbose:
                print('bifurcate \n',
                      f'reward: {reward}\n')

        else:
            # get_reward should receive one and only one of either terminated, bifurcate, or delta_density_diff
            if matching_precision is None:
                raise RuntimeError("matching_precision must not be none if the action is neither bifurcate nor terminate.")

            M = matching_precision
            q = self.paths[self.head_id][-1]
            q_ = self.paths[self.head_id][-2]
            q__ = self.paths[self.head_id][-3]
            P = - torch.sum((q - q_)**2)/(2*sigmaf**2) - torch.sum((q - 2*q_ + q__)**2) / (2*sigmab**2) #- 1.0 # prior likelihood
            P_norm = self.step_size**2 / (2*sigmaf**2) + (2*self.step_size)**2 / (2*sigmab**2)
            P = P/P_norm
            # reward = self.alpha*(M - self_overlap) + self.beta*P # TODO: self_overlap needs to be corrected.
            reward = self.alpha*M + self.beta*P
            if verbose:
                print(f'matching precision: {matching_precision}\n',
                    f'matching reward: {self.alpha*M}\n',
                    f'prior likelihood: {P}\n',
                    f'prior reward: {self.beta*P}')

        return torch.tensor([reward], device=DEVICE, dtype=torch.float32)


    def step(self, action_id, verbose=False):
        """ Take a tracking step for one streamline. Add the new position to the path
        and the bundle density mask, and compute the reward and new state observation.

        Parameters
        ----------
        action_id : int
            Action index
        
        Returns
        -------
        observation : tuple
            The img patch centered at the new streamline head (a tensor with shape c x h x w x d), and the
            last three streamline positions.
        reward : torch.Tensor
            Tensor with shape 1
        terminated : bool

        """
        terminate_path = False
        terminated = False

        # check if the agent chose to mark the position as a bifurcation point
        # Add the last segment (last two points) from the current path to a new path in the paths list
        if action_id == len(self.action_space)+1: # bifurcate
            if len(self.paths) + len(self.finished_paths) == 100: # too many branches
                terminated = True
                observation = None
                reward = torch.tensor([0.0], device=DEVICE, dtype=torch.float32)
            else:
                self.paths.append(self.paths[self.head_id][-2:])
                # draw point in bifurcation points channel
                self.img.draw_point(self.paths[self.head_id][-1], radius=3, channel=4)

                reward = self.get_reward(bifurcate=True, verbose=verbose)
                # reward = torch.tensor([0.0], device=DEVICE, dtype=torch.float32)
                observation = self.get_state()
                # don't move to the new path head until after taking a step on the current path
            
        # check if the action is terminate path
        elif action_id == len(self.action_space):
            terminate_path = True
            
        else: # decide if path terminates accidentally
            direction = self.action_space[action_id]
            new_position = self.paths[self.head_id][-1] + self.step_size*direction

            out_of_bound = any([x >= y or x < 0 for x,y in zip(torch.round(new_position), self.img.data.shape[1:])])
            if out_of_bound:
                terminate_path = True
            else:
                out_of_mask = 1 - self.mask[(0,)+tuple([int(np.round(x)) for x in new_position])]
                too_long = len(self.paths[self.head_id]) > self.max_len
                terminate_path = too_long or out_of_mask

        if terminate_path:
            # observation = None #TODO: change for testing, 6/18
            observation = self.get_state(terminate=True)
            reward = self.get_reward(terminate=True, verbose=verbose)

            # remove the path from 'paths' and add it to 'ended_paths'
            self.finished_paths.append(self.paths.pop(self.head_id))
            # if that was the last path in the list, then terminate the episode
            if len(self.paths) == 0:
                terminated = True
            # otherwise, move to the next path
            else:
                self.head_id = (self.head_id + 1)%len(self.paths)

        elif action_id < len(self.action_space): # otherwise take a step
            # add new position to path
            self.paths[self.head_id] = torch.cat((self.paths[self.head_id], new_position[None]))

            # draw the segment on the state input image as a single pixel wide segment whose value is proportional to the branch length
            segment = self.paths[self.head_id][-2:, :3]
            travel_dist = np.linalg.norm(self.paths[self.head_id][-1] - self.paths[self.head_id][0])
            # self.img.draw_line_segment(segment, width=self.step_width, channel=3, value=travel_dist/100.0, binary=True) # divide the brightness value of the segment by 100 to make the values closer to a range of [0,1]

            # make the new segment patch
            # X = make_line_segment(segment, width=self.step_width, value=travel_dist/100.0, binary=True) #TODO: test without terminate distance reward
            X = make_line_segment(segment, width=self.step_width, binary=True)
            # get the current bundle_density patch centered at the same point as the new segment
            center = torch.round(segment[0]).to(int)
            patch_radius = int((X.shape[0]-1)/2)
            # bundle_density_patch, padding = self.bundle_density.crop(center, patch_radius, interp=False, pad=False)
            img_patch, padding = self.img.crop(center, patch_radius, interp=False, pad=False)
            # crop the new segment if it overlaps the image boundary.
            new_patch = X[padding[0]:X.shape[0]-padding[1], padding[2]:X.shape[1]-padding[3], padding[4]:X.shape[2]-padding[5]]
            true_patch, _ = self.true_density.crop(center, patch_radius, interp=False, pad=False) # patch centered at previous step position

            # get error: precision is TP/(TP+FP) and self_overlap is fraction of positives that were already labeled positive.
            precision, self_overlap = binary_matching_error(new_patch, img_patch[3], true_patch.squeeze(), step_width=self.step_width)
            reward = self.get_reward(matching_precision=precision, self_overlap=self_overlap, verbose=verbose)

            # draw the segment on the state input path image.
            img_patch[3] = torch.maximum(new_patch, img_patch[3])

            observation = self.get_state()

            self.head_id = (self.head_id + 1)%len(self.paths)

        return observation, reward, terminated


    def reset(self):
        # reset the agent to the next image or seed and reset the path with new random previous step directions.
        self.n_resets += 1
        # load the next image
        img_idx = self.n_resets // len(self.seeds) % len(self.img_files)
        img, true_density, mask = list(torch.load(self.img_files[img_idx]).values())
        self.img = Image(img)       
        self.mask = mask
        self.true_density = Image(true_density)
        # start with next seed point
        seed_id = self.n_resets % len(self.seeds)
        center = torch.Tensor(self.seeds[seed_id])
        offsets = torch.randn((self.n_seeds, 3))
        offsets /= torch.sum(offsets**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        bundle_seeds = center[None] + r * offsets

        self.paths = [*torch.Tensor(bundle_seeds)[:,None]] # a list of N paths. each path is a 1 x 3 tensor

        last_step = torch.randn((len(self.paths),) + (1,3))
        last_step /= torch.norm(last_step)
        self.paths = [torch.cat((point - step*self.step_size, point)) for point, step in zip(self.paths, last_step)]

        self.head_id = 0
        self.finished_paths = []

        # # reset bundle density
        # self.bundle_density = torch.zeros_like(self.true_density.data)
        # self.bundle_density = Image(self.bundle_density)
        if self.branching:
            self.img.data = torch.cat((self.img.data, torch.zeros((2,)+self.img.data.shape[1:])), dim=0) # add 2 channels for path, and bifurcation points.
        else:
            self.img.data = torch.cat((self.img.data, torch.zeros((1,)+self.img.data.shape[1:])), dim=0) # add 1 channel for path.
        for i in range(len(self.paths)):
            segment = torch.stack((self.paths[i][0], self.paths[i][1]), dim=0)
            # self.img.draw_line_segment(segment, width=self.step_width, channel=3, value=self.step_size/100.0, binary=True)
            self.img.draw_line_segment(segment, width=self.step_width, channel=3, binary=True) # TODO: test without terminate distance reward

        return