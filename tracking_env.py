#!/usr/bin/env python

"""
Deep reinforcement learning tractography environment

Author: Bryson Gray
2024
"""

import numpy as np
import torch
import scipy
from skimage.draw import line_nd
from skimage.filters import gaussian
from image import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    old_mad = torch.mean(torch.abs(true_density - old_density))
    new_mad = torch.mean(torch.abs(true_density - new_density))

    return new_mad - old_mad


class Environment():
    """ Loads a volume. Takes a list of seeds which
    which will be cycled through on every call of reset.

    Parameters
    ----------
    img : torch.Tensor
        The image to perform tracking on. Tensor with shape c x h x w x d.
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

    def __init__(self, img, radius, seeds, mask, true_density, actions, n_seeds=1, step_size=1.0,\
                 step_width=1.0, max_len=10000, alpha=1.0, beta=1e-3, friction=1e-4):

        self.head_id = 0
        self.n_resets = 0 # count number of resets
        self.img = Image(img)
        self.radius = radius
        self.seeds = seeds        
        self.mask = mask
        self.true_density = Image(true_density)
        self.n_seeds = n_seeds # the number of paths per bundle
        self.step_size = step_size
        self.step_width = step_width
        self.max_len = max_len
        self.action_space = actions
        if not isinstance(self.action_space, torch.Tensor):
            self.action_space = torch.tensor(self.action_space)

        seed_id = self.n_resets % len(self.seeds)
        center = torch.Tensor(self.seeds[seed_id])
        self.r = 4.0 # radius around center to randomly place starting points
        g = torch.Generator()
        g.manual_seed(0)
        offsets = torch.rand((self.n_seeds, 3), generator=g)
        offsets /= torch.sum(offsets**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        bundle_seeds = center[None] + r * offsets

        self.paths = [*torch.Tensor(bundle_seeds)[:,None]] # a list of N paths. each path is a 1 x 3 tensor
        self.alpha = alpha
        self.beta = beta
        self.friction = friction

        # initialize last two step directions randomly for each streamline
        g = torch.Generator()
        g.manual_seed(self.n_resets)
        last_steps = [*2*torch.rand(((len(self.paths),)+(1,3)), generator=g)-1.0] # list of len N paths, of 1x3 tensors
        last_steps = [x / np.sqrt(x[0,0]**2+x[0,1]**2+x[0,2]**2) for x in last_steps] # unit normalize directions
        self.paths = [torch.cat((point - 2*step*self.step_size, point - step*self.step_size, point)) for point, step in zip(self.paths, last_steps)]
        
        self.finished_paths = []

        # we will want to save completed paths
        self.finished_paths = []

        # initialize bundle density map
        self.bundle_density = torch.zeros_like(true_density)
        self.img.data = torch.cat((self.img.data, self.bundle_density.clone()), dim=0)
        self.bundle_density = Image(self.bundle_density)
        for i in range(len(self.paths)):
            # add_bundle_point(bundle_density, self.paths[i][0], self.ball)
            for j in range(len(self.paths[i])-1):
                segment = torch.stack((self.paths[i][j], self.paths[i][j+1]), dim=0)
                self.img.draw_line_segment(segment, width=0)
                self.bundle_density.draw_line_segment(segment, width=self.step_width)


    def get_state(self):
        """ Get the state for the current step at streamline 'head_id'. The state consists of an image patch and
        streamline density patch centered on the streamline head plus the last three streamline positions.

        Returns
        -------
        patch : torch.Tensor
            Tensor with shape (n x c x h x w x d) where the first channels are the input image and the last channel is the streamline density.
        last_steps : torch.Tensor
         Tensor with shape 3 x 3 (3 step positions by 3 euclidean coordinates)
        """
        patch, _ = self.img.crop(self.paths[self.head_id][-1], self.radius, pad=True, value=0.0)
        patch = patch.detach().clone()
        
        last_steps = self.paths[self.head_id][-2:].detach().clone() # 2 x 3 tensor of last three streamline positions

        return patch[None], last_steps[None]


    def get_reward(self, terminated, delta_density_diff, verbose=False):
        """ Get the reward for the current state. The reward depends on the streamline smoothness and
        the change in distance between the streamline density and true denisty maps.

        Parameters
        ----------
        terminated : bool
        alpha : float
            Weight for density matching.
        beta : float
            Weight for streamline smoothness.
        friction = 
            
        Returns
        -------
        reward : torch.Tensor
            Tensor with shape 1
        """
        if terminated:
            reward = 0
        else:
            prev_step = (self.paths[self.head_id][-2]-self.paths[self.head_id][-3]) / self.step_size
            current_step = (self.paths[self.head_id][-1]-self.paths[self.head_id][-2]) / self.step_size
            cos_angle = torch.dot(current_step, prev_step).to(float)
            # note that delta density difference is a change in error,
            # so negative change is good, hence the flipped (positive) exponent which is normally negative for sigmoid function.
            # it is also shifted down so that zero change yields zero.
            # sigmoid_diff = 2e3 / (1 + np.exp(delta_density_diff / 5e-3)) - 1e3 # calibrated so that a good step is around 1.
            m = -2.5e3 #-52631.
            b = 0.2 #-0.05263
            diff = m*delta_density_diff + b
            # sigmoid_diff = np.max([sigmoid_diff, 0.0]) # do not give negative values for matching
            reward = self.alpha*diff + self.beta*(cos_angle-1) - self.friction 
            if verbose:
                print(f'sigmoid_diff: {diff}','\n',
                      f'matching reward: {self.alpha*diff}','\n',
                      f'cos_angle: {cos_angle}','\n',
                      f'smoothing reward: {self.beta*(cos_angle-1)}', '\n',
                      f'friction reward: {-self.friction}')

        return torch.tensor([reward], device=DEVICE, dtype=torch.float32)


    def step(self, action, verbose=False):
        """ Take a tracking step for one streamline. Add the new position to the path and
        and the bundle density mask, and compute the reward and new state observation.

        Parameters
        ----------
        action : torch.Tensor
            Tensor with shape 3.
        
        Returns
        -------
        observation : tuple
            The img patch centered at the new streamline head (a tensor with shape c x h x w x d), and the
            last three streamline positions.
        reward : torch.Tensor
            Tensor with shape 1
        terminated : bool

        """
        terminated = False

        new_position = self.paths[self.head_id][-1] + self.step_size*action
        # decide if path terminates
        out_of_bound = any([x >= y or x < 0 for x,y in zip(torch.round(new_position), self.img.data.shape[1:])])
        if out_of_bound:
            terminate_path = True
        else:
            out_of_mask = 1 - self.mask[(0,)+tuple([int(x) for x in torch.round(new_position)])]
            too_long = len(self.paths[self.head_id]) > self.max_len
            self_terminate = not any(action)
            terminate_path = too_long or self_terminate or out_of_mask

        if terminate_path:
            observation = None
            reward = torch.tensor([0.], device=DEVICE)
            # remove the path from 'paths' and add it to 'finished_paths'
            self.finished_paths.append(self.paths.pop(self.head_id))
            # if that was the last path in the list, then terminate the episode
            if len(self.paths) == 0:
                terminated = True
            else:
                self.head_id = (self.head_id + 1)%len(self.paths)
        else:
            # add new position to path
            self.paths[self.head_id] = torch.cat((self.paths[self.head_id], new_position[None]))

            # get true density
            center = self.paths[self.head_id][-2]
            r = self.radius + int(np.ceil(self.step_size))
            true_density_patch, _ = self.true_density.crop(center, radius=r) # patch centered at previous step position
            
            # get old patch centered on old streamline head
            old_density_patch, _ = self.bundle_density.crop(center, radius=r)
            old_density_patch = old_density_patch.clone() # need to make a copy or else this will be modified by adding a point to img

            # draw the segment
            segment = self.paths[self.head_id][-2:, :3]
            self.img.draw_line_segment(segment, width=0)
            self.bundle_density.draw_line_segment(segment, width=self.step_width)

            # get the new patch centered on the old streamline head
            new_density_patch, _ = self.bundle_density.crop(center, radius=r)

            # find the change in error mean(|true - new|) - mean(|true - old|)
            delta_density_diff = density_error_change(true_density_patch, old_density_patch, new_density_patch)
            reward = self.get_reward(terminated, delta_density_diff, verbose=verbose)
            observation = self.get_state()

            self.head_id = (self.head_id + 1)%len(self.paths)

        return observation, reward, terminated


    def reset(self):
        # reset the bundle with new random previous step directions.
        self.n_resets += 1

        # start with next seed point
        seed_id = self.n_resets % len(self.seeds)
        center = torch.Tensor(self.seeds[seed_id])
        self.r = 5.0 # radius around center to randomly place starting points
        g = torch.Generator()
        g.manual_seed(0)
        offsets = torch.rand((self.n_seeds, 3), generator=g)
        offsets /= torch.sum(offsets**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        bundle_seeds = center[None] + r * offsets

        self.paths = [*torch.Tensor(bundle_seeds)[:,None]] # a list of N paths. each path is a 1 x 3 tensor

        # initialize last two step directions randomly for each streamline
        g = torch.Generator()
        g.manual_seed(self.n_resets)
        last_steps = [*2*torch.rand(((len(self.paths),)+(1,3)), generator=g)-1.0] # list of len N paths, of 1x3 tensors
        last_steps = [x / np.sqrt(x[0,0]**2+x[0,1]**2+x[0,2]**2) for x in last_steps] # unit normalize directions
        self.paths = [torch.cat((point - 2*step*self.step_size, point - step*self.step_size, point)) for point, step in zip(self.paths, last_steps)]

        self.finished_paths = []

        # reset bundle density
        self.bundle_density = torch.zeros_like(self.true_density.data)
        self.bundle_density = Image(self.bundle_density)
        self.img.data[-1] = torch.zeros_like(self.bundle_density.data[0])
        for i in range(len(self.paths)):
            for j in range(len(self.paths[i])-1):
                segment = torch.stack((self.paths[i][j], self.paths[i][j+1]), dim=0)
                self.img.draw_line_segment(segment, width=0)
                self.bundle_density.draw_line_segment(segment, width=self.step_width)

        return