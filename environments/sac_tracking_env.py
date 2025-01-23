#!/usr/bin/env python

"""
Soft actor critic reinforcement learning tractography environment

Author: Bryson Gray
2024
"""

import os
from typing import Literal

from matplotlib import category
import torch

from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent))
from data.image import Image
import env_utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Environment():
    """ Loads a volume. Takes a list of seeds which
    which will be cycled through on every call of reset.

    Parameters
    ----------
    img_path : str
        Path to image file or directory.
    radius : int
        The radius of a state input image patch along each axis (total width is 2*radius + 1).
    seeds : ndarray
        N x 3 array of starting streamline coordinates
    n_seeds : int, optional
        Number of seeds per bundle. Default is 1
    step_size : float, optional
        Distance taken in each step in units of pixels. Default is 1.0.
    step_width : float, optional
        Width of the path segments drawn by the agent. Default is 1.0.
    max_len : int, optional
        Maximum number of steps allowed per streamline. Default is 10000
    alpha : float, optional
        Density matching reward weight. Default is 1.0
    beta : float, optional
        Smoothness reward weight. Default is 1e-3.
    friction : float, optional
        Friction reward weight. Default is 1e-4.
    branching : bool, optional
        If true, model branching is allowed. Default is True.
    
    TODO: update attrubutes
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
    step_size : float
        Distance taken in each step in units of pixels.
    max_len : int
        Maximum number of steps allowed per streamline
    n_resets : int
        Count of the number of episode resets during training.

    """

    def __init__(
            self,
            img_path: str,
            radius: int,
            seeds: list[tuple[int, int, int]] | None = None,
            step_size: float = 1.0,
            step_width: float = 1.0,
            max_len: int = 10000,
            alpha: float = 1.0,
            beta: float = 1e-3,
            friction: float = 1e-4,
            branching: bool = True,
            classifier=None):
        
        if os.path.isdir(img_path):
            self.img_files = [os.path.join(img_path, f) for f in os.listdir(img_path)]
        else:
            self.img_files = [img_path]

        self.img_idx = 0
        print(f"Loading image {self.img_files[self.img_idx]}")
        neuron_data = torch.load(self.img_files[self.img_idx], weights_only=False)
        img = neuron_data["image"]
        self.img = Image(img.to(device=DEVICE))
        neuron_density = neuron_data["neuron_density"]
        self.true_density = Image(neuron_density.to(device=DEVICE))
        section_labels = neuron_data["section_labels"]
        self.section_labels = Image(section_labels.to(device=DEVICE))
        self.mask = neuron_data["branch_mask"]
        self.seeds = neuron_data["seeds"]
        self.graph = neuron_data["graph"]
        
        self.radius = radius
        # make copies of the branch and terminal points so these can be changed while saving the originals
        self.step_size = step_size
        self.step_width = step_width
        self.max_len = max_len
        self.branching = branching
        self.classifier = classifier

        self.seed_idx = 0
        seed = torch.Tensor(self.seeds[self.seed_idx])
        self.r = 0.0 # radius around center to randomly place starting points
        offset = torch.randn((1, 3))
        offset /= torch.sum(offset**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        seed = seed[None] + r * offset
        seed = seed.to(device=DEVICE)

        self.paths = [seed] # a list initialized with 1 path, a 1 x 3 tensor.
        self.roots = [seed[0]] # a list of path start points.
        self.path_labels = [0] # a list of path labels. 0 means the path is not yet labeled.
        self.alpha = alpha
        self.beta = beta
        self.friction = friction

        # we will want to save completed paths
        self.finished_paths = []

        if self.branching:
            self.img.data = torch.cat((self.img.data, torch.zeros((2,)+self.img.data.shape[1:], device=DEVICE)), dim=0) # add 2 channels for path, and bifurcation points.
        else:
            self.img.data = torch.cat((self.img.data, torch.zeros((1,)+self.img.data.shape[1:], device=DEVICE)), dim=0) # add 1 channel for path.
        
        self.head_id = 0 # head id keeps track of the current path since there may be multiple paths per episode 
        self.img.draw_point(self.paths[self.head_id][-1], radius=(self.step_width-1)//2, channel=3, binary=False)

    
    def __step_prior(self, sigmaf: float = 1.5, sigmab: float = 1.5) -> float:
        prior = 0.0
        if len(self.paths[self.head_id]) > 2: # ignore the prior for the first step.
            q = self.paths[self.head_id][-1]
            q_ = self.paths[self.head_id][-2]
            q__ = self.paths[self.head_id][-3]
            prior = - torch.sum((q - q_)**2).item()/(2*sigmaf**2) - torch.sum((q - 2*q_ + q__)**2).item() / (2*sigmab**2)
        
        return prior
    

    def __get_status(self, new_position):

        status = "step"
        terminate_path = False
        # decide if path terminates accidentally
        out_of_image = any([x >= y or x < 0 for x,y in zip(torch.round(new_position), self.img.data.shape[1:])])
        if out_of_image:
            terminate_path = True
            status = "out_of_image"
        else:
            # out_of_mask = ~self.mask[(0,)+tuple([int(torch.round(x)) for x in new_position])]
            turn_around = False
            if len(self.paths[self.head_id]) > 1:
                s = torch.stack((self.paths[self.head_id][-1], new_position)) - self.paths[self.head_id][-2:]
                cos_dist = torch.dot(s[1]/torch.linalg.norm(s[1]), s[0]/torch.linalg.norm(s[0]))
                angle = torch.arccos(cos_dist)
                turn_around = angle > 3*torch.pi/4
            # label = None
            # if self.classifier is not None:
            #     patch, _ = self.img.crop(new_position, self.radius, pad=True, value=0.0)
            #     out = self.classifier(patch[None])
            #     label = torch.argmax(out).squeeze()
                # out = torch.nn.functional.sigmoid(out.squeeze())
                # label = out > 0.5

            too_long = len(self.paths[self.head_id]) > self.max_len

            if too_long:
                terminate_path = True
                status = "too_long"
            # elif label is not None:
            #     if not label:
            #         terminate_path = True
            #         status = "choose_stop"
            elif turn_around:
                terminate_path = True
                status = "choose_stop"
            # elif out_of_mask:
            #     terminate_path = True
            #     status = "out_of_mask"
        return terminate_path, status
    

    def get_state(self, terminate=False):
        """ Get the state for the current step at streamline 'head_id'. The state consists of an image patch and
        streamline density patch centered on the streamline head plus the last three streamline positions.

        Returns
        -------
        patch : torch.Tensor
            Tensor with shape (c x h x w x d) where the first channels are the input image.
        last_step : torch.Tensor
            Tensor with shape 1 x 3 
        """
        if terminate:
            patch = torch.zeros((self.img.data.shape[0],)+(2*self.radius + 1,)*3, device=DEVICE)
        else:
            patch, _ = self.img.crop(self.paths[self.head_id][-1], self.radius, pad=True, value=0.0)
            patch = patch.clone()

        return patch[None]


    def get_reward(self, category: Literal["step", "out_of_image", "out_of_mask", "too_long", "choose_stop", "bifurcate"],
                   step_accuracy: float = 0.0,
                   verbose: bool = False) -> torch.Tensor:
        """ Get the reward for the current state. The reward depends on the streamline smoothness and
        the change in distance between the streamline density and true denisty maps.

        Parameters
        category: str
            Reward category.
        step_accuracy: float, Optional
            
        Returns
        -------
        reward : torch.Tensor
            Tensor with shape 1
        """

        if category == "out_of_image":
            reward = 0.0 
            if verbose:
                print('out_of_image \n',
                      f'reward: {reward}\n')
        elif category == "out_of_mask":
            reward = 0.0 
            if verbose:
                print('out_of_mask \n',
                      f'reward: {reward}\n')
        elif category == "too_long":
            reward = 0.0
            if verbose:
                print('too_long \n',
                      f'reward: {reward}\n')
        elif category == "choose_stop":
            reward = 0.0
            if verbose:
                print('choose_stop \n',
                      f'reward: {reward}\n')
        elif category == "bifurcate":
            reward = 0.0
            if verbose:
                print('bifurcate \n',
                      f'reward: {reward}\n')
        elif category == "step":
            prior = self.__step_prior()
            reward = self.alpha * step_accuracy + self.beta * prior

        else:
            raise NameError(f"category: {category} was not recognized.")

        return torch.tensor([reward], device=DEVICE, dtype=torch.float32)


    def step(self, action, max_paths=100, verbose=False):
        """ Take a tracking step for one streamline. Add the new position to the path
        and the bundle density mask, and compute the reward and new state observation.

        Parameters
        ----------
        direction : torch.Tensor
            Direction vector with shape (3,).
        choice : int
            Categorical choice of {step, branch, terminate}.
        max_branches : int
            Maximum number of paths allowed.
        
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

        direction = action
            
        new_position = self.paths[self.head_id][-1] + direction

        terminate_path, status = self.__get_status(new_position)


        if terminate_path:
            reward = self.get_reward(status, verbose=verbose)
            observation = self.get_state(terminate=True)
            # remove the path from 'paths' and add it to 'ended_paths'
            self.finished_paths.append(self.paths.pop(self.head_id).cpu())
            self.path_labels.pop(self.head_id)
            # if that was the last path in the list, then terminate the episode
            if len(self.paths) == 0:
                terminated = True
            # otherwise, move to the next path
            else:
                self.head_id = (self.head_id + 1)%len(self.paths)

        else: # otherwise take a step
            # add new position to path
            self.paths[self.head_id] = torch.cat((self.paths[self.head_id], new_position[None]))
            # draw the segment on the state input image
            segment = self.paths[self.head_id][-2:, :3]
            old_patch, new_patch = self.img.draw_line_segment(segment, width=self.step_width, binary=False)
            # get reward
            center = torch.round(segment[0]).to(dtype=torch.int)
            segment_length = torch.linalg.norm(segment[1] - segment[0])
            L = int(torch.ceil(segment_length)) + 1 # The radius of the patch is the whole line length since the line starts at patch center.
            overhang = int(2*self.step_width) # include space beyond the end of the line
            patch_radius = L + overhang
            true_patch, _ = self.true_density.crop(center, patch_radius, interp=False, pad=False)

            # mask out competing paths
            # true_patch_label, _ = self.section_labels.crop(center, patch_radius, interp=False, pad=False)

            # new_label = int(true_patch_label[0, patch_radius, patch_radius, patch_radius].item())
            # current_label = self.path_labels[self.head_id]
            # # here mask out any sections that are not the current section or its children
            # if current_label != 0:
            #     section_ids = [current_label, *self.graph[current_label]]
            #     section = torch.zeros_like(true_patch)
            #     for id in section_ids:
            #         section += torch.where(true_patch_label == id, 1, 0)
            #     true_patch_masked = true_patch * section
            #     if new_label != current_label and new_label in section_ids:
            #         self.path_labels[self.head_id] = new_label
            # else:
            #     true_patch_masked = true_patch
            #     if new_label != current_label:
            #         self.path_labels[self.head_id] = new_label

            true_patch_masked = true_patch # don't mask
            step_accuracy = -env_utils.density_error_change(true_patch_masked[0], old_patch, new_patch)
            reward = self.get_reward(status, step_accuracy, verbose)

            observation = self.get_state() 

            # self.head_id = (self.head_id + 1)%len(self.paths) # only move to the next path if the current path is terminated.

            # decide if path branches
            if self.classifier is not None:
                out = self.classifier(observation[:,:3, 10:25, 10:25, 10:25])
                out = torch.nn.functional.sigmoid(out.squeeze())
                if out > 0.5: # create branch
                    distances = torch.linalg.norm(torch.stack(self.roots) - new_position, dim=1)
                    if not torch.any(distances < 3.0):
                        self.paths.append(new_position[None])
                        self.path_labels.append(0)
                        self.roots.append(new_position)

        return observation, reward, terminated


    def reset(self, move_to_next=True):
        if move_to_next:
            # reset the agent to the next image or seed and reset the path.
            self.seed_idx += 1
            self.seed_idx = self.seed_idx % len(self.seeds) # type: ignore
            if self.seed_idx == 0:
                self.img_idx += 1
                self.img_idx = self.img_idx % len(self.img_files)

                # load the next image
                print(f"Loading image {self.img_files[self.img_idx]}")
                neuron_data = torch.load(self.img_files[self.img_idx], weights_only=False)
                img = neuron_data["image"]
                self.img = Image(img.to(device=DEVICE))
                neuron_density = neuron_data["neuron_density"]
                self.true_density = Image(neuron_density.to(device=DEVICE))
                section_labels = neuron_data["section_labels"]
                self.section_labels = Image(section_labels.to(device=DEVICE))
                self.mask = neuron_data["branch_mask"]
                self.seeds = neuron_data["seeds"]
                self.graph = neuron_data["graph"]

        seed = torch.tensor(self.seeds[self.seed_idx]) # type: ignore
        self.r = 0.0 # radius around center to randomly place starting points
        offset = torch.randn((1, 3))
        offset /= torch.sum(offset**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        seed = seed[None] + r * offset
        seed = seed.to(device=DEVICE)

        self.paths = [seed] # a list initialized with 1 path, a 1 x 3 tensor.
        self.roots = [seed[0]]
        self.path_labels = [0] # a list of path labels. 0 means the path is not yet labeled.
        self.finished_paths = []

        if self.branching:
            self.img.data = torch.cat((self.img.data[:3], torch.zeros((2,)+self.img.data.shape[1:], device=DEVICE)), dim=0) # add 2 channels for path, and bifurcation points.
        else:
            self.img.data = torch.cat((self.img.data[:3], torch.zeros((1,)+self.img.data.shape[1:], device=DEVICE)), dim=0) # add 1 channel for path.

        self.head_id = 0
        self.img.draw_point(self.paths[self.head_id][-1], radius=(self.step_width-1)//2, channel=3, binary=False)

        return