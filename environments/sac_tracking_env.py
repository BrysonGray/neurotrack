#!/usr/bin/env python

"""
Soft actor critic reinforcement learning tractography environment

Author: Bryson Gray
2024
"""

from glob import glob
import json
import numpy as np
import os
from pathlib import Path
import sys
import tifffile as tf
from typing import Literal
import torch

script_path = Path(os.path.abspath(__file__))
parent_dir = script_path.parent.parent
sys.path.append(str(parent_dir))
from data_prep.image import Image
from environments import env_utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Environment():
    """
    This class represents the environment for soft actor critic (SAC) reinforcement learning applied to tracing antomical structures.
    It provides methods to initialize the environment, take steps, compute rewards, and reset the environment.
    """

    def __init__(
            self,
            img_path: str,
            radius: int,
            step_size: float = 1.0,
            step_width: float = 1.0,
            max_len: int = 10000,
            alpha: float = 1.0,
            beta: float = 1e-3,
            friction: float = 1e-4,
            classifier=None):
        """
        Initialize the SAC tracking environment.
        
        Parameters
        ----------
        img_path : str
            Path to the image file or directory containing image files.
        radius : int
            Radius around the center to randomly place starting points.
        step_size : float, optional
            Step size for tracking, by default 1.0.
        step_width : float, optional
            Step width for tracking, by default 1.0.
        max_len : int, optional
            Maximum length of the path, by default 10000.
        alpha : float, optional
            Alpha parameter for tracking, by default 1.0.
        beta : float, optional
            Beta parameter for tracking, by default 1e-3.
        friction : float, optional
            Friction parameter for tracking, by default 1e-4.
        classifier : optional
            Classifier for tracking, by default None.
            
        Attributes
        ----------
        img_files : list
            List of image file paths.
        img_idx : int
            Index of the current image file.
        img : Image
            Image object loaded from the image file.
        true_density : Image
            Image object representing the true neuron density.
        section_labels : Image
            Image object representing the section labels.
        mask : tensor
            Branch mask tensor.
        seeds : list
            List of seed points.
        graph : object
            Graph object representing the neuron structure.
        radius : int
            Radius around the center to randomly place starting points.
        step_size : float
            Step size for tracking.
        step_width : float
            Step width for tracking.
        max_len : int
            Maximum length of the path.
        classifier : optional
            Classifier for tracking.
        seed_idx : int
            Index of the current seed point.
        r : float
            Radius around the center to randomly place starting points.
        paths : list
            List of paths, each initialized with a 1 x 3 tensor.
        roots : list
            List of path start points.
        path_labels : list
            List of path labels. 0 means the path is not yet labeled.
        alpha : float
            Weight for the accuracy compononent of reward.
        beta : float
            Weight for the prior compenent of the reward.
        friction : float
            Friction weight for reward.
        finished_paths : list
            List of completed paths.
        head_id : int
            Head ID keeps track of the current path since there may be multiple paths per episode.
        """
    
        
        if os.path.isdir(img_path):
            self.img_files = [os.path.join(img_path, f) for f in os.listdir(img_path)]
        else:
            self.img_files = [img_path]

        self.img_idx = 0

        self.__load_data(self.img_files[self.img_idx])

        # TODO: remove old data format loader in next version update
        # neuron_data = torch.load(self.img_files[self.img_idx], weights_only=False)
        # img = neuron_data["image"]
        # self.img = Image(img)
        # neuron_density = neuron_data["neuron_density"]
        # self.true_density = Image(neuron_density)
        # section_labels = neuron_data["section_labels"]
        # self.section_labels = Image(section_labels)
        # self.mask = neuron_data["branch_mask"]
        # self.seeds = neuron_data["seeds"]
        # self.graph = neuron_data["graph"]
        
        self.radius = radius
        # make copies of the branch and terminal points so these can be changed while saving the originals
        self.step_size = step_size
        self.step_width = step_width
        self.max_len = max_len
        self.classifier = classifier

        self.seed_idx = 0
        seed = torch.Tensor(self.seeds[self.seed_idx])
        self.r = 0.0 # radius around center to randomly place starting points
        offset = torch.randn((1, 3))
        offset /= torch.sum(offset**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        seed = seed[None] + r * offset

        self.paths = [seed] # a list initialized with 1 path, a 1 x 3 tensor.
        self.roots = [seed[0]] # a list of path start points.
        i,j,k = [int(round(x.item())) for x in seed[0]]
        self.path_labels = [int(self.section_labels.data[0, i, j, k].item())] # a list with the current labels for each path.
        self.prev_children = [[]] # Keep track of the previous section's children for computing a section mask in reward calculation.
        self.alpha = alpha
        self.beta = beta
        self.friction = friction

        # we will want to save completed paths
        self.finished_paths = []

        self.img.data = torch.cat((self.img.data, torch.zeros((1,)+self.img.data.shape[1:])), dim=0) # add 1 channel for path.
        
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
            turn_around = False
            if len(self.paths[self.head_id]) > 1:
                s = torch.stack((self.paths[self.head_id][-1], new_position)) - self.paths[self.head_id][-2:]
                cos_dist = torch.dot(s[1]/torch.linalg.norm(s[1]), s[0]/torch.linalg.norm(s[0]))
                angle = torch.arccos(cos_dist)
                turn_around = angle > 3*np.pi/4

            too_long = len(self.paths[self.head_id]) > self.max_len

            if too_long:
                terminate_path = True
                status = "too_long"

            elif turn_around:
                terminate_path = True
                status = "choose_stop"

        return terminate_path, status
    

    def __load_data(self, path):
        img_file = glob(os.path.join(path, "*image.tif"))[0]
        img = tf.imread(img_file)
        self.img = Image(torch.from_numpy(img))
        density_file = glob(os.path.join(path, "*density.tif"))[0]
        density = tf.imread(density_file)
        self.true_density = Image(torch.from_numpy(density))
        section_labels_file = glob(os.path.join(path, "*sections.tif"))[0]
        section_labels = tf.imread(section_labels_file)
        self.section_labels = Image(torch.from_numpy(section_labels))
        seeds = glob(os.path.join(path, "*seeds.txt"))[0]
        with open(seeds, 'r') as f:
            self.seeds = [[int(x) for x in line.strip().split(' ')] for line in f if line.strip()]
        graph_file = glob(os.path.join(path, '*section_graph.json'))[0]
        with open(graph_file, 'r') as f:
            graph = json.load(f)
            # Convert all keys from string to int
            self.graph = {int(k): v for k, v in graph.items()}
    

    def get_state(self, terminate=False):
        """ Get the state for the current step at streamline 'head_id'. The state consists of an image patch and
        streamline density patch centered on the streamline head.
        
        Parameters
        ----------
        terminate : bool, optional
            If True, returns a zero tensor representing the terminated state. If False, returns the current state.
        
        Returns
        -------
        patch : torch.Tensor
            Tensor with shape (c x h x w x d) where the first channels are the input image.
        """
        if terminate:
            patch = torch.zeros((self.img.data.shape[0],)+(2*self.radius + 1,)*3)
        else:
            patch, _ = self.img.crop(self.paths[self.head_id][-1], self.radius, pad=True, value=0.0)
            patch = patch.clone()

        return patch[None]


    def get_reward(self, category: Literal["step", "out_of_image", "out_of_mask", "too_long", "choose_stop", "bifurcate"],
                   step_accuracy: float = 0.0,
                   verbose: bool = False) -> torch.Tensor:
        """
        Calculate the reward based on the given category and step accuracy.
        
        Parameters
        ----------
        category : Literal["step", "out_of_image", "out_of_mask", "too_long", "choose_stop", "bifurcate"]
            The category of the action taken.
        step_accuracy : float, optional
            The accuracy of the step taken, by default 0.0.
        verbose : bool, optional
            If True, prints detailed information about the reward calculation, by default False.
            
        Returns
        -------
        torch.Tensor
            The calculated reward as a tensor.
            
        Raises
        ------
        NameError
            If the provided category is not recognized.
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

        return torch.tensor([reward], dtype=torch.float32)


    def step(self, action, max_paths=100, verbose=False):
        """
        Perform a single step in the environment.
        
        Parameters
        ----------
        action : torch.Tensor
            The action to be taken, representing the direction of movement.
        max_paths : int, optional
            The maximum number of paths allowed, by default 100. Not currently implemented.
        verbose : bool, optional
            If True, additional information will be printed, by default False.
            
        Returns
        -------
        observation : torch.Tensor
            The current state of the environment after the step.
        reward : float
            The reward obtained from taking the step.
        terminated : bool
            Whether the episode has terminated.
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

            # check for max branches
            if len(self.finished_paths) > 80: # TODO: Max branches should be an argument 
                terminated = True
            # if that was the last path in the list, then we need to decide what to do
            elif len(self.paths) == 0:
                # if the path started at the seed and only took one step, then terminate the episode
                if torch.all(self.finished_paths[-1][0] == self.roots[0]) and len(self.finished_paths[-1]) <= 2:
                    terminated = True
                # otherwise, return to the seed point
                else:
                    # reset to the initial seed
                    self.paths.append(self.roots[0][None])
                    self.roots.append(self.roots[0])
                    i,j,k = [int(round(x.item())) for x in self.roots[0]]
                    self.path_labels = [int(self.section_labels.data[0, i, j, k].item())]
                    self.head_id = 0
                    terminated = False
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
            segment_vec = segment[1] - segment[0]
            segment_length = torch.linalg.norm(segment_vec)
            L = int(torch.ceil(segment_length)) + 1 # The radius of the patch is the whole line length since the line starts at patch center.
            overhang = int(2*self.step_width) # include space beyond the end of the line
            patch_radius = L + overhang
            density_patch, _ = self.true_density.crop(center, patch_radius, interp=False, pad=False)

            # mask out competing paths
            labels_patch, _ = self.section_labels.crop(center, patch_radius, interp=False, pad=False)
            end_point = patch_radius + segment_vec
            new_label_idx = (0, int(round(end_point[0].item())), int(round(end_point[1].item())), int(round(end_point[2].item())))
            new_label = int(labels_patch[new_label_idx].item())
            current_label = self.path_labels[self.head_id]

            # Optimize section masking
            if current_label != 0:
                # Pre-compute the section IDs
                prev_children = self.prev_children[self.head_id]
                graph_current = self.graph[current_label]
                section_ids = [current_label] + [x for x in graph_current if x not in prev_children]
                
                # Create mask using vectorized operations
                section_mask = torch.zeros_like(density_patch, dtype=torch.bool)
                for id in section_ids:
                    section_mask |= (labels_patch == id)
                
                true_patch_masked = density_patch * section_mask.float()
                
                # Update label if needed
                if new_label != current_label and new_label in section_ids:
                    self.path_labels[self.head_id] = new_label
                    self.prev_children[self.head_id] = graph_current
            else:
                true_patch_masked = density_patch
                if new_label != 0:
                    self.path_labels[self.head_id] = new_label

            true_patch_masked = density_patch # don't mask
            step_accuracy = -env_utils.density_error_change(true_patch_masked[0], old_patch, new_patch)
            reward = self.get_reward(status, step_accuracy, verbose)

            observation = self.get_state() 

            # self.head_id = (self.head_id + 1)%len(self.paths) # only move to the next path if the current path is terminated.

            # decide if path branches
            if self.classifier is not None:
                out = self.classifier(observation[:,:3, 10:25, 10:25, 10:25].to(DEVICE))
                out = torch.sigmoid(out.squeeze())
                if out > 0.5: # create branch
                    distances = torch.linalg.norm(torch.stack(self.roots) - new_position, dim=1)
                    if not torch.any(distances < 3.0):
                        self.paths.append(new_position[None])
                        self.path_labels.append(0)
                        self.prev_children.append(self.prev_children[self.head_id])
                        self.roots.append(new_position)

        return observation, reward, terminated


    def reset(self, move_to_next=True):
        """
        Resets the environment state.
        
        Parameters
        ----------
        move_to_next : bool, optional
            If True, move to the next image or seed and reset the path. Default is True.
            
        Returns
        -------
        None
        """
        
        if move_to_next:
            # reset the agent to the next image or seed and reset the path.
            self.seed_idx += 1
            self.seed_idx = self.seed_idx % len(self.seeds) # type: ignore
            if self.seed_idx == 0:
                self.img_idx += 1
                self.img_idx = self.img_idx % len(self.img_files)

                self.__load_data(self.img_files[self.img_idx])
                
                # TODO: remove in next version
                # # load the next image
                # neuron_data = torch.load(self.img_files[self.img_idx], weights_only=False)
                # img = neuron_data["image"]
                # self.img = Image(img)
                # neuron_density = neuron_data["neuron_density"]
                # self.true_density = Image(neuron_density)
                # section_labels = neuron_data["section_labels"]
                # self.section_labels = Image(section_labels)
                # self.mask = neuron_data["branch_mask"]
                # self.seeds = neuron_data["seeds"]
                # self.graph = neuron_data["graph"]

        seed = torch.tensor(self.seeds[self.seed_idx]) # type: ignore
        self.r = 0.0 # radius around center to randomly place starting points
        offset = torch.randn((1, 3))
        offset /= torch.sum(offset**2, dim=1)**0.5
        r = self.r * torch.rand(1)
        seed = seed[None] + r * offset

        self.paths = [seed] # a list initialized with 1 path, a 1 x 3 tensor.
        self.roots = [seed[0]] # a list of path start points.
        i,j,k = [int(round(x.item())) for x in seed[0]]
        self.path_labels = [int(self.section_labels.data[0, i, j, k].item())] # a list with the current labels for each path.
        self.prev_children = [[]] # Keep track of the previous section's children for computing a section mask in reward calculation.
        self.finished_paths = []
        self.img.data = torch.cat((self.img.data[:3], torch.zeros((1,)+self.img.data.shape[1:])), dim=0) # add 1 channel for path.

        self.head_id = 0
        self.img.draw_point(self.paths[self.head_id][-1], radius=(self.step_width-1)//2, channel=3, binary=False)

        return
    
if __name__ == "__main__":
    pass