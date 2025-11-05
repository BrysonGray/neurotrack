#!/usr/bin/env python

"""
Neuron tracking environment for SAC reinforcement learning

Author: Bryson Gray
2024
"""

import numpy as np
import os
from pathlib import Path
import sys
import tifffile as tf
import torch
from typing import Dict, Tuple, Literal
import warnings

script_path = Path(os.path.abspath(__file__))
parent_dir = script_path.parent.parent
sys.path.append(str(parent_dir))

from data_prep import load
from data_prep.image import Image
from environments import env_utils
from neurotrack.data.neuron_data import Dataset, DataLoader, DataGenerator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuronTrackingEnvironment:
    """
    Enhanced neuron tracking environment using DataLoader for neuron sampling.
    
    The environment automatically samples neuron data from the dataloader when reset() is called.
    It loads TIFF images, SWC subtrees, and reward masks from file paths, then sets up the 
    tracking environment with appropriate seeds and visualization channels.
    """
    
    def __init__(self, dataloader: DataLoader,
                 radius: int = 17, step_size: float = 2.0, step_width: float = 4.0,
                 max_len: int = 10000, max_paths: int = 1000, alpha: float = 1.0,
                 beta: float = 0.2, friction: float = 0.0, branching: bool = False,
                 repeat_starts: bool = False, section_masking: bool = False, classifier=None):
        """
        Initialize the enhanced SAC tracking environment.
        
        Parameters:
        -----------
        dataloader : DataLoader
            DataLoader object for sampling neurons
        radius : int
            Radius around the center to randomly place starting points
        step_size : float
            Step size for tracking
        step_width : float
            Step width for tracking
        max_len : int
            Maximum length of the path
        max_paths : int
            Maximum number of paths allowed
        alpha : float
            Alpha parameter for tracking
        beta : float
            Beta parameter for tracking
        friction : float
            Friction parameter for tracking
        repeat_starts : bool
            Whether to repeatedly restart at the beginning of a completed path
        section_masking : bool
            Whether to mask out all sections except the current section and its descendants
        classifier : optional
            Branch classifier for tracking
        """
        self.dataloader = dataloader
        self.current_neuron_info = None
        
        # Store initialization parameters without calling parent __init__ yet
        self.radius = radius
        self.step_size = step_size
        self.step_width = step_width
        self.max_len = max_len
        self.max_paths = max_paths
        self.alpha = alpha
        self.beta = beta
        self.friction = friction
        self.branching = branching
        self.repeat_starts = repeat_starts
        self.section_masking = section_masking
        self.classifier = classifier
        
        # Initialize other attributes that will be set when neuron data is loaded
        self.img = None
        self.true_density = None
        self.seeds = []
        self.paths = []
        self.roots = []
        self.path_labels = []
        self.prev_children = []
        self.finished_paths = []
        self.head_id = 0
    
    def _process_file_paths(self, file_data: Dict) -> Dict:
        """
        Process file paths from dataloader.sample() into neuron data format.
        
        Parameters:
        -----------
        file_data : Dict
            Dictionary containing file paths with keys:
            - neuron_name: name of neuron
            - img_path: path to image TIFF file
            - swc_path: path to SWC subtree file
            - reward_mask_path: path to reward mask TIFF file
            - complexity: neuron complexity level
            - morphology: morphology complexity category
        
        Returns:
        --------
        Dict containing:
        - image: processed image tensor
        - subtree: subtree data
        - reward_mask: reward mask tensor  
        - metadata: additional information
        """
        # Load image
        img_array = tf.imread(file_data['img_path'])
        if img_array.ndim == 3:
            img_array = img_array[None]  # Add channel dimension
        image_tensor = torch.from_numpy(img_array)
        
        # Load subtree
        subtree = load.swc(file_data['swc_path'], verbose=False)
        
        # Load reward mask
        reward_array = tf.imread(file_data['reward_mask_path'])
        if reward_array.ndim == 3:
            reward_array = reward_array[None]  # Add channel dimension  
        reward_tensor = torch.from_numpy(reward_array)
        
        return {
            'image': image_tensor,
            'subtree': subtree,
            'reward_mask': reward_tensor,
            'metadata': file_data
        }
    
    def _setup_environment(self, neuron_data: Dict):
        """
        Setup environment with neuron data.
        
        Parameters:
        -----------
        neuron_data : Dict
            Dictionary containing:
            - image: processed image tensor
            - subtree: subtree data
            - reward_mask: reward mask tensor
            - metadata: additional information
        """
        
        # Setup image
        img_tensor = neuron_data['image']
        self.img = Image(img_tensor)
        
        # Setup reward mask as true density
        reward_tensor = neuron_data['reward_mask']
        self.true_density = Image(reward_tensor)
        
        # Generate seeds from subtree endpoints/branch points
        subtree = neuron_data['subtree']
        edge_list = load.undirected_edge_list(subtree)
        
        # Find endpoints and branch points as seeds
        n_endpoints = sum(1 for neighbors in edge_list.values() if len(neighbors) == 1)
        seeds = []
        for node_id, neighbors in edge_list.items():
            if len(seeds) >= n_endpoints//2:  # Limit number of seeds to half the endpoints
                break
            if len(neighbors) == 1:  # Endpoints only
                node = next(row for row in subtree if row[0] == node_id)
                seeds.append([int(node[4]), int(node[3]), int(node[2])])  # z, y, x to match image coords
        
        # If no seeds found, raise an error
        if not seeds:
            raise ValueError(f"No valid seed points found in subtree. Edge list: {edge_list}")
        
        self.seeds = seeds[:min(10, len(seeds))]  # Limit to 10 seeds
        
        # Initialize paths and other state variables
        self.paths = [torch.tensor(seed).float().unsqueeze(0) for seed in self.seeds]
        self.roots = [torch.tensor(seed).float() for seed in self.seeds]
        self.path_labels = [0] * len(self.paths)
        self.prev_children = [[]] * len(self.paths)
        self.finished_paths = []
        
        # Add channel for path visualization
        if self.img.data.shape[0] == 1:
            self.img.data = torch.cat((
                self.img.data, 
                torch.zeros((1,) + self.img.data.shape[1:], dtype=self.img.data.dtype)
            ), dim=0)
        
        self.head_id = 0
        if self.paths:
            self.img.draw_point(
                self.paths[self.head_id][-1], 
                radius=(self.step_width / 2.35), 
                channel=-1, 
                mode="gaussian", 
                binary=False
            )
    
    def _step_prior(self, sigmaf: float = 1.5, sigmab: float = 1.5) -> float:
        """
        Calculate the prior probability for the current step based on path smoothness.
        
        Parameters:
        -----------
        sigmaf : float
            Standard deviation for forward smoothness penalty
        sigmab : float  
            Standard deviation for backward smoothness penalty
            
        Returns:
        --------
        float
            Prior probability value
        """
        prior = 0.0
        if len(self.paths[self.head_id]) > 2:  # ignore the prior for the first step
            q = self.paths[self.head_id][-1]
            q_ = self.paths[self.head_id][-2]
            q__ = self.paths[self.head_id][-3]
            prior = - torch.sum((q - q_)**2).item()/(2*sigmaf**2) - torch.sum((q - 2*q_ + q__)**2).item() / (2*sigmab**2)
        
        return prior
    
    def _get_status(self, new_position):
        """
        Determine the status of a proposed new position and whether to terminate the path.
        
        Parameters:
        -----------
        new_position : torch.Tensor
            The proposed new position
            
        Returns:
        --------
        tuple
            (terminate_path: bool, status: str) indicating whether to terminate and the reason
        """
        status = "step"
        terminate_path = False
        
        # Check if out of image bounds
        out_of_image = any([x >= y or x < 0 for x,y in zip(new_position, self.img.data.shape[1:])])
        if out_of_image:
            terminate_path = True
            status = "out_of_image"
        else:
            # Check for turn around (sharp angle change)
            turn_around = False
            if len(self.paths[self.head_id]) > 1:
                s = torch.stack((self.paths[self.head_id][-1], new_position)) - self.paths[self.head_id][-2:]
                cos_dist = torch.dot(s[1]/torch.linalg.norm(s[1]), s[0]/torch.linalg.norm(s[0]))
                angle = torch.arccos(cos_dist)
                turn_around = angle > 3*np.pi/4

            # Check if path is too long
            too_long = len(self.paths[self.head_id]) > self.max_len

            if too_long:
                terminate_path = True
                status = "too_long"
            elif turn_around:
                terminate_path = True
                status = "choose_stop"

        return terminate_path, status
    
    def reset(self, move_to_next: bool = True, dataset_index=None):
        """
        Reset environment state by sampling new neuron data from dataloader.
        
        Parameters:
        -----------
        move_to_next : bool
            Deprecated parameter, kept for compatibility. Does nothing.
            New neuron data is automatically sampled from the dataloader.
        """
        if move_to_next:
            # Sample new neuron data from dataloader
            if dataset_index is not None:
                if not isinstance(dataset_index, int):
                    raise TypeError(f"dataset index must be int but got {type(dataset_index)}")
                if dataset_index < 0 or dataset_index >= len(self.dataloader.dataset):
                    raise ValueError(f"dataset index {dataset_index} is out of range [0, {len(self.dataloader.dataset)-1}]")
                file_data = self.dataloader.dataset[dataset_index]
                self.dataloader.current_idx = dataset_index
            else:
                file_data = self.dataloader.sample()
        else:
            file_data = self.dataloader.dataset[self.dataloader.current_idx]
        
        # Process file paths into neuron data format
        neuron_data = self._process_file_paths(file_data)
        self.current_neuron_info = file_data

        # Setup environment with new neuron data
        self._setup_environment(neuron_data)
    
    def get_state(self, terminate=False):
        """Get the state for the current step at streamline 'head_id'."""
        if self.img is None:
            raise ValueError("No neuron data loaded. Call reset() first.")
            
        if terminate:
            patch = torch.zeros((self.img.data.shape[0],)+(2*self.radius + 1,)*3)
        else:
            center = self.paths[self.head_id][-1]
            patch, _ = self.img.crop(center, self.radius, pad=True, value=0.0)
            patch = patch.clone()
            if patch.dtype == torch.uint8:
                patch = patch / torch.tensor(255.0, dtype=torch.float32)

        return patch[None]
    
    def step(self, action, verbose=False, training=True):
        """
        Perform a single step in the environment.
        
        Parameters:
        -----------
        action : torch.Tensor
            The action to be taken, representing the direction of movement
        verbose : bool
            If True, additional information will be printed
        training : bool
            Whether in training mode (affects branching behavior)
            
        Returns:
        --------
        tuple
            (observation, reward, terminated) - the new state, reward, and termination flag
        """
        if self.img is None:
            raise ValueError("No neuron data loaded. Call reset() first.")
        
        terminate_path = False
        terminated = False

        direction = action
        new_position = self.paths[self.head_id][-1] + direction

        terminate_path, status = self._get_status(new_position)

        if terminate_path:
            reward = self.get_reward(status, verbose=verbose)
            observation = self.get_state(terminate=True)
            
            # Remove the path from 'paths' and add it to 'finished_paths'
            self.finished_paths.append(self.paths.pop(self.head_id).cpu())
            self.path_labels.pop(self.head_id)

            # Check for max branches
            if len(self.finished_paths) > self.max_paths:
                terminated = True
            elif training and self.repeat_starts and len(self.finished_paths[-1]) > 4:
                # If the path took more than three steps, add a new path at the same root
                self.paths.append(self.roots[self.head_id][None])
                self.roots.append(self.roots[self.head_id])
                self.path_labels.append(0)
            elif len(self.paths) == 0:
                terminated = True

            if not terminated:
                self.head_id = (self.head_id + 1) % len(self.paths)

        else:  # Take a step
            # Add new position to path
            self.paths[self.head_id] = torch.cat((self.paths[self.head_id], new_position[None]))
            
            # Draw the segment on the state input image
            segment = self.paths[self.head_id][-2:, :3]
            old_patch, new_patch = self.img.draw_line_segment(segment, width=self.step_width, channel=-1, mask=False)
            if self.img.data.dtype == torch.uint8:
                old_patch = old_patch / torch.tensor(255.0, dtype=torch.float32)
                new_patch = new_patch / torch.tensor(255.0, dtype=torch.float32)
                
            # Get reward
            segment_vec = segment[1] - segment[0]
            L = int(torch.abs(segment_vec).max().item())  # Radius is the whole line length
            overhang = int(2*self.step_width)  # Include space beyond the end of the line
            patch_radius = L + overhang
            center = segment[0]
            density_patch, _ = self.true_density.crop(center, patch_radius, interp=False, pad=False)

            # For now, use simple density without section masking
            # TODO: Add section masking support if needed
            true_patch_masked = density_patch

            step_accuracy = -env_utils.density_error_change(true_patch_masked[0], old_patch, new_patch)
            reward = self.get_reward(status, step_accuracy, verbose)

            observation = self.get_state()

            # Create new branches during training
            if training and self.branching:
                distances = torch.linalg.norm(torch.stack(self.roots) - new_position, dim=1)
                if not torch.any(distances < 12.0):
                    self.paths.append(new_position[None])
                    self.path_labels.append(0)
                    self.prev_children.append([])
                    self.roots.append(new_position)

        return observation, reward, terminated
    
    def get_reward(self, category: Literal["step", "out_of_image", "out_of_mask", "too_long", "choose_stop", "bifurcate"], 
                   step_accuracy: float = 0.0, verbose: bool = False) -> torch.Tensor:
        """
        Calculate the reward based on the given category and step accuracy.
        
        Parameters:
        -----------
        category : Literal
            The category of the action taken
        step_accuracy : float
            The accuracy of the step taken
        verbose : bool
            If True, prints detailed information about the reward calculation
            
        Returns:
        --------
        torch.Tensor
            The calculated reward as a tensor
            
        Raises:
        -------
        NameError
            If the provided category is not recognized
        """
        if category == "out_of_image":
            reward = 0.0 
            if verbose:
                print('out_of_image \n', f'reward: {reward}\n')
        elif category == "out_of_mask":
            reward = 0.0 
            if verbose:
                print('out_of_mask \n', f'reward: {reward}\n')
        elif category == "too_long":
            reward = 0.0
            if verbose:
                print('too_long \n', f'reward: {reward}\n')
        elif category == "choose_stop":
            reward = 0.0
            if verbose:
                print('choose_stop \n', f'reward: {reward}\n')
        elif category == "bifurcate":
            reward = 0.0
            if verbose:
                print('bifurcate \n', f'reward: {reward}\n')
        elif category == "step":
            prior = self._step_prior()
            reward = self.alpha * step_accuracy + self.beta * prior
            if verbose:
                print(f'step \n accuracy: {step_accuracy}\n prior: {prior}\n reward: {reward}\n')
        else:
            raise NameError(f"category: {category} was not recognized.")

        return torch.tensor([reward], dtype=torch.float32)


# Convenience function to create a complete setup
def create_neuron_tracking_environment(data_dir: str, 
                                     complexity: float = 0.0,
                                     rng_seed: int = 0, 
                                     **env_kwargs) -> NeuronTrackingEnvironment:
    """
    Create a neuron tracking environment setup.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing neuron data CSV file and associated TIFF/SWC files
    complexity : float
        Initial complexity parameter for neuron sampling
    rng_seed : int
        Random seed for reproducibility
    **env_kwargs : dict
        Additional arguments for environment initialization
    
    Returns:
    --------
    NeuronTrackingEnvironment
        Configured environment ready for use. Call reset() to load first neuron.
    
    Note:
    -----
    The environment will automatically sample and load neuron data when reset() is called.
    Data is loaded from file paths returned by the DataLoader.
    """

    rng = np.random.default_rng(rng_seed)

    # Create dataset
    dataset = Dataset(data_dir=data_dir, rng=rng)

    # Create dataloader
    dataloader = DataLoader(dataset=dataset, complexity=complexity, rng=rng)
    
    # Create environment
    environment = NeuronTrackingEnvironment(
        dataloader=dataloader,
        **env_kwargs
    )
    
    return environment


if __name__ == "__main__":
    # Example usage
    data_dir = "/home/brysongray/data/neurotrack_data/gold166/gold166_swc_processed_subset"
    
    # Create environment
    env = create_neuron_tracking_environment(data_dir, complexity=0.2)
    
    # Reset to load first neuron from dataloader
    env.reset()
    
    # Environment is now ready for training/inference
    print(f"Loaded neuron: {env.current_neuron_data['metadata']['neuron_name']}")
    print(f"Image shape: {env.img.data.shape}")
    print(f"Number of seeds: {len(env.seeds)}")
