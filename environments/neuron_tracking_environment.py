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
from environments.tracking_reward import (
    _get_nearest_node, _get_termination_nodes, _init_visited,
    _compute_target_point, _distance_reward, _add_to_visited, remove_visited
)
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
                 radius: int = 17, step_size: float = 4.0, step_width: float = 4.0,
                 max_len: int = 10000, max_paths: int = 1000, friction: float = 0.0,
                 gamma=0.99, branching: bool = False, repeat_starts: bool = False, section_masking: bool = False, classifier=None):
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
        self.gamma = gamma
        self.friction = friction
        self.branching = branching
        self.repeat_starts = repeat_starts
        self.section_masking = section_masking
        self.classifier = classifier
        
        # Initialize other attributes that will be set when neuron data is loaded
        self.img = None
        self.neuron_mask = None
        self.seeds = []
        self.paths = []
        self.roots = []
        self.finished_paths = []
        self.termination_points = [] # list of lists of termination nodes for each path
        self.visited = {}
        self.id_to_idx = {}
        self.edge_list = {}
        self.full_tree = None
        self.unvisited_tree = None
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
        self.neuron_mask = neuron_data['reward_mask']
        if self.neuron_mask.ndim == 4:
            self.neuron_mask = self.neuron_mask[0]  # Remove channel dimension if present

        # Generate seeds from subtree endpoints/branch points
        subtree = torch.tensor(neuron_data['subtree']) # remember subtree is in x,y,z order
        self.edge_list = load.undirected_edge_list(subtree)
        self.full_tree = subtree
        self.full_tree[:, 2:5] = self.full_tree[:, 2:5].flip(dims=(1,))  # Convert to z, y, x order
        self.unvisited_tree = self.full_tree.clone()
        self.id_to_idx = {int(node_id): idx for idx, node_id in enumerate(self.unvisited_tree[:, 0].tolist())}

        # Find endpoints and branch points as seeds
        end_nodes = [k for k, v in self.edge_list.items() if len(v) == 1]
        n_seeds = len(end_nodes) // 2 # Limit number of seeds to half the endpoints and a maximum of 10
        n_seeds = min(n_seeds, 10)
        mask = torch.isin(self.full_tree[:,0], torch.tensor(end_nodes))
        seeds = self.full_tree[mask][:n_seeds, 2:5]

        # If no seeds found, raise an error
        if len(seeds) == 0:
            raise ValueError(f"No valid seed points found in subtree. Edge list: {self.edge_list}")
        
        self.paths = [[p] for p in seeds.unbind(0)]
        self.roots = list(seeds.unbind(0))

        self.termination_points = []
        for root in self.roots:
            nearest_node = _get_nearest_node(root.numpy(), swc_list=self.full_tree, current_section_id=None)
            termination_nodes = _get_termination_nodes(nearest_node, self.edge_list)
            mask = torch.isin(self.full_tree[:, 0], torch.tensor(termination_nodes))
            term_pts = self.full_tree[mask][:, 2:5]
            self.termination_points.append(term_pts)
        self.visited = _init_visited(self.full_tree)

        # Add channel for path visualization
        if self.img.data.shape[0] == 1:
            self.img.data = torch.cat((
                self.img.data, 
                torch.zeros((1,) + self.img.data.shape[1:], dtype=self.img.data.dtype)
            ), dim=0)
        
        if self.paths:
            self.img.draw_point(
                self.paths[self.head_id][-1], 
                radius=(self.step_width / 2.35), 
                channel=-1, 
                mode="gaussian", 
                binary=False
            )

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
            status: str in {"out_of_image", "out_of_mask", "too_long", "choose_stop", "continue"}
        """
        
        # Check if out of image bounds
        bounds = torch.as_tensor(self.img.data.shape[1:], device=new_position.device, dtype=new_position.dtype)
        out_of_image = (new_position < 0).any() or (new_position >= bounds).any()
        if out_of_image:
            status = "out_of_image"
        else:
            # Check for small step (stalling)
            delta = new_position - self.paths[self.head_id][-1]
            step_size2 = (delta * delta).sum()
            stall_threshold2 = 1.0  # squared threshold (1.0^2)
            stall = step_size2 < stall_threshold2
            if stall:
                status = "choose_stop"
            elif len(self.paths[self.head_id]) > self.max_len: # Check if path is too long
                status = "too_long"
            # If new position is out of neuron mask, truncate.
            elif not self.neuron_mask[new_position[0].int(), new_position[1].int(), new_position[2].int()]:
                status = "out_of_mask"
            else:
                status = "continue"

        return status

    def _terminate_path(self, training) -> bool:
        """
        Remove current path and move to next path. Determine if episode should terminate.

        Parameters
        ----------
        training : bool
            Whether in training mode (affects branching behavior)

        Returns
        -------
        terminate_episode : bool
            True if there are no remaining paths after termination
        """

        terminate_episode = False
        # Convert list of points to stacked tensor once when finalizing the path
        finished_path = torch.stack(self.paths.pop(self.head_id), dim=0)
        self.finished_paths.append(finished_path)
        self.termination_points.pop(self.head_id)
        # Check for max branches
        if len(self.finished_paths) > self.max_paths:
            terminate_episode = True
        elif training and self.repeat_starts and len(self.finished_paths[-1]) > 4:
            # If the path took more than three steps, add a new path at the same root
            self.paths.append([self.roots[self.head_id]])
            self.roots.append(self.roots[self.head_id])
        elif len(self.paths) == 0:
            terminate_episode = True
        if not terminate_episode: # Move to next path
            self.head_id = (self.head_id + 1) % len(self.paths)

        return terminate_episode
    
    def step(self, action: torch.Tensor, verbose: bool = False, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor, bool]:
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
            (observation, reward, terminated, truncated, info) - the new state, reward, termination flag, truncation flag, and additional information
        """
        if self.img is None:
            raise ValueError("No neuron data loaded. Call reset() first.")
        
        with torch.no_grad():
            terminated = False # Path termination flag
            truncated = False # Path truncation flag
            info = {'terminate_episode': False}

            direction = action
            new_position = self.paths[self.head_id][-1] + direction

            status = self._get_status(new_position)

            if status in ["out_of_image", "choose_stop"]: # then terminate path
                terminated = True
                # Terminate the branch, but the episode may continue.
                # Reward is negative squared distance to nearest termination point.
                tp = self.termination_points[self.head_id]
                reward = _distance_reward(new_position, tp)
                observation = self.get_state(terminate=True)
                # terminate path
                info['terminate_episode'] = self._terminate_path(training)
            
            else: # Take step
                # Add new position to path
                self.paths[self.head_id].append(new_position)
                
                # Draw the segment on the state input image
                segment = torch.stack(self.paths[self.head_id][-2:], dim=0)
                self.img.draw_line_segment(segment, width=self.step_width, channel=-1, mask=False)

                # get new observation
                observation = self.get_state()

                if status in ["out_of_mask", "too_long"]:  # Truncate path
                    truncated = True

                    # Get reward
                    if status == "out_of_mask":
                        reward = torch.tensor([-289.0], dtype=torch.float32)  # -17^2
                    else:  # too_long. reward is negative squared distance to target
                        target_points = _compute_target_point(self.paths[self.head_id][-2], self.unvisited_tree, self.step_size, edge_list=self.edge_list, id_to_idx=self.id_to_idx)
                        reward = _distance_reward(self.paths[self.head_id][-1], target_points)

                    # terminate path
                    info['terminate_episode'] = self._terminate_path(training)

                else:
                    # get reward
                    target_points = _compute_target_point(self.paths[self.head_id][-2], self.unvisited_tree, self.step_size, edge_list=self.edge_list, id_to_idx=self.id_to_idx)
                    reward = _distance_reward(self.paths[self.head_id][-1], target_points)

                    # update visited edges
                    self.visited = _add_to_visited(self.paths[self.head_id][-2], self.paths[self.head_id][-1], self.unvisited_tree, self.visited, edge_list=self.edge_list, id_to_idx=self.id_to_idx)
                    self.unvisited_tree, self.visited, self.edge_list = remove_visited(self.unvisited_tree, self.visited, self.edge_list, id_to_idx=self.id_to_idx)
                    if self.unvisited_tree.shape[0] > 0:
                        self.id_to_idx = {int(node_id): idx for idx, node_id in enumerate(self.unvisited_tree[:, 0].tolist())}
                    else:
                        self.id_to_idx = {}


        return observation, reward, terminated, truncated, info

    
    def reset(self, move_to_next: bool = True, dataset_index=None):
        """
        Reset environment state by sampling new neuron data from dataloader.
        
        Parameters:
        -----------
        move_to_next : bool
            Deprecated parameter, kept for compatibility. Does nothing.
            New neuron data is automatically sampled from the dataloader.
        """

        # Clear attributes that will be set when neuron data is loaded
        self.img = None
        self.neuron_mask = None
        self.seeds = []
        self.paths = []
        self.roots = []
        self.finished_paths = []
        self.termination_points = [] # list of lists of termination nodes for each path
        self.visited = {}
        self.id_to_idx = {}
        self.edge_list = {}
        self.full_tree = None
        self.unvisited_tree = None
        self.head_id = 0

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
            
        with torch.no_grad():
            if terminate:
                patch = torch.zeros((self.img.data.shape[0],)+(2*self.radius + 1,)*3)
            else:
                center = self.paths[self.head_id][-1]
                patch, _ = self.img.crop(center, self.radius, pad=True, value=0.0)
                patch = patch.clone()
                if patch.dtype == torch.uint8:
                    patch = patch / 255.0

        return patch[None]


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
    print(f"Loaded neuron: {env.current_neuron_info['neuron_name']}")
    print(f"Image shape: {env.img.data.shape}")
    print(f"Number of seeds: {len(env.seeds)}")
