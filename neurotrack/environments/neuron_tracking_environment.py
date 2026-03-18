#!/usr/bin/env python

"""
Neuron tracking environment for SAC reinforcement learning

Author: Bryson Gray
2024
"""

import numpy as np
import tifffile as tf
import torch
from typing import Dict, Tuple, Literal, Optional, List

from neurotrack.data import loading as load
from neurotrack.data import tree
from neurotrack.data.image import Image, to_uint8
from neurotrack.environments.tracking_reward import (
    _get_nearest_node, _get_termination_nodes, _init_visited,
    _compute_target_point, distance_reward, _add_to_visited,
    remove_visited, _get_connected_nodes)
from torch.utils.data import DataLoader as TorchDataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuronTrackingEnvironment:
    """
    Enhanced neuron tracking environment using NeuronPatchDataset.
    
    The environment automatically samples neuron patch data from the dataset when reset() is called.
    Patches contain pre-processed images, subtrees, and masks ready for tracking.
    """
    
    def __init__(self, dataset,
                 radius: int = 17, target_step_len: float = 4.0, step_width: float = 4.0,
                 stall_threshold: float = 1.0,
                 max_len: int = 10000, max_paths: int = 1000, gamma=0.99, branching: bool = False,
                 repeat_starts: bool = False, start_idx: int = 0,
                 inference_mode: bool = False):
        """
        Initialize the enhanced SAC tracking environment.
        
        Parameters:
        -----------
        dataset : NeuronPatchDataset or torch.utils.data.Dataset
            Dataset object for sampling neuron patches
        radius : int
            Radius around the center to randomly place starting points
        target_step_len : float
            Target step length for tracking
        step_width : float
            Step width for tracking
        stall_threshold : float
            Minimum action magnitude before a step is treated as a stop choice.
        max_len : int
            Maximum length of the path
        max_paths : int
            Maximum number of paths allowed
        gamma : float
            Discount factor for reward computation
        branching : bool
            Whether to enable branching during training
        repeat_starts : bool
            Whether to repeatedly restart at the beginning of a completed path
        """
        self.dataset = dataset
        self.current_patch_idx = start_idx
        self.current_neuron_info = None
        
        # Store initialization parameters
        self.radius = radius
        self.target_step_len = target_step_len
        dataset_step_width = getattr(dataset, "step_width", None)
        self.step_width = float(dataset_step_width) if dataset_step_width is not None else step_width
        self.stall_threshold = float(stall_threshold)
        if self.stall_threshold < 0.0:
            raise ValueError("stall_threshold must be non-negative")
        self.stall_threshold2 = self.stall_threshold * self.stall_threshold
        self.max_len = max_len
        self.max_paths = max_paths
        self.gamma = gamma
        self.branching = branching
        self.repeat_starts = repeat_starts
        self.inference_mode = inference_mode
        
        # Initialize other attributes that will be set when neuron data is loaded
        self.img = None
        self.neuron_mask = None
        self.seeds = []
        self.paths = []
        self.roots = torch.empty((0, 3), dtype=torch.float32)
        self._zero_state_patch: Optional[torch.Tensor] = None
        self.finished_paths = []
        self.section_nodes = None
        self.termination_points = None # termination points for the current path
        self.cut_ends = []
        self.visited = {}
        self.id_to_idx = {}
        self.adj_dict = {}
        self.full_tree = None
        self.unvisited_tree = None
        self.target_vectors = None
        self.section_assigned = False
        self.has_ground_truth = False

    def _set_roots(self, roots: List[torch.Tensor]) -> None:
        """Set cached branch-root tensor representation."""
        if len(roots) == 0:
            self.roots = torch.empty((0, 3), dtype=torch.float32)
            return
        self.roots = torch.stack([r.to(dtype=torch.float32) for r in roots], dim=0)

    def _append_root(self, root: torch.Tensor) -> None:
        """Append a branch root while keeping cached tensor in sync."""
        root_t = root.to(dtype=torch.float32)
        if self.roots.numel() == 0:
            self.roots = root_t.unsqueeze(0)
            return
        if self.roots.device != root_t.device:
            root_t = root_t.to(self.roots.device)
        self.roots = torch.cat((self.roots, root_t.unsqueeze(0)), dim=0)

    def _zero_target_vectors(self, device=None) -> torch.Tensor:
        """Return a single zero target vector candidate."""
        if device is None:
            device = torch.device("cpu")
            if self.unvisited_tree is not None:
                device = self.unvisited_tree.device
            elif self.img is not None:
                device = self.img.data.device
        return torch.zeros((1, 3), dtype=torch.float32, device=device)

    def _target_vectors_from_points(self, position: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        """Convert absolute target points to target vectors relative to a position."""
        position_t = position.to(dtype=torch.float32)
        if target_points is None:
            return self._zero_target_vectors(device=position_t.device)
        target_points_t = torch.as_tensor(target_points, dtype=torch.float32, device=position_t.device).view(-1, 3)
        if target_points_t.numel() == 0:
            return self._zero_target_vectors(device=position_t.device)
        return target_points_t - position_t.unsqueeze(0)

    def _setup_environment(self, patch_data: Dict):
        """
        Setup environment with patch data from NeuronPatchDataset.
        
        Parameters:
        -----------
        patch_data : Dict
            Dictionary containing:
            - image: composite image (already processed)
            - neuron_tree: subtree data (already in proper format)
            - neuron_mask: neuron area mask
            - name: filename
            - image_idx: source image index
            - global_idx: global patch index
        """
        
        # Setup image
        img_tensor = patch_data['image']
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension if needed
        if img_tensor.dtype != torch.uint8:
            img_tensor = to_uint8(img_tensor)
        self.img = Image(img_tensor)
        
        # Setup neuron mask and tree if available
        neuron_mask_data = patch_data.get('neuron_mask', None)
        neuron_tree_data = patch_data.get('neuron_tree', None)
        self.has_ground_truth = (not self.inference_mode) and (neuron_mask_data is not None) and (neuron_tree_data is not None)

        if self.has_ground_truth:
            self.neuron_mask = neuron_mask_data
            if self.neuron_mask.ndim == 4:
                self.neuron_mask = self.neuron_mask[0]  # Remove channel dimension if present

            # Get subtree - already in proper format from dataset
            subtree = torch.as_tensor(neuron_tree_data, dtype=torch.float32)  # Already in x,y,z order
            self.adj_dict = load.adjacency_dict(subtree)
            self.full_tree = subtree.clone()
            self.full_tree[:, 2:5] = self.full_tree[:, 2:5].flip(dims=(1,))  # Convert to z, y, x order
            self.unvisited_tree = self.full_tree.clone()
            self.id_to_idx = {int(node_id): idx for idx, node_id in enumerate(self.unvisited_tree[:, 0].tolist())}
        else:
            self.neuron_mask = None
            self.adj_dict = {}
            self.full_tree = torch.empty((0, 7), dtype=torch.float32)
            self.unvisited_tree = self.full_tree.clone()
            self.id_to_idx = {}

        # Determine seed point(s) from dataset output only.
        seed_points_data = patch_data.get('seed_points', None)
        if seed_points_data is None:
            raise ValueError("patch_data must include 'seed_points' with shape (N, 3) in (z, y, x) order.")

        seeds = torch.as_tensor(seed_points_data, dtype=torch.float32)
        if seeds.ndim == 1:
            seeds = seeds.unsqueeze(0)
        if seeds.ndim != 2 or seeds.shape[1] != 3:
            raise ValueError("patch_data['seed_points'] must have shape (N, 3) in (z, y, x) order.")
        
        self.paths = [[p] for p in seeds.unbind(0)]
        self._set_roots(list(seeds.unbind(0)))
        self.cut_ends = []
        if self.has_ground_truth:
            # Initialize visited edges
            self.visited = _init_visited(self.full_tree)
            # Initialize first path: assign section nodes, termination points, target vectors, and section_assigned flag.
            self._init_path()
        else:
            self.visited = {}
            self.termination_points = torch.empty((0, 3), dtype=torch.float32)
            self.target_vectors = self._zero_target_vectors()
            self.section_nodes = None
            self.section_assigned = False

        if self.paths:
            self.img.draw_point(
                self.paths[0][-1], 
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
            delta = new_position - self.paths[0][-1]
            step_size2 = (delta * delta).sum()
            stall_threshold2 = self.stall_threshold2
            stall = step_size2 < stall_threshold2
            if stall:
                status = "choose_stop"
            elif len(self.paths[0]) > self.max_len: # Check if path is too long
                status = "too_long"
            # If new position is out of neuron mask, truncate.
            elif self.neuron_mask is not None and not self.neuron_mask[new_position[0].int(), new_position[1].int(), new_position[2].int()]:
                status = "out_of_mask"
            else:
                status = "continue"

        return status


    def _step_without_ground_truth(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, bool, Dict]:
        """Step function for inference-only mode without neuron tree or mask."""
        terminated = False
        truncated = False
        info = {
            'terminate_episode': False,
            'current_target_vectors': self._zero_target_vectors(device=action.device),
            'next_target_vectors': self._zero_target_vectors(device=action.device),
            'status': "continue",
        }

        current_position = self.paths[0][-1]
        new_position = current_position + action
        status = self._get_status(new_position)
        info['status'] = status

        if status in ["out_of_image", "choose_stop"]:
            terminated = True
            reward = torch.tensor(0.0, dtype=torch.float32)
            info['terminate_episode'] = self._terminate_path(training=False)
            observation = self.get_state(terminate=True)
            return observation, reward, terminated, truncated, info

        if status == "too_long":
            truncated = True
            reward = torch.tensor(0.0, dtype=torch.float32)
            info['terminate_episode'] = self._terminate_path(training=False)
            observation = self.get_state(terminate=True)
            return observation, reward, terminated, truncated, info

        self.paths[0].append(new_position)
        segment = torch.stack(self.paths[0][-2:], dim=0)
        self.img.draw_line_segment(segment, width=self.step_width, channel=-1, mask=True)
        if self.branching:
            roots_tensor = self.roots
            if roots_tensor.device != new_position.device:
                roots_tensor = roots_tensor.to(new_position.device)
            distances = ((roots_tensor - new_position) ** 2).sum(dim=1)
            if not torch.any(distances < 49.0):  # no branches within 7 pixels (7^2 to avoid sqrt)
                self.paths.append([new_position])
                self._append_root(new_position)
                
        observation = self.get_state()
        reward = torch.tensor(0.0, dtype=torch.float32)
        return observation, reward, terminated, truncated, info
    

    def _init_path(self):
        """
        Assign new section nodes. If there are cut ends, sections nodes will begin from nearby cut ends. If not, from the nearest node within the agents observable window.
        If no unvisited nodes remain, or no nearby nodes, section nodes will be None.
        """
        termination_nodes = []
        # Set default termination points, target vectors, and section nodes.
        self.termination_points = torch.empty((0, 3), dtype=torch.float32, device=self.unvisited_tree.device)
        self.target_vectors = self._zero_target_vectors(device=self.unvisited_tree.device)
        self.section_nodes = None
        self.section_assigned = False
        if self.unvisited_tree.shape[0] > 0:
            # use the nearby cut ends if available
            if self.cut_ends and len(self.cut_ends) > 0:
                cut_ends_indices = [self.id_to_idx[int(v)] for v in self.cut_ends]
                swc_filtered = self.unvisited_tree[cut_ends_indices]
                node_coords = swc_filtered[:, 2:5]  # shape (M,3)
                # Compute distances
                dists2 = torch.sum((node_coords - self.paths[0][0].unsqueeze(0)) ** 2, dim=1)
                # get cut ends within 17 pixels i.e. 17^2 = 289
                close_mask = dists2 < 289.0
                if torch.any(close_mask):
                    close_cut_ends_nodes = torch.tensor(self.cut_ends)[close_mask]
                    termination_nodes = []
                    self.section_nodes = []
                    for ce in close_cut_ends_nodes:
                        connected_nodes, terminals = _get_connected_nodes(int(ce), adj_dict=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
                        self.section_nodes.extend(connected_nodes)
                        termination_nodes.extend(terminals)
                    mask = torch.isin(self.full_tree[:, 0], torch.tensor(termination_nodes))
                    self.termination_points = self.full_tree[mask][:, 2:5]
                    self.section_assigned = True
                    target_points = _compute_target_point(self.paths[0][0], self.unvisited_tree, self.target_step_len, adj_dict=self.adj_dict,
                                                        id_to_idx=self.id_to_idx, terminal_points=None, valid_nodes=close_cut_ends_nodes.tolist())
                    self.target_vectors = self._target_vectors_from_points(self.paths[0][0], target_points)
            else: # otherwise use the nearest node
                nearest_node = _get_nearest_node(self.paths[0][0], self.unvisited_tree, id_to_idx=self.id_to_idx)
                nearest_node_coords = self.unvisited_tree[self.id_to_idx[nearest_node], 2:5]
                dists2 = torch.sum((nearest_node_coords - self.paths[0][0]) ** 2)
                if dists2 < 289.0:  # only assign section if within 17 pixels
                    # Initialize section nodes for new path
                    self.section_nodes, terminals = _get_connected_nodes(nearest_node, adj_dict=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
                    mask = torch.isin(self.full_tree[:, 0], torch.tensor(terminals))
                    self.termination_points = self.full_tree[mask][:, 2:5]
                    self.section_assigned = True
                    target_points = _compute_target_point(self.paths[0][0], self.unvisited_tree, self.target_step_len, adj_dict=self.adj_dict,
                                                        id_to_idx=self.id_to_idx, terminal_points=None, valid_nodes=self.section_nodes)
                    self.target_vectors = self._target_vectors_from_points(self.paths[0][0], target_points)


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
        finished_path = torch.stack(self.paths.pop(0), dim=0)
        self.finished_paths.append(finished_path)
        # Check for max branches
        if len(self.finished_paths) > self.max_paths:
            terminate_episode = True
        elif training and self.repeat_starts and len(self.finished_paths[-1]) > 4:
            # If the path took more than three steps, add a new path at the same root
            self.paths.append([finished_path[0]])
            self._append_root(finished_path[0])
        elif len(self.paths) == 0:
            terminate_episode = True
        if not terminate_episode: # Move to next path
            self._init_path()

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

        if not self.has_ground_truth:
            return self._step_without_ground_truth(action)
        
        with torch.no_grad():
            terminated = False # Path termination flag
            truncated = False # Path truncation flag
            info = {'terminate_episode': False,
                    'current_target_vectors': None,
                    'next_target_vectors': None,}

            direction = action
            current_position = self.paths[0][-1]
            new_position = current_position + direction
            status = self._get_status(new_position)
            info['current_target_vectors'] = self.target_vectors
            info['status'] = status

            if status in ["out_of_image", "choose_stop"]: # then terminate path
                terminated = True
                # Terminate the branch, but the episode may continue.
                # Reward is negative squared distance to nearest termination point times 1 / (1 - gamma).
                reward = distance_reward(direction, self.target_vectors, terminated=True, gamma=self.gamma)
                # terminate path
                info['terminate_episode'] = self._terminate_path(training)
                observation = self.get_state(terminate=True)
            
            else: # Take step
                # Add new position to path
                self.paths[0].append(new_position)
                
                # Draw the segment on the state input image
                segment = torch.stack(self.paths[0][-2:], dim=0)
                self.img.draw_line_segment(segment, width=self.step_width, channel=-1, mask=True)

                # get new observation
                observation = self.get_state()

                # Get reward
                reward = distance_reward(direction, self.target_vectors, terminated=False)

                if status in ["out_of_mask", "too_long"]:  # Truncate path
                    truncated = True
                    # terminate path
                    info['terminate_episode'] = self._terminate_path(training)
                else:
                    # Check if section_nodes should be assigned (cut ends entered window)
                    if self.section_nodes is None and self.cut_ends and len(self.cut_ends) > 0:
                        cut_ends_indices = [self.id_to_idx[int(v)] for v in self.cut_ends]
                        swc_filtered = self.unvisited_tree[cut_ends_indices]
                        node_coords = swc_filtered[:, 2:5]  # shape (M,3)
                        dists2 = torch.sum((node_coords - new_position.unsqueeze(0)) ** 2, dim=1)
                        close_mask = dists2 < 289.0  # within 17 pixels
                        if torch.any(close_mask):
                            # Assign section nodes from nearby cut ends
                            close_cut_ends = swc_filtered[close_mask][:, 0].tolist()
                            self.section_nodes = []
                            termination_nodes = []
                            for node in close_cut_ends:
                                connected_nodes, terminals = _get_connected_nodes(int(node), adj_dict=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
                                self.section_nodes.extend(connected_nodes)
                                termination_nodes.extend(terminals)
                            if termination_nodes:
                                mask = torch.isin(self.full_tree[:, 0], torch.tensor(termination_nodes))
                                self.termination_points = self.full_tree[mask][:, 2:5]
                            self.section_assigned = True
                    
                    # update visited edges
                    if self.cut_ends and len(self.cut_ends) > 0:
                        # use nearest cut end as starting point
                        cut_ends_indices = [self.id_to_idx[int(v)] for v in self.cut_ends]
                        swc_filtered = self.unvisited_tree[cut_ends_indices]
                        node_coords = swc_filtered[:, 2:5]  # shape (M,3)
                        # Compute distances
                        dists2 = torch.sum((node_coords - new_position.unsqueeze(0)) ** 2, dim=1)
                        start_pos = node_coords[torch.argmin(dists2)]
                    else:
                        start_pos = current_position

                    neuron_end_point = None
                    if self.section_nodes is not None:
                        potentially_visited, neuron_end_point = _add_to_visited(start_pos, new_position, self.unvisited_tree, self.visited, adj_dict=self.adj_dict,
                                                    id_to_idx=self.id_to_idx, valid_nodes=self.section_nodes)
                    if neuron_end_point is not None:
                        dist_to_neuron_sq = ((neuron_end_point - new_position)**2).sum()
                        if dist_to_neuron_sq < 289.0:  # within 17 pixels
                            self.visited = potentially_visited
                            self.unvisited_tree, self.visited, self.adj_dict, changed_nodes = remove_visited(self.unvisited_tree, self.visited, self.adj_dict, id_to_idx=self.id_to_idx)
                        
                            # Update id_to_idx mapping
                            if self.unvisited_tree.shape[0] > 0:
                                self.id_to_idx = {int(node_id): idx for idx, node_id in enumerate(self.unvisited_tree[:, 0].tolist())}
                            else:
                                self.id_to_idx = {}

                        
                            self.cut_ends.extend(changed_nodes)
                            # remove nodes from cut ends that no longer exist
                            valid_keys = set(self.id_to_idx.keys())
                            self.cut_ends = list({
                                int(ce)
                                for ce in self.cut_ends
                                if int(ce) in valid_keys
                                and len(self.adj_dict.get(int(ce), [])) > 0
                            })

                            # TEST: try updating section nodes based on proximity to the new position. Use any cut ends within 17 pixels.
                            # Update the section nodes
                            if self.unvisited_tree.shape[0] > 0 and len(self.cut_ends) > 0:
                                cut_ends_indices = [self.id_to_idx[int(v)] for v in self.cut_ends]
                                swc_filtered = self.unvisited_tree[cut_ends_indices]
                                node_coords = swc_filtered[:, 2:5]  # shape (M,3)
                                # Compute distances
                                dists2 = torch.sum((node_coords - new_position.unsqueeze(0)) ** 2, dim=1)
                                # get cut ends within 17 pixels i.e. 17^2 = 289
                                close_mask = dists2 < 289.0
                                close_cut_ends = swc_filtered[close_mask][:, 0].tolist()
                                self.section_nodes = []
                                self.termination_nodes = []
                                for node in close_cut_ends:
                                    connected_nodes, terminals = _get_connected_nodes(int(node), adj_dict=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
                                    self.section_nodes.extend(connected_nodes)
                                    self.termination_nodes.extend(terminals)
                                if self.termination_nodes:
                                    mask = torch.isin(self.full_tree[:, 0], torch.tensor(self.termination_nodes))
                                    self.termination_points = self.full_tree[mask][:, 2:5]
                            elif self.unvisited_tree.shape[0] == 0:
                                self.section_nodes = None

                    # Create new branches during training
                    if self.branching:
                        roots_tensor = self.roots
                        if roots_tensor.device != new_position.device:
                            roots_tensor = roots_tensor.to(new_position.device)
                        distances = ((roots_tensor - new_position) ** 2).sum(dim=1)
                        if not torch.any(distances < 49.0):  # no branches within 7 pixels (7^2 to avoid sqrt)
                            self.paths.append([new_position])
                            self._append_root(new_position)

                    # Compute next target vector
                    valid_nodes = self.cut_ends if len(self.cut_ends) > 0 else self.section_nodes
                    target_points = _compute_target_point(new_position, self.unvisited_tree, self.target_step_len, adj_dict=self.adj_dict,
                                    id_to_idx=self.id_to_idx, terminal_points=self.termination_points, valid_nodes=valid_nodes)
                    
                    # if no neuron nodes or termination points are within radius, then if the path has already been assigned
                    # a section but walked away from it, trucate the path. If the path has no section assigned, i.e. it started
                    # at a new path with no neuron within the agent window, then the target point is the current position.
                    target_out_of_window = len(target_points) == 0 or ((target_points - new_position)**2).sum(dim=1).min() > self.radius**2
                    if target_out_of_window: # target out of window
                        if self.section_assigned: # if no target points found in window and section already assigned, truncate.
                            # truncate path
                            truncated = True
                            info['terminate_episode'] = self._terminate_path(training)
                        else: # if no target points found and no section assigned, set target to current position.
                            self.target_vectors = self._zero_target_vectors(device=direction.device)
                            info['next_target_vectors'] = self.target_vectors
                    else: # target points found in window
                        if not self.section_assigned: # if target points found and no section assigned, assign section.
                            self.section_assigned = True
                        self.target_vectors = self._target_vectors_from_points(new_position, target_points)
                        info['next_target_vectors'] = self.target_vectors

        return observation, reward, terminated, truncated, info

    
    def reset(self, move_to_next: bool = True, dataset_index=None, return_state: bool = False):
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
        self.roots = torch.empty((0, 3), dtype=torch.float32)
        self._zero_state_patch = None
        self.finished_paths = []
        self.termination_points = None # termination points for the current path
        self.visited = {}
        self.id_to_idx = {}
        self.adj_dict = {}
        self.full_tree = None
        self.unvisited_tree = None
        self.target_vectors = None
        self.has_ground_truth = False

        if move_to_next:
            # Sample new patch data from dataset
            if dataset_index is not None:
                if not isinstance(dataset_index, int):
                    raise TypeError(f"dataset index must be int but got {type(dataset_index)}")
                self.current_patch_idx = dataset_index
            else:
                # Move to next patch
                self.current_patch_idx += 1
            patch_data = self.dataset[self.current_patch_idx]
        else:
            # Reuse current patch
            patch_data = self.dataset[self.current_patch_idx]
        
        # Store patch info
        self.current_neuron_info = {
            'neuron_name': patch_data.get('neuron_name', 'unknown'),
            'image_idx': patch_data.get('image_idx', -1),
            'global_idx': patch_data.get('global_idx', self.current_patch_idx)
        }

        # Setup environment with patch data
        self._setup_environment(patch_data)

        if return_state:
            return self.get_state()
        return None
    

    def get_state(self, terminate=False):
        """Get the state for the current step at streamline 'head_id'."""
        if self.img is None:
            raise ValueError("No neuron data loaded. Call reset() first.")
            
        with torch.no_grad():
            if terminate:
                patch_shape = (self.img.data.shape[0],) + (2 * self.radius + 1,) * 3
                if (
                    self._zero_state_patch is None
                    or tuple(self._zero_state_patch.shape) != patch_shape
                    or self._zero_state_patch.dtype != self.img.data.dtype
                    or self._zero_state_patch.device != self.img.data.device
                ):
                    self._zero_state_patch = torch.zeros(
                        patch_shape,
                        dtype=self.img.data.dtype,
                        device=self.img.data.device,
                    )
                patch = self._zero_state_patch
            else:
                center = self.paths[0][-1]
                radius = int(self.radius)
                patch, _ = self.img.crop(center, radius, pad=True, value=0.0)

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
    

def create_neuron_tracking_environment(dataset, **env_kwargs) -> NeuronTrackingEnvironment:
    """
    Create a neuron tracking environment setup.
    
    Parameters:
    -----------
    dataset : NeuronPatchDataset or torch.utils.data.Dataset
        Dataset containing neuron patches
    **env_kwargs : dict
        Additional arguments for environment initialization
    
    Returns:
    --------
    NeuronTrackingEnvironment
        Configured environment ready for use. Call reset() to load first patch.
    
    Note:
    -----
    The environment will automatically sample and load patch data when reset() is called.
    Patches contain pre-processed images, subtrees, and masks.
    """
    
    # Create environment
    environment = NeuronTrackingEnvironment(
        dataset=dataset,
        **env_kwargs
    )
    
    return environment


if __name__ == "__main__":
    # Example usage
    from neurotrack.data.datasets import NeuronPatchDataset
    import numpy as np
    
    # Create dataset
    dataset = NeuronPatchDataset(
        swc_dir="data_cache",
        img_dir="simple_data",
        crop_size=64,
        patches_per_image=10,
        alpha=0.2,
        rng=np.random.default_rng(42)
    )
    
    # Create environment
    env = create_neuron_tracking_environment(dataset)
    
    # Reset to load first patch
    env.reset()
    
    # Environment is now ready for training/inference
    print(f"Loaded patch: {env.current_neuron_info['neuron_name']}")
    print(f"Image shape: {env.img.data.shape}")
    print(f"Number of seeds: {len(env.seeds)}")
