#!/usr/bin/env python

"""
Neuron tracking environment for SAC reinforcement learning

Author: Bryson Gray
2024
"""

import numpy as np
import tifffile as tf
import torch
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional, List
import warnings

from neurotrack.core.pipeline_config import flexible_image_key_lookup
from neurotrack.data import loading as load
from neurotrack.data import tree
from neurotrack.data.image import Image
from neurotrack.data.seed_io import load_seeds_json
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
                 max_len: int = 10000, max_paths: int = 1000, gamma=0.99, branching: bool = False,
                 repeat_starts: bool = False, start_idx: int = 0,
                 inference_mode: bool = False, seeds_path: Optional[str] = None,
                 auto_seed_selection_mode: Literal["remote_endnode", "root_nodes"] = "remote_endnode",
                 seed_points_by_image: Optional[Dict[str, List[List[float]]]] = None):
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
        seeds_path : Optional[str]
            Optional path to a seed JSON file keyed by relative image path.
        auto_seed_selection_mode : Literal["remote_endnode", "root_nodes"]
            Automatic seed strategy when configured seeds are unavailable.
            - "remote_endnode": pick end node farthest from branch points (current behavior)
            - "root_nodes": use all nodes whose parent id is -1
        seed_points_by_image : Optional[Dict[str, List[List[float]]]]
            Optional in-memory seeds keyed by relative image path.
        section_masking : bool
            Whether to mask out all sections except the current section and its descendants
        """
        self.dataset = dataset
        self.current_patch_idx = start_idx
        self.current_neuron_info = None
        
        # Store initialization parameters
        self.radius = radius
        self.target_step_len = target_step_len
        self.step_width = step_width
        self.max_len = max_len
        self.max_paths = max_paths
        self.gamma = gamma
        self.branching = branching
        self.repeat_starts = repeat_starts
        self.inference_mode = inference_mode
        self.seeds_path = seeds_path
        self.auto_seed_selection_mode = "remote_endnode" if auto_seed_selection_mode == "remote_end_node" else auto_seed_selection_mode
        allowed_seed_modes = {"remote_endnode", "root_nodes"}
        if self.auto_seed_selection_mode not in allowed_seed_modes:
            raise ValueError(
                f"auto_seed_selection_mode must be one of {sorted(allowed_seed_modes)} but got "
                f"'{auto_seed_selection_mode}'."
            )
        if seed_points_by_image is not None:
            self.seed_points_by_image = dict(seed_points_by_image)
        elif seeds_path is not None:
            self.seed_points_by_image = load_seeds_json(seeds_path)
        else:
            self.seed_points_by_image = {}
        
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
        self.target_vector = None
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
            subtree = torch.tensor(neuron_tree_data)  # Already in x,y,z order
            self.adj_dict = load.adjacency_dict(subtree)
            self.full_tree = subtree
            self.full_tree[:, 2:5] = self.full_tree[:, 2:5].flip(dims=(1,))  # Convert to z, y, x order
            self.unvisited_tree = self.full_tree.clone()
            self.id_to_idx = {int(node_id): idx for idx, node_id in enumerate(self.unvisited_tree[:, 0].tolist())}
        else:
            self.neuron_mask = None
            self.adj_dict = {}
            self.full_tree = torch.empty((0, 7), dtype=torch.float32)
            self.unvisited_tree = self.full_tree.clone()
            self.id_to_idx = {}

        # Determine seed point(s)
        seeds = None
        relative_image_path = patch_data.get('relative_image_path', None)
        if relative_image_path is not None:
            _seed_data = flexible_image_key_lookup(self.seed_points_by_image, relative_image_path)
            if _seed_data is not None:
                seeds = torch.as_tensor(_seed_data, dtype=torch.float32)
                if seeds.ndim != 2 or seeds.shape[1] != 3:
                    raise ValueError(
                        f"Seed points for '{relative_image_path}' must have shape (N, 3) in (z, y, x) order."
                    )

        if seeds is None:
            if self.seeds_path is not None:
                warnings.warn(
                    "No configured seeds were found for this patch; falling back to automatic seed selection.",
                    RuntimeWarning,
                )

            if self.has_ground_truth and len(self.adj_dict) > 0:
                seed_mode = self.auto_seed_selection_mode
                if seed_mode == "root_nodes":
                    root_mask = self.full_tree[:, 6] == -1
                    root_coords = self.full_tree[root_mask][:, 2:5]
                    if root_coords.shape[0] > 0:
                        seeds = root_coords
                    else:
                        warnings.warn(
                            "auto_seed_selection_mode='root_nodes' found no root nodes; "
                            "falling back to remote end-node selection.",
                            RuntimeWarning,
                        )
                        seed_mode = "remote_endnode"

                if seeds is None and seed_mode == "remote_endnode":
                    branch_nodes = [k for k, v in self.adj_dict.items() if len(v) > 2]
                    end_nodes = [k for k, v in self.adj_dict.items() if len(v) == 1]
                    branch_indices = [self.id_to_idx[int(n)] for n in branch_nodes]
                    end_indices = [self.id_to_idx[int(n)] for n in end_nodes]
                    branch_coords = self.full_tree[branch_indices, 2:5]
                    end_coords = self.full_tree[end_indices, 2:5]
                    if branch_coords.shape[0] == 0:
                        seeds = end_coords[0].unsqueeze(0)
                    else:
                        compare_coords = branch_coords
                        if branch_coords.shape[0] > 30:
                            indices = torch.randperm(branch_coords.shape[0], device=branch_coords.device)[:30]
                            compare_coords = branch_coords[indices]
                        dists_sq = torch.sum((end_coords.float().unsqueeze(1) - compare_coords.float().unsqueeze(0)) ** 2, dim=2)
                        min_dists_sq, _ = torch.min(dists_sq, dim=1)
                        farthest_end_idx = torch.argmax(min_dists_sq)
                        seeds = end_coords[farthest_end_idx].unsqueeze(0)
            else:
                spatial_shape = torch.tensor(self.img.data.shape[1:], dtype=torch.float32)
                seeds = (spatial_shape / 2.0).unsqueeze(0)
        
        self.paths = [[p] for p in seeds.unbind(0)]
        self._set_roots(list(seeds.unbind(0)))
        self.cut_ends = []
        if self.has_ground_truth:
            # Initialize visited edges
            self.visited = _init_visited(self.full_tree)
            # Initialize first path: Assign section nodes, termination points, target vector, and section_assigned flag
            self._init_path()
        else:
            self.visited = {}
            self.termination_points = torch.empty((0, 3), dtype=torch.float32)
            self.target_vector = torch.zeros((3,), dtype=torch.float32)
            self.section_nodes = None
            self.section_assigned = False

        # Add channel for path visualization
        if self.img.data.shape[0] == 1:
            self.img.data = torch.cat((
                self.img.data, 
                torch.zeros((1,) + self.img.data.shape[1:], dtype=self.img.data.dtype)
            ), dim=0)
        
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
            stall_threshold2 = 0.25 # squared threshold (0.5^2)
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
            'current_target_vector': torch.zeros_like(action),
            'next_target_vector': torch.zeros_like(action),
        }

        current_position = self.paths[0][-1]
        new_position = current_position + action
        status = self._get_status(new_position)

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
        # set default termination points, target vector and section nodes to empty
        self.termination_points = torch.empty((0, 3), dtype=torch.float32, device=self.unvisited_tree.device)
        self.target_vector = torch.zeros((3,), dtype=torch.float32, device=self.unvisited_tree.device)
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
                        connected_nodes, terminals = _get_connected_nodes(int(ce), edge_list=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
                        self.section_nodes.extend(connected_nodes)
                        termination_nodes.extend(terminals)
                    mask = torch.isin(self.full_tree[:, 0], torch.tensor(termination_nodes))
                    self.termination_points = self.full_tree[mask][:, 2:5]
                    self.section_assigned = True
                    target_points = _compute_target_point(self.paths[0][0], self.unvisited_tree, self.target_step_len, edge_list=self.adj_dict,
                                                        id_to_idx=self.id_to_idx, terminal_points=None, valid_nodes=close_cut_ends_nodes.tolist())
                    target_vectors = target_points - self.paths[0][0].unsqueeze(0)
                    # assign target vector to closest target point
                    self.target_vector = target_vectors[torch.argmin((target_vectors**2).sum(dim=1))]
            else: # otherwise use the nearest node
                nearest_node = _get_nearest_node(self.paths[0][0], self.unvisited_tree, id_to_idx=self.id_to_idx)
                nearest_node_coords = self.unvisited_tree[self.id_to_idx[nearest_node], 2:5]
                dists2 = torch.sum((nearest_node_coords - self.paths[0][0]) ** 2)
                if dists2 < 289.0:  # only assign section if within 17 pixels
                    # Initialize section nodes for new path
                    self.section_nodes, terminals = _get_connected_nodes(nearest_node, edge_list=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
                    mask = torch.isin(self.full_tree[:, 0], torch.tensor(terminals))
                    self.termination_points = self.full_tree[mask][:, 2:5]
                    self.section_assigned = True
                    target_points = _compute_target_point(self.paths[0][0], self.unvisited_tree, self.target_step_len, edge_list=self.adj_dict,
                                                        id_to_idx=self.id_to_idx, terminal_points=None, valid_nodes=self.section_nodes)
                    target_vectors = target_points - self.paths[0][0].unsqueeze(0)
                    # assign target vector to closest target point
                    self.target_vector = target_vectors[torch.argmin((target_vectors**2).sum(dim=1))]


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
                    'current_target_vector': None,
                    'next_target_vector': None,}

            direction = action
            current_position = self.paths[0][-1]
            new_position = current_position + direction
            status = self._get_status(new_position)
            info['current_target_vector'] = self.target_vector

            if status in ["out_of_image", "choose_stop"]: # then terminate path
                terminated = True
                # Terminate the branch, but the episode may continue.
                # Reward is negative squared distance to nearest termination point times 1 / (1 - gamma).
                reward = distance_reward(direction, self.target_vector, terminated=True, gamma=self.gamma)
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
                reward = distance_reward(direction, self.target_vector, terminated=False)

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
                                connected_nodes, terminals = _get_connected_nodes(int(node), edge_list=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
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
                        potentially_visited, neuron_end_point = _add_to_visited(start_pos, new_position, self.unvisited_tree, self.visited, edge_list=self.adj_dict,
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
                            valid_keys = self.id_to_idx.keys()
                            self.cut_ends = list(set([ce for ce in self.cut_ends if ce in valid_keys]))

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
                                    connected_nodes, terminals = _get_connected_nodes(int(node), edge_list=self.adj_dict, max_dist=12.0, swc_list=self.unvisited_tree, id_to_idx=self.id_to_idx)
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
                    target_points = _compute_target_point(new_position, self.unvisited_tree, self.target_step_len, edge_list=self.adj_dict,
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
                            info['next_target_vector'] = torch.zeros_like(direction)
                    else: # target points found in window
                        if not self.section_assigned: # if target points found and no section assigned, assign section.
                            self.section_assigned = True
                        target_vectors = target_points - new_position
                        self.target_vector = target_vectors[torch.argmin(((target_vectors - direction)**2).sum(dim=1))]
                        info['next_target_vector'] = self.target_vector

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
        self.roots = torch.empty((0, 3), dtype=torch.float32)
        self._zero_state_patch = None
        self.finished_paths = []
        self.termination_points = None # termination points for the current path
        self.visited = {}
        self.id_to_idx = {}
        self.adj_dict = {}
        self.full_tree = None
        self.unvisited_tree = None
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
                if patch.dtype == torch.uint8:
                    patch = patch.to(dtype=torch.float32) * (1.0 / 255.0)

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
