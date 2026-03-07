#!/usr/bin/env python

"""
Neuron data handling components for SAC tracking

Author: Bryson Gray
2024
"""

import csv
import json
import shutil
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tifffile as tf
import torch
from torch.utils.data import Dataset as TorchDataset, Sampler
from typing import Dict, List, Tuple, Optional, Union, Literal
import warnings
from dataclasses import dataclass
from collections import deque

from neurotrack.data import rendering as draw
from neurotrack.data import loading as load
from neurotrack.data import generation as generate
from neurotrack.data.image import to_uint8


@dataclass
class DrawingComplexityConfig:
    """Configuration for complexity-based drawing parameters."""

    width_correlation_rho_range: Tuple[float, float] = (0.5, 1.0)
    segment_intensity_correlation_rho_range: Tuple[float, float] = (0.9, 1.0)
    # Foreground parameters
    foreground_mean_range: Tuple[float, float] = (0.5, 1.0)
    foreground_std_range: Tuple[float, float] = (0.0, 0.35)
    # Background parameters
    background_mean_range: Tuple[float, float] = (0.0, 0.1)
    background_std_range: Tuple[float, float] = (0.0, 0.04)
    # Spatial noise parameters
    spatial_noise_amplitude_range: Tuple[float, float] = (0.0, 1.0)

    # Vignette magnitude [min, max]
    vignette_magnitude_range: Tuple[float, float] = (0.0, 2.0)
    
    def interpolate_config(self, complexity: float) -> 'draw.DrawingConfig':
        """
        Interpolate drawing configuration based on complexity (0.0 to 1.0).
        Higher complexity = more artifacts and variation.
        """
        complexity = max(0.0, min(1.0, complexity))
        
        def lerp(min_val, max_val, t):
            return min_val + t * (max_val - min_val)
        
        return draw.DrawingConfig(
            width=3.0,
            blur=1.0,
            sharpness=2.0,
            mask_threshold=0.1,
            rgb=False,
            width_correlation=True,
            width_correlation_rho=lerp(*self.width_correlation_rho_range, 1.0 - complexity),
            segment_intensity_correlation=True,
            segment_intensity_correlation_rho=lerp(*self.segment_intensity_correlation_rho_range, 1.0 - complexity),
            foreground_mean=lerp(*self.foreground_mean_range, 1.0 - complexity),  # Less mean for higher complexity
            foreground_std=lerp(*self.foreground_std_range, complexity),
            background_mean=lerp(*self.background_mean_range, complexity),  # More background for complexity
            background_std=lerp(*self.background_std_range, complexity),
            spatial_noise_scale=3.0,  # Fixed scale
            spatial_noise_amplitude=lerp(*self.spatial_noise_amplitude_range, complexity),
            vignette_magnitude=lerp(*self.vignette_magnitude_range, complexity)
        )


class Dataset:
    """
    Dataset class that stores neuron file paths and complexity information.
    
    Complexity is represented as a tuple containing:
    - scalar: degree of contrast artifacts in the image (0.0 to 1.0)
    - category: overall complexity of neuron morphology ("simple", "moderate", "complex")
    """
    
    def __init__(self, data_dir: str = None, rng=None):
        """
        Initialize Dataset from directory containing CSV file.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing a single CSV file with neuron data
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.entries = []
        self.rng = rng or np.random.default_rng()

        if data_dir and os.path.exists(data_dir):
            data_path = Path(data_dir)
            
            # Find CSV files in the directory
            csv_files = list(data_path.glob("*.csv"))
            
            if len(csv_files) == 0:
                # No CSV files found
                raise ValueError(f"No CSV files found in {data_dir}. Directory must contain exactly one CSV file.")
            elif len(csv_files) == 1:
                # Exactly one CSV file - load it as data entries
                csv_file = csv_files[0]
                self.load_from_csv(str(csv_file), data_dir)
            else:
                # Multiple CSV files - error
                csv_names = [f.name for f in csv_files]
                raise ValueError(f"Multiple CSV files found in {data_dir}: {csv_names}. "
                               f"Directory should contain exactly one CSV file.")
        else:
            warnings.warn("No valid data_dir provided. Dataset is empty.")
    
    def load_from_csv(self, csv_path: str, data_dir: str):
        """Load dataset from CSV file."""
        df = pd.read_csv(csv_path)
        required_columns = ['neuron_name', 'complexity', 'morphology']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        data_path = Path(data_dir)
        
        for _, row in df.iterrows():
            neuron_name = str(row['neuron_name'])
            
            # Construct file paths based on neuron_name
            img_path = data_path / f"{neuron_name}_image.tif"
            swc_path = data_path / f"{neuron_name}_subtree.swc"
            reward_mask_path = data_path / f"{neuron_name}_reward_mask.tif"
            
            # Verify files exist
            if not (img_path.exists() and swc_path.exists() and reward_mask_path.exists()):
                print(f"Warning: Missing files for {neuron_name}, skipping entry")
                continue
            
            self.entries.append({
                'neuron_name': neuron_name,
                'img_path': str(img_path),
                'swc_path': str(swc_path),
                'reward_mask_path': str(reward_mask_path),
                'complexity': float(row['complexity']),
                'morphology': str(row['morphology'])
            })
    
    def save_to_csv(self, csv_path: str):
        """Save dataset to CSV file."""
        if not self.entries:
            print("No entries to save")
            return
            
        with open(csv_path, 'w', newline='') as csvfile:
            # Use the standard format with neuron_name
            fieldnames = ['neuron_name', 'img_path', 'swc_path', 'reward_mask_path', 'complexity', 'morphology']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.entries)
    
    def get_complexity_distribution(self) -> Dict:
        """Get distribution of complexity levels."""
        morphology_counts = {}
        complexities = []
        
        for entry in self.entries:
            morph = entry['morphology']
            morphology_counts[morph] = morphology_counts.get(morph, 0) + 1
            complexities.append(entry['complexity'])
        
        return {
            'morphology_distribution': morphology_counts,
            'complexity_stats': {
                'mean': np.mean(complexities),
                'std': np.std(complexities),
                'min': np.min(complexities),
                'max': np.max(complexities)
            }
        }
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.entries[idx]


class DataLoader:
    """
    DataLoader that samples neurons based on their complexity with adjustable sampling priority.
    """
    
    def __init__(self, dataset: Dataset, complexity: float = 0.0, morphology: Literal["simple", "moderate", "complex", "full", "any"] = "any", stochastic_complexity=True, rng=None):
        """
        Initialize DataLoader.
        
        Parameters:
        -----------
        dataset : Dataset
            Dataset to sample from
        complexity : float
            Complexity parameter (0.0 to 1.0) determining sampling priority.
            0.0 = prioritize simple neurons, 1.0 = prioritize complex neurons
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.dataset = dataset
        self.complexity = complexity
        self.morphology = morphology
        self.stochastic_complexity = stochastic_complexity
        self.current_idx = 0
        self.COMPLEXITY_DECAY_RATE = 8.0  # Rate of exponential decay for complexity weighting
        self.rng = rng or np.random.default_rng()
        self._update_sampling_weights()
    
    def _update_sampling_weights(self):
        """Update sampling weights based on current complexity parameter."""
        if len(self.dataset) == 0:
            self.weights = []
            return
            
        weights = []
        for entry in self.dataset.entries:
            # Combine artifact level and morphology complexity
            complexity = entry['complexity']
            
            if self.stochastic_complexity:
                # Weight based on how close the neuron complexity is to target complexity
                weight = np.exp(-self.COMPLEXITY_DECAY_RATE * abs(complexity - self.complexity)) # exponential decay
            else:
                # if self.complexity == complexity:
                if np.isclose(self.complexity, complexity):
                    weight = 1.0
                else:
                    weight = 0.0

            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights] if total_weight > 0 else weights
    
    def set_complexity(self, complexity: float):
        """Update complexity parameter and recalculate weights."""
        self.complexity = max(0.0, min(1.0, complexity))
        self._update_sampling_weights()
    
    def set_morphology(self, morphology: Literal["simple", "moderate", "complex", "full", "any"]):
        """Set morphology filter for sampling."""

        if morphology not in ["simple", "moderate", "complex", "full", "any"]:
            warnings.warn(f"Invalid morphology '{morphology}', setting to 'any'")
            morphology = "any"

        self.morphology = morphology
    
    def sample(self) -> Dict:
        """Sample a neuron based on current complexity weights."""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # filter for morphology if needed
        if self.morphology != "any":
            filtered_indices = [i for i, entry in enumerate(self.dataset.entries) if entry.get('morphology') == self.morphology]
            if not filtered_indices:
                raise ValueError(f"No entries found with morphology '{self.morphology}'")
        else:
            filtered_indices = list(range(len(self.dataset)))
        
        if not self.weights:
            # Fallback to uniform sampling
            idx = self.rng.choice(filtered_indices)
        else:
            # Adjust weights for filtered indices
            filtered_weights = [self.weights[i] for i in filtered_indices]
            total_weight = sum(filtered_weights)
            if total_weight > 0:
                filtered_weights = [w / total_weight for w in filtered_weights]
            else:
                filtered_weights = None  # fallback to uniform if all weights are zero
            idx = self.rng.choice(filtered_indices, p=filtered_weights)
        self.current_idx = idx
        
        return self.dataset[idx]
    
    def __iter__(self):
        """Iterate through dataset in order."""
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Get next item in sequential order."""
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        item = self.dataset[self.current_idx]
        self.current_idx += 1
        return item


class Dataset:
    """
    Dataset class that stores neuron image file paths and SWC data.
    """

    def __init__(self, image_dir, swc_dir):
        """
        Initialize Dataset from directories containing images and SWC files.

        Parameters:
        -----------
        image_dir : str
            Directory containing neuron image files
        swc_dir : str
            Directory containing SWC files
        """
        self.image_dir = Path(image_dir)
        self.swc_dir = Path(swc_dir)

        # Find all SWC files in the directory
        self.swc_files = list(self.swc_dir.rglob("*.swc"))
        if not self.swc_files:
            raise ValueError(f"No SWC files found in directory: {swc_dir}")

        # Map SWC files to corresponding image files
        self.data_entries = []
        for swc_file in self.swc_files:
            neuron_name = swc_file.stem
            img_file = self.image_dir / f"{neuron_name}_image.tif"
            if img_file.exists():
                self.data_entries.append({
                    'neuron_name': neuron_name,
                    'swc_path': str(swc_file),
                    'img_path': str(img_file)
                })
            else:
                print(f"Warning: No matching image file for {swc_file.name}, skipping.")

    def __len__(self) -> int:
        return len(self.data_entries)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data_entries[idx]


class DataGenerator:
    """
    DataGenerator handles generating neuron images from SWC files on-the-fly.
    """
    
    def __init__(self, cache_dir: str = None,
                 complexity_config: Optional[DrawingComplexityConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize DataGenerator.
        
        Parameters:
        -----------
        cache_dir : str, optional
            Directory to cache generated data
        complexity_config : DrawingComplexityConfig, optional
            Configuration for complexity-based drawing parameters
        rng_seed : int, optional
            Random number generator seed
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.complexity_config = complexity_config or DrawingComplexityConfig()
        
        # Set up RNG
        self.rng = rng or np.random.default_rng()

        self.renderer = draw.NeuronRenderer(rng=self.rng)
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_subtrees(self, swc_list: List, target_path_len: int, num_subtrees: int, 
                    mode: str = "no_branch") -> List[List]:
        """
        Extract subtrees from SWC data using the get_subtrees function from the notebook.
        
        Parameters:
        -----------
        swc_list : List
            List of SWC nodes
        num_edges : int
            Number of edges in each subtree
        num_subtrees : int
            Number of subtrees to extract
        mode : str
            Branching mode: "no_branch", "one_branch", "any_branch"
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        adj_dict = load.adjacency_dict(swc_list)
        
        # Identify branch points and endpoints
        branch_points = [node for node, neighbors in adj_dict.items() if len(neighbors) > 2]
        end_points = [node for node, neighbors in adj_dict.items() if len(neighbors) == 1]
        critical_points = branch_points + end_points
        
        subtrees = []
        swc_list = np.array(swc_list)
        
        if mode == "no_branch":
            for start_node in self.rng.permutation(end_points):
                attempts = 0
                max_attempts = 100  # Prevent infinite loops
                while len(subtrees) < num_subtrees and attempts < max_attempts:
                    attempts += 1
                    current_node = start_node
                    path = [current_node]
                    subtree = [swc_list[swc_list[:,0] == current_node][0].tolist()]
                    path_len = 0.0
                    while path_len < target_path_len:
                        neighbors = adj_dict[current_node]
                        next_nodes = [n for n in neighbors if n not in path]
                        
                        if not next_nodes:
                            break  # Dead end
                            
                        next_node = self.rng.choice(next_nodes)
                        path.append(next_node)
                        current_node = next_node
                        subtree.append(swc_list[swc_list[:,0] == current_node][0].tolist())
                        path_len += np.linalg.norm(np.array(subtree[-1][2:5]) - np.array(subtree[-2][2:5]))

                    if path_len >= target_path_len:
                        subtrees.append(subtree)
                        break
        
        elif mode == "one_branch":
            for start_node in self.rng.permutation(branch_points):
                attempts = 0
                max_attempts = 100
                
                while len(subtrees) < num_subtrees and attempts < max_attempts:
                    attempts += 1
                    
                    # Get all neighbors of the branch point
                    neighbors = edge_list[start_node]
                    
                    # Initialize the path with the branch point
                    path = [start_node]
                    subtree = [swc_list[swc_list[:,0] == start_node][0].tolist()]
                    total_len = 0.0

                    # # Calculate edges per branch - divide equally among neighbors
                    # edges_per_branch = max(1, num_edges // len(neighbors))
                    # remaining_nodes = num_edges
                    len_per_branch = max(1, target_path_len // len(neighbors))
                    remaining_len = target_path_len
                    
                    # Walk along each neighbor's path
                    for neighbor in neighbors:
                        if neighbor in path:
                            continue
                            
                        # Start walking from this neighbor
                        current_node = neighbor
                        path.append(current_node)
                        subtree.append(swc_list[swc_list[:,0] == current_node][0].tolist())
                        branch_len = np.linalg.norm(np.array(subtree[-1][2:5]) - np.array(subtree[0][2:5]))
                        total_len += branch_len
                        remaining_len = target_path_len - total_len
                        # remaining_nodes -= 1
                        # Continue walking this branch until we hit the limit or an endpoint
                        # while len(branch_path) < edges_per_branch and remaining_nodes > 0:
                        while branch_len < len_per_branch and remaining_len > 0:
                            next_neighbors = [n for n in edge_list[current_node] if n not in path]
                            
                            # Stop if we reach an endpoint or have no more neighbors
                            if not next_neighbors:
                                break
                                
                            # Move to next node
                            current_node = self.rng.choice(next_neighbors)
                            path.append(current_node)
                            subtree.append(swc_list[swc_list[:,0] == current_node][0].tolist())
                            step_len = np.linalg.norm(np.array(subtree[-1][2:5]) - np.array(subtree[-2][2:5]))
                            total_len += step_len
                            branch_len += step_len
                            remaining_len = target_path_len - total_len
                            # remaining_nodes -= 1
                    
                    # Add subtree if we've collected enough nodes
                    if total_len >= target_path_len * 0.7:  # Ensure we have at least some minimum path
                        subtrees.append(subtree)
                        break
        
        elif mode == "any_branch":
            for start_node in self.rng.permutation(critical_points):
                attempts = 0
                max_attempts = 100
                
                while len(subtrees) < num_subtrees and attempts < max_attempts:
                    attempts += 1
                    current_nodes = [start_node]
                    path = [start_node]
                    subtree = [swc_list[swc_list[:,0] == start_node][0].tolist()]
                    path_len = 0.0
                    
                    while path_len < target_path_len:
                        # Get all neighbors for current active nodes
                        all_next_nodes = []
                        for current_node in current_nodes:
                            neighbors = edge_list[current_node]
                            next_nodes = [n for n in neighbors if n not in path]
                            all_next_nodes.extend(next_nodes)
                        
                        # Remove duplicates and nodes already in path
                        all_next_nodes = list(set(all_next_nodes) - set(path))
                        
                        if not all_next_nodes:
                            break  # No more nodes to explore
                        
                        # Randomly select next node to add
                        next_node = self.rng.choice(all_next_nodes)
                        path.append(next_node)
                        subtree.append(swc_list[swc_list[:,0] == next_node][0].tolist())
                        path_len += np.linalg.norm(np.array(subtree[-1][2:5]) - np.array(subtree[-2][2:5]))

                        # Update current_nodes - keep nodes with unexplored neighbors
                        new_current_nodes = []
                        for current_node in current_nodes:
                            neighbors = edge_list[current_node]
                            available_neighbors = [n for n in neighbors if n not in path]
                            # Keep node active if it has unexplored neighbors and isn't an endpoint
                            if available_neighbors and current_node not in end_points:
                                new_current_nodes.append(current_node)
                        
                        # Add the new node as an active node
                        new_current_nodes.append(next_node)
                        current_nodes = new_current_nodes
                    
                    # Accept any subtree that reaches the target length
                    if path_len >= target_path_len:
                        subtrees.append(subtree)
                        break
        
        return subtrees
    

    def crop_around_subtree(self, volume: torch.Tensor, subtree: List, 
                           padding: int = 10) -> Tuple[torch.Tensor, List]:
        """
        Crop image around subtree coordinates and shift subtree coordinates.
        """
        # Get coordinates from subtree
        subtree_array = np.array(subtree)
        coords = subtree_array[:, 2:5]  # x, y, z columns
        
        # Calculate bounding box with padding
        min_coords = np.min(coords, axis=0) - padding
        max_coords = np.max(coords, axis=0) + padding
        
        # Convert to integers and ensure within bounds
        volume_shape = volume.shape
        min_coords = np.maximum(0, min_coords.astype(int))
        max_coords = np.minimum(volume_shape[1:][::-1], max_coords.astype(int))
        
        # Crop the image
        cropped_image = volume[:, 
                              min_coords[2]:max_coords[2], 
                              min_coords[1]:max_coords[1], 
                              min_coords[0]:max_coords[0]]
        
        # Shift subtree coordinates
        shifted_subtree = []
        for node in subtree:
            shifted_node = (
                node[0],  # node_id
                node[1],  # node_type  
                node[2] - min_coords[0],  # x
                node[3] - min_coords[1],  # y
                node[4] - min_coords[2],  # z
                node[5],  # radius
                node[6]   # parent_id
            )
            shifted_subtree.append(shifted_node)
        
        return cropped_image, shifted_subtree
    
    def generate_reward_mask(self, subtree: List, shape: Tuple[int, ...], width=35.0) -> torch.Tensor:
        """Generate reward mask from subtree."""
        # Parse subtree into sections format
        sections, _ = load.parse_swc(subtree)
        
        # Create reward mask (same as true density)
        # reward_mask = self.renderer.draw_density(sections, shape, width=width)
        reward_mask = self.renderer.draw_density(sections, shape, width=width, mask=True)
        return reward_mask.data
    
    def simulate_neuron_image(self, subtree: List, shape: Tuple[int, ...], 
                             config: draw.DrawingConfig) -> torch.Tensor:
        """
        Simulate neuron image from subtree using draw.neuron.
        
        Parameters:
        -----------
        subtree : List
            Subtree SWC data
        shape : Tuple[int, ...]
            Output image shape
        config : DrawingConfig
            Drawing configuration
        """

        # Parse subtree into sections format
        sections, _ = load.parse_swc(subtree)
        
        # Generate simulated neuron image
        img = self.renderer.draw_neuron(sections, shape, config)
        return img.data
    
    def load_files(self, swc_path: str, img_path: Optional[str] = None) -> Tuple[List, Optional[torch.Tensor]]:
        """Load SWC file and optional image file."""
        # Load SWC
        swc_list = load.swc(swc_path)
        
        # Load image if provided
        img_tensor = None
        if img_path and os.path.exists(img_path):
            img_array = tf.imread(img_path)
            if img_array.ndim == 3:
                img_array = img_array[None]  # Add channel dimension
            # img_tensor = torch.from_numpy(img_array.astype(np.float32)) / 255.0 #TODO: Keep original dtype?
            img_tensor = torch.from_numpy(img_array)
        
        return swc_list, img_tensor
    
    def generate_data(self, subtrees_per_swc: int = 1, 
                     complexity_range: Tuple[float, float] = (0.0, 1.0),
                     n_steps: int = 6,
                     morphology: Literal["simple", "moderate", "complex", "full", "any"] = "any",
                     swc_dir: Optional[str] = None, img_dir: Optional[str] = None,
                     dataset_size: int = 100, output_dir: Optional[str] = None) -> Dict:
        """
        Generate processed data from SWC files or synthetic data and save to output directory.
        
        Parameters:
        -----------
        subtrees_per_swc : int
            Number of subtrees to generate per SWC file (or per synthetic neuron)
        complexity_range : Tuple[float, float]
            Range of complexity values to sample from (min, max)
        n_steps : int
            Number of steps in which to divide complexity range for sampling.
        morphology : Literal["simple", "moderate", "complex", "full", "any"]
            Morphology complexity level to filter neurons. "any" means no filtering.
        swc_dir : str, optional
            Directory containing SWC files to process. If None, synthetic data will be generated.
        img_dir : str, optional
            Directory containing corresponding TIFF images for SWC files
        dataset_size : int
            Number of synthetic neurons to generate (only used if swc_dir is None)
        output_dir : str, optional
            Output directory to save generated data. Uses cache_dir by default.
        
        Returns:
        --------
        Dict containing:
        - processed_files: Number of files/neurons processed
        - total_subtrees: Total number of subtrees generated
        - output_dir: Path to output directory
        """
        if output_dir is None:
            output_dir_path = self.cache_dir
        else:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

        # check for existing csv in output dir
        existing_csv = list(output_dir_path.glob("*.csv"))
        existing_neuron_names = set()
        if existing_csv:
            # get neuron names from existing csv
            existing_df = pd.read_csv(existing_csv[0])
            existing_neuron_names = set(existing_df['neuron_name'].tolist())
        
        # Determine data source: files from directory or synthetic generation
        if swc_dir is not None:
            # Use SWC files from directory
            swc_dir_path = Path(swc_dir)
            # Find all SWC files in the directory
            swc_files = list(swc_dir_path.rglob("*.swc"))
            if not swc_files:
                raise ValueError(f"No SWC files found in directory: {swc_dir}")
            
            # If img_dir provided, check for matching TIFF files
            if img_dir:
                img_dir_path = Path(img_dir)
                if not img_dir_path.exists():
                    raise ValueError(f"Image directory not found: {img_dir}")
                
                # Filter SWC files to only those with matching TIFF images
                matched_swc_files = []
                for swc_file in swc_files:
                    # Look for corresponding TIFF file
                    tiff_patterns = [
                        swc_file.stem + "_image.tif",
                        swc_file.stem + ".tif",
                        swc_file.stem + "_image.tiff",
                        swc_file.stem + ".tiff"
                    ]
                    
                    tiff_found = False
                    for pattern in tiff_patterns:
                        tiff_path = img_dir_path / pattern
                        if tiff_path.exists():
                            matched_swc_files.append((swc_file, str(tiff_path)))
                            tiff_found = True
                            break
                    
                    if not tiff_found:
                        print(f"No matching TIFF image found for {swc_file.name}")
                
                if not matched_swc_files:
                    raise ValueError("No SWC files have matching TIFF images in the image directory")
                
                swc_files = matched_swc_files
            
            data_items = swc_files
            use_synthetic = False
        else:
            # Generate synthetic data
            data_items = [f"simulated_img{i}" for i in range(dataset_size)]
            use_synthetic = True
        
        processed_entries = []
        total_subtrees = 0
        
        print(f"Processing {len(data_items)} {'SWC files' if not use_synthetic else 'synthetic neurons'}...")
        
        complexity_values = np.linspace(complexity_range[0], complexity_range[1], n_steps)
        for i, item in enumerate(data_items):
            if not use_synthetic and img_dir:
                swc_file, img_path = item
                item_name = swc_file.stem
            elif not use_synthetic:
                swc_file = item
                img_path = None
                item_name = swc_file.stem
            else:
                # Synthetic data
                item_name = item
                swc_file = None
                img_path = None
            
            try:
                # Sample complexity from range
                # complexity = self.rng.uniform(*complexity_range)
                # complexity = complexity_values[i // (len(data_items) // n_steps)]
                # config = self.complexity_config.interpolate_config(complexity)
                
                # Determine subtree extraction mode and set complexity parameters
                # morphology = self._complexity_to_category(complexity)
                if morphology == "any":
                    morphology_ = self.rng.choice(["simple", "moderate", "complex"])
                else:
                    morphology_ = morphology
                subtree_mode = {"simple": "no_branch",
                                "moderate": "one_branch",
                                "complex": "any_branch",
                                "full": "any_branch"}[morphology_]
                # target_path_len = 50.0 + {"no_branch": 100.0,
                #                            "one_branch": 500.0,
                #                            "any_branch": 5000.0}[subtree_mode] * complexity
                target_path_len = {"no_branch": 100.0,
                                    "one_branch": 500.0,
                                    "any_branch": 5000.0}[subtree_mode]
                
                if use_synthetic:
                    # Generate synthetic SWC data
                    if subtree_mode == "any_branch":
                        num_branches = self.rng.choice([2, 3, 4])
                    else:
                        num_branches = 1 if subtree_mode == "one_branch" else 0
                    
                    # Estimate reasonable shape based on path length
                    estimated_shape = (300, 300, 300)  # Default shape for synthetic data. This will be cropped later.
                    kappa = 1000.0 if complexity < 0.1 else 20.0
                    swc_list = generate.make_swc_list(
                        size=estimated_shape,
                        length=target_path_len,
                        step_size=1.0,
                        kappa=kappa,
                        num_branches=num_branches,
                        rng=self.rng
                    )
                    img_tensor = None
                    subtrees = [swc_list]  # Always 1 subtree per synthetic SWC
                else:
                    # Load files from disk
                    swc_list, img_tensor = self.load_files(str(swc_file), img_path)
                
                    if morphology_ == "full":
                        subtrees = [swc_list]  # Use full neuron
                        padding = 10  # Padding only for full neurons
                    else:
                        # Extract subtrees from loaded SWC
                        if img_tensor is None:
                            subtrees = self.get_subtrees(swc_list, target_path_len, subtrees_per_swc, mode=subtree_mode)
                        else:
                            subtrees = []
                            morphologies = []
                            box_size = 200.0
                            swc_array = np.array(swc_list)
                            centers = swc_array[np.random.choice(np.arange(len(swc_array)), size=subtrees_per_swc, replace=False)]
                            for center in centers:
                                center_point = center[2:5]
                                in_box_mask = np.all(
                                    (swc_array[:, 2:5] >= (center_point - box_size/2)) &
                                    (swc_array[:, 2:5] <= (center_point + box_size/2)),
                                    axis=1
                                )
                                subtree = swc_array[in_box_mask].tolist()
                                subtree_adj_dict = load.adjacency_dict(subtree)
                                # only keep tree connected to the center node
                                center_node = center[0]
                                visited = set()
                                to_visit = [center_node]
                                while to_visit:
                                    node = to_visit.pop()
                                    if node not in visited:
                                        visited.add(node)
                                        neighbors = subtree_adj_dict.get(node, [])
                                        to_visit.extend(neighbors)
                                subtree = [node for node in subtree if node[0] in visited]
                                # get number of branches
                                num_branches = sum(1 for neighbors in subtree_adj_dict.values() if len(neighbors) > 2)
                                morphology_ = "simple" if num_branches == 0 else "moderate" if num_branches == 1 else "complex"
                                subtrees.append(subtree)
                                morphologies.append(morphology_)
                                padding = 0  # No padding for subtrees extracted from image
                
                # Process and save each subtree individually to avoid memory issues
                for i, subtree in enumerate(subtrees):
                    # Generate data for this single subtree
                    if img_tensor is not None:

                        # Create unique prefix for this file and subtree
                        file_prefix = f"{item_name}_subtree_{i:02d}"
                        if file_prefix in existing_neuron_names:
                            # try incrementing i until unique
                            j = 1
                            while f"{item_name}_subtree_{i+j:02d}" in existing_neuron_names:
                                j += 1
                            file_prefix = f"{item_name}_subtree_{i+j:02d}"
                            print(f"Adjusted file prefix to avoid duplicate: {file_prefix}")
                        existing_neuron_names.add(file_prefix)

                        # Use real image - crop around subtree
                        cropped_image, shifted_subtree = self.crop_around_subtree(img_tensor, subtree, padding=padding)
                        if padding == 0:
                            # pad after cropping
                            # pad image with random noise based on image stats
                            pad = 10
                            img_mean = float(torch.mean(cropped_image.float()))
                            img_std = float(torch.std(cropped_image.float()))
                            if img_std == 0.0:
                                img_std = 1e-6

                            device = cropped_image.device
                            dtype = cropped_image.dtype

                            # Compute padded shape (assumes cropped_image shape is [C, Z, Y, X])
                            padded_shape = list(cropped_image.shape)
                            for dim in range(1, len(padded_shape)):
                                padded_shape[dim] += 2 * pad
                            padded_shape = tuple(padded_shape)

                            # Create noise in float on the correct device, clamp and cast once
                            noise = torch.normal(mean=img_mean, std=img_std, size=padded_shape, device=device)
                            if dtype == torch.uint8:
                                noise = noise.clamp(0, 255).to(dtype)
                            else:
                                noise = noise.clamp(0.0, 1.0).to(dtype)

                            # Copy original image into the center of the noisy padded tensor (in-place)
                            z0, y0, x0 = pad, pad, pad
                            z1 = z0 + cropped_image.shape[1]
                            y1 = y0 + cropped_image.shape[2]
                            x1 = x0 + cropped_image.shape[3]
                            noise[:, z0:z1, y0:y1, x0:x1] = cropped_image
                            cropped_image = noise

                            # Shift subtree coordinates accordingly
                            shifted_subtree = np.array(shifted_subtree)
                            shifted_subtree[:, 2:5] = shifted_subtree[:, 2:5] + pad
                            shifted_subtree = shifted_subtree.tolist()

                        reward_mask = self.generate_reward_mask(shifted_subtree, cropped_image.shape[1:], width=35.0) # TODO: mask width from config
                        
                        result = {
                            'image': cropped_image,
                            'subtree': shifted_subtree,
                            'reward_mask': reward_mask
                        }
                        # Since real images have natural artifacts, the complexity only reflects morphology
                        # Adjust complexity to reflect this.
                        complexity = 1.0

                        # Save immediately to free memory
                        self.save_data(result, str(output_dir_path), prefix=file_prefix)
                        total_subtrees += 1

                        # Track entry for CSV - one row per subtree
                        if "morphologies" in locals():
                            morphology_ = morphologies[i]
                        processed_entries.append({
                            'neuron_name': file_prefix,
                            'complexity': complexity,
                            'morphology': morphology_
                        })
                    else:
                        # Simulate image from subtree
                        # Estimate shape from subtree coordinates
                        subtree = np.array(subtree)
                        coords = subtree[:, 2:5]                
                        shape_estimate = tuple(int(x) + 20 for x in np.ptp(coords, axis=0))[::-1] # Reverse for z,y,x

                        # shift coords
                        subtree[:, 2:5] = subtree[:, 2:5] - coords.min(axis=0) + 10.0
                        subtree = subtree.tolist()
                        for complexity in complexity_values:

                            # Create unique prefix for this file and subtree
                            file_prefix = f"{item_name}_subtree_{i:02d}"
                            if file_prefix in existing_neuron_names:
                                # try incrementing i until unique
                                j = 1
                                while f"{item_name}_subtree_{i+j:02d}" in existing_neuron_names:
                                    j += 1
                                file_prefix = f"{item_name}_subtree_{i+j:02d}"
                                print(f"Adjusted file prefix to avoid duplicate: {file_prefix}")
                            existing_neuron_names.add(file_prefix)

                            config = self.complexity_config.interpolate_config(complexity)
                            simulated_image = self.simulate_neuron_image(subtree, shape_estimate, config=config)
                            # reward_mask = self.generate_reward_mask(subtree, shape_estimate, width=float(config.width))
                            reward_mask = self.generate_reward_mask(subtree, shape_estimate, width=35.0)
                            
                            result = {
                                'image': simulated_image,
                                'subtree': subtree,
                                'reward_mask': reward_mask
                            }
                    
                            # Save immediately to free memory
                            self.save_data(result, str(output_dir_path), prefix=file_prefix)
                            total_subtrees += 1
                    
                            # Track entry for CSV - one row per subtree
                            if "morphologies" in locals():
                                morphology_ = morphologies[i]
                            processed_entries.append({
                                'neuron_name': file_prefix,
                                'complexity': complexity,
                                'morphology': morphology_
                            })
                    
                    # Clear result to free memory
                    del result
                
                print(f"Processed {item_name}: {len(subtrees)} subtrees generated")
                
            except Exception as e:
                print(f"Failed to process {item_name}: {str(e)}")
        
        # Save entries to CSV
        if processed_entries:
            # check for existing csv
            if existing_csv:
                # append to existing csv
                df_new = pd.DataFrame(processed_entries)
                df_combined = pd.concat([existing_df, df_new], ignore_index=True)
                csv_path = output_dir_path / existing_csv[0].name
                df_combined.to_csv(csv_path, index=False)
                print(f"Appended entry data to existing CSV: {csv_path}")
            else:
                # create new csv
                csv_path = output_dir_path / "generated_data_entries.csv"
                df = pd.DataFrame(processed_entries)
                df.to_csv(csv_path, index=False)
                print(f"Entry data saved to: {csv_path}")

        print(f"\nProcessing complete!")
        print(f"Total subtrees generated: {total_subtrees}")
        print(f"Results saved to: {output_dir_path}")
        
        return {
            'processed_files': len(data_items),
            'total_subtrees': total_subtrees,
            'output_dir': str(output_dir_path)
        }

    def _complexity_to_category(self, complexity: float) -> str:
        """Convert numeric complexity to category string."""
        if complexity < 0.33:
            return "simple"
        elif complexity < 0.67:
            return "moderate"
        elif complexity < 0.9:
            return "complex"
        else:
            return "full"
    
    # def _complexity_to_mode(self, complexity: float) -> str:
    #     """Convert numeric complexity to subtree extraction mode."""
    #     if complexity < 0.33:
    #         return "no_branch"
    #     elif complexity < 0.67:
    #         return "one_branch"
    #     else:
    #         return "any_branch"
    
    def save_data(self, data: Dict, save_dir: str, prefix: str = "neuron_data"):
        """Save only image, subtree, and reward mask to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save image
        img_path = save_path / f"{prefix}_image.tif"
        tf.imwrite(str(img_path), data['image'].numpy())
        
        # Save reward mask
        mask_path = save_path / f"{prefix}_reward_mask.tif"
        tf.imwrite(str(mask_path), data['reward_mask'].numpy())
        
        # Save subtree as SWC
        swc_path = save_path / f"{prefix}_subtree.swc"
        with open(swc_path, 'w') as f:
            for node in data['subtree']:
                f.write(f"{node[0]} {node[1]} {node[2]:.3f} {node[3]:.3f} {node[4]:.3f} {node[5]:.3f} {node[6]}\n")
    
    def empty_cache(self):
        """Empty the cache directory."""
        if self.cache_dir and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Cache directory {self.cache_dir} emptied.")
        else:
            print("No cache directory to empty.")


# Convenience function to create data components
def create_neuron_data_components(data_dir: str, complexity: float = 0.0,
                                 complexity_config: Optional[DrawingComplexityConfig] = None,
                                 cache_dir: str = None, 
                                 rng: Optional[np.random.Generator] = None) -> Tuple[Dataset, DataLoader, DataGenerator]:
    """
    Create neuron data handling components.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing neuron data with a single CSV file
    complexity : float
        Initial complexity parameter
    complexity_config : DrawingComplexityConfig, optional
        Configuration for complexity-based drawing parameters
    cache_dir : str, optional
        Directory for caching generated data
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns:
    --------
    Tuple[Dataset, DataLoader, DataGenerator]
        Configured data components
    """
    # Create dataset
    dataset = Dataset(data_dir=data_dir, rng=rng)
    
    # Create dataloader
    dataloader = DataLoader(dataset=dataset, complexity=complexity, rng=rng)
    
    # Create data generator
    data_generator = DataGenerator(
        cache_dir=cache_dir,
        complexity_config=complexity_config,
        rng=rng
    )
    
    return dataset, dataloader, data_generator


if __name__ == "__main__":
    # Example usage
    swc_dir = "/home/brysongray/data/neurotrack_data/gold166/gold166_swc_processed_subset"
    output_dir = "/home/brysongray/neurotrack/generated_data_output"
    img_dir = None  # Set to path containing TIFF images if available
    
    # Create data generator
    data_generator = DataGenerator(
        cache_dir=None,  # No caching for this example
        complexity_config=DrawingComplexityConfig(),
        rng=np.random.default_rng(42)  # Fixed seed for reproducibility
    )
    
    print("DataGenerator created successfully!")
    
    # Example batch data generation from directory
    if os.path.exists(swc_dir):
        print(f"Processing SWC files from: {swc_dir}")
        results = data_generator.generate_data(
            output_dir=output_dir,
            subtrees_per_swc=2,  # Generate 2 subtrees per SWC file
            complexity_range=(0.2, 0.8),  # Medium complexity range
            swc_dir=swc_dir,
            img_dir=img_dir  # Optional image directory
        )
        
        print(f"Processing results:")
        print(f"  - Files processed: {results['processed_files']}")
        print(f"  - Total subtrees: {results['total_subtrees']}")
        print(f"  - Output directory: {results['output_dir']}")
        
        # Example of loading the generated dataset
        print(f"\n--- Loading generated dataset ---")
        dataset = Dataset(data_dir=output_dir)
        dataloader = DataLoader(dataset=dataset, complexity=0.5)
        
        print(f"Generated dataset size: {len(dataset)}")
        print(f"Complexity distribution: {dataset.get_complexity_distribution()}")
        
        # Sample from the generated dataset
        if len(dataset) > 0:
            sample_entry = dataloader.sample()
            print(f"Sample generated entry: {sample_entry}")
    else:
        print(f"SWC directory not found: {swc_dir}")
        print("Generating synthetic data instead...")
        
        # Example synthetic data generation
        results = data_generator.generate_data(
            output_dir=output_dir,
            subtrees_per_swc=1,  # Generate 1 subtree per synthetic neuron
            complexity_range=(0.2, 0.8),  # Medium complexity range
            swc_dir=None,  # No SWC directory - use synthetic data
            dataset_size=50  # Generate 50 synthetic neurons
        )
        
    # Example of creating dataset from existing generated data
    print(f"\n--- Example: Loading existing generated dataset ---")
    test_data_dir = "/home/brysongray/neurotrack/data_dir_test"
    if os.path.exists(test_data_dir):
        try:
            dataset = Dataset(data_dir=test_data_dir)
            dataloader = DataLoader(dataset=dataset, complexity=0.5)
            
            print(f"Test dataset size: {len(dataset)}")
            if len(dataset) > 0:
                print(f"Complexity distribution: {dataset.get_complexity_distribution()}")
                sample_entry = dataloader.sample()
                print(f"Sample entry: {sample_entry}")
        except Exception as e:
            print(f"Failed to load test dataset: {e}")
    else:
        print(f"Test data directory not found: {test_data_dir}")



