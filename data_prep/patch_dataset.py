from collections import deque
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler
import tifffile as tf
from data_prep import load, draw


class NeuronPatchDataset(TorchDataset):
    """
    PyTorch Dataset for efficiently loading cropped patches from neuron images.
    
    Implements a lazy loading strategy with caching to minimize memory usage:
    - Loads one full TIFF image at a time
    - Extracts patches on-demand using deterministic random seeds
    - Caches the most recently loaded image to avoid redundant I/O
    - Each idx deterministically maps to the same patch (reproducible)
    
    The key insight: idx // patches_per_image determines which image to load,
    and idx % patches_per_image determines which patch from that image.
    A deterministic RNG seeded with idx ensures reproducibility.
    """
    
    def __init__(
        self,
        swc_dir: List[str],
        img_dir: List[str],
        crop_size: int = 128,
        patches_per_image: int = 10,
        alpha: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        crop_patches: bool = True
    ):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        swc_dir : str
            Directory containing SWC files
        img_dir : str
            Directory containing TIFF image files
        crop_size : int
            Size of cropped patches (assumed cubic)
        patches_per_image : int
            Number of patches to extract from each image before moving to next
        alpha : float
            opacity for rendering neuron masks
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        crop_patches : bool
            If True, extract random patches. If False, return full images.
        """
        swc_dir = Path(swc_dir)
        self.swc_files = list(swc_dir.glob("*.swc"))
        img_dir = Path(img_dir)
        self.img_files_unordered = list(img_dir.glob("*.tif"))
        if len(self.swc_files) != len(self.img_files_unordered):
            raise ValueError(f"Number of SWC files ({len(self.swc_files)}) must match TIFF files ({len(self.img_files_unordered)})")
        # Order files so names match
        self.img_files = []
        for swc_file in self.swc_files:
            swc_name = swc_file.stem
            matching_imgs = [f for f in self.img_files_unordered if f.stem == swc_name or f.stem.startswith(swc_name + "_")]
            if not matching_imgs:
                raise ValueError(f"No matching TIFF files found for SWC file {swc_file}")
            if len(matching_imgs) > 1:
                raise ValueError(f"Multiple matching TIFF files found for SWC file {swc_file}: {matching_imgs}")
            self.img_files.append(matching_imgs[0])

        self.crop_size = crop_size
        self.patches_per_image = patches_per_image
        self.rng = rng or np.random.default_rng(0)
        self.alpha = alpha
        self.crop_patches = crop_patches
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        
        # Cache for currently loaded image
        self._cached_image_idx: Optional[int] = None
        self._cached_image: Optional[torch.Tensor] = None
        self._cached_swc_data: Optional[Dict] = None
        
        # Total dataset size = number of images * patches per image (or just images if not cropping)
        if self.crop_patches:
            self.total_size = len(self.img_files) * self.patches_per_image
        else:
            self.total_size = len(self.img_files)

        # Base seed for reproducibility
        self._base_seed = self.rng.integers(0, 2**31)
        self.renderer = draw.NeuronRenderer(rng=self.rng)
        
    def _load_image(self, idx: int) -> torch.Tensor:
        """Load a full image."""
        img_path = self.img_files[idx]
        
        img = tf.imread(img_path)

        # normalize to [0, 1]
        img_max = img.max()
        img_min = img.min()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
            img = img.astype(np.float32)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        # if 'float' not in str(img.dtype):
        #     img = img.astype(np.float32) / np.iinfo(img.dtype).max
        
        img = torch.from_numpy(img)
        
        return img
    
    def _load_swc(self, idx: int) -> Dict:
        """Load SWC neuron data."""
        swc_path = self.swc_files[idx]
        
        swc_data = load.swc(swc_path, rotate=False, verbose=False)
        
        return swc_data
    
    def _extract_random_subtree(self, swc_data: List, patch_rng: np.random.Generator) -> List:
        """Extract a random subtree using the provided RNG for reproducibility."""
        swc_array = np.array(swc_data)
        center = swc_array[patch_rng.integers(len(swc_array))]
        center_point = center[2:5]
        in_box_mask = np.all(
            (swc_array[:, 2:5] >= (center_point - self.crop_size/2)) &
            (swc_array[:, 2:5] <= (center_point + self.crop_size/2)),
            axis=1
        )
        subtree = swc_array[in_box_mask].tolist()
        subtree_edge_list = load.undirected_edge_list(subtree)
        # only keep tree connected to the center node
        center_node = center[0]
        visited = set()
        to_visit = [center_node]
        while to_visit:
            node = to_visit.pop()
            if node not in visited:
                visited.add(node)
                neighbors = subtree_edge_list.get(node, [])
                to_visit.extend(neighbors)
        subtree = [node for node in subtree if node[0] in visited]

        return subtree
    
    def _extract_random_patch(self, image: torch.Tensor, swc_data: List, 
                              patch_rng: np.random.Generator, pad: int = 10) -> Dict[str, torch.Tensor]:
        """
        Extract a random cropped patch from the given image using provided RNG.
        
        Parameters:
        -----------
        image : torch.Tensor
            The full image to extract from
        swc_data : List
            SWC neuron data
        patch_rng : np.random.Generator
            Random number generator for this specific patch
        
        Returns:
        --------
        Dict containing:
            - 'image': Cropped image patch
            - 'neuron_tree': Associated subtree data
            - 'neuron_mask': Neuron area mask
        """
        # extract subtrees from one example
        subtree = self._extract_random_subtree(swc_data, patch_rng)

        cropped_img, cropped_subtree = crop_around_subtree(image, subtree, padding=1) # pad=1 to ensure cropped img never has zero size in any dim
        # shifted_subtree = np.array(subtree)
        # shifted_subtree[:, 2:5] = shifted_subtree[:, 2:5] + pad
        padded_img, padded_subtree = pad_with_noise(cropped_img, cropped_subtree)

        # create mask
        sections, _ = load.parse_swc(padded_subtree, verbose=False, transpose=True)
        neuron_mask = self.renderer.draw_density(sections, padded_img.shape[-3:], width=3.0, mask=True)
        neuron_area_mask = self.renderer.draw_density(sections, padded_img.shape[-3:], width=35.0, mask=True)
        
        composite_img = self.alpha * padded_img + (1 - self.alpha) * neuron_mask.data

        
        return {
            'image': composite_img,
            'neuron_tree': padded_subtree,
            'neuron_mask': neuron_area_mask.data
        }
    
    def __len__(self) -> int:
        """Return total number of patches (or images if not cropping) in the dataset."""
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patch (or full image) from the dataset.
        
        If crop_patches is True:
            Uses idx to deterministically select which image and patch to extract.
            The same idx will always return the same patch.
        
        If crop_patches is False:
            Returns the full image and SWC data at the given index.
        
        Parameters:
        -----------
        idx : int
            Index of the patch/image to retrieve
            
        Returns:
        --------
        Dict containing image (patch or full), neuron tree, and neuron mask
        """
        
        if self.crop_patches:
            # Original patch extraction logic
            image_idx = (idx // self.patches_per_image) % len(self.img_files)
            
            # Create a deterministic RNG for this specific patch
            patch_seed = self._base_seed + idx
            patch_rng = np.random.default_rng(patch_seed)
            
            # Load image if not cached or if different image
            if self._cached_image_idx != image_idx:
                self._cached_image = self._load_image(image_idx)
                self._cached_swc_data = self._load_swc(image_idx)
                self._cached_image_idx = image_idx
            
            # Extract patch using the deterministic RNG
            patch = self._extract_random_patch(
                self._cached_image, 
                self._cached_swc_data,
                patch_rng
            )
            
            # Add metadata
            patch['neuron_name'] = self.swc_files[image_idx].stem
            patch['image_idx'] = image_idx
            patch['global_idx'] = idx
            
            return patch
        else:
            # Return full image without cropping
            image_idx = idx % len(self.img_files)
            
            # Load image if not cached or if different image
            if self._cached_image_idx != image_idx:
                self._cached_image = self._load_image(image_idx)
                self._cached_swc_data = self._load_swc(image_idx)
                self._cached_image_idx = image_idx
            
            # Create full neuron mask and composite
            sections, _ = load.parse_swc(self._cached_swc_data, verbose=False, transpose=True)
            neuron_mask = self.renderer.draw_density(sections, self._cached_image.shape[-3:], width=3.0, mask=True)
            neuron_area_mask = self.renderer.draw_density(sections, self._cached_image.shape[-3:], width=35.0, mask=True)
            
            composite_img = self.alpha * self._cached_image + (1 - self.alpha) * neuron_mask.data
            
            result = {
                'image': composite_img,
                'neuron_tree': self._cached_swc_data,
                'neuron_mask': neuron_area_mask.data,
                'neuron_name': self.swc_files[image_idx].stem,
                'image_idx': image_idx,
                'global_idx': idx
            }
            
            return result


class PatchSampler(Sampler):
    """
    Custom sampler that respects the queue-based loading strategy.
    
    Simply yields indices sequentially to ensure patches are consumed
    in the order they're loaded into the queue.
    """
    
    def __init__(self, data_source: NeuronPatchDataset):
        """
        Initialize sampler.
        
        Parameters:
        -----------
        data_source : NeuronPatchDataset
            The dataset to sample from
        """
        self.data_source = data_source
    
    def __iter__(self):
        """Yield indices sequentially."""
        return iter(range(len(self.data_source)))
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data_source)


class ShuffledPatchSampler(Sampler):
    """
    Sampler that shuffles indices but respects queue batches.
    
    Shuffles groups of patches (from the same image) rather than
    individual patches to maintain queue efficiency.
    """
    
    def __init__(self, data_source: NeuronPatchDataset, rng: Optional[np.random.Generator] = None):
        """
        Initialize sampler.
        
        Parameters:
        -----------
        data_source : NeuronPatchDataset
            The dataset to sample from
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.data_source = data_source
        self.rng = rng or np.random.default_rng()
        self.patches_per_image = data_source.patches_per_image
    
    def __iter__(self):
        """
        Yield shuffled indices that respect queue batches.
        
        Shuffles the order of images, but keeps patches from the same
        image together to maintain loading efficiency.
        """
        n_images = len(self.data_source.img_files)
        
        # Shuffle image order
        image_indices = self.rng.permutation(n_images)
        
        # Yield patch indices in shuffled image order
        for img_idx in image_indices:
            start_idx = img_idx * self.patches_per_image
            for patch_offset in range(self.patches_per_image):
                yield start_idx + patch_offset
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data_source)


def crop_around_subtree(image: torch.Tensor, subtree: List, 
                        padding: int = 10) -> Tuple[torch.Tensor, List]:
    """
    Crop image around subtree coordinates and shift subtree coordinates.
    """
    # Validate subtree is not empty
    if not subtree or len(subtree) == 0:
        raise ValueError("Cannot crop around empty subtree")
    
    # Get coordinates from subtree
    subtree_array = np.array(subtree)
    coords = subtree_array[:, 2:5]  # x, y, z columns
    
    # Calculate bounding box with padding
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding
    
    # Convert to integers and ensure within bounds
    image_shape = image.shape
    min_coords = np.maximum(0, min_coords.astype(int))
    max_coords = np.minimum(image_shape[-3:][::-1], max_coords.astype(int))
    
    # Validate that crop will have positive dimensions
    crop_shape = max_coords - min_coords
    if np.any(crop_shape <= 0):
        raise ValueError(
            f"Crop would result in empty or invalid dimensions. "
            f"Bounding box: min={min_coords}, max={max_coords}, "
            f"crop_shape={crop_shape}, image_shape={image_shape[-3:][::-1]}"
        )
    
    # Crop the image
    cropped_image = image[..., min_coords[2]:max_coords[2], 
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


def pad_with_noise(cropped_image, subtree, pad=10):
    # pad after cropping
    # pad image with random noise based on image stats
    # img_mean = float(torch.mean(cropped_image.float()))
    # img_std = float(torch.std(cropped_image.float()))
    # if img_std == 0.0:
    #     img_std = 1e-6
    img_median = float(torch.median(cropped_image.float()))
    img_mad = float(torch.median(torch.abs(cropped_image.float() - img_median)))
    
    # Handle NaN or zero MAD cases
    if torch.isnan(torch.tensor(img_mad)) or img_mad == 0.0 or img_mad < 0.0:
        print(f"Warning: Image MAD is invalid ({img_mad}). Setting to small constant.")
        img_mad = 1e-6

    device = cropped_image.device
    dtype = cropped_image.dtype

    # Compute padded shape (assumes cropped_image shape is [C, Z, Y, X])
    padded_shape = torch.tensor(cropped_image.shape)
    padded_shape[-3:] += torch.tensor((2*pad,)*3)

    # Create noise in float on the correct device, clamp and cast once
    # noise = torch.normal(mean=img_mean, std=img_std, size=tuple(padded_shape), device=device)
    noise = torch.normal(mean=img_median, std=img_mad, size=tuple(padded_shape), device=device)
    if dtype == torch.uint8:
        noise = noise.clamp(0, 255).to(dtype)
    else:
        noise = noise.clamp(0.0, 1.0).to(dtype)

    # Copy original image into the center of the noisy padded tensor (in-place)
    z0, y0, x0 = pad, pad, pad
    zdim, ydim, xdim = (0,1,2) if len(cropped_image.shape) == 3 else (1,2,3)
    z1 = z0 + cropped_image.shape[zdim]
    y1 = y0 + cropped_image.shape[ydim]
    x1 = x0 + cropped_image.shape[xdim]
    noise[..., z0:z1, y0:y1, x0:x1] = cropped_image
    cropped_image = noise

    # Shift subtree coordinates accordingly
    subtree = np.array(subtree)
    subtree[:, 2:5] = subtree[:, 2:5] + pad
    subtree = subtree.tolist()

    return cropped_image, subtree