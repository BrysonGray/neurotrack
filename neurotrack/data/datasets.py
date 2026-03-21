from collections import deque
from PIL.ImageOps import scale
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import warnings
from qtpy.QtCore import center
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler
import tifffile as tf
from neurotrack.core.pipeline_config import flexible_image_key_lookup
from neurotrack.data import loading as load
from neurotrack.data import rendering as draw
from neurotrack.data.image import Image, to_uint8
from neurotrack.data.seed_io import load_seeds_json


class _ResamplePatch(Exception):
    """Internal signal indicating the caller should sample a different patch."""


class NeuronPatchDataset(TorchDataset):
    """
    PyTorch Dataset for efficiently loading cropped patches from neuron images.
    
    Implements a lazy loading strategy with caching to minimize memory usage:
    - Loads one full TIFF image at a time
    - Extracts patches on-demand using deterministic random seeds
    - Caches the most recently loaded image to avoid redundant I/O
    - Each idx deterministically maps to the same patch (reproducible)
    
    idx // patches_per_image determines which image to load,
    and idx % patches_per_image determines which patch from that image.
    A deterministic RNG seeded with idx ensures reproducibility.
    """
    
    def __init__(
        self,
        swc_dir: Optional[Union[str, Path]],
        img_dir: Union[str, Path],
        crop_size: int = 64,
        patches_per_image: int = 10,
        alpha: float = 0.5,
        step_width: float = 3.0,
        rng: Optional[np.random.Generator] = None,
        crop_patches: bool = True,
        inference_mode: bool = False,
        seeds_path: Optional[Union[str, Path]] = None,
        seed_points_by_image: Optional[Dict[str, List[List[float]]]] = None,
        root_sampling_probability: Optional[float] = None,
        soma_sample_radius: float = 0.0,
        random_offset: float = 0.0,
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
        step_width : float
            Width used for initialization path rendering in dataset-generated
            predicted path channel.
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        crop_patches : bool
            If True, extract random patches. If False, return full images.
        inference_mode : bool
            If True, do not extract SWC data and do not generate neuron masks.
        seeds_path : Optional[str or Path]
            Optional path to a seed JSON file keyed by relative image path.
        seed_points_by_image : Optional[Dict[str, List[List[float]]]]
            Optional in-memory seeds keyed by relative image path.
        root_sampling_probability : Optional[float]
            If set, probability of sampling the subtree center from root nodes
            (parent_id == -1) instead of all SWC nodes when extracting cropped
            patches. If None, center sampling is fully random over all SWC nodes.
        soma_sample_radius : float
            Radius within which to sample soma points.
        random_offset : float
            Random offset to add to sampled points.

        """
        self.swc_files = []
        if swc_dir is not None:
            swc_dir = Path(swc_dir)
            self.swc_files = [file for file in swc_dir.rglob("*.swc") if file.is_file()]
        img_dir = Path(img_dir)
        self.img_dir = img_dir
        if inference_mode:
            self.img_files_unordered = sorted([f for f in img_dir.rglob("*.tif") if f.is_file()])
        else:
            self.img_files_unordered = [f for f in img_dir.rglob("*.tif") if f.is_file()]
        self.img_files = []

        if len(self.img_files_unordered) == 0:
            raise ValueError(f"No TIFF files found in image directory: {img_dir}")

        if len(self.swc_files) > 0:
            if len(self.swc_files) != len(self.img_files_unordered):
                raise ValueError(f"Number of SWC files ({len(self.swc_files)}) must match TIFF files ({len(self.img_files_unordered)})")
            # Order files so names match
            for swc_file in self.swc_files:
                swc_stem = swc_file.stem
                matching_imgs = [f for f in self.img_files_unordered if f.stem == swc_stem or f.stem.startswith(swc_stem + "_") or swc_stem.startswith(f.stem + "_")]
                if not matching_imgs:
                    raise ValueError(f"No matching TIFF files found for SWC file {swc_file}")
                if len(matching_imgs) > 1:
                    raise ValueError(f"Multiple matching TIFF files found for SWC file {swc_file}: {matching_imgs}")
                self.img_files.append(matching_imgs[0])
        else:
            self.img_files = sorted(self.img_files_unordered)

        self.crop_size = crop_size
        self.patches_per_image = patches_per_image
        self.rng = rng or np.random.default_rng(0)
        self.alpha = alpha
        self.step_width = float(step_width)
        self.crop_patches = crop_patches
        self.has_swc = len(self.swc_files) > 0
        self.inference_mode = inference_mode
        self.root_sampling_probability = None if root_sampling_probability is None else float(root_sampling_probability)
        self.soma_sample_radius = float(soma_sample_radius)
        self.random_offset = float(random_offset)
        self.seeds_path = str(seeds_path) if seeds_path is not None else None
        if seed_points_by_image is not None:
            self.seed_points_by_image = dict(seed_points_by_image)
        elif seeds_path is not None:
            self.seed_points_by_image = load_seeds_json(seeds_path)
        else:
            self.seed_points_by_image = {}

        if self.crop_patches and not self.has_swc:
            raise ValueError("crop_patches=True requires SWC files.")

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        if self.root_sampling_probability is not None and not (0.0 <= self.root_sampling_probability <= 1.0):
            raise ValueError("root_sampling_probability must be between 0.0 and 1.0")
        
        # Cache for currently loaded image
        self._cached_image_idx: Optional[int] = None
        self._cached_image: Optional[torch.Tensor] = None
        self._cached_swc_data: Optional[List] = None
        self._cached_swc_array: Optional[np.ndarray] = None
        self._cached_swc_adj_dict: Optional[Dict[int, List[int]]] = None
        self._warned_center_seed_fallback_no_swc = False
        
        # Total dataset size = number of images * patches per image (or just images if not cropping)
        if self.crop_patches:
            self.total_size = len(self.img_files) * self.patches_per_image
        else:
            self.total_size = len(self.img_files)

        # Base seed for reproducibility
        self._base_seed = self.rng.integers(0, 2**31)
        self.renderer = draw.NeuronRenderer(rng=self.rng)

    def _normalize_seed_rows(self, seed_rows: List[List[float]], context: str) -> torch.Tensor:
        """Validate and convert seed rows to a float32 tensor in (z, y, x) order."""
        seeds = torch.as_tensor(seed_rows, dtype=torch.float32)
        if seeds.ndim == 1:
            if seeds.numel() == 0:
                return seeds.reshape(0, 3)
            seeds = seeds.unsqueeze(0)
        if seeds.ndim != 2 or seeds.shape[1] != 3:
            raise ValueError(f"Seed points for '{context}' must have shape (N, 3) in (z, y, x) order.")
        return seeds

    def _get_configured_seed_points(self, relative_image_path: str) -> Optional[torch.Tensor]:
        """Return configured seeds for an image path if available."""
        seed_rows = flexible_image_key_lookup(self.seed_points_by_image, relative_image_path)
        if seed_rows is None:
            return None
        seeds = self._normalize_seed_rows(seed_rows, context=relative_image_path)
        if seeds.shape[0] == 0:
            return None
        return seeds

    @staticmethod
    def _get_root_seed_points_from_swc(swc_data: List) -> Optional[torch.Tensor]:
        """Return root node coordinates from SWC as seeds in (z, y, x) order."""
        if swc_data is None or len(swc_data) == 0:
            return None
        swc_array = np.asarray(swc_data)
        root_rows = swc_array[swc_array[:, 6] == -1]
        if len(root_rows) == 0:
            return None
        root_xyz = torch.as_tensor(root_rows[:, 2:5], dtype=torch.float32)
        return root_xyz.flip(dims=(1,))

    @staticmethod
    def _get_center_seed_point_from_shape(shape_zyx: torch.Size) -> torch.Tensor:
        """Return one center seed in (z, y, x) for a 3D shape."""
        return (torch.as_tensor(shape_zyx, dtype=torch.float32) / 2.0).unsqueeze(0)

    @staticmethod
    def _find_path_ids(
        adjacency: Dict[int, List[int]],
        start_node_id: int,
        target_node_id: int,
    ) -> List[int]:
        """Find a path between two node ids in an undirected adjacency graph."""
        start = int(start_node_id)
        target = int(target_node_id)
        if start == target:
            return [start]

        parents = {start: None}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, []):
                neighbor = int(neighbor)
                if neighbor in parents:
                    continue
                parents[neighbor] = node
                if neighbor == target:
                    queue.clear()
                    break
                queue.append(neighbor)

        if target not in parents:
            return []

        path_ids = []
        current = target
        while current is not None:
            path_ids.append(int(current))
            current = parents[current]
        path_ids.reverse()
        return path_ids

    def _build_predicted_path_channel(
        self,
        subtree: List,
        seed_node_id: int,
        seed_point_xyz: torch.Tensor,
        spatial_shape_zyx: torch.Size,
        add_prev_path: bool = True,
    ) -> torch.Tensor:
        """Build path channel from smallest-id subtree node to the seed node."""
        spatial_shape = tuple(int(v) for v in spatial_shape_zyx)
        path_channel = torch.zeros(spatial_shape, dtype=torch.uint8)

        if not subtree:
            return path_channel, subtree

        id_to_node = {int(node[0]): node for node in subtree}
        seed_id = int(seed_node_id)
        seed_node = id_to_node.get(seed_id)
        if seed_node is None:
            return path_channel, subtree

        # If seed is a root node or add_prev_path is False, draw a seed marker instead of a path.
        if int(seed_node[6]) == -1 or not add_prev_path:
            seed_point = torch.as_tensor((seed_node[4], seed_node[3], seed_node[2]), dtype=torch.float32)
            path_image = Image(torch.zeros((1,) + spatial_shape, dtype=torch.uint8))
            path_image.draw_point(seed_point, radius=self.step_width, channel=0, mode="mask")
            return path_image.data[0], subtree

        start_node_id = min(id_to_node.keys())
        adj_dict = load.adjacency_dict(subtree)
        path_ids = self._find_path_ids(
            adjacency=adj_dict,
            start_node_id=start_node_id,
            target_node_id=seed_id,
        )
        if len(path_ids) < 2:
            return path_channel, subtree
        path_id_set = set(path_ids)

        path_image = Image(torch.zeros((1,) + spatial_shape, dtype=torch.uint8))
        for idx in range(len(path_ids) - 1):
            node_a = id_to_node[path_ids[idx]]
            node_b = id_to_node[path_ids[idx + 1]]
            point_a = torch.as_tensor((node_a[4], node_a[3], node_a[2]), dtype=torch.float32)
            point_b = torch.as_tensor((node_b[4], node_b[3], node_b[2]), dtype=torch.float32)
            segment = torch.stack((point_a, point_b), dim=0)
            path_image.draw_line_segment(segment, width=self.step_width, channel=0, mask=True)
        
        # lastly, draw the small segment from the seed node to the path if it is not zero length to ensure the seed point is included in the path channel
        seed_point_zyx = seed_point_xyz.flip(dims=(0,))
        if not torch.all(seed_point_zyx == point_b):
            segment = torch.stack((seed_point_zyx, point_b), dim=0)
            path_image.draw_line_segment(segment, width=self.step_width, channel=0, mask=True)

        # remove path_ids from subtree.
        # set removed node neighbors that are not removed to have parent_id = -1
        neighbors_to_update = set()
        for node_id in path_ids:
            neighbor_nodes = adj_dict.get(int(node_id), [])
            for neighbor in neighbor_nodes:
                if int(neighbor) not in path_id_set:
                    neighbors_to_update.add(int(neighbor))
        updated_subtree = []
        for node in subtree:
            if int(node[0]) in path_id_set:
                continue
            if int(node[0]) in neighbors_to_update:
                updated_node = list(node)
                updated_node[6] = -1
                updated_subtree.append(updated_node)
            else:
                updated_subtree.append(node)

        # Keep subtree edge-based: drop any isolated leftovers after path removal.
        if updated_subtree:
            updated_adj = load.adjacency_dict(updated_subtree)
            connected_ids = {
                int(node_id)
                for node_id, neighbors in updated_adj.items()
                if len(neighbors) > 0
            }
            if connected_ids:
                updated_subtree = [node for node in updated_subtree if int(node[0]) in connected_ids]
            else:
                updated_subtree = []

        return path_image.data[0], updated_subtree

    @staticmethod
    def _append_path_channel(image: torch.Tensor, path_channel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Append a path channel to image tensor with shape [C,Z,Y,X] or [Z,Y,X]."""
        if image.ndim == 3:
            image_cf = image.unsqueeze(0)
        elif image.ndim == 4:
            image_cf = image
        else:
            raise ValueError(f"Image must be 3D or 4D, got shape {tuple(image.shape)}")

        if path_channel is None:
            path_channel = torch.zeros(image_cf.shape[-3:], dtype=image_cf.dtype, device=image_cf.device)
        else:
            if path_channel.ndim != 3:
                raise ValueError(f"Path channel must be 3D [Z,Y,X], got shape {tuple(path_channel.shape)}")
            path_channel = path_channel.to(device=image_cf.device, dtype=image_cf.dtype)

        return torch.cat((image_cf, path_channel.unsqueeze(0)), dim=0)

    @staticmethod
    def _iter_parent_child_segments(swc_data: Union[List, np.ndarray]) -> List[torch.Tensor]:
        """Return SWC parent-child segments in (z, y, x) coordinate order."""
        swc_array = np.asarray(swc_data)
        if swc_array.size == 0:
            return []

        id_to_row = {int(row[0]): row for row in swc_array}
        segments: List[torch.Tensor] = []
        for row in swc_array:
            parent_id = int(row[6])
            if parent_id == -1:
                continue
            parent_row = id_to_row.get(parent_id)
            if parent_row is None:
                continue

            parent_zyx = torch.as_tensor((parent_row[4], parent_row[3], parent_row[2]), dtype=torch.float32)
            child_zyx = torch.as_tensor((row[4], row[3], row[2]), dtype=torch.float32)
            segments.append(torch.stack((parent_zyx, child_zyx), dim=0))

        return segments

    def _draw_tree_mask(
        self,
        swc_data: Union[List, np.ndarray],
        spatial_shape_zyx: torch.Size,
        width: float,
    ) -> Image:
        """Draw a binary neuron mask directly from SWC parent-child segments."""
        mask_image = Image(torch.zeros((1,) + tuple(int(v) for v in spatial_shape_zyx), dtype=torch.uint8))
        for segment in self._iter_parent_child_segments(swc_data):
            mask_image.draw_line_segment(segment, width=width, channel=0, mask=True)
        return mask_image

    @staticmethod
    def _convex_blend_uint8(base_image: torch.Tensor, overlay_image: torch.Tensor, alpha: float) -> torch.Tensor:
        """Blend two images using integer arithmetic and return uint8 output."""
        if base_image.shape != overlay_image.shape:
            raise ValueError(
                f"Cannot blend images with different shapes: {tuple(base_image.shape)} vs {tuple(overlay_image.shape)}"
            )

        if base_image.dtype != torch.uint8:
            base_u8 = to_uint8(base_image)
        else:
            base_u8 = base_image

        if overlay_image.dtype != torch.uint8:
            overlay_u8 = to_uint8(overlay_image)
        else:
            overlay_u8 = overlay_image

        if overlay_u8.device != base_u8.device:
            overlay_u8 = overlay_u8.to(base_u8.device)

        alpha_q = int(round(float(alpha) * 255.0))
        alpha_q = max(0, min(255, alpha_q))
        inv_alpha_q = 255 - alpha_q

        blended = (
            base_u8.to(dtype=torch.int32) * alpha_q
            + overlay_u8.to(dtype=torch.int32) * inv_alpha_q
            + 127
        ) // 255
        return blended.to(dtype=torch.uint8)
        
    def _load_image(self, idx: int) -> torch.Tensor:
        """Load a full image."""
        img_path = self.img_files[idx]
        
        img = tf.imread(img_path)
        img = img.squeeze()
        if img.dtype != np.uint8:
            print(f"Warning: Image {img_path} has dtype {img.dtype}, converting to uint8")

        img = torch.from_numpy(to_uint8(img))
        
        return img
    
    def _load_swc(self, idx: int) -> List:
        """Load SWC neuron data."""
        if not self.has_swc:
            raise RuntimeError("SWC data requested but dataset was initialized without SWC files")
        swc_path = self.swc_files[idx]
        
        swc_data = load.swc(swc_path, rotate=False, verbose=False)
        
        return swc_data
    
    def _extract_subtree(
        self,
        swc_data: Union[List, np.ndarray],
        center_node_id: int,
        center_point: np.ndarray,
        swc_adj_dict: Optional[Dict[int, List[int]]] = None,
        image_shape_xyz: Optional[Tuple[int, int, int]] = None,
    ) -> List:
        """Extract the connected subtree within crop bounds around a center SWC node."""

        swc_array = np.asarray(swc_data)
        min_coords = np.floor(center_point - self.crop_size // 2).astype(int)
        max_coords = np.ceil(center_point + self.crop_size // 2).astype(int)
        if image_shape_xyz is not None:
            shape_xyz = np.asarray(image_shape_xyz, dtype=np.int64)
            min_coords = np.maximum(0, min_coords)
            max_coords = np.minimum(shape_xyz, max_coords)

        in_box_mask = np.all(
            (swc_array[:, 2:5] >= min_coords) &
            (swc_array[:, 2:5] < max_coords),
            axis=1
        )
        in_box_nodes = swc_array[in_box_mask]
        if in_box_nodes.shape[0] == 0:
            return []

        if swc_adj_dict is None:
            swc_adj_dict = load.adjacency_dict(swc_array)

        # Keep only the connected component containing center_id under in-box node constraints.
        in_box_ids = set(in_box_nodes[:, 0].astype(np.int64).tolist())
        if center_node_id not in in_box_ids:
            return in_box_nodes.tolist()

        visited = set()
        to_visit = [center_node_id]
        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            visited.add(node)
            neighbors = swc_adj_dict.get(node, [])
            for neighbor in neighbors:
                if neighbor in in_box_ids and neighbor not in visited:
                    to_visit.append(neighbor)

        connected_ids = np.array(list(visited), dtype=np.int64)
        connected_mask = np.isin(in_box_nodes[:, 0].astype(np.int64), connected_ids)
        subtree = in_box_nodes[connected_mask].tolist()

        # TODO: This results in abberant segments. Need to figure out a more robust way to add boundary nodes that doesn't create weird artifacts.
        # # get subtree neighbors that are just outside the box and add a node where the segment intersects the box boundary
        # boundary_segments = set() # set of (node_id, neighbor_id) tuples for segments crossing the boundary
        # for node in subtree:
        #     node_id = int(node[0])
        #     neighbors = swc_adj_dict.get(node_id, [])
        #     for neighbor in neighbors:
        #         if neighbor not in visited:
        #             boundary_segments.add((node_id,neighbor))
        # for inside_node, outside_node in boundary_segments:
        #     outside_row = swc_array[id_to_idx[outside_node]]
        #     outside_point = outside_row[2:5]
        #     inside_point = swc_array[id_to_idx[inside_node]][2:5]
        #     segment = outside_point - inside_point
        #     if np.all(segment == 0):
        #         continue
        #     direction = segment / np.linalg.norm(segment)

        #     # find the scale factor to move from inside_point to the box boundary in the direction of the segment
        #     half = self.crop_size / 2.0
        #     box_min = center_point - half
        #     box_max = center_point + half
            
        #     candidates = []
        #     for axis in range(3):
        #         if direction[axis] > 0:
        #             t = (box_max[axis] - inside_point[axis]) / direction[axis]
        #         elif direction[axis] < 0:
        #             t = (box_min[axis] - inside_point[axis]) / direction[axis]
        #         else:
        #             continue
        #         if t >= 0:
        #             candidates.append(t)
            
        #     if not candidates:
        #         continue
            
        #     scale = min(candidates)

        #     intersection_point = inside_point + direction * scale
        #     new_node = np.array([
        #         outside_row[0],
        #         outside_row[1],
        #         intersection_point[0],
        #         intersection_point[1],
        #         intersection_point[2],
        #         outside_row[5],
        #         node_id,
        #     ])
        #     subtree.append(new_node.tolist())

        return subtree
    
    def _sample_near_soma(self, swc_data: Union[List, np.ndarray], patch_rng: np.random.Generator, image_shape_xyz: Tuple[int, int, int], radius: float = 0.0, random_offset: float = 0.0) -> np.ndarray:
        """Sample a point within radius of the soma (root nodes) with some random offset."""
        swc_array = np.asarray(swc_data)
        root_nodes = swc_array[swc_array[:, 6] == -1]
        if len(root_nodes) == 0:
            raise _ResamplePatch("No root nodes found in SWC data for soma-biased sampling.")
        soma_center = root_nodes[0, 2:5]
        if radius > 0.0:
            in_radius_mask = np.sum((swc_array[:, 2:5] - soma_center) ** 2, axis=1) <= radius ** 2
            candidates = swc_array[in_radius_mask]
            if len(candidates) == 0:
                raise _ResamplePatch(f"No nodes found within radius {radius} of soma for sampling.")
            center_node = candidates[patch_rng.integers(len(candidates))]
        else:
            center_node = root_nodes[0]
        seed_node_id = int(center_node[0])
        if random_offset > 0.0:
            # try to add random offset while keeping the point within image bounds if image_shape_xyz is provided

            offset = patch_rng.uniform(-random_offset, random_offset, size=3)
            seed_point_xyz = center_node[2:5] + offset
            seed_point_xyz = np.maximum(seed_point_xyz, 0)
            seed_point_xyz = np.minimum(seed_point_xyz, np.array(image_shape_xyz) - 1)
        else:
            seed_point_xyz = center_node[2:5]

        return seed_point_xyz, seed_node_id
        

    def _extract_random_patch(
        self,
        image: torch.Tensor,
        swc_data: Union[List, np.ndarray],
        patch_rng: np.random.Generator,
        swc_adj_dict: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract a cropped patch using RNG-selected center with optional root bias.

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
        # Choose center node (default random over all nodes, with optional root bias)
        swc_array = np.asarray(swc_data)
        image_shape_xyz = tuple(image.shape[-3:][::-1])
        sample_from_root = self.root_sampling_probability is not None and patch_rng.random() < self.root_sampling_probability
        if sample_from_root:
            seed_point_xyz, seed_node_id = self._sample_near_soma(swc_array, patch_rng, radius=self.soma_sample_radius, random_offset=self.random_offset, image_shape_xyz=image_shape_xyz)
        else:
            center_node = swc_array[patch_rng.integers(len(swc_array))]
            seed_node_id = int(center_node[0])
            seed_point_xyz = center_node[2:5]
            if self.random_offset > 0.0:
                offset = patch_rng.uniform(-self.random_offset, self.random_offset, size=3)
                seed_point_xyz = seed_point_xyz + offset

        subtree = self._extract_subtree(
            swc_array,
            center_node_id=seed_node_id,
            center_point=seed_point_xyz,
            swc_adj_dict=swc_adj_dict,
            image_shape_xyz=image_shape_xyz,
        )
        if len(subtree) == 0:
            raise _ResamplePatch("Extracted subtree is empty before cropping.")

        try:
            cropped_img, shifted_subtree = crop_around_subtree(
                image,
                subtree,
                center_point=seed_point_xyz,
                size=self.crop_size,
            )
        except ValueError as exc:
            raise _ResamplePatch(str(exc)) from exc

        seed_xyz = None
        for node in shifted_subtree:
            if int(node[0]) == seed_node_id:
                seed_xyz = torch.as_tensor(node[2:5], dtype=torch.float32)
                break
        if seed_xyz is None:
            raise _ResamplePatch(f"Seed node {seed_node_id} was not found in cropped subtree.")

        if self.inference_mode:
            image = cropped_img
            neuron_area_mask = None

        else:
            # create mask
            neuron_area_mask = self._draw_tree_mask(
                swc_data=shifted_subtree,
                spatial_shape_zyx=cropped_img.shape[-3:],
                width=35.0,
            )
            neuron_area_mask = neuron_area_mask.data
            if self.alpha < 1.0:
                neuron_mask = self._draw_tree_mask(
                    swc_data=shifted_subtree,
                    spatial_shape_zyx=cropped_img.shape[-3:],
                    width=self.step_width,
                )
                image = self._convex_blend_uint8(cropped_img, neuron_mask.data.squeeze(), alpha=self.alpha)
            else:
                image = cropped_img

        path_channel, shifted_subtree = self._build_predicted_path_channel(
            subtree=shifted_subtree,
            seed_node_id=seed_node_id,
            seed_point_xyz=seed_xyz,
            spatial_shape_zyx=cropped_img.shape[-3:],
            add_prev_path=True # always add the previous path.
        )
        if len(shifted_subtree) == 0:
            raise _ResamplePatch("Extracted subtree is empty after path pruning.")

        image = self._append_path_channel(image, path_channel=path_channel)

        return {
            'image': image,
            'neuron_tree': shifted_subtree,
            'neuron_mask': neuron_area_mask,
            'seed_point_xyz': seed_xyz,
            'seed_node_id': seed_node_id,
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
            base_image_idx = (idx // self.patches_per_image) % len(self.img_files)
            
            # Create a deterministic RNG for this specific patch
            patch_seed = self._base_seed + idx
            patch_rng = np.random.default_rng(patch_seed)

            # Retry patch sampling when pruning removes all subtree nodes.
            attempts_per_image = max(8, min(64, int(self.patches_per_image)))
            last_error: Optional[Exception] = None

            for image_offset in range(len(self.img_files)):
                image_idx = (base_image_idx + image_offset) % len(self.img_files)

                # Load image/SWC cache for this image index.
                if self._cached_image_idx != image_idx:
                    self._cached_image = self._load_image(image_idx)
                    self._cached_swc_data = self._load_swc(image_idx)
                    self._cached_swc_array = np.asarray(self._cached_swc_data, dtype=np.float32)
                    self._cached_swc_adj_dict = load.adjacency_dict(self._cached_swc_array)
                    self._cached_image_idx = image_idx

                swc_data = self._cached_swc_array if self._cached_swc_array is not None else self._cached_swc_data
                if swc_data is None:
                    raise RuntimeError("SWC cache is unexpectedly empty for crop_patches mode")

                for _ in range(attempts_per_image):
                    try:
                        patch = self._extract_random_patch(
                            self._cached_image,
                            swc_data,
                            patch_rng,
                            swc_adj_dict=self._cached_swc_adj_dict,
                        )
                    except _ResamplePatch as exc:
                        last_error = exc
                        continue

                    seed_xyz = patch.pop('seed_point_xyz')
                    patch['seed_points'] = seed_xyz.flip(dims=(0,)).unsqueeze(0)

                    # Add metadata
                    patch['neuron_name'] = self.img_files[image_idx].stem
                    patch['relative_image_path'] = self.img_files[image_idx].relative_to(self.img_dir).as_posix()
                    patch['image_idx'] = image_idx
                    patch['global_idx'] = idx

                    return patch

            total_attempts = attempts_per_image * len(self.img_files)
            raise RuntimeError(
                "Failed to sample a valid patch after "
                f"{total_attempts} attempts starting at image index {base_image_idx}. "
                f"Last sampling error: {last_error}"
            )
        else:
            # Return full image without cropping
            image_idx = idx % len(self.img_files)
            
            # Load image if not cached or if different image
            if self._cached_image_idx != image_idx:
                self._cached_image = self._load_image(image_idx)
                self._cached_swc_data = self._load_swc(image_idx) if self.has_swc else None
                if self._cached_swc_data is not None:
                    self._cached_swc_array = np.asarray(self._cached_swc_data, dtype=np.float32)
                    self._cached_swc_adj_dict = load.adjacency_dict(self._cached_swc_array)
                else:
                    self._cached_swc_array = None
                    self._cached_swc_adj_dict = None
                self._cached_image_idx = image_idx

            if self.has_swc and not self.inference_mode:
                # Create full neuron mask and composite
                neuron_area_mask = self._draw_tree_mask(
                    swc_data=self._cached_swc_data,
                    spatial_shape_zyx=self._cached_image.shape[-3:],
                    width=35.0,
                )
                if self.alpha < 1.0:
                    neuron_mask = self._draw_tree_mask(
                        swc_data=self._cached_swc_data,
                        spatial_shape_zyx=self._cached_image.shape[-3:],
                        width=self.step_width,
                    )
                    composite_img = self._convex_blend_uint8(self._cached_image, neuron_mask.data.squeeze(), alpha=self.alpha)
                else:
                    composite_img = self._cached_image
                neuron_tree = self._cached_swc_data
                neuron_mask_data = neuron_area_mask.data
                neuron_name = self.swc_files[image_idx].stem
            else:
                composite_img = self._cached_image
                neuron_tree = None
                neuron_mask_data = None
                neuron_name = self.img_files[image_idx].stem

            relative_image_path = self.img_files[image_idx].relative_to(self.img_dir).as_posix()

            configured_seeds = self._get_configured_seed_points(relative_image_path)

            if configured_seeds is not None:
                seed_points = configured_seeds
            elif self.has_swc and self._cached_swc_data is not None:
                root_seeds = self._get_root_seed_points_from_swc(self._cached_swc_data)
                if root_seeds is not None:
                    seed_points = root_seeds
                else:
                    seed_points = self._get_center_seed_point_from_shape(self._cached_image.shape[-3:])
            else:
                if not self._warned_center_seed_fallback_no_swc:
                    warnings.warn(
                        "No SWC and no configured seeds were provided; using image center as fallback seed.",
                        RuntimeWarning,
                    )
                    self._warned_center_seed_fallback_no_swc = True
                seed_points = self._get_center_seed_point_from_shape(self._cached_image.shape[-3:])
            
            result = {
                'image': self._append_path_channel(composite_img),
                'neuron_tree': neuron_tree,
                'neuron_mask': neuron_mask_data,
                'neuron_name': neuron_name,
                'relative_image_path': relative_image_path,
                'seed_points': seed_points,
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
                        center_point: np.ndarray, size: int) -> Tuple[torch.Tensor, List]:
    """
    Crop image around subtree coordinates and shift subtree coordinates.
    """
    # Validate subtree is not empty
    if not subtree or len(subtree) == 0:
        raise ValueError("Cannot crop around empty subtree")
    
    # Get coordinates from subtree
    subtree_array = np.array(subtree)
    
    # Crop around the center of the subtree with given size
    if np.any(center_point < 0):
        raise ValueError(f"Center point has negative coordinates: {center_point}")
    min_coords = np.floor(center_point - size // 2).astype(int)
    max_coords = np.ceil(center_point + size // 2).astype(int)
    
    # Compute padding if crop goes out of bounds
    image_shape = image.shape[-3:][::-1] # convert to X, Y, Z order

    min_coords = np.maximum(0, min_coords)
    max_coords = np.minimum(image_shape, max_coords)
    
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
            node[2] - min_coords[0], # x
            node[3] - min_coords[1], # y
            node[4] - min_coords[2], # z
            node[5],  # radius
            node[6]   # parent_id
        )
        shifted_subtree.append(shifted_node)
    
    return cropped_image, shifted_subtree

