"""
Refactored draw.py - Simplified and more readable neuron drawing functionality.

This module provides clean interfaces for drawing neuron structures with clear
separation of concerns and improved readability.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from skimage.filters import gaussian 
import torch
import imageio

import sys
sys.path.append(str(Path(__file__).parent))
from image import Image
import load


@dataclass
class DrawingConfig:
    """Configuration for neuron drawing parameters."""
    width: float = 3.0
    noise: float = 0.05
    rgb: bool = True
    binary: bool = False
    random_brightness: bool = False
    random_sharpness: Optional[float] = None
    neuron_color: Optional[Tuple[float, float, float]] = None
    background_color: Optional[Tuple[float, float, float]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.neuron_color and len(self.neuron_color) != 3:
            raise ValueError("neuron_color must be a 3-tuple of floats")
        if self.background_color and len(self.background_color) != 3:
            raise ValueError("background_color must be a 3-tuple of floats")


@dataclass 
class GifConfig:
    """Configuration for GIF generation."""
    save_gif: bool = False
    gif_path: Optional[str] = None
    gif_axis: int = 0
    gif_frames: int = 100


class NeuronRenderer:
    """Main class for rendering neuron structures."""
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        """Initialize the renderer with optional random number generator."""
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def draw_density(self, segments: Union[List, np.ndarray, torch.Tensor], 
                    shape: Tuple[int, ...], width: Optional[float] = None, 
                    scale: float = 1.0) -> Image:
        """
        Draw neuron density image from segments.
        
        Args:
            segments: Neuron segments as coordinates
            shape: Output image shape
            width: Line width (auto-detected if None)
            scale: Pixel scale factor
            
        Returns:
            Image object with neuron density
        """
        density = Image(torch.zeros((1,) + shape))
        segments = self._ensure_tensor(segments)
        
        for segment in segments:
            line_width = self._get_segment_width(segment, width, scale)
            density.draw_line_segment(segment[:, :3], width=line_width, channel=0)
        
        return density
    
    def draw_mask(self, density: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        """
        Create binary mask from density image.
        
        Args:
            density: Neuron density tensor
            threshold: Threshold for mask creation
            
        Returns:
            Binary mask tensor
        """
        peak = density.amax()
        mask = torch.zeros_like(density, dtype=torch.bool)
        mask[density > peak * np.exp(-0.5 * threshold)] = True
        return mask
    
    def draw_section_labels(self, sections: Dict, shape: Tuple[int, ...], 
                          width: float = 3.0) -> Image:
        """
        Draw discrete labels for neuron sections.
        
        Args:
            sections: Dictionary of section_id -> segments
            shape: Output image shape
            width: Line width
            
        Returns:
            Image with section labels
        """
        labels = Image(torch.zeros((1,) + shape))
        
        for section_id, section_segments in sections.items():
            for segment in section_segments:
                segment_width = self._get_segment_width(segment, width)
                labels.draw_line_segment(
                    segment[:, :3], width=segment_width, 
                    channel=0, mask=True, value=section_id
                )
        
        return labels
    
    def draw_path(self, img: Image, path: Union[List, np.ndarray, torch.Tensor], 
                 width: float, binary: bool = False) -> Image:
        """
        Draw a path on an image.
        
        Args:
            img: Target image
            path: Path coordinates
            width: Line width
            binary: Whether to use binary drawing
            
        Returns:
            Modified image
        """
        path = self._ensure_tensor(path)
        segments = torch.stack((path[:-1], path[1:]), dim=1)
        
        for segment in segments:
            img.draw_line_segment(segment[:, :3], width=width, mask=binary, channel=0)
        
        return img
    
    def draw_neuron(self, segments: List[np.ndarray], shape: Tuple[int, ...], 
                   config: DrawingConfig, gif_config: Optional[GifConfig] = None) -> Image:
        """
        Draw complete neuron image with all effects.
        
        Args:
            segments: List of neuron segments
            shape: Output image shape  
            config: Drawing configuration
            gif_config: Optional GIF generation config
            
        Returns:
            Rendered neuron image
        """
        img = Image(torch.zeros((1,) + shape))
        gif_frames = []
        gif_steps = self._setup_gif_steps(len(segments), gif_config)
        
        # Draw segments
        for idx, segment in enumerate(segments):
            self._draw_single_segment(img, segment, config, idx)
            self._maybe_capture_gif_frame(img, idx, gif_steps, gif_frames, gif_config)
        
        # Apply post-processing
        img = self._apply_color_effects(img, config)
        img = self._apply_noise_and_scaling(img, config)
        
        # Save GIF if requested
        self._maybe_save_gif(gif_frames, gif_config)
        
        return img
    
    def neuron_from_swc(self, swc_list: List, config: DrawingConfig, 
                       shape: Optional[Tuple[int, ...]] = None, 
                       dropout: bool = False, adjust: bool = False) -> Dict:
        """
        Generate neuron image from SWC data.
        
        Args:
            swc_list: SWC neuron data
            config: Drawing configuration
            shape: Output shape (auto-calculated if None)
            dropout: Whether to add signal dropout
            adjust: Whether to adjust coordinates
            
        Returns:
            Dictionary with image, density, labels, etc.
        """
        # Parse SWC data
        sections, graph = load.parse_swc(swc_list)
        branches, terminals = load.get_critical_points(swc_list, sections)
        scale = 1.0
        
        if adjust:
            sections, branches, terminals, scale = load.adjust_neuron_coords(
                sections, branches, terminals
            )
        
        # Prepare segments
        segments = self._consolidate_segments(sections)
        
        if shape is None:
            shape = self._calculate_shape(segments)
        
        # Generate images
        img = self.draw_neuron(segments, shape, config)
        density = self.draw_density(segments, shape, width=config.width)
        section_labels = self.draw_section_labels(sections, shape, width=2*config.width)
        
        # Apply dropout if requested
        if dropout:
            img = self._apply_dropout(img, section_labels, config.width)
        
        # Prepare output
        root_key = min(sections.keys())
        seed = sections[root_key][0, 0, :3].round().astype(np.uint16).tolist()
        
        return {
            "image": img.data,
            "neuron_density": density.data,
            "section_labels": section_labels.data,
            "branches": branches,
            "seeds": [seed],
            "scale": scale,
            "graph": graph
        }
    
    # Private helper methods
    def _ensure_tensor(self, data: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert data to torch tensor."""
        if isinstance(data, list):
            return torch.tensor(data)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return data
    
    def _get_segment_width(self, segment: torch.Tensor, default_width: Optional[float] = None, 
                          scale: float = 1.0) -> float:
        """Get width for a segment, handling various cases."""
        if default_width is not None:
            return default_width
        elif segment.shape[1] == 4:  # Width included in segment data
            return ((segment[0, 3] + segment[1, 3]) / 2).item() / scale
        else:
            return 3.0
    
    def _setup_gif_steps(self, n_segments: int, gif_config: Optional[GifConfig]) -> set:
        """Setup which steps to capture for GIF."""
        if not gif_config or not gif_config.save_gif:
            return set()
        return set(np.linspace(0, n_segments-1, gif_config.gif_frames, dtype=int))
    
    def _draw_single_segment(self, img: Image, segment: np.ndarray, 
                           config: DrawingConfig, idx: int) -> None:
        """Draw a single segment with all effects."""
        # Calculate segment properties
        value = self._get_segment_value(config, idx)
        sharpness = self._get_segment_sharpness(config)
        width = self._get_segment_width(torch.from_numpy(segment), config.width)
        
        # Draw the segment
        img.draw_line_segment(
            segment[:, :3], width=width, sharpness=sharpness, 
            mask=config.binary, channel=0, value=value
        )
    
    def _get_segment_value(self, config: DrawingConfig, idx: int) -> float:
        """Get brightness value for segment."""
        if config.random_brightness:
            y0 = 0.5
            return y0 + (1.0 - y0) * self.rng.uniform(0.0, 1.0)
        return 1.0
    
    def _get_segment_sharpness(self, config: DrawingConfig) -> float:
        """Get sharpness value for segment."""
        if config.random_sharpness:
            sharpness = self.rng.normal(1.0, config.random_sharpness)
            return np.clip(sharpness, 1.0, 6.0)
        return 1.0
    
    def _maybe_capture_gif_frame(self, img: Image, idx: int, gif_steps: set, 
                               gif_frames: List, gif_config: Optional[GifConfig]) -> None:
        """Capture GIF frame if needed."""
        if gif_config and idx in gif_steps:
            arr = img.data[0].cpu().numpy()  # Fixed index
            mip = arr.max(axis=gif_config.gif_axis)
            mip = ((mip - mip.min()) / (mip.ptp() + 1e-8) * 255).astype(np.uint8)
            gif_frames.append(mip)
    
    def _apply_color_effects(self, img: Image, config: DrawingConfig) -> Image:
        """Apply RGB color effects."""
        if not config.rgb:
            return img
            
        # Apply neuron color
        if config.neuron_color:
            neuron_color = torch.tensor(config.neuron_color, dtype=torch.float32)
            img.data = neuron_color[:, None, None, None] * img.data
        
        # Apply background color
        if config.background_color:
            background_color = torch.tensor(config.background_color, dtype=torch.float32)
            img.data = img.data + torch.ones_like(img.data) * background_color[:, None, None, None]
        
        return img
    
    def _apply_noise_and_scaling(self, img: Image, config: DrawingConfig) -> Image:
        """Apply noise and intensity scaling."""
        # Determine max value
        max_val = self.rng.uniform(0.2, 1.0) if config.random_brightness else 1.0
        
        # Add noise
        noise = torch.from_numpy(
            self.rng.standard_normal(size=img.data.shape, dtype=np.float32) * config.noise
        )
        img.data += noise
        
        # Scale to [0, max_val]
        data_range = img.data.amax() - img.data.amin()
        if data_range > 0:
            img.data = (img.data - img.data.amin()) / data_range * max_val
        
        return img
    
    def _consolidate_segments(self, sections: Dict) -> np.ndarray:
        """Consolidate all sections into segment array."""
        segments = [section for section in sections.values()]
        return np.concatenate(segments)
    
    def _calculate_shape(self, segments: np.ndarray) -> Tuple[int, ...]:
        """Calculate image shape from segments."""
        shape = np.ceil(np.max(segments[..., :3], axis=(0, 1)))
        shape = shape.astype(np.uint16)
        shape = shape + np.array([10, 10, 10])
        return tuple(shape.tolist())
    
    def _apply_dropout(self, img: Image, section_labels: Image, width: float) -> Image:
        """Apply random signal dropout."""
        neuron_coords = torch.nonzero(section_labels.data)
        dropout_density = 0.001
        size = int(dropout_density * len(neuron_coords))
        
        if size > 0:
            rand_ints = self.rng.integers(0, len(neuron_coords), size=(size,))
            dropout_points = neuron_coords[rand_ints]
            dropout_points = dropout_points[:, 1:].T
            
            dropout_img = torch.zeros_like(img.data)
            dropout_img[:, dropout_points[0], dropout_points[1], dropout_points[2]] = 1.0
            dropout_img = gaussian(dropout_img, sigma=0.5*width)
            dropout_img /= dropout_img.max()
            img.data = img.data * (1. - dropout_img)
        
        return img
    
    def _maybe_save_gif(self, gif_frames: List, gif_config: Optional[GifConfig]) -> None:
        """Save GIF if frames were captured."""
        if gif_config and gif_config.save_gif and gif_config.gif_path and gif_frames:
            imageio.mimsave(gif_config.gif_path, gif_frames, duration=0.05)


# Convenience functions for backward compatibility
def draw_neuron_density(segments, shape, width=None, scale=1.0):
    """Convenience function - delegates to NeuronRenderer."""
    renderer = NeuronRenderer()
    return renderer.draw_density(segments, shape, width, scale)


def draw_neuron_mask(density, threshold=1.0):
    """Convenience function - delegates to NeuronRenderer."""
    renderer = NeuronRenderer()
    return renderer.draw_mask(density.data if hasattr(density, 'data') else density, threshold)


def draw_section_labels(sections, shape, width=3):
    """Convenience function - delegates to NeuronRenderer."""
    renderer = NeuronRenderer()
    return renderer.draw_section_labels(sections, shape, width)


def draw_path(img, path, width, binary):
    """Convenience function - delegates to NeuronRenderer."""
    renderer = NeuronRenderer()
    return renderer.draw_path(img, path, width, binary)


def draw_neuron(segments, shape, noise, width=None, random_sharpness=None,
               rgb=True, neuron_color=None, background_color=None, random_brightness=False,
               binary=False, rng=None, save_gif=False, gif_path=None, gif_axis=0):
    """Convenience function - delegates to NeuronRenderer with legacy interface."""
    renderer = NeuronRenderer(rng)
    
    config = DrawingConfig(
        width=width or 3.0,
        noise=noise,
        rgb=rgb,
        binary=binary,
        random_brightness=random_brightness,
        random_sharpness=random_sharpness,
        neuron_color=neuron_color,
        background_color=background_color
    )
    
    gif_config = GifConfig(
        save_gif=save_gif,
        gif_path=gif_path,
        gif_axis=gif_axis
    ) if save_gif else None
    
    return renderer.draw_neuron(segments, shape, config, gif_config)


def neuron_from_swc(swc_list, width=3, noise=0.05, shape=None, dropout=False, adjust=False, 
                   rgb=False, background_color=None, neuron_color=None, random_brightness=False,
                   random_sharpness=False, binary=False, rng=None):
    """Convenience function - delegates to NeuronRenderer with legacy interface."""
    renderer = NeuronRenderer(rng)
    
    config = DrawingConfig(
        width=width,
        noise=noise,
        rgb=rgb,
        binary=binary,
        random_brightness=random_brightness,
        random_sharpness=random_sharpness,
        neuron_color=neuron_color,
        background_color=background_color
    )
    
    return renderer.neuron_from_swc(swc_list, config, shape, dropout, adjust)


if __name__ == "__main__":
    # Example usage
    renderer = NeuronRenderer()
    config = DrawingConfig(width=5.0, noise=0.1, rgb=True)
    print("NeuronRenderer ready for use!")
