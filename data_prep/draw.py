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
from scipy.ndimage import gaussian_filter, median_filter

import sys
sys.path.append(str(Path(__file__).parent))
from image import Image
import load


@dataclass
class DrawingConfig:
    """Configuration for neuron drawing parameters."""
    width: float = 4.0
    rgb: bool = False
    random_width: bool = False
    neuron_color: Optional[Tuple[float, float, float]] = None
    background_color: Optional[Tuple[float, float, float]] = None
    
    foreground_mean: float = 0.8
    foreground_std: float = 0.1
    background_mean: float = 0.2
    background_std: float = 0.05
    mask_threshold: float = 0.1  # Fraction of max value for foreground/background mask
    spatial_noise_scale: float = 10.0  # Scale for spatial noise features
    spatial_noise_amplitude: float = 1.0  # Amplitude multiplier for spatial noise contribution
    noise_method: str = 'gaussian_convolution'  # Method for spatial noise: 'gaussian_convolution', 'fractal', 'sparse_kernel'
    blur: float = 1.0  # Sigma for optional Gaussian smoothing applied during post-processing
    sharpness: float = 1.0  # Sharpness parameter for line drawing edges
    vignette_magnitude: float = 0.2  # Strength of vignette effect (0 disables)
    width_correlation: bool = False
    width_correlation_rho: float = 0.0  # target lag-1 correlation for widths when enabled
    segment_intensity_correlation: bool = False
    segment_intensity_correlation_rho: float = 0.0  # target lag-1 correlation for per-segment intensity when enabled
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.neuron_color and len(self.neuron_color) != 3:
            raise ValueError("neuron_color must be a 3-tuple of floats")
        if self.background_color and len(self.background_color) != 3:
            raise ValueError("background_color must be a 3-tuple of floats")
        if not 0 <= self.mask_threshold <= 1:
            raise ValueError("mask_threshold must be between 0 and 1")
        if self.spatial_noise_amplitude < 0:
            raise ValueError("spatial_noise_amplitude must be non-negative")
        if self.noise_method not in ['gaussian_convolution', 'fractal', 'sparse_kernel']:
            raise ValueError("noise_method must be one of: 'gaussian_convolution', 'fractal', 'sparse_kernel'")
        # Validate correlations
        if self.width_correlation and not (-1.0 <= float(self.width_correlation_rho) <= 1.0):
            raise ValueError("width_correlation_rho must be in (-1.0, 1.0) when width_correlation=True")
        if self.segment_intensity_correlation and not (-1.0 <= float(self.segment_intensity_correlation_rho) <= 1.0):
            raise ValueError("segment_intensity_correlation_rho must be in (-1.0, 1.0) when segment_intensity_correlation=True")


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

    def draw_density(self, sections: Dict, shape: Tuple[int, ...], 
                    width: Optional[float] = None, mask: bool = False, scale: float = 1.0) -> Image:
        """
        Draw neuron density image from sections.
        
        Args:
            sections: Dictionary of section_id -> segments
            shape: Output image shape
            width: Line width (auto-detected if None)
            mask: Whether to create a binary mask
            scale: Pixel scale factor
            
        Returns:
            Image object with neuron density
        """
        density = Image(torch.zeros((1,) + shape))
        segments = self._consolidate_segments(sections)
        
        for segment in segments:
            line_width = self._get_segment_width(torch.from_numpy(segment), width, scale)
            density.draw_line_segment(segment[:, :3], width=line_width, mask=mask, channel=0)
        
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
    
    def draw_neuron(self, sections: Dict, shape: Tuple[int, ...], 
                   config: DrawingConfig, gif_config: Optional[GifConfig] = None) -> Image:
        """
        Draw complete neuron image with all effects.
        
        Args:
            sections: Dictionary of section_id -> segments
            shape: Output image shape
            config: Drawing configuration
            gif_config: Optional GIF generation config
            
        Returns:
            Rendered neuron image
        """
        segments = self._consolidate_segments(sections)
        if shape is None:
            shape = self._calculate_shape(segments)
        bg_mean = getattr(config, 'background_mean', 0.0)
        img = Image(torch.ones((1,) + shape) * bg_mean)

        gif_frames = []
        gif_steps = self._setup_gif_steps(len(segments), gif_config)

        # Optionally prepare correlated sequences for width and per-segment intensity
        n = len(segments)
        width_seq = None
        value_seq = None
        if config.width_correlation: # and abs(float(config.width_correlation_rho)) > 0.0:
            ar = self._generate_ar1_sequence(n, float(config.width_correlation_rho), start_val=0.0)
            base_w = float(config.width)
            # 50% variation around base width, clamp min at 2.0 px
            width_seq = np.clip(base_w + 0.5 * base_w * ar, 2.0, 14.0)
        if config.segment_intensity_correlation: # and abs(float(config.segment_intensity_correlation_rho)) > 0.0:
            ar = self._generate_ar1_sequence(n, float(config.segment_intensity_correlation_rho), start_val=0.0)
            fg_mean = float(config.foreground_mean)
            # 40% variation around fg_mean, clamp min at 0.15
            value_seq = np.clip(fg_mean + 0.4 * fg_mean * ar, 0.15, None)

        # Draw segments
        for idx, segment in enumerate(segments):
            width_override = float(width_seq[idx]) if width_seq is not None else None
            value_override = float(value_seq[idx]) if value_seq is not None else None
            self._draw_single_segment(img, segment, config, width_override=width_override, value_override=value_override)
            self._maybe_capture_gif_frame(img, idx, gif_steps, gif_frames, gif_config)
        
        # Compute masks once (based on raw drawn density) for post-processing
        peak = img.data.max()
        threshold = bg_mean + float(config.mask_threshold) * (float(peak) - bg_mean)
        foreground_mask = img.data > threshold

        # Apply post-processing
        # img = self._apply_color_effects(img, config)
        img = self._add_spatial_noise(img, config, foreground_mask)
        img = self._add_gaussian_noise(img, config, foreground_mask)
        img = self._add_vignette(img, config, foreground_mask)

        # Final clamp and optional Gaussian blur
        img.data = torch.clamp(img.data, 0.0, 1.0)
        cfg_blur = config.blur if hasattr(config, 'blur') else None
        if cfg_blur is not None and cfg_blur > 0.0:
            cpu_arr = img.data.detach().cpu().numpy()
            for c in range(cpu_arr.shape[0]):
                cpu_arr[c] = gaussian_filter(cpu_arr[c], sigma=cfg_blur, radius=1)
            img.data = torch.from_numpy(cpu_arr).to(device=img.data.device, dtype=img.data.dtype)

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
        img = self.draw_neuron(sections, shape, config)
        density = self.draw_density(sections, shape, width=config.width)
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
    
    def _setup_gif_steps(self, n_segments: int, gif_config: Optional[GifConfig]) -> set:
        """Setup which steps to capture for GIF."""
        if not gif_config or not gif_config.save_gif:
            return set()
        return set(np.linspace(0, n_segments-1, gif_config.gif_frames, dtype=int))

    def _generate_spatially_correlated_noise(self, shape: Tuple[int, ...], scale: float = 10.0, 
                                          method: str = "gaussian_convolution") -> torch.Tensor:
        """Generate spatially correlated 3D noise using stable methods.
        
        Args:
            shape: Output shape (depth, height, width)
            scale: Correlation length scale (higher = more clumpy)
            method: Method to use - "gaussian_convolution", "fractal", or "sparse_kernel"
            
        Returns:
            Normalized noise tensor in [-1, 1] range
        """
        depth, height, width = [int(s) for s in shape]
        
        if method == "gaussian_convolution":
            return self._gaussian_convolution_noise(depth, height, width, scale)
        elif method == "fractal":
            return self._fractal_noise(depth, height, width, scale)
        elif method == "sparse_kernel":
            return self._sparse_kernel_noise(depth, height, width, scale)
        else:
            # Fallback to Gaussian convolution
            return self._gaussian_convolution_noise(depth, height, width, scale)
    
    def _gaussian_convolution_noise(self, depth: int, height: int, width: int, scale: float) -> torch.Tensor:
        """Generate noise by convolving white noise with Gaussian kernel."""
        # Generate white noise
        noise = self.rng.normal(0, 1, (depth, height, width)).astype(np.float32)
        
        # Calculate sigma for desired correlation scale
        sigma = max(0.5, scale / 3.0)  # Scale controls correlation length
        
        # Apply 3D Gaussian filter for spatial correlation
        from scipy.ndimage import gaussian_filter
        correlated_noise = gaussian_filter(noise, sigma=sigma, mode='reflect')
        
        # Convert to tensor and normalize to [-1, 1]
        t = torch.from_numpy(correlated_noise)
        std_val = torch.std(t)
        if std_val > 0:
            t = t / (3 * std_val)  # Normalize to roughly [-1, 1] (3-sigma rule)
        return torch.clamp(t, -1.0, 1.0)
    
    def _fractal_noise(self, depth: int, height: int, width: int, scale: float) -> torch.Tensor:
        """Generate fractal noise using octave-based approach."""
        noise = torch.zeros((depth, height, width), dtype=torch.float32)
        
        # Multi-octave fractal noise
        amplitude = 1.0
        frequency = 1.0 / max(1.0, scale)
        
        for octave in range(4):  # 4 octaves for good detail
            # Generate noise at this frequency
            octave_noise = self._generate_octave_noise(depth, height, width, frequency)
            noise += amplitude * octave_noise
            
            # Update for next octave
            amplitude *= 0.5
            frequency *= 2.0
        
        # Normalize to [-1, 1]
        max_abs = torch.abs(noise).max()
        if max_abs > 0:
            noise = noise / max_abs
        return torch.clamp(noise, -1.0, 1.0)
    
    def _generate_octave_noise(self, depth: int, height: int, width: int, frequency: float) -> torch.Tensor:
        """Generate a single octave of noise using interpolated grid."""
        # Create a coarser grid based on frequency
        grid_scale = max(1, int(1.0 / frequency))
        grid_d = max(2, depth // grid_scale)
        grid_h = max(2, height // grid_scale)
        grid_w = max(2, width // grid_scale)
        
        # Generate random values at grid points
        grid_values = self.rng.normal(0, 1, (grid_d, grid_h, grid_w)).astype(np.float32)
        
        # Interpolate to full resolution using scipy
        from scipy.ndimage import zoom
        zoom_factors = (depth / grid_d, height / grid_h, width / grid_w)
        interpolated = zoom(grid_values, zoom_factors, order=1, mode='reflect')
        
        # Ensure exact output shape (zoom can sometimes be off by 1)
        if interpolated.shape != (depth, height, width):
            interpolated = interpolated[:depth, :height, :width]
            if interpolated.shape != (depth, height, width):
                # Pad if needed
                pad_d = depth - interpolated.shape[0]
                pad_h = height - interpolated.shape[1] 
                pad_w = width - interpolated.shape[2]
                interpolated = np.pad(interpolated, 
                                    ((0, pad_d), (0, pad_h), (0, pad_w)), 
                                    mode='reflect')
        
        return torch.from_numpy(interpolated)
    
    def _sparse_kernel_noise(self, depth: int, height: int, width: int, scale: float) -> torch.Tensor:
        """Generate noise using sparse random kernels - good for clumpy patterns."""
        noise = torch.zeros((depth, height, width), dtype=torch.float32)
        
        # Number of "clumps" based on volume and scale
        volume = depth * height * width
        n_clumps = max(10, int(volume / (scale ** 3)))
        
        # Kernel size based on scale
        kernel_size = max(3, int(scale // 2))
        
        for _ in range(n_clumps):
            # Random center position
            center_z = self.rng.integers(kernel_size, depth - kernel_size)
            center_y = self.rng.integers(kernel_size, height - kernel_size)
            center_x = self.rng.integers(kernel_size, width - kernel_size)
            
            # Random intensity
            intensity = self.rng.normal(0, 1)
            
            # Create Gaussian blob around center
            z_range = slice(max(0, center_z - kernel_size), min(depth, center_z + kernel_size))
            y_range = slice(max(0, center_y - kernel_size), min(height, center_y + kernel_size))
            x_range = slice(max(0, center_x - kernel_size), min(width, center_x + kernel_size))
            
            # Create coordinate grids for this region
            z_coords = torch.arange(z_range.start, z_range.stop, dtype=torch.float32) - center_z
            y_coords = torch.arange(y_range.start, y_range.stop, dtype=torch.float32) - center_y
            x_coords = torch.arange(x_range.start, x_range.stop, dtype=torch.float32) - center_x
            
            zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            
            # Gaussian kernel
            sigma = kernel_size / 3.0
            kernel = intensity * torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))
            
            # Add to noise
            noise[z_range, y_range, x_range] += kernel
        
        # Normalize to [-1, 1]
        max_abs = torch.abs(noise).max()
        if max_abs > 0:
            noise = noise / max_abs
        return torch.clamp(noise, -1.0, 1.0)
        
    def _add_spatial_noise(self, img: Image, config: DrawingConfig, foreground_mask: torch.Tensor) -> Image:
        """Add spatially correlated noise contribution to foreground/background means.

        Masks are precomputed once in draw_neuron and passed here for consistency.
        """
        background_mask = ~foreground_mask

        # Get noise method from config (with fallback)
        noise_method = getattr(config, 'noise_method', 'gaussian_convolution')
        
        # Generate spatially correlated noise (Z,Y,X) -> (C,Z,Y,X)
        noise_shape = img.data.shape[1:]
        spatial_noise = self._generate_spatially_correlated_noise(
            noise_shape, config.spatial_noise_scale, method=noise_method
        ).unsqueeze(0)

        result_img = torch.zeros_like(img.data)
        amp = float(getattr(config, 'spatial_noise_amplitude', 1.0))
        fg_mean = float(getattr(config, 'foreground_mean', 1.0))
        bg_mean = float(getattr(config, 'background_mean', 0.0))
        if foreground_mask.any():
            result_img[foreground_mask] = img.data[foreground_mask] + (amp * fg_mean * spatial_noise)[foreground_mask]
        if background_mask.any():
            result_img[background_mask] = img.data[background_mask] + (amp * bg_mean * spatial_noise)[background_mask]

        img.data = result_img
        return img

    def _add_gaussian_noise_tensor(
        self,
        img: torch.Tensor,
        foreground_mask: torch.Tensor,
        fg_std: float,
        bg_std: float,
    ) -> torch.Tensor:
        """Add zero-mean Gaussian noise with different std for foreground/background.

        Args:
            img: image tensor (C, Z, Y, X)
            foreground_mask: bool mask same shape as img
            background_mask: bool mask same shape as img
            fg_std: standard deviation for foreground gaussian noise
            bg_std: standard deviation for background gaussian noise
        Returns:
            Tensor of same shape as img with noise added.
        """
        if fg_std <= 0 and bg_std <= 0:
            return img
        device = img.device
        dtype = img.dtype
        fg_noise = torch.randn_like(img, device=device, dtype=dtype) * float(fg_std)
        bg_noise = torch.randn_like(img, device=device, dtype=dtype) * float(bg_std)
        gauss = torch.where(foreground_mask, fg_noise, bg_noise)
        return img + gauss

    def _add_gaussian_noise(self, img: Image, config: DrawingConfig, foreground_mask: torch.Tensor) -> Image:
        """Add Gaussian noise using foreground_std/background_std.
        """
        img.data = self._add_gaussian_noise_tensor(
            img=img.data,
            foreground_mask=foreground_mask,
            fg_std=float(config.foreground_std),
            bg_std=float(config.background_std),
        )
        return img

    def _add_vignette(
        self,
        img: Image,
        config: DrawingConfig,
        foreground_mask: torch.Tensor,
    ) -> Image:
        """Apply a vignette effect: attenuate intensity farther from the center.

        The vignette is applied multiplicatively as a smooth radial falloff. By default
        the background is dimmed slightly more than the foreground to preserve neuron
        visibility while still creating a realistic peripheral fade.

        Args:
            img: Image object (C, Z, Y, X) in [0, 1] range typically.
            foreground_mask: Bool mask (C, Z, Y, X) identifying foreground voxels.
            magnitude: Base vignette strength in [0, 1]. 0 disables the effect.

        Returns:
            Image with vignette applied in-place and returned.
        """
        mag = float(max(0.0, config.vignette_magnitude))
        if mag <= 0.0:
            return img

        # Build a normalized radial distance map r in [0, ~1] with shape (Z, Y, X)
        _, Z, Y, X = img.data.shape
        device = img.data.device
        dtype = img.data.dtype

        z = torch.linspace(0, Z - 1, Z, device=device, dtype=dtype)
        y = torch.linspace(0, Y - 1, Y, device=device, dtype=dtype)
        x = torch.linspace(0, X - 1, X, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

        cz = (Z - 1) / 2.0
        cy = (Y - 1) / 2.0
        cx = (X - 1) / 2.0
        # Normalize distances along each axis by half-extent, then combine
        rz = (zz - cz) / max(1.0, cz)
        ry = (yy - cy) / max(1.0, cy)
        rx = (xx - cx) / max(1.0, cx)
        r = torch.sqrt(rz * rz + ry * ry + rx * rx)
        # Scale to roughly [0, 1] across the volume (sqrt(3) at corners)
        r = torch.clamp(r / np.sqrt(3.0), 0.0, 1.0)

        # Smooth falloff curve; higher exponent keeps center brighter
        exponent = 1.5
        base_falloff = r.pow(exponent)

        # Compute per-voxel attenuation factors for FG and BG
        # Apply slightly weaker vignette on foreground to retain neuron contrast
        fg_mag = 0.6 * mag
        bg_mag = mag
        factor_fg = torch.clamp(1.0 - fg_mag * base_falloff, 0.0, 1.0)
        factor_bg = torch.clamp(1.0 - bg_mag * base_falloff, 0.0, 1.0)

        # Broadcast factors to (C, Z, Y, X)
        factor_fg = factor_fg.unsqueeze(0).expand(img.data.shape)
        factor_bg = factor_bg.unsqueeze(0).expand(img.data.shape)

        factors = torch.where(foreground_mask, factor_fg, factor_bg)
        img.data = img.data * factors
        return img
    
    def _draw_single_segment(self, img: Image, segment: np.ndarray, 
                           config: DrawingConfig, width_override: Optional[float] = None, value_override: Optional[float] = None) -> None:
        """Draw a single segment with all effects."""
        # Calculate segment properties
        value = value_override if value_override is not None else self._get_segment_value(config)
        width = width_override if width_override is not None else self._get_segment_width(torch.from_numpy(segment), config.width)
        
        # Draw the segment
        img.draw_line_segment(segment[:, :3], width=width, mask=True, channel=0, value=value, sharpness=getattr(config, 'sharpness', 2.0))

    def _generate_ar1_sequence(self, n: int, rho: float, start_val: Optional[float] = None) -> np.ndarray:
        """Generate an AR(1) sequence with lag-1 correlation rho and unit variance.

        x_t = rho * x_{t-1} + sqrt(1-rho^2) * eps_t, eps_t ~ N(0,1), x_0 ~ N(0,1)
        """
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=float)
        rho = float(rho)
        seq = np.zeros((n,), dtype=float)
        seq[0] = start_val if start_val is not None else self.rng.normal(0.0, 1.0)
        sigma = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        eps = self.rng.normal(0.0, 1.0, size=n-1) if n > 1 else np.array([], dtype=float)
        for t in range(1, n):
            seq[t] = rho * seq[t-1] + sigma * eps[t-1]
        return seq

    def _get_segment_value(self, config: DrawingConfig, default_value: int = 1.0) -> float:
        """Get brightness value for segment."""
        if config.foreground_mean:
            value = config.foreground_mean
        else:
            value = default_value
        return value

    def _get_segment_width(self, segment: torch.Tensor, default_width: Optional[float] = None, 
                          scale: float = 1.0) -> float:
        """Get width for a segment, handling various cases."""
        if default_width is not None:
            return default_width
        elif segment.shape[1] == 4:  # Width included in segment data
            return ((segment[0, 3] + segment[1, 3]) / 2).item() / scale
        else:
            return 3.0

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
        pass  # Not implemented
            
        # Apply neuron color
        if config.neuron_color:
            neuron_color = torch.tensor(config.neuron_color, dtype=torch.float32)
            img.data = neuron_color[:, None, None, None] * img.data
        
        # Apply background color
        if config.background_color:
            background_color = torch.tensor(config.background_color, dtype=torch.float32)
            img.data = img.data + torch.ones_like(img.data) * background_color[:, None, None, None]
        
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


if __name__ == "__main__":
    # Minimal example
    renderer = NeuronRenderer()
    config = DrawingConfig(width=5.0, rgb=True)
    print("NeuronRenderer ready for use!")
