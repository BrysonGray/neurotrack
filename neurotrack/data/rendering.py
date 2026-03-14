"""
Neuron drawing and rendering utilities.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from skimage.filters import gaussian
import torch
import imageio
from scipy.ndimage import gaussian_filter, median_filter

from neurotrack.data.image import Image, to_uint8
from neurotrack.data import loading as load


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

    @staticmethod
    def _coerce_segments(sections: Union[Dict, np.ndarray]) -> np.ndarray:
        """Normalize section-like inputs to an array with shape (N, 2, >=3)."""
        if isinstance(sections, dict):
            if len(sections) == 0:
                return np.empty((0, 2, 4), dtype=np.float32)
            segments = np.concatenate([section for section in sections.values()])
        else:
            segments = np.asarray(sections)

        if segments.size == 0:
            return np.empty((0, 2, 4), dtype=np.float32)
        if segments.ndim != 3 or segments.shape[1] != 2 or segments.shape[2] < 3:
            raise ValueError(f"Expected segments with shape (N, 2, >=3), got {tuple(segments.shape)}")

        return segments

    def draw_density(self, sections: Union[Dict, np.ndarray], shape: Tuple[int, ...], 
                    width: Optional[float] = 3.0, mask: bool = False) -> Image:
        """
        Draw neuron density image from sections.
        
        Args:
            sections: Dictionary of section_id -> segments or array-like segments
            shape: Output image shape
            width: Line width (defaults to 3.0)
            mask: Whether to create a binary mask
            scale: Pixel scale factor
            
        Returns:
            Image object with neuron density
        """
        density = Image(torch.zeros((1,) + shape))
        segments = self._coerce_segments(sections)
        if segments.size == 0:
            return density

        segments_t = torch.as_tensor(segments[:, :, :3], dtype=torch.float32)
        for segment in segments_t:
            density.draw_line_segment(segment, width=width, mask=mask, channel=0)

        return density

    def draw_density_pair(
        self,
        sections: Union[Dict, np.ndarray],
        shape: Tuple[int, ...],
        widths: Tuple[float, float],
        mask: bool = False,
    ) -> Tuple[Image, Image]:
        """Draw two density images over the same segments in one segment traversal."""
        density_a = Image(torch.zeros((1,) + shape))
        density_b = Image(torch.zeros((1,) + shape))
        segments = self._coerce_segments(sections)
        if segments.size == 0:
            return density_a, density_b

        segments_t = torch.as_tensor(segments[:, :, :3], dtype=torch.float32)
        width_a, width_b = widths
        for segment in segments_t:
            density_a.draw_line_segment(segment, width=width_a, mask=mask, channel=0)
            density_b.draw_line_segment(segment, width=width_b, mask=mask, channel=0)

        return density_a, density_b
    
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
                segment_width = self._get_segment_width(segment, default_width=width)
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
    
    def draw_neuron(self, sections: Dict, 
                   config: DrawingConfig,
                   shape: Optional[Tuple[int, ...]] = None,
                   gif_config: Optional[GifConfig] = None) -> Image:
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
        segments = np.concatenate([section for section in sections.values()])
        if shape is None:
            shape = self._calculate_shape(segments)
        img = Image(torch.zeros((1,) + shape))

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
            width_override = float(width_seq[idx]) if width_seq is not None else getattr(config, 'width', 3.0)
            value_override = float(value_seq[idx]) if value_seq is not None else getattr(config, 'foreground_mean', 1.0)
            img.draw_line_segment(segment[:, :3], width=width_override, mask=True, channel=0, value=value_override)
            self._maybe_capture_gif_frame(img, idx, gif_steps, gif_frames, gif_config)
        
        # Compute masks once (based on raw drawn density) for post-processing
        foreground_mask = img.data > 0.0
        bg_mean = getattr(config, 'background_mean', 0.0)
        img.data[~foreground_mask] = bg_mean


        # Apply post-processing
        img = self._add_combined_noise(img, config, foreground_mask)
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
        
        sigma = max(0.1, scale)
        noise = gaussian_filter(noise, sigma=sigma, mode='reflect')
        # Standardize to zero mean, unit variance
        noise = noise - noise.mean()
        std = noise.std()
        if std > 1e-6:
            noise = noise / std
        else:
            noise = np.zeros_like(noise)

        noise = torch.from_numpy(noise)

        return noise
    
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
        
    def _add_combined_noise(self, img: Image, config: DrawingConfig, foreground_mask: torch.Tensor) -> Image:
        """Add combined spatially correlated and Gaussian noise.
        
        Ensures total noise has zero mean and approximates target standard deviation.
        """
        noise_shape = img.data.shape[1:]
        noise_method = getattr(config, 'noise_method', 'gaussian_convolution')
        spatial_noise = self._generate_spatially_correlated_noise(
            noise_shape, config.spatial_noise_scale, method=noise_method
        ).unsqueeze(0) # (1, Z, Y, X)
            
        gaussian_noise = torch.randn_like(img.data)
        
        # Combine and scale for Foreground and Background
        # We want Total = k * (w_s * S + w_g * G)
        # Var(Total) = k^2 * (w_s^2 + w_g^2) = target_std^2
        # Let w_s = spatial_noise_amplitude, w_g = 1.0
        # k = target_std / sqrt(w_s^2 + 1)
        
        w_s = float(getattr(config, 'spatial_noise_amplitude', 1.0))
        w_g = 1.0
        norm_factor = np.sqrt(w_s**2 + w_g**2)
        
        # Foreground
        fg_std = float(getattr(config, 'foreground_std', 0.1))
        fg_k = fg_std / norm_factor if norm_factor > 0 else 0
        
        # Background
        bg_std = float(getattr(config, 'background_std', 0.05))
        bg_k = bg_std / norm_factor if norm_factor > 0 else 0
        
        # Calculate total noise field
        total_noise = (w_s * spatial_noise + w_g * gaussian_noise)
        
        # Apply scaling
        final_noise = torch.zeros_like(img.data)
        
        if foreground_mask.any():
            final_noise[foreground_mask] = (total_noise * fg_k)[foreground_mask]
            
        background_mask = ~foreground_mask
        if background_mask.any():
            final_noise[background_mask] = (total_noise * bg_k)[background_mask]
            
        # Add to image
        img.data = img.data + final_noise
        
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
    
    def _maybe_capture_gif_frame(self, img: Image, idx: int, gif_steps: set, 
                               gif_frames: List, gif_config: Optional[GifConfig]) -> None:
        """Capture GIF frame if needed."""
        if gif_config and idx in gif_steps:
            arr = img.data[0].cpu().numpy()  # Fixed index
            mip = arr.max(axis=gif_config.gif_axis)
            mip = to_uint8(mip)
            gif_frames.append(mip)
    
    def _apply_color_effects(self, img: Image, config: DrawingConfig) -> Image:
        """Apply RGB color effects."""
        pass  # Not implemented    
    
    def _calculate_shape(self, segments: np.ndarray) -> Tuple[int, ...]:
        """Calculate image shape from segments."""
        shape = np.ceil(np.max(segments[..., :3], axis=(0, 1)))
        shape = shape.astype(np.uint16)
        shape = shape + np.array([10, 10, 10])
        return tuple(shape.tolist())
    
    def _maybe_save_gif(self, gif_frames: List, gif_config: Optional[GifConfig]) -> None:
        """Save GIF if frames were captured."""
        if gif_config and gif_config.save_gif and gif_config.gif_path and gif_frames:
            imageio.mimsave(gif_config.gif_path, gif_frames, duration=0.05)


if __name__ == "__main__":
    # Minimal example
    renderer = NeuronRenderer()
    config = DrawingConfig(width=5.0, rgb=True)
    print("NeuronRenderer ready for use!")
