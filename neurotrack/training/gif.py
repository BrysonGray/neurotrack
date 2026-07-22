"""
Functions to save animated GIFs of model tracing progress during training.
"""

import os
from pathlib import Path
import torch
from typing import List

from neurotrack.data.image import Image, to_uint8
from PIL import Image as PILImage
import math
import numpy as np


def trace_gif(image: torch.Tensor, paths: List[torch.Tensor], step_width: float, output_path: str, n_frames: int) -> None:

    image = image.view(-1, *image.shape[-3:])
    if not image.ndim == 4:
        raise ValueError("Image must be 4D tensor (C, D, H, W)")
    if not image.shape[0] in [1, 3]:
        raise ValueError("Image must have 1 or 3 channels")
    
    # Add channel for path visualization
    image = torch.cat((
        image, 
        torch.zeros((1,) + image.shape[1:], dtype=image.dtype)
    ), dim=0)

    img = Image(image)
    frames = []
    # path = torch.cat(paths, dim=0)
    n_segments = sum(len(p)-1 for p in paths)
    if n_segments < 1:
        return
    else:
        segments_per_frame = math.ceil(n_segments / n_frames) if n_frames > 0 else n_segments
        segments_drawn = 0
    for path in paths:
        segments = torch.stack((path[:-1], path[1:]), dim=1)  # (N-1, 2, 3)
        for segment in segments:
            img.draw_line_segment(segment, width=step_width, channel=-1, mask=True)
            segments_drawn += 1
            if (segments_drawn % segments_per_frame != 0 and segments_drawn != n_segments):
                continue
            else:
                # compose a 2D RGB frame from current 3D image tensor
                data = img.data.clone().cpu()
                frames.append(_compose_frame_from_image_tensor(data))

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as animated GIF
    if len(frames) == 0:
        # no frames to save
        return
    elif len(frames) == 1:
        frames[0].save(output_path)
    else:
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def _compose_frame_from_image_tensor(data: torch.Tensor) -> PILImage:
    """Compose a single HxWx3 PIL image from an Image.data tensor.

    Expects data shape (C, D, H, W) or (C, H, W) where the last channel is the path overlay.
    Performs a max-projection over the depth axis and overlays the path channel as an
    opaque red top layer.
    """
    # Handle 3D vs 4D
    if data.ndim == 3:
        # (C, H, W)
        C, H, W = data.shape
        # treat as single-slice volume
        base = data[:-1] if C > 1 else data[:1]
        path = data[-1]
        # expand depth dimension for uniform handling
        base_mip = base
    elif data.ndim == 4:
        C, D, H, W = data.shape
        base = data[:-1]
        path = data[-1]
        # max projection over depth for each base channel
        if base.shape[0] == 0:
            # no base channels, create zeros
            base_mip = torch.zeros((1, H, W), dtype=data.dtype)
        else:
            base_mip = torch.amax(base, dim=1)
        path = torch.amax(path, dim=0)
    else:
        raise ValueError(f"Unsupported tensor shape for composing frame: {data.shape}")

    # ensure base_mip has channel dimension
    if base_mip.ndim == 2:
        base_mip = base_mip.unsqueeze(0)

    # normalize to uint8
    base_np = to_uint8(base_mip).numpy()

    if isinstance(path, torch.Tensor):
        path_np = to_uint8(path).numpy()

    # compose RGB
    # base_np shape: (Cbase, H, W)
    Cbase = base_np.shape[0]
    if Cbase >= 3:
        rgb = np.stack([base_np[0], base_np[1], base_np[2]], axis=-1)
    else:
        # replicate grayscale
        gray = base_np[0]
        rgb = np.stack([gray, gray, gray], axis=-1)

    # Draw the traced path on top of the base image so it remains fully visible.
    path_img = path_np if path_np.ndim == 2 else path_np[0]
    path_mask = path_img > 0
    rgb[..., 0] = np.where(path_mask, path_img, rgb[..., 0])
    rgb[..., 1] = np.where(path_mask, 0, rgb[..., 1])
    rgb[..., 2] = np.where(path_mask, 0, rgb[..., 2])

    return PILImage.fromarray(rgb.astype(np.uint8))