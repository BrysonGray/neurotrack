import os
import numpy as np
from PIL import Image

from neurotrack.data.loading import swc, parse_swc, map_tiff_to_swc


def compute_mip_along_axis(volume, axis):
    """Compute maximum intensity projection along specified axis."""
    return np.max(volume, axis=axis)


def normalize_for_display(image_2d):
    """Normalize 2D image to 0-255 range for display."""
    img_min = image_2d.min()
    img_max = image_2d.max()
    if img_max > img_min:
        normalized = ((image_2d - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image_2d, dtype=np.uint8)
    return normalized


def save_image_to_file(image_2d, output_dir, filename):
    """Save 2D numpy array as PNG file."""
    normalized = normalize_for_display(image_2d)
    pil_img = Image.fromarray(normalized)
    output_path = os.path.join(output_dir, filename)
    pil_img.save(output_path, format='PNG')
    return filename


def save_mask_to_file_rgba(mask_2d, output_dir, filename, color=(255, 0, 0)):
    """Save 2D mask as RGBA PNG file with specified color."""
    # Handle channel dimension if present - squeeze all singleton dimensions
    while len(mask_2d.shape) > 2:
        mask_2d = np.squeeze(mask_2d)
    
    # Normalize mask to 0-255
    mask_normalized = normalize_for_display(mask_2d)
    
    # Create RGBA image (red mask)
    h, w = mask_normalized.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = color[0]  # Red channel
    rgba[:, :, 1] = color[1]  # Green channel  
    rgba[:, :, 2] = color[2]  # Blue channel
    rgba[:, :, 3] = mask_normalized  # Alpha channel (mask intensity)
    
    pil_img = Image.fromarray(rgba, mode='RGBA')
    output_path = os.path.join(output_dir, filename)
    pil_img.save(output_path, format='PNG')
    return filename


def draw_2d_projection_mask(sections, shape_2d, axis, width=3.0):
    """
    Draw a 2D mask by projecting neuron segments onto a 2D plane.
    Much faster than generating a full 3D mask.
    
    Args:
        sections: Dictionary of section_id -> segments array
        shape_2d: Shape of the 2D output (height, width)
        axis: Which axis to project along (0, 1, or 2)
        width: Line width for drawing
    
    Returns:
        2D numpy array mask
    """
    mask = np.zeros(shape_2d, dtype=np.float32)
    
    # Axis mapping: which dimensions to keep for projection
    # axis 0: project along Z, keep Y,X -> dimensions [1,2]
    # axis 1: project along Y, keep Z,X -> dimensions [0,2]  
    # axis 2: project along X, keep Z,Y -> dimensions [0,1]
    dim_map = {
        0: (1, 2),  # Y, X
        1: (0, 2),  # Z, X
        2: (0, 1)   # Z, Y
    }
    d1, d2 = dim_map[axis]
    
    # Draw each section
    for section_id, segments in sections.items():
        for segment in segments:
            # segment has shape (N, 3) for N points along the segment
            # Each point is [x, y, z] but we need to project to 2D
            for i in range(len(segment) - 1):
                p1 = segment[i]
                p2 = segment[i + 1]
                
                # Project to 2D by selecting the relevant dimensions
                y1, x1 = int(p1[d1]), int(p1[d2])
                y2, x2 = int(p2[d1]), int(p2[d2])
                
                # Draw line between points using Bresenham's algorithm
                draw_line_2d(mask, y1, x1, y2, x2, width)
    
    return mask


def draw_line_2d(mask, y1, x1, y2, x2, width):
    """Draw a thick line on a 2D mask using simple rasterization."""
    # Bresenham's line algorithm with thickness
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    h, w = mask.shape
    radius = int(width / 2)
    
    while True:
        # Draw a circle at current point
        for dy_offset in range(-radius, radius + 1):
            for dx_offset in range(-radius, radius + 1):
                if dy_offset**2 + dx_offset**2 <= radius**2:
                    ny, nx = y1 + dy_offset, x1 + dx_offset
                    if 0 <= ny < h and 0 <= nx < w:
                        mask[ny, nx] = 1.0
        
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy