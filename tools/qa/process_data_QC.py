#!/usr/bin/env python3
"""
Process TIFF image volumes and generate maximum intensity projections.
Creates a JSON manifest for the viewer and saves all images to the images/ folder.
"""

import os
import json
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image

from neurotrack.data.loading import swc, parse_swc, map_tiff_to_swc
from neurotrack.data.rendering import NeuronRenderer


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


def process_directory(tiff_dir, swc_dir, section_title, images_output_dir):
    """
    Process a directory of TIFF files and return image data.
    
    Args:
        tiff_dir: Directory containing TIFF files
        swc_dir: Directory containing SWC morphology files
        section_title: Title for this section (e.g., "Training Set")
        images_output_dir: Directory to save image files
    
    Returns:
        Dictionary with section metadata and list of image data
    """
    tiff_dir = Path(tiff_dir)
    swc_dir = Path(swc_dir)
    tiff_to_swc_map = map_tiff_to_swc(tiff_dir, swc_dir, use_fixed=False, verbose=True)
    tiff_files = sorted([f for f in tiff_dir.rglob("*.tif") if f.is_file()])
    GOLD166_CONVERTED_PATH = Path('/home/brysongray/data/neurotrack_data/gold166/gold166_converted/')
    TIFF_NAME_TO_DATASET = {f.name: f.parent.parent.name for f in GOLD166_CONVERTED_PATH.rglob('*.tif') if f.is_file()}
    
    if not tiff_files:
        print(f"No TIFF files found in {tiff_dir}")
        return {
            "title": section_title,
            "directory": str(tiff_dir),
            "images": []
        }
    
    # Initialize renderer
    renderer = NeuronRenderer(rng=np.random.default_rng())
    
    print(f"\n{section_title}: Found {len(tiff_files)} TIFF files")
    
    section_data = {
        "title": section_title,
        "directory": str(tiff_dir),
        "images": []
    }
    
    # Process each TIFF file
    for idx, tiff_path in enumerate(tiff_files, 1):
        print(f"  Processing {idx}/{len(tiff_files)}: {tiff_path.name}")
        
        try:
            # Load TIFF volume
            volume = tifffile.imread(str(tiff_path))
            shape = volume.shape
            # Remove singleton dimension if present
            if len(shape) != 3:
                if len(shape) == 4 and shape[0] == 1:
                    volume = volume[0]
                    shape = volume.shape
                else:
                    raise ValueError(f"Expected 3D volume, got shape {shape}")
            
            # Find matching SWC file
            tiff_stem = tiff_path.stem
            # Remove '_image' suffix if present
            if tiff_stem.endswith('_image'):
                tiff_stem = tiff_stem[:-6]

            swc_file = tiff_to_swc_map.get(tiff_path, None)
            dataset_name = TIFF_NAME_TO_DATASET.get(tiff_path.name, "Unknown Dataset")
            
            if not swc_file:
                print(f"    Warning: No matching SWC file found for {tiff_path.name}")
                # Continue without mask
                mip_axis0 = compute_mip_along_axis(volume, axis=0)
                mip_axis1 = compute_mip_along_axis(volume, axis=1)
                mip_axis2 = compute_mip_along_axis(volume, axis=2)
                
                # Save images to files
                base_name = tiff_path.stem
                img_file_axis0 = save_image_to_file(mip_axis0, images_output_dir, f"{base_name}_axis0.png")
                img_file_axis1 = save_image_to_file(mip_axis1, images_output_dir, f"{base_name}_axis1.png")
                img_file_axis2 = save_image_to_file(mip_axis2, images_output_dir, f"{base_name}_axis2.png")
                
                # Add image data to section
                section_data["images"].append({
                    "filename": tiff_path.name,
                    "dataset_name": dataset_name,
                    "file_stem": tiff_stem,
                    "shape": str(shape),
                    "swc_file": None,
                    "img_axis0": img_file_axis0,
                    "img_axis1": img_file_axis1,
                    "img_axis2": img_file_axis2,
                    "mask_axis0": None,
                    "mask_axis1": None,
                    "mask_axis2": None,
                    "shape_axis0": str(mip_axis0.shape),
                    "shape_axis1": str(mip_axis1.shape),
                    "shape_axis2": str(mip_axis2.shape),
                })
                continue
            
            print(f"    Matched with SWC: {swc_file.name}")
            
            # Load SWC and generate sections
            swc_data = swc(swc_file, rotate=False, verbose=False)
            sections, _ = parse_swc(swc_data, verbose=False, transpose=True)
            
            # Compute MIPs of the image first
            mip_img_axis0 = compute_mip_along_axis(volume, axis=0)
            mip_img_axis1 = compute_mip_along_axis(volume, axis=1)
            mip_img_axis2 = compute_mip_along_axis(volume, axis=2)
            
            # Generate 2D masks directly for each MIP projection (MUCH faster than 3D)
            # For axis 0 projection (project along Z), draw in XY plane
            mip_mask_axis0 = draw_2d_projection_mask(sections, mip_img_axis0.shape, axis=0, width=3.0)
            mip_mask_axis1 = draw_2d_projection_mask(sections, mip_img_axis1.shape, axis=1, width=3.0)
            mip_mask_axis2 = draw_2d_projection_mask(sections, mip_img_axis2.shape, axis=2, width=3.0)
            
            # Save images and masks to files
            base_name = tiff_path.stem
            img_file_axis0 = save_image_to_file(mip_img_axis0, images_output_dir, f"{base_name}_axis0.png")
            img_file_axis1 = save_image_to_file(mip_img_axis1, images_output_dir, f"{base_name}_axis1.png")
            img_file_axis2 = save_image_to_file(mip_img_axis2, images_output_dir, f"{base_name}_axis2.png")
            
            mask_file_axis0 = save_mask_to_file_rgba(mip_mask_axis0, images_output_dir, f"{base_name}_mask_axis0.png")
            mask_file_axis1 = save_mask_to_file_rgba(mip_mask_axis1, images_output_dir, f"{base_name}_mask_axis1.png")
            mask_file_axis2 = save_mask_to_file_rgba(mip_mask_axis2, images_output_dir, f"{base_name}_mask_axis2.png")
            
            # Add image data to section
            section_data["images"].append({
                "filename": tiff_path.name,
                "dataset_name": dataset_name,
                "file_stem": tiff_stem,
                "shape": str(shape),
                "swc_file": swc_file.name,
                "img_axis0": img_file_axis0,
                "img_axis1": img_file_axis1,
                "img_axis2": img_file_axis2,
                "mask_axis0": mask_file_axis0,
                "mask_axis1": mask_file_axis1,
                "mask_axis2": mask_file_axis2,
                "shape_axis0": str(mip_img_axis0.shape),
                "shape_axis1": str(mip_img_axis1.shape),
                "shape_axis2": str(mip_img_axis2.shape),
            })
            
        except Exception as e:
            print(f"  Error processing {tiff_path.name}: {e}")
            import traceback
            traceback.print_exc()
            # Add error entry to data
            section_data["images"].append({
                "filename": tiff_path.name,
                "dataset_name": TIFF_NAME_TO_DATASET.get(tiff_path.name, "Unknown Dataset"),
                "file_stem": tiff_path.stem[:-6] if tiff_path.stem.endswith('_image') else tiff_path.stem,
                "error": str(e)
            })
    
    return section_data


def process_dataset(dataset_configs, output_dir="../data_QC"):
    """
    Process TIFF datasets and generate image data manifest.
    
    Args:
        dataset_configs: List of tuples (tiff_dir, swc_dir, section_title)
        output_dir: Output directory for images and manifest (default: ../data_QC)
    """
    
    # Create output directories
    output_dir = os.path.abspath(output_dir)
    images_output_dir = os.path.join(output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Process each directory and collect data
    all_sections = []
    for tiff_dir, swc_dir, section_title in dataset_configs:
        section_data = process_directory(tiff_dir, swc_dir, section_title, images_output_dir)
        all_sections.append(section_data)
    
    # Generate JSON manifest
    manifest = {
        "title": "TIFF Volume Viewer",
        "description": "Orthogonal MIP projections with neuron masks",
        "sections": all_sections
    }
    
    manifest_path = os.path.join(output_dir, "image_data.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nImage data manifest saved to: {manifest_path}")
    print(f"Total images processed: {sum(len(s['images']) for s in all_sections)}")
    print(f"\nTo view the results:")
    print(f"  1. Open {os.path.join(output_dir, 'viewer.html')} in a web browser")
    print(f"  2. Make sure viewer.html, viewer.css, viewer.js, and image_data.json are in {output_dir}")
    print(f"  3. The images/ folder should be in {output_dir} as well")


if __name__ == "__main__":
    import sys
    
    # Default directories to process (tiff_dir, swc_dir, title)
    # dataset_configs = [
    #     (
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_training_set/images",
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_training_set/morphology",
    #         "Training Set"
    #     ),
    #     (
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_validation_set/images",
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_validation_set/morphology",
    #         "Validation Set"
    #     ),
    #     (
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_test_set/images",
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_test_set/morphology",
    #         "Test Set"
    #     )
    # ]

    dataset_configs = [
        (
            "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/images",
            "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/morphology",
            "Whole Set"
        )
    ]
    
    # Output to outputs/data_QC directory
    output_dir = str(Path(__file__).resolve().parents[2] / "outputs" / "data_QC")
    
    process_dataset(dataset_configs, output_dir)
