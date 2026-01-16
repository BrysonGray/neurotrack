#!/usr/bin/env python3
"""
Generate an HTML viewer for TIFF image volumes with maximum intensity projections.
Displays MIP along the first (0th) dimension with shape and filename info.
"""

import os
import base64
from io import BytesIO
from pathlib import Path
import sys
import numpy as np
import tifffile
from PIL import Image

# Add neurotrack to path to import data_prep modules directly
sys.path.insert(0, '/home/brysongray/neurotrack')
from data_prep.load import swc, parse_swc
from data_prep.draw import NeuronRenderer


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


def image_to_base64(image_2d):
    """Convert 2D numpy array to base64 encoded PNG string."""
    normalized = normalize_for_display(image_2d)
    pil_img = Image.fromarray(normalized)
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def mask_to_base64_rgba(mask_2d, color=(255, 0, 0)):
    """Convert 2D mask to base64 encoded RGBA PNG with specified color."""
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
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


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


def process_directory(tiff_dir, swc_dir, html_parts, section_title):
    """
    Process a directory of TIFF files and add to HTML parts.
    
    Args:
        tiff_dir: Directory containing TIFF files
        swc_dir: Directory containing SWC morphology files
        html_parts: List to append HTML strings to
        section_title: Title for this section (e.g., "Training Set")
    """
    tiff_dir = Path(tiff_dir)
    swc_dir = Path(swc_dir)
    tiff_files = sorted(tiff_dir.glob("*.tif*"))
    
    if not tiff_files:
        print(f"No TIFF files found in {tiff_dir}")
        html_parts.extend([
            f"    <h2 style='text-align: center; color: #e74c3c; margin-top: 40px;'>{section_title}</h2>",
            f"    <p style='text-align: center; color: #7f8c8d;'>No TIFF files found in {tiff_dir}</p>",
        ])
        return
    
    # Initialize renderer
    renderer = NeuronRenderer(rng=np.random.default_rng())
    
    print(f"\n{section_title}: Found {len(tiff_files)} TIFF files")
    
    html_parts.extend([
        f"    <h2 style='text-align: center; color: #2c3e50; margin-top: 40px; border-top: 3px solid #3498db; padding-top: 20px;'>{section_title}</h2>",
        f"    <p style='text-align: center; color: #7f8c8d;'>Directory: {tiff_dir}</p>",
        f"    <p style='text-align: center; color: #7f8c8d;'>Total files: {len(tiff_files)}</p>",
    ])
    
    # Process each TIFF file
    for idx, tiff_path in enumerate(tiff_files, 1):
        print(f"  Processing {idx}/{len(tiff_files)}: {tiff_path.name}")
        
        try:
            # Load TIFF volume
            volume = tifffile.imread(str(tiff_path))
            shape = volume.shape
            
            # Find matching SWC file
            tiff_stem = tiff_path.stem
            # Remove '_image' suffix if present
            if tiff_stem.endswith('_image'):
                tiff_stem = tiff_stem[:-6]
            
            swc_files = list(swc_dir.glob("*.swc"))
            matching_swc = [f for f in swc_files if f.stem == tiff_stem or f.stem.startswith(tiff_stem + "_")]
            
            if not matching_swc:
                print(f"    Warning: No matching SWC file found for {tiff_path.name}")
                # Continue without mask
                mip_axis0 = compute_mip_along_axis(volume, axis=0)
                mip_axis1 = compute_mip_along_axis(volume, axis=1)
                mip_axis2 = compute_mip_along_axis(volume, axis=2)
                
                img_base64_axis0 = image_to_base64(mip_axis0)
                img_base64_axis1 = image_to_base64(mip_axis1)
                img_base64_axis2 = image_to_base64(mip_axis2)
                
                html_parts.extend([
                    "    <div class='image-container'>",
                    f"        <div class='filename'>{tiff_path.name}</div>",
                    f"        <div class='shape-info'>Volume Shape: {shape} | <span style='color: orange;'>No SWC file found</span></div>",
                    "        <div class='mip-grid'>",
                    "            <div class='mip-column'>",
                    f"                <div class='mip-label'>Axis 0 Projection<br/>Shape: {mip_axis0.shape}</div>",
                    f"                <img class='mip-image' src='data:image/png;base64,{img_base64_axis0}' alt='{tiff_path.name} axis 0'>",
                    "            </div>",
                    "            <div class='mip-column'>",
                    f"                <div class='mip-label'>Axis 1 Projection<br/>Shape: {mip_axis1.shape}</div>",
                    f"                <img class='mip-image' src='data:image/png;base64,{img_base64_axis1}' alt='{tiff_path.name} axis 1'>",
                    "            </div>",
                    "            <div class='mip-column'>",
                    f"                <div class='mip-label'>Axis 2 Projection<br/>Shape: {mip_axis2.shape}</div>",
                    f"                <img class='mip-image' src='data:image/png;base64,{img_base64_axis2}' alt='{tiff_path.name} axis 2'>",
                    "            </div>",
                    "        </div>",
                    "    </div>",
                ])
                continue
            
            if len(matching_swc) > 1:
                print(f"    Warning: Multiple matching SWC files found for {tiff_path.name}, using first: {matching_swc[0].name}")
            
            swc_file = matching_swc[0]
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
            
            # Convert to base64 - images
            img_base64_axis0 = image_to_base64(mip_img_axis0)
            img_base64_axis1 = image_to_base64(mip_img_axis1)
            img_base64_axis2 = image_to_base64(mip_img_axis2)
            
            # Convert to base64 - masks (in red color)
            mask_base64_axis0 = mask_to_base64_rgba(mip_mask_axis0)
            mask_base64_axis1 = mask_to_base64_rgba(mip_mask_axis1)
            mask_base64_axis2 = mask_to_base64_rgba(mip_mask_axis2)
            
            # Create unique IDs for this row
            row_id = f"row_{section_title.replace(' ', '_')}_{idx}"
            
            # Add to HTML with overlay support
            html_parts.extend([
                "    <div class='image-container'>",
                f"        <div class='filename'>{tiff_path.name}</div>",
                f"        <div class='shape-info'>Volume Shape: {shape} | SWC: {swc_file.name}</div>",
                "        <div class='slider-container'>",
                f"            <label for='slider_{row_id}'>Neuron Mask Opacity: <span id='value_{row_id}'>0.5</span></label>",
                f"            <input type='range' min='0' max='100' value='50' class='opacity-slider' id='slider_{row_id}' ",
                f"                   oninput='updateOpacity(\"{row_id}\", this.value)'>",
                "        </div>",
                "        <div class='mip-grid'>",
                "            <div class='mip-column'>",
                f"                <div class='mip-label'>Axis 0 Projection<br/>Shape: {mip_img_axis0.shape}</div>",
                f"                <div class='image-stack'>",
                f"                    <img class='mip-image base-image' src='data:image/png;base64,{img_base64_axis0}' alt='{tiff_path.name} axis 0'>",
                f"                    <img class='mip-image overlay-image' id='mask_{row_id}_axis0' src='data:image/png;base64,{mask_base64_axis0}' alt='mask'>",
                "                </div>",
                "            </div>",
                "            <div class='mip-column'>",
                f"                <div class='mip-label'>Axis 1 Projection<br/>Shape: {mip_img_axis1.shape}</div>",
                f"                <div class='image-stack'>",
                f"                    <img class='mip-image base-image' src='data:image/png;base64,{img_base64_axis1}' alt='{tiff_path.name} axis 1'>",
                f"                    <img class='mip-image overlay-image' id='mask_{row_id}_axis1' src='data:image/png;base64,{mask_base64_axis1}' alt='mask'>",
                "                </div>",
                "            </div>",
                "            <div class='mip-column'>",
                f"                <div class='mip-label'>Axis 2 Projection<br/>Shape: {mip_img_axis2.shape}</div>",
                f"                <div class='image-stack'>",
                f"                    <img class='mip-image base-image' src='data:image/png;base64,{img_base64_axis2}' alt='{tiff_path.name} axis 2'>",
                f"                    <img class='mip-image overlay-image' id='mask_{row_id}_axis2' src='data:image/png;base64,{mask_base64_axis2}' alt='mask'>",
                "                </div>",
                "            </div>",
                "        </div>",
                "    </div>",
            ])
            
        except Exception as e:
            print(f"  Error processing {tiff_path.name}: {e}")
            import traceback
            traceback.print_exc()
            html_parts.extend([
                "    <div class='image-container'>",
                f"        <div class='filename'>{tiff_path.name}</div>",
                f"        <div class='shape-info' style='color: red;'>Error: {e}</div>",
                "    </div>",
            ])


def generate_html_viewer(dataset_configs, output_html="tiff_viewer.html"):
    """
    Generate HTML file displaying MIP projections of all TIFF files from multiple directories.
    
    Args:
        dataset_configs: List of tuples (tiff_dir, swc_dir, section_title)
        output_html: Output HTML filename
    """
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <title>TIFF Volume Viewer - Orthogonal MIP Projections with Neuron Masks</title>",
        "    <style>",
        "        body {",
        "            font-family: Arial, sans-serif;",
        "            margin: 20px;",
        "            background-color: #f5f5f5;",
        "        }",
        "        h1 {",
        "            color: #333;",
        "            text-align: center;",
        "        }",
        "        h2 {",
        "            color: #2c3e50;",
        "            text-align: center;",
        "            margin-top: 40px;",
        "        }",
        "        .image-container {",
        "            background-color: white;",
        "            margin: 20px auto;",
        "            padding: 20px;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            max-width: 1400px;",
        "        }",
        "        .filename {",
        "            font-size: 18px;",
        "            font-weight: bold;",
        "            color: #2c3e50;",
        "            margin-bottom: 10px;",
        "            text-align: center;",
        "        }",
        "        .shape-info {",
        "            font-size: 14px;",
        "            color: #7f8c8d;",
        "            margin-bottom: 15px;",
        "            text-align: center;",
        "        }",
        "        .slider-container {",
        "            text-align: center;",
        "            margin: 15px 0;",
        "        }",
        "        .opacity-slider {",
        "            width: 300px;",
        "            margin: 0 10px;",
        "        }",
        "        .slider-container label {",
        "            font-size: 14px;",
        "            color: #34495e;",
        "        }",
        "        .mip-grid {",
        "            display: grid;",
        "            grid-template-columns: repeat(3, 1fr);",
        "            gap: 15px;",
        "            margin-top: 15px;",
        "        }",
        "        .mip-column {",
        "            display: flex;",
        "            flex-direction: column;",
        "            align-items: center;",
        "        }",
        "        .mip-label {",
        "            font-size: 12px;",
        "            font-weight: bold;",
        "            color: #34495e;",
        "            margin-bottom: 10px;",
        "            text-align: center;",
        "        }",
        "        .image-stack {",
        "            position: relative;",
        "            width: 100%;",
        "        }",
        "        .mip-image {",
        "            width: 100%;",
        "            height: auto;",
        "            border: 1px solid #ddd;",
        "            border-radius: 4px;",
        "        }",
        "        .base-image {",
        "            display: block;",
        "        }",
        "        .overlay-image {",
        "            position: absolute;",
        "            top: 0;",
        "            left: 0;",
        "            opacity: 0.5;",
        "            pointer-events: none;",
        "        }",
        "    </style>",
        "    <script>",
        "        function updateOpacity(rowId, value) {",
        "            const opacity = value / 100;",
        "            document.getElementById('mask_' + rowId + '_axis0').style.opacity = opacity;",
        "            document.getElementById('mask_' + rowId + '_axis1').style.opacity = opacity;",
        "            document.getElementById('mask_' + rowId + '_axis2').style.opacity = opacity;",
        "            document.getElementById('value_' + rowId).textContent = opacity.toFixed(2);",
        "        }",
        "    </script>",
        "</head>",
        "<body>",
        "    <h1>TIFF Volume Viewer - Orthogonal MIP Projections with Neuron Masks</h1>",
        "    <p style='text-align: center; color: #7f8c8d;'>Three orthogonal views per volume (MIP along each axis)</p>",
        "    <p style='text-align: center; color: #7f8c8d;'>Use sliders to adjust neuron mask overlay opacity</p>",
    ]
    
    # Process each directory
    for tiff_dir, swc_dir, section_title in dataset_configs:
        process_directory(tiff_dir, swc_dir, html_parts, section_title)
    
    html_parts.extend([
        "</body>",
        "</html>",
    ])
    
    # Write HTML file
    html_content = "\n".join(html_parts)
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"\nHTML viewer saved to: {output_html}")
    print(f"Open this file in a web browser to view the images.")


if __name__ == "__main__":
    import sys
    
    # Default directories to process (tiff_dir, swc_dir, title)
    dataset_configs = [
        (
            "/home/brysongray/data/neurotrack_data/gold166/gold166_training_set/images",
            "/home/brysongray/data/neurotrack_data/gold166/gold166_training_set/morphology",
            "Training Set"
        ),
        (
            "/home/brysongray/data/neurotrack_data/gold166/gold166_validation_set/images",
            "/home/brysongray/data/neurotrack_data/gold166/gold166_validation_set/morphology",
            "Validation Set"
        ),
        (
            "/home/brysongray/data/neurotrack_data/gold166/gold166_test_set/images",
            "/home/brysongray/data/neurotrack_data/gold166/gold166_test_set/morphology",
            "Test Set"
        )
    ]
    
    output_file = "tiff_viewer.html"
    
    generate_html_viewer(dataset_configs, output_file)
