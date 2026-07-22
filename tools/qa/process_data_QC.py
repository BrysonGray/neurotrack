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
from qc_utils import compute_mip_along_axis, save_image_to_file, save_mask_to_file_rgba, draw_2d_projection_mask

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
    tiff_to_swc_map = map_tiff_to_swc(tiff_dir, swc_dir, use_fixed=True, verbose=True, fixed_suffix="_filtered.swc")
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
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/train_set/images",
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/train_set/morphology",
    #         "Training Set"
    #     ),
    #     (
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/validate_set/images",
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/validate_set/morphology",
    #         "Validation Set"
    #     ),
    #     (
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/test_set/images",
    #         "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/test_set/morphology",
    #         "Test Set"
    #     )
    # ]

    dataset_configs = [
        # (
        #     "/home/brysongray/data/neurotrack_data/gold166/gold166_cleaned/all/images",
        #     "/home/brysongray/data/neurotrack_data/gold166/gold166_cleaned/all/morphology_soma_removed_deduped",
        #     "Whole Set"
        # )
        (
            "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/train_set_soma_removed/images_synthetic_simple",
            "/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/train_set_soma_removed/morphology",
            "Training Set (Soma Removed Synthetic Simple)"
        )
    ]
    
    # Output to outputs/data_QC directory
    output_dir = str(Path(__file__).resolve().parents[2] / "outputs" / "cropped_synthetic_simple_QC")
    
    process_dataset(dataset_configs, output_dir)
