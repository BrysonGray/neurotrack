#!/usr/bin/env python3
"""
Process neuron data by applying scaling, inhomogeneity correction, and cropping.

This script processes TIFF images and corresponding SWC files by:
1. Loading and cropping images
2. Applying scaling transformations
3. Performing inhomogeneity correction
4. Saving processed data

Usage:
    python process_neuron_data.py --tifs_path /path/to/tifs --swc_path /path/to/swc \
                                  --tifs_out /path/to/output/tifs --swc_out /path/to/output/swc \
                                  --scaling_dict /path/to/scaling_dict.npy \
                                  [--correct_inhomogeneity] [--sync]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tf
import torch
from tqdm import tqdm

from neurotrack.data import loading as load
from neurotrack.data import rendering as draw
from neurotrack.data import save, data_utils
from neurotrack.data.rendering import NeuronRenderer, DrawingConfig
from neurotrack.data.image import to_uint8


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process neuron data with scaling and inhomogeneity correction')
    
    parser.add_argument('--tifs_path', type=str, required=True,
                        help='Path to input TIFF files directory')
    parser.add_argument('--swc_path', type=str, required=True,
                        help='Path to input SWC files directory')
    parser.add_argument('--tifs_out', type=str, required=True,
                        help='Path to output TIFF files directory')
    parser.add_argument('--swc_out', type=str, required=True,
                        help='Path to output SWC files directory')
    parser.add_argument('--scaling_dict', type=str, required=False,
                        help='Path to scaling dictionary .npy file')
    parser.add_argument('--correct_inhomogeneity', action='store_true',
                        help='Apply inhomogeneity correction')
    parser.add_argument('--sync', action='store_true',
                        help='Skip files that already exist in output')
    parser.add_argument('--compression', action='store_true',
                        help='Use zlib compression when saving TIFF files')
    
    return parser.parse_args()


def process_neuron_data(tifs_path, swc_path, tifs_out, swc_out, scaling_dict_path=None,
                       plot=False, correct_inhomogeneity=False,
                       sync=True, compression=False):
    """
    Process neuron data with scaling and inhomogeneity correction.
    
    Parameters
    ----------
    tifs_path : str
        Path to input TIFF files directory
    swc_path : str
        Path to input SWC files directory
    tifs_out : str
        Path to output TIFF files directory
    swc_out : str
        Path to output SWC files directory
    scaling_dict_path : str, Optional
        Path to scaling dictionary .npy file
    correct_inhomogeneity : bool, default True
        Apply inhomogeneity correction
    sync : bool, default True
        Skip files that already exist in output
    """
    
    # Get aligned TIFF->SWC file mapping
    img_to_swc_map = load.map_tiff_to_swc(tifs_path, swc_path, use_fixed=False, verbose=False)
    tif_files = list(img_to_swc_map.keys())
    
    tifs_out = Path(tifs_out)
    swc_out = Path(swc_out)

    # Load scaling dictionary
    if scaling_dict_path is not None:
        scaling_dict = np.load(scaling_dict_path, allow_pickle=True).item()
    else:
        scaling_dict = None
    
    def _lookup_scale(img_path: Path, scaling: dict | None) -> float:
        if scaling is None:
            return 1.0
        for key in (img_path, str(img_path), img_path.name, img_path.stem):
            if key in scaling:
                return scaling[key]
        return 1.0

    # Process each TIFF file
    for i in tqdm(range(len(tif_files)), desc="Processing files"):
        background_threshold = None
        
        if plot:
            plt.close("all")
            fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        
        # Load image
        img_path = tif_files[i]
        print(f"Loading image: {img_path}")
        
        # Check if output files already exist
        name = tif_files[i].stem
        tif_out_path = tifs_out / name / f"{name}.tif"
        swc_out_path = swc_out / name / f"{name}.swc"
        
        if sync:
            tif_exists = tif_out_path.exists()
            swc_exists = swc_out_path.exists()
            if tif_exists and swc_exists:
                print(f"Skipping {tif_files[i]}")
                continue
        
        # Create output directories
        tif_out_path.parent.mkdir(parents=True, exist_ok=True)
        swc_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and process image
        img = tf.imread(img_path)
        img = to_uint8(img)
        
        # Get matching SWC file
        swc_file = img_to_swc_map[img_path]
        swc_list = load.swc(str(swc_file), verbose=False)
        
        # Crop image and adjust SWC coordinates
        img, swc_list = load.crop_and_adjust_coords(img, swc_list)
        
        # Apply scaling
        scale = _lookup_scale(img_path, scaling_dict)
        if scale != 1.0:
            img, swc_list = load.scale_and_adjust_coords(img, swc_list, scale)
        
        # Inhomogeneity correction
        if correct_inhomogeneity:
            if background_threshold is None:
                img_flat = img.flatten()[::27, None]  # Downsample for efficiency
                # Global intensity normalization
                img_flat = img_flat.astype(np.float32) / img.max()
                mu, sigmas = data_utils.kmeans(img_flat, k=2, tolerance=1e-3, 
                                             init_means=np.array([1.0, 0.0]), max_iter=100)
                del img_flat
                background_threshold = mu[0] + sigmas[0] * 8
            
            img = data_utils.inhomogeneity_correction(img, background_threshold=background_threshold)
        
        # Save processed data
        if compression:
            tf.imwrite(tif_out_path, img, compression='zlib')
        else:
            tf.imwrite(tif_out_path, img)
        save.write_swc(swc_list, swc_out_path)


def main():
    """Main function to run the script."""
    args = parse_args()
    
    process_neuron_data(
        tifs_path=args.tifs_path,
        swc_path=args.swc_path,
        tifs_out=args.tifs_out,
        swc_out=args.swc_out,
        scaling_dict_path=args.scaling_dict,
        correct_inhomogeneity=args.correct_inhomogeneity,
        sync=args.sync,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
