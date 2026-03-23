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
                                  [--plot] [--correct_inhomogeneity] [--sync] [--save_out]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tf
import torch
from glob import glob
from tqdm import tqdm

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep import load, draw, save, data_utils
from data_prep.draw import NeuronRenderer, DrawingConfig


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
    parser.add_argument('--scaling_dict', type=str, required=True,
                        help='Path to scaling dictionary .npy file')
    parser.add_argument('--scales_df', type=str, default=None,
                        help='Path to scaling CSV file (required for plotting)')
    parser.add_argument('--plot', action='store_true',
                        help='Enable plotting visualization')
    parser.add_argument('--correct_inhomogeneity', action='store_true',
                        help='Apply inhomogeneity correction')
    parser.add_argument('--sync', action='store_true',
                        help='Skip files that already exist in output')
    parser.add_argument('--save_out', action='store_true',
                        help='Save processed files to output directories')
    
    return parser.parse_args()


def process_neuron_data(tifs_path, swc_path, tifs_out, swc_out, scaling_dict_path,
                       plot=False, correct_inhomogeneity=False,
                       sync=True, save_out=True):
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
    scaling_dict_path : str
        Path to scaling dictionary .npy file
    plot : bool, default False
        Enable plotting visualization
    correct_inhomogeneity : bool, default True
        Apply inhomogeneity correction
    sync : bool, default True
        Skip files that already exist in output
    save_out : bool, default True
        Save processed files to output directories
    """
    
    # Load scaling dictionary
    scaling_dict = np.load(scaling_dict_path, allow_pickle=True).item()
    
    # Get list of files
    tif_files = os.listdir(tifs_path)
    swc_files = os.listdir(swc_path)
    
    # Process each TIFF file
    for i in tqdm(range(len(tif_files)), desc="Processing files"):
        background_threshold = None
        
        if plot:
            plt.close("all")
            fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        
        # Load image
        img_path = os.path.join(tifs_path, tif_files[i])
        print(f"Loading image: {img_path}")
        
        if save_out:
            name = tif_files[i].split('.tif')[0]
            tif_out_path = os.path.join(tifs_out, name, name + "_image.tif")
            swc_out_path = os.path.join(swc_out, name, name + ".swc")
            
            if sync:
                tif_exists = os.path.exists(tif_out_path)
                swc_exists = os.path.exists(swc_out_path)
                if tif_exists and swc_exists:
                    print(f"Skipping {tif_files[i]}")
                    continue
            
            # Create output directories
            os.makedirs(os.path.split(tif_out_path)[0], exist_ok=True)
            os.makedirs(os.path.split(swc_out_path)[0], exist_ok=True)
        
        # Load and process image
        img = tf.imread(img_path)
        
        if img.dtype == "uint16":
            # Convert to uint8
            img = img - img.min()
            img = img / img.max()
            img = np.round(img * 255).astype(np.uint8)
        
        # Get matching SWC file
        swc_file = [f for f in swc_files if tif_files[i].split('.')[0] == f.split('.')[0]][0]
        swc_list = load.swc(os.path.join(swc_path, swc_file), verbose=False)
        
        # Crop image and adjust SWC coordinates
        img, swc_list = load.crop_and_adjust_coords(img, swc_list)
        
        # Apply scaling
        scale = scaling_dict[tif_files[i]]
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
        if save_out:
            tf.imwrite(tif_out_path, img, compression='zlib')
            save.write_swc(swc_list, swc_out_path)
        
        # Plotting
        if plot:
            
            # Draw density
            sections, section_graph = load.parse_swc(swc_list)
            branches, terminals = load.get_critical_points(swc_list, sections)
                        
            # Use the new NeuronRenderer with clean config
            renderer = NeuronRenderer()
            config = DrawingConfig(width=3.0, rgb=False)
            density = renderer.draw_neuron(sections, shape=img.shape, config=config)
            
            # Get a random branch point
            branch_point = branches[torch.randint(0, len(branches), (1,)).squeeze(0)]
            
            big_image = img.max(0)
            big_density = density.data[0].numpy().max(0)
            if big_image.dtype == np.uint8:
                big_image = big_image.astype(np.float32) / 255.0
            
            ax[0].imshow(big_image, cmap='gray', vmin=0.0, vmax=big_image.max())
            ax[0].set_title(f"{tif_files[i]}", fontsize=8)
            
            # Plot seed point
            seed = swc_list[swc_list[:, 6] == -1][0, 2:5]
            ax[0].scatter(seed[2], seed[1], c='r')
            
            # Draw rectangles
            rect_size1 = 19
            rect_size2 = 35
            rect1 = plt.Rectangle((round(branch_point[2]) - rect_size1 // 2, 
                                 round(branch_point[1]) - rect_size1 // 2), 
                                rect_size1, rect_size1, linewidth=1, 
                                edgecolor='r', facecolor='none')
            rect2 = plt.Rectangle((round(branch_point[2]) - rect_size2 // 2, 
                                 round(branch_point[1]) - rect_size2 // 2), 
                                rect_size2, rect_size2, linewidth=1, 
                                edgecolor='g', facecolor='none')
            ax[0].add_patch(rect1)
            ax[0].add_patch(rect2)
            
            # Small image views
            small_image = big_image[round(branch_point[1]) - rect_size2 // 2:round(branch_point[1]) + rect_size2 // 2, 
                                  round(branch_point[2]) - rect_size2 // 2:round(branch_point[2]) + rect_size2 // 2]
            small_density = big_density[round(branch_point[1]) - rect_size2 // 2:round(branch_point[1]) + rect_size2 // 2, 
                                      round(branch_point[2]) - rect_size2 // 2:round(branch_point[2]) + rect_size2 // 2]
            
            ax[1].imshow(small_image, cmap='gray', interpolation="none", vmin=0.0, vmax=big_image.max())
            ax[1].imshow(small_density, cmap='hot', interpolation="none", alpha=0.25)
            
            rect1 = plt.Rectangle((small_image.shape[1] // 2 - rect_size1 // 2, 
                                 small_image.shape[0] // 2 - rect_size1 // 2), 
                                rect_size1-2, rect_size1-2, linewidth=1, 
                                edgecolor='r', facecolor='none')
            rect2 = plt.Rectangle((small_image.shape[1] // 2 - rect_size2 // 2, 
                                 small_image.shape[0] // 2 - rect_size2 // 2), 
                                rect_size2-2, rect_size2-2, linewidth=2, 
                                edgecolor='g', facecolor='none')
            ax[1].add_patch(rect1)
            ax[1].add_patch(rect2)
            
            plt.show()
            plt.close()


def main():
    """Main function to run the script."""
    args = parse_args()
    
    process_neuron_data(
        tifs_path=args.tifs_path,
        swc_path=args.swc_path,
        tifs_out=args.tifs_out,
        swc_out=args.swc_out,
        scaling_dict_path=args.scaling_dict,
        plot=args.plot,
        correct_inhomogeneity=args.correct_inhomogeneity,
        sync=args.sync,
        save_out=args.save_out
    )


if __name__ == "__main__":
    main()
