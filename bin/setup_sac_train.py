#!/usr/bin/env python3

""" Setup input directory for training SAC neuron tracking model.
"""

import argparse
import shutil
import numpy as np
import os
from pathlib import Path
import sys
import tifffile as tf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_prep import draw, load, tree

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup input directory for training SAC neuron tracking model.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory (contains tiffs).")
    parser.add_argument("--swc_dir", type=str, required=True, help="Path to the SWC directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--save_seeds", action="store_true", help="Save seed points.")
    parser.add_argument("--save_branches", action="store_true", help="Save branch points.")
    parser.add_argument("--save_density", action="store_true", help="Save density maps.")
    parser.add_argument("--random_seeds", action="store_true", help="Use random seeds from SWC.")
    parser.add_argument("--copy_image", action="store_true", help="Copy image files to output directory.")
    parser.add_argument("--remove_soma", action="store_true", help="Remove soma from SWC.")
    parser.add_argument("--sync", action="store_true", help="Sync with existing output.")
    parser.add_argument("--use_symlinks", action="store_true", help="Use symlinks for output files.")
    return parser.parse_args()


def setup(image_dir, swc_dir, output_dir, save_seeds, save_branches,
          save_density, random_seeds, copy_image, sync, use_symlinks, remove_soma):
    
    swc_files = sorted(os.listdir(swc_dir))
    tif_files = os.listdir(image_dir)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    swc_lists = []
    sections_list = []
    for swc_file in tqdm(swc_files, desc="Processing SWC files"):
        fname = swc_file.split('.')[0]
        seeds_out = os.path.join(output_dir, fname, f"{fname}_seeds.txt")
        branches_out = os.path.join(output_dir, fname, f"{fname}_branches.txt")
        density_out = os.path.join(output_dir, fname, f"{fname}_density.tif")
        image_out = os.path.join(output_dir, fname, f"{fname}_image.tif")

        if sync:
            save_seeds_ = save_seeds and not os.path.exists(seeds_out)
            save_branches_ = save_branches and not os.path.exists(branches_out)
            save_density_ = save_density and not os.path.exists(density_out)
            copy_image_ = copy_image and not os.path.exists(image_out)
            if not any((save_seeds_, save_branches_, save_density_)):
                continue
        else:
            save_seeds_, save_branches_, save_density_, copy_image_ = save_seeds, save_branches, save_density, copy_image

        # Load the SWC file
        swc_list = load.swc(os.path.join(swc_dir, swc_file), verbose=False)
        seeds = np.array(swc_list[0][4:1:-1])
        
        if remove_soma:
            swc_list, seeds = tree.remove_soma(swc_list, max_radius=7.0)
            seeds = seeds[:, ::-1]  # Reorder seeds to match image format (z, y, x)
        
        if random_seeds:
            random_seed_ids = np.random.choice(len(swc_list), 10)
            new_seeds = np.array(swc_list)[random_seed_ids][:, 4:1:-1]
            seeds = np.concatenate((seeds, new_seeds), 0)

        # Parse the SWC file to get sections and section graph
        sections, section_graph = load.parse_swc(swc_list)
        # get branches list from sections_graph
        # This returns reordered coordinates, i.e. xyz -> zyx.
        branches, terminals = load.get_critical_points(swc_list, sections)

        swc_lists.append(swc_list)
        sections_list.append(sections)

        if save_branches_:
            branches_outdir = os.path.split(branches_out)[0]
            if not os.path.exists(branches_outdir):
                os.makedirs(branches_outdir, exist_ok=True)
            with open(branches_out, 'w') as f:
                for branch_point in branches:
                    # Convert the branch points coordinates to string and write to file
                    f.write(f"{branch_point[0]} {branch_point[1]} {branch_point[2]}\n")

        if save_seeds_:
            seeds_outdir = os.path.split(seeds_out)[0]
            if not os.path.exists(seeds_outdir):
                os.makedirs(seeds_outdir, exist_ok=True)        
            with open(seeds_out, 'w') as f:
                for seed in seeds:
                    # Convert the seed points coordinates to string and write to file
                    f.write(f"{round(seed[0])} {round(seed[1])} {round(seed[2])}\n")

        if save_density_:
            matching_files = [f for f in tif_files if fname == f.split('_image.tif')[0]]
            tif_img = tf.imread(os.path.join(image_dir, matching_files[0]))
            shape = tif_img.shape
            del tif_img

            segments = []
            for section in sections.values():
                segments.append(section)
            segments = np.concatenate(segments)

            density = draw.draw_neuron(segments, shape=shape, width=3.0, noise=0.0, rgb=False)
            density_outdir = os.path.split(density_out)[0]
            if not os.path.exists(density_outdir):
                os.makedirs(density_outdir, exist_ok=True)
            tf.imwrite(density_out, density.data.numpy(), compression='zlib')
        
        if copy_image_:
            # copy tiff to folder
            split_key = '_image.tif' if "_image.tif" in tif_files[0] else '.'
            tif_file = [f for f in tif_files if fname == f.split(split_key)[0]]
            if tif_file:
                tif_file = tif_file[0]
                tif_outdir = os.path.join(output_dir, fname)
                if not os.path.exists(tif_outdir):
                    os.makedirs(tif_outdir, exist_ok=True)
                if use_symlinks:
                    os.symlink(os.path.join(image_dir, tif_file), os.path.join(tif_outdir, f"{fname}_image.tif"))
                else:
                    shutil.copy(os.path.join(image_dir, tif_file), tif_outdir)

    return

def main():
    args = parse_args()
    setup(args.image_dir, args.swc_dir, args.output_dir, args.save_seeds,
          args.save_branches, args.save_density, args.random_seeds,
          args.copy_image, args.sync, args.use_symlinks, args.remove_soma)

if __name__ == "__main__":
    main()
