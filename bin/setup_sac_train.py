#!/usr/bin/env python3

""" Setup input directory for training SAC neuron tracking model.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import shutil
import sys
import tifffile as tf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_prep import draw, load, tree

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup input directory for training SAC neuron tracking model.")
    parser.add_argument("--swc_dir", type=str, required=True, help="Path to the SWC directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--image_dir", type=str, required=False, help="Path to the image directory (contains tiffs).")
    parser.add_argument("--save_seeds", action="store_true", help="Save seed points.")
    parser.add_argument("--save_branches", action="store_true", help="Save branch points.")
    parser.add_argument("--save_density", action="store_true", help="Save neuron density maps as compressed TIFF files (requires image directory).")
    parser.add_argument("--save_section_labels", action="store_true", help="Save section labels.")
    parser.add_argument("--random_seeds", action="store_true", help="Use random seeds from SWC.")
    parser.add_argument("--copy_image", action="store_true", help="Copy image files to output directory (requires image directory).")
    parser.add_argument("--remove_soma", action="store_true", help="Remove soma from SWC.")
    parser.add_argument("--sync", action="store_true", help="Sync with existing output.")
    parser.add_argument("--use_symlinks", action="store_true", help="Use symlinks for output files.")
    return parser.parse_args()


def setup(
    image_dir,
    swc_dir,
    output_dir,
    save_seeds,
    save_branches,
    save_density,
    save_section_labels,
    random_seeds,
    copy_image,
    sync,
    use_symlinks,
    remove_soma
):
    """Set up input directory for training SAC neuron tracking model.
    
    Args:
        image_dir: Path to the image directory containing TIFF files.
        swc_dir: Path to the SWC directory.
        output_dir: Path to the output directory.
        save_seeds: Whether to save seed points.
        save_branches: Whether to save branch points.
        save_density: Whether to save neuron density maps.
        save_section_labels: Whether to save section labels.
        random_seeds: Whether to use random seeds from SWC.
        copy_image: Whether to copy image files to output directory.
        sync: Whether to sync with existing output.
        use_symlinks: Whether to use symlinks for output files.
        remove_soma: Whether to remove soma from SWC.
    """
    tif_files = []
    swc_dir = Path(swc_dir)
    swc_files = sorted([f for f in swc_dir.rglob("*.swc")])
    
    if image_dir:
        image_dir = Path(image_dir)
        tif_files = [f for f in image_dir.rglob("*image.tif")]
        if not tif_files:
            tif_files = [f for f in image_dir.rglob("*.tif")]
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in {image_dir}")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True)

    swc_lists = []
    sections_list = []
    
    for swc_file in tqdm(swc_files, desc="Processing SWC files"):
        fname = swc_file.stem
        file_dir = output_dir / fname
        seeds_out = file_dir / f"{fname}_seeds.txt"
        branches_out = file_dir / f"{fname}_branches.txt"
        density_out = file_dir / f"{fname}_density.tif"
        image_out = file_dir / f"{fname}_image.tif"
        section_labels_out = file_dir / f"{fname}_sections.tif"
        section_graph_out = file_dir / f"{fname}_section_graph.json"

        if sync:
            save_seeds_ = save_seeds and not seeds_out.exists()
            save_branches_ = save_branches and not branches_out.exists()
            save_density_ = save_density and not density_out.exists()
            save_section_labels_ = save_section_labels and not section_labels_out.exists()
            copy_image_ = copy_image and not image_out.exists()
            if not any((save_seeds_, save_branches_, save_density_, copy_image_)):
                continue
        else:
            save_seeds_ = save_seeds
            save_branches_ = save_branches
            save_density_ = save_density
            save_section_labels_ = save_section_labels
            copy_image_ = copy_image

        # Load the SWC file
        swc_list = load.swc(swc_file, verbose=False)
        seeds = np.array(swc_list[0][4:1:-1])
        
        if remove_soma:
            swc_list, seeds = tree.remove_soma(swc_list, max_radius=7.0)
            seeds = seeds[:, ::-1]  # Reorder seeds to match image format (z, y, x)
        
        if random_seeds:
            random_seed_ids = np.random.choice(len(swc_list), 10)
            # new_seeds = np.array(swc_list)[random_seed_ids][:, 4:1:-1]
            # seeds = np.concatenate((seeds, new_seeds), 0)
            seeds = np.array(swc_list)[random_seed_ids][:, 4:1:-1]

        # Parse the SWC file to get sections and section graph
        sections, section_graph = load.parse_swc(swc_list)
        # Get branches list from sections_graph
        # This returns reordered coordinates, i.e. xyz -> zyx.
        branches, terminals = load.get_critical_points(swc_list, sections)

        swc_lists.append(swc_list)
        sections_list.append(sections)

        if save_branches_:
            branches_out.parent.mkdir(parents=True, exist_ok=True)
            with open(branches_out, 'w') as f:
                for branch_point in branches:
                    # Convert the branch points coordinates to string and write to file
                    f.write(f"{branch_point[0]} {branch_point[1]} {branch_point[2]}\n")

        if save_seeds_:
            seeds_out.parent.mkdir(parents=True, exist_ok=True)
            with open(seeds_out, 'w') as f:
                for seed in seeds:
                    # Convert the seed points coordinates to string and write to file
                    f.write(f"{round(seed[0])} {round(seed[1])} {round(seed[2])}\n")

        if save_density_ or save_section_labels_:
            if not tif_files:
                raise ValueError("Image directory must be provided to save density maps or section labels.")
            matching_files = [f for f in tif_files if fname == f.stem.split('_image')[0]]
            if not matching_files:
                matching_files = [f for f in tif_files if fname == f.stem]
            if not matching_files:
                print(f"Warning: No matching image file found for {fname}")
                continue

            tif_img = tf.imread(matching_files[0])
            shape = tif_img.shape
            del tif_img

            if save_section_labels_:
                renderer = draw.NeuronRenderer()
                labels = renderer.draw_section_labels(sections, shape=shape, width=5.0)
                section_labels_out.parent.mkdir(parents=True, exist_ok=True)
                tf.imwrite(section_labels_out, labels.data.numpy(), compression='zlib')
                with open(section_graph_out, 'w') as f:
                    json.dump(section_graph, f, indent=4)

            if save_density_:
                renderer = draw.NeuronRenderer()
                config = draw.DrawingConfig(width=3.0, rgb=False)
                density = renderer.draw_neuron(sections, shape=shape, config=config)
                density_out.parent.mkdir(parents=True, exist_ok=True)
                tf.imwrite(density_out, density.data.numpy(), compression='zlib')

        if copy_image_:
            if not tif_files:
                raise ValueError("Image directory must be provided to copy image files.")
            # Find matching image file
            matching_files = [f for f in tif_files if fname == f.stem.split('_image')[0]]
            if not matching_files:
                matching_files = [f for f in tif_files if fname == f.stem]
            if not matching_files:
                print(f"Warning: No matching image file found for {fname}")
                continue
                
            tif_file = matching_files[0]
            file_dir.mkdir(parents=True, exist_ok=True)
            dest_file = file_dir / f"{fname}_image.tif"
            
            if use_symlinks:
                if dest_file.exists() or dest_file.is_symlink():
                    dest_file.unlink()
                dest_file.symlink_to(tif_file)
            else:
                shutil.copy(tif_file, dest_file)

def main():
    args = parse_args()
    setup(args.image_dir, args.swc_dir, args.output_dir, args.save_seeds,
          args.save_branches, args.save_density, args.save_section_labels,
          args.random_seeds, args.copy_image, args.sync, args.use_symlinks, args.remove_soma)

if __name__ == "__main__":
    main()
