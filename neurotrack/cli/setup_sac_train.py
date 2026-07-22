#!/usr/bin/env python3

""" Setup input directory for training SAC neuron tracking model.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import shutil
import tifffile as tf
from tqdm import tqdm

from neurotrack.data import rendering as draw
from neurotrack.data import loading as load
from neurotrack.data import tree, save
from neurotrack.data import DataGenerator, DrawingComplexityConfig

def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup input directory for training SAC neuron tracking model.")
    parser.add_argument("--swc_dir", type=str, required=False, help="Path to the SWC directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--image_dir", type=str, required=False, help="Path to the image directory (contains tiffs).")
    parser.add_argument("--remove_soma", action="store_true", help="Remove soma from SWC.")
    parser.add_argument("--rng_seed", type=int, default=1, help="Random seed for data generator.")
    parser.add_argument("--complexity_range", type=float, nargs=2, default=(0.0, 1.0), help="Range of drawing complexities to use.")
    parser.add_argument("--morphology", type=str, default="all", help="Complexity of neuron morphology to generate (e.g., 'simple', 'moderate', 'complex', 'full', or 'all').")
    parser.add_argument("--subtrees_per_swc", type=int, default=1, help="Number of subtrees to draw per SWC.")
    parser.add_argument("--dataset_size", type=int, default=100, help="Number of synthetic neurons to generate if no SWC directory is provided.")
    return parser.parse_args()


def setup(
    image_dir,
    swc_dir,
    output_dir,
    remove_soma,
    rng_seed,
    complexity_range=(0.0, 1.0),
    morphology="all",
    subtrees_per_swc=1,
    dataset_size=100
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
    if remove_soma and swc_dir is not None:
        for swc_file in list(Path(swc_dir).rglob("*.swc")):
            swc_list = load.swc(swc_file)
            swc_list, seeds = tree.remove_soma(swc_list, max_radius=7.0)
            # setup temporary directory for soma-removed SWCs
            temp_swc_dir = Path(output_dir) / "temp_swc"
            temp_swc_dir.mkdir(parents=True, exist_ok=True)
            save.write_swc(swc_list, temp_swc_dir / swc_file.name, exist_ok=True)
        swc_dir = str(temp_swc_dir)

    complexity_config = DrawingComplexityConfig()
    rng = np.random.default_rng(rng_seed)
    data_generator = DataGenerator(cache_dir=output_dir, complexity_config=complexity_config, rng=rng)
    data_generator.generate_data(subtrees_per_swc=subtrees_per_swc, complexity_range=complexity_range, morphology=morphology, swc_dir=swc_dir, img_dir=image_dir, dataset_size=dataset_size)
    if remove_soma and swc_dir is not None:
        # remove temporary directory
        shutil.rmtree(temp_swc_dir)

def main():
    args = parse_args()
    setup(args.image_dir, args.swc_dir, args.output_dir, args.remove_soma, args.rng_seed, args.complexity_range, args.morphology, args.subtrees_per_swc, args.dataset_size)

if __name__ == "__main__":
    main()