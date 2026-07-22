"""
Collect neuron image patches and branch labels.
"""

import argparse
from datetime import datetime
from glob import glob
import numpy as np
import os

from neurotrack.data import loading as load
from neurotrack.data import collect, tree

DATE = datetime.now().strftime("%m-%d-%y")

def main():
    """
    Collects neuron image patches and branch labels based on provided arguments.
    Parses command line arguments for labels directory, images directory, output directory,
    output filename base, adjustment flag, and number of samples to collect from each image file.
    Loads SWC files from the labels directory, collects random sample points, and saves image patches centered at those points.
    
    Arguments
    ---------
    -l, --labels : str
        Path to labels directory (contains swc files).
    -i, --images : str
        Path to images directory.
    -o, --out : str
        Path to output directory.
    -n, --name : str
        Output filename base.
    -a, --adjust : bool
        Set to true if neuron coordinates were rescaled to draw images.
    --n_samples : int
        Number of samples to collect from each image file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels', type=str, help='Path to labels directory (contains swc files).')
    parser.add_argument('-i', '--images', type=str, help='Path to images directory.')
    parser.add_argument('-o','--out', type=str, help="Path to output directory.")
    parser.add_argument('-n', '--name', type=str, help='Output filename base.')
    parser.add_argument('-a', '--adjust', action='store_true', default=False, help='Set to true if neuron coordinates were rescaled to draw images.')
    parser.add_argument('-r', '--remove_soma', action='store_true', default=False, help='Remove edges near the soma from the swc.')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to collect from each image file.')
    args = parser.parse_args()
    labels_dir = args.labels
    image_dir = args.images
    out_dir = args.out
    name = args.name
    adjust = args.adjust
    remove_soma = args.remove_soma
    samples_per_file = args.n_samples
    
    # get sample points from swc files
    files = [f for x in os.walk(labels_dir) for f in glob(os.path.join(x[0], '*.swc'))]
    swc_lists = []
    for f in files:
        swc_list = load.swc(f)
        swc_list = np.array(swc_list)
        if adjust:
            min = np.min(swc_list[:, 2:5], axis=0)
            max = np.max(swc_list[:, 2:5], axis=0)
            vol = np.prod(max - min)
            scale = np.round((5e7 / vol)**(1/3)) # scale depends on the volume
            swc_list[...,:3] = (swc_list[...,:3] - min) * scale + np.array([10.0, 10.0, 10.0])
        if remove_soma:
            swc_list, _ = tree.remove_soma(swc_list, max_radius=7.0)
        swc_lists.append(swc_list)
    fnames = [f.split('/')[-1].split('.')[0] for f in files]
    
    sample_points = collect.swc_random_points(samples_per_file, swc_lists, fnames, adjust=adjust)
    
    # save sample patches from the images centered at the sample points
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    collect.collect_data(sample_points, image_dir, out_dir, name, DATE)
    
    return


if __name__ == "__main__":
    main()