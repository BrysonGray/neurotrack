import argparse
from datetime import datetime
from glob import glob
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
from data_prep import load, collect

DATE = datetime.now().strftime("%m-%d-%y")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels', type=str, help='Path to labels directory (contains swc files).')
    parser.add_argument('-i', '--images', type=str, help='Path to images directory.')
    parser.add_argument('-o','--out', type=str, help="Path to output directory.")
    parser.add_argument('-n', '--name', type=str, help='Output filename base.')
    parser.add_argument('-a', '--adjust', action='store_true', default=False, help='Set to true if neuron coordinates were rescaled to draw images.')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to collect from each image file.')
    args = parser.parse_args()
    labels_dir = args.labels
    image_dir = args.images
    out_dir = args.out
    name = args.name
    adjust = args.adjust
    samples_per_file = args.n_samples
    
    # get sample points from swc files
    files = [f for x in os.walk(labels_dir) for f in glob(os.path.join(x[0], '*.swc'))]
    swc_lists = []
    for f in files:
        swc_lists.append(load.swc(f))
    fnames = [f.split('/')[-1].split('.')[0] for f in files]
    
    sample_points = collect.swc_random_points(samples_per_file, swc_lists, fnames, adjust=adjust)
    
    # save sample patches from the images centered at the sample points
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    collect.collect_data(sample_points, image_dir, out_dir, name, DATE)
    
    return


if __name__ == "__main__":
    main()