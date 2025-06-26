import argparse
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parents[1]))
from data_prep import load, collect

DATE = datetime.now().strftime("%m-%d-%y")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', type=str, help='Path to samples_points.npy file (contains swc files).')
    parser.add_argument('-i', '--images', type=str, help='Path to images directory.')
    parser.add_argument('-o','--out', type=str, help="Path to output directory.")

    args = parser.parse_args()
    sample_points = args.samples
    img_dir = args.images
    out_dir = args.out

    if not os.path.exists(os.path.join(out_dir, 'observations')):
        os.makedirs(os.path.join(out_dir, 'observations'), exist_ok=True)

    sample_points = np.load(sample_points, allow_pickle=True)
    sample_points = sample_points.item()
    radii = torch.arange(6,55,6)
    collect.save_spherical_patches(sample_points, img_dir, out_dir, radii)
    
    return


if __name__ == "__main__":
    main()