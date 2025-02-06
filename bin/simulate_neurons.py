"""
Generate and save simulated neuron images, either from existing neuron swc files or
by generating new simulated neuron trees based on the provided parameters.
"""


import argparse
from glob import glob
import json
import numpy as np
import os
from pathlib import Path
import sys
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from data_prep import generate, draw, load


def main():

    """
    Main function to simulate neuron trees or load existing neuron trees from SWC files,
    draw neuron images, and save them to the specified output directory.

    JSON Configuration Parameters
    -----------------------------
    labels_dir : str, optional
        Directory containing SWC files of existing neuron trees. If not provided, neuron trees will be simulated.
    out : str
        Output directory to save the generated neuron images.
    width : int
        Width of the generated neuron images.
    random_contrast : bool
        Whether to apply random contrast to the neuron images.
    dropout : float
        Density of intensity dropout points for the neuron images.
    random_brightness : bool
        Whether to apply random signal to noise ratio to the neuron images.
    noise : float
        Amount of noise to add to the neuron images.
    binary : bool
        Whether to draw the neuron images as a binary mask.
    seed : int
        Seed for the random number generator.
    count : int, optional
        Number of neuron trees to simulate. Required if `labels_dir` is not provided.
    size : int, optional
        Size of the simulated neuron trees. Required if `labels_dir` is not provided.
    length : int, optional
        Length of the simulated neuron trees. Required if `labels_dir` is not provided.
    stepsize : float, optional
        Step size for the simulated neuron trees. Required if `labels_dir` is not provided.
    uniform_len : bool, optional
        Whether to use uniform length for the simulated neuron trees. Required if `labels_dir` is not provided.
    kappa : float, optional
        Kappa parameter for the simulated neuron trees. Required if `labels_dir` is not provided.
    random_start : bool, optional
        Whether to use random starting points for the simulated neuron trees. Required if `labels_dir` is not provided.
    branches : int, optional
        Number of branches for the simulated neuron trees. Required if `labels_dir` is not provided.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--json', type=str, help='Path to input parameters json file.')
    args = parser.parse_args()
    parameters_file = args.json
    with open(parameters_file) as f:
        parameters = json.load(f)
    
    labels_dir = parameters["labels_dir"] if "labels_dir" in parameters else None
    out = parameters["out"]
    if not os.path.exists(out):
        os.makedirs(out)
    width = parameters["width"]
    random_contrast = parameters["random_contrast"]
    dropout = parameters["dropout"]
    random_brightness = parameters["random_brightness"]
    noise = parameters["noise"]
    binary = parameters["binary"]
    seed = parameters["seed"]
    rng = np.random.default_rng(seed)

    if labels_dir is not None: # Load existing neuron trees as swc files
        print(f"Loading existing neuron trees as swc files...\n\
            labels_dir: {labels_dir}")
        files = [f for x in os.walk(labels_dir) for f in glob(os.path.join(x[0], "*.swc"))]
        swc_lists = []
        for f in files:
            swc_lists.append(load.swc(f))
        print("done")

    else: # Generate simulated neuron trees
        count = parameters["count"]
        size = (parameters["size"],)*3
        length = parameters["length"]
        stepsize = parameters["stepsize"]
        uniform_len = parameters["uniform_len"]
        kappa = parameters["kappa"]
        random_start = parameters["random_start"]
        branches = parameters["branches"]

        print(f"Generating simulated neuron trees...\n"
              f"    size: {size}\n"
              f"    length: {length}\n"
              f"    step size: {stepsize}\n"
              f"    uniform_len: {uniform_len}\n"
              f"    kappa: {kappa}\n"
              f"    random_start: {random_start}\n"
              f"    branches: {branches}")

        swc_lists = []
        for i in tqdm(range(count)):
            swc_list = generate.make_swc_list(size,
                                    length=length,
                                    step_size=stepsize,
                                    kappa=kappa,
                                    uniform_len=uniform_len,
                                    random_start=random_start,
                                    rng=rng,
                                    num_branches=branches) # make simulated neuron paths.
            swc_lists.append(swc_list)
        print("done\n")

    print(
        f"Drawing neuron images and saving to {out}..."
        f"    width: {width}\n"
        f"    random_contrast: {random_contrast}\n"
        f"    random_brightness: {random_brightness}\n"
        f"    dropout: {dropout}\n"
        f"    noise: {noise}\n"
        f"    binary: {binary}\n"
        f"    seed: {seed}\n"
    )
    
    for i in tqdm(range(len(swc_lists))):
        color = np.array([1.0, 1.0, 1.0])
        background = np.array([0., 0., 0.])
        if random_contrast:
            color = np.random.rand(3)
            color /= np.linalg.norm(color)
            background = np.random.rand(3)
            background = background / np.linalg.norm(background) * 0.01
        swc_data = draw.neuron_from_swc(swc_lists[i],
                                        width=width,
                                        noise=noise,
                                        adjust=False,
                                        neuron_color=color,
                                        background_color=background,
                                        random_brightness=random_brightness,
                                        dropout=dropout,
                                        binary=binary) # Use simulated paths to draw the image.

        torch.save(swc_data, os.path.join(out, f"img_{i}.pt"))

    print("done")

    return


if __name__ == "__main__":
    main()