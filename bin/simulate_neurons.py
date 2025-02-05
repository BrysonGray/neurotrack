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

if __name__ == "__main__":

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

        print(f"Generating simulated neuron trees...\n\
            size: {size}\n\
            length: {length}\n\
            step size: {stepsize}\n\
            uniform_len: {uniform_len}\n\
            kappa: {kappa}\n\
            random_start: {random_start}\n\
            branches: {branches}")

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
            print("done")

    print(f"width: {width}\n\
          random_contrast: {random_contrast}\n\
          random_brightness: {random_brightness}\n\
          dropout: {dropout}\n\
          noise: {noise}\n\
          binary: {binary}\n\
          seed: {seed}\n\
          Drawing neuron images and saving to {out}...")
    
    for swc_list in swc_lists:
        color = np.array([1.0, 1.0, 1.0])
        background = np.array([0., 0., 0.])
        if random_contrast:
            color = np.random.rand(3)
            color /= np.linalg.norm(color)
            background = np.random.rand(3)
            background = background / np.linalg.norm(background) * 0.01
        swc_data = draw.neuron_from_swc(swc_list,
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