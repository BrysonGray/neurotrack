from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parent))
import load
from image import Image


def swc_random_points(samples_per_file, swc_lists, file_names, adjust=False, rng=None):
    """
    Choose random points near the neuron coordinates from swc data.
    
    Parameters
    ----------
    samples_per_file : int
        Number of samples to take from each swc file
    swc_lists : list
        A list of neuron tree data each represented as a list of nodes.
    file_names : list
        A list of file names corresponding to each swc list in 'swc_lists'.
    adjust : bool, optional
        Whether the images generated from swc file data preserved swc coordinates as voxel indices or adjusted
        the coordinates with relation to voxel indices. Default is False.
    rng : numpy.random.Generator, optional
        Sets random number generator for random point selection.
    
    Returns
    -------
    sample_points : dict
        Dictionary whose keys are the file names from 'file_names' and values are numpy arrays
        of random points chosen near the neuron coordinates.
    """
    
    if rng is None:
        rng = np.random.default_rng()
    sample_points = {}
    for fname, swc_list in zip(file_names,swc_lists):
        sections, section_graph, branches, terminals, scale = load.parse_swc_list(swc_list, adjust=adjust)
        rand_sections = rng.choice(list(sections.keys()), size=samples_per_file)
        points = []
        for j in rand_sections:
            section_flat = sections[j].flatten(0,1) # type: ignore # 
            random_point = rng.choice(np.arange(len(section_flat)))
            random_point = section_flat[random_point]
            # random translation vector from normal distribution about random_point
            translation = rng.uniform(low=0.0, high=1.0, size=(3,))*8.0 - 4.0
            random_point += translation
            points.append(random_point)
        points = np.array(points)
        
        sample_points[fname] = points
    
    return sample_points


def collect_data(sample_points, image_dir, out_dir, name, date, rng=None):
    """
    Collect data from images and save labels.

    Parameters
    ----------
    sample_points : dict
        Dictionary whose keys are the file names and values are numpy arrays of random points.
    image_dir : str
        Directory containing the images.
    out_dir : str
        Directory to save the output data.
    name : str
        Name for the output files.
    date : str
        Date for the output files.
    rng : numpy.random.Generator, optional
        Random number generator for data collection.
    """

    if rng is None:
        rng = np.random.default_rng()

    os.makedirs(os.path.join(out_dir,"observations"), exist_ok=True)
    image_files = os.listdir(image_dir)
    annotations = {}
    obs_id = 0
    for f in image_files:
        points = sample_points[f.split('.')[0]]
        data = torch.load(os.path.join(image_dir, f), weights_only=True)
        img = data["image"]
        img = Image(img)
        branch_mask = data["branch_mask"]
        for point in points:
            patch, _ = img.crop(torch.tensor(point), 7, pad=True, value=0.0)
            i,j,k = [int(np.round(x)) for x in point]
            label = branch_mask[0, i, j, k].item()
            fname = f"obs_{obs_id}.pt"
            torch.save(patch, os.path.join(os.path.join(out_dir, "observations"), fname))
            annotations[fname] = label
            obs_id += 1

    # save annotations
    # split into test and training data
    data_permutation = torch.randperm(len(annotations))
    test_idxs = data_permutation[:len(data_permutation)//5].tolist()
    training_idxs = data_permutation[len(data_permutation)//5:].tolist()
    training_annotations = {list(annotations)[i]: list(annotations.values())[i] for i in training_idxs}
    test_annotations = {list(annotations)[i]: list(annotations.values())[i] for i in test_idxs}
    # save 
    df = pd.DataFrame.from_dict(training_annotations, orient="index")
    df.to_csv(os.path.join(out_dir, f"branch_classifier_{name}_{date}_training_labels.csv"))
    df = pd.DataFrame.from_dict(test_annotations, orient="index")
    df.to_csv(os.path.join(out_dir, f"branch_classifier_{name}_{date}_test_labels.csv"))

    return

if __name__ == "__main__":
    pass