from datetime import datetime
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import tifffile as tf
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
import load, draw
from image import Image, extract_spherical_patch


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


def random_points_from_mask(mask, branches, samples_per_neuron, rng=None):
    
    if len(branches) == 0:
       raise Exception("Branches list must not be empty.")         
    if rng is None:
        rng = np.random.default_rng()

    # create branch mask
    branch_mask = Image(torch.zeros_like(mask, dtype=torch.bool))
    for point in branches:
        branch_mask.draw_point(point[:3], radius=point[3].item(), binary=True, value=1, channel=0)

    # get a random sample of branch coordinates
    branch_coords = np.argwhere(branch_mask.data[0])
    branch_coords = rng.choice(branch_coords, int(samples_per_neuron/2), replace=False, axis=1)

    # get a random sample of non-branch neuron coordinates
    neuron_nonbranch = mask[0] & ~branch_mask.data[0]
    neuron_nonbranch_coords = np.argwhere(neuron_nonbranch)
    neuron_nonbranch_coords = rng.choice(neuron_nonbranch_coords, int(samples_per_neuron/4), replace=False, axis=1)

    # background is voxels not in the neuron mask or branch mask
    background = ~(mask[0] & branch_mask.data[0])
    del mask

    # get sample_per_neuron/4 background samples
    # background mask has too many voxels to use argwhere. Instead, sample randomly and keep foreground coordinates
    background_coords = []
    shape = background.shape
    i = 0
    while i < int(samples_per_neuron/4):
        # randomly sample a point in the image
        z = rng.integers(0, shape[0])
        y = rng.integers(0, shape[1])
        x = rng.integers(0, shape[2])
        # if the point is in background, add it to the list
        if background[z, y, x]:
            background_coords.append([z, y, x])
            i += 1

    del background

    non_branch_coords = np.concatenate((background_coords, neuron_nonbranch_coords.T), axis=0)

    return branch_coords.T, non_branch_coords


def save_spherical_patches(img, branch_coords, non_branch_coords, out_dir, resolution=(180, 360), start_id=0, annotations=None):

    obs_id = start_id
    if annotations is None:
        annotations = {}

    # Sample spherical patches from the corresponding neuron image
    # img = tf.imread(img_path)
    img = img / img.max()
    permutations = [[0,1,2],
                    [0,2,1],
                    [1,2,0],
                    [1,0,2],
                    [2,0,1],
                    [2,1,0]]
    # Create meshgrid for spherical coordinates
    theta_res, phi_res = resolution
    theta = np.linspace(0, np.pi, theta_res)
    phi = np.linspace(0, 2*np.pi, phi_res)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to cartesian coordinates (points on a unit sphere)
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    # Sample branch positive patches
    for i in range(len(branch_coords)):
        spherical_patches = []
        for r in range(3,55,3):
            for perm in permutations:
                patch = extract_spherical_patch(img, x, y, z, branch_coords[i], radius=r, permutation=perm)
                spherical_patches.append(patch)
        patch = np.stack(spherical_patches, axis=0)
        fname = f"obs_{obs_id}.pt"
        torch.save(torch.from_numpy(patch), os.path.join(os.path.join(out_dir, "observations"), fname))
        annotations[fname] = 1
        obs_id += 1
    # Sample non branch patches
    for i in range(len(non_branch_coords)):
        spherical_patches = []
        for r in range(3,55,3):
            for perm in permutations:
                patch = extract_spherical_patch(img, non_branch_coords[i], radius=r, permutation=perm)
                spherical_patches.append(patch)
        patch = np.stack(spherical_patches, axis=0)
        fname = f"obs_{obs_id}.pt"
        torch.save(torch.from_numpy(patch), os.path.join(os.path.join(out_dir, "observations"), fname))
        annotations[fname] = 0
        obs_id += 1

    return annotations, obs_id


def save_coordinates_and_annotations(swc_dir, img_dir, out_dir, samples_per_neuron=100, seed=0):
    rng = np.random.default_rng(seed)

    sample_points = {}
    annotations = {}
    swc_files = os.listdir(swc_dir)
    img_files = os.listdir(img_dir)
    for i in tqdm(range(len(swc_files))):
        swc_list = load.swc(os.path.join(swc_dir, swc_files[i]), verbose=False)
        img_name = [img_file for img_file in img_files if img_file.split('.tif')[0] in swc_files[i]]
        try:
            img_name = img_name[0]
        except IndexError:
            continue
        img_path = os.path.join(img_dir, img_name)
        img = tf.imread(img_path)
        shape = img.shape
        del img

        sections, sections_graph = load.parse_swc(swc_list)
        branches, terminals = load.get_critical_points(swc_list, sections)

        segments = []
        for section in sections.values():
            segments.append(section)
        segments = np.concatenate(segments)

        density = draw.draw_neuron_density(segments, shape)
        mask = draw.draw_neuron_mask(density, threshold=5.0)
        del density

        branch_coords, non_branch_coords = random_points_from_mask(mask, branches, samples_per_neuron, rng=rng)
        sample_points[img_name] = np.concatenate((branch_coords, non_branch_coords))
        current_size = len(annotations)
        for i in range(current_size,len(sample_points)+current_size):
            k = i - current_size < len(branch_coords)
            annotations[f"obs_{i}.pt"] = k

        # overwrite sample points and annotations after every file
        np.save(os.path.join(out_dir, "sample_points.npy"), sample_points)
        # save annotations
        # split into test and training data
        name = "gold166"
        data_permutation = torch.randperm(len(annotations))
        test_idxs = data_permutation[:len(data_permutation)//5].tolist()
        training_idxs = data_permutation[len(data_permutation)//5:].tolist()
        training_annotations = {list(annotations)[i]: list(annotations.values())[i] for i in training_idxs}
        test_annotations = {list(annotations)[i]: list(annotations.values())[i] for i in test_idxs}
        # save
        date = datetime.now().strftime("%m-%d-%y")
        df = pd.DataFrame.from_dict(training_annotations, orient="index")
        df.to_csv(os.path.join(out_dir, f"branch_classifier_{name}_{date}_training_labels.csv"))
        df = pd.DataFrame.from_dict(test_annotations, orient="index")
        df.to_csv(os.path.join(out_dir, f"branch_classifier_{name}_{date}_test_labels.csv"))

    return



def spherical_patch_dataset(swc_dir, img_dir, out_dir, samples_per_neuron=100, sync=False, seed=0):

    rng = np.random.default_rng(seed)
    obs_id = 0
    ncomplete = 0
    if sync:
        existing_files = os.listdir(os.path.join(out_dir, "observations"))
        ids = [int(f.split('.')[0].split('_')[1]) for f in existing_files]
        obs_id = max(ids) + 1
        ncomplete = obs_id//1000

    annotations = {}
    swc_files = os.listdir(swc_dir)
    img_files = os.listdir(img_dir)
    for i in tqdm(range(ncomplete, len(swc_files))):
        swc_list = load.swc(os.path.join(swc_dir, swc_files[i]), verbose=False)
        img_name = [img_file for img_file in img_files if img_file.split('.tif')[0] in swc_files[i]]
        try:
            img_name = img_name[0]
        except IndexError:
            continue
        img_path = os.path.join(img_dir, img_name)
        img = tf.imread(img_path)
        shape = img.shape
        # del img

        sections, sections_graph = load.parse_swc(swc_list)
        branches, terminals = load.get_critical_points(swc_list, sections)
        if len(branches)==0:
            continue
        segments = []
        for section in sections.values():
            segments.append(section)
        segments = np.concatenate(segments)

        density = draw.draw_neuron_density(segments, shape)
        mask = draw.draw_neuron_mask(density, threshold=5.0)
        del density

        branch_coords, non_branch_coords = random_points_from_mask(mask, branches, samples_per_neuron, rng=rng)
        
        annotations, obs_id = save_spherical_patches(img, branch_coords, non_branch_coords, out_dir, start_id=obs_id, annotations=annotations)

    # save annotations
    # split into test and training data
    name = "gold166"
    data_permutation = torch.randperm(len(annotations))
    test_idxs = data_permutation[:len(data_permutation)//5].tolist()
    training_idxs = data_permutation[len(data_permutation)//5:].tolist()
    training_annotations = {list(annotations)[i]: list(annotations.values())[i] for i in training_idxs}
    test_annotations = {list(annotations)[i]: list(annotations.values())[i] for i in test_idxs}
    # save
    date = datetime.now().strftime("%m-%d-%y")
    df = pd.DataFrame.from_dict(training_annotations, orient="index")
    df.to_csv(os.path.join(out_dir, f"branch_classifier_{name}_{date}_training_labels.csv"))
    df = pd.DataFrame.from_dict(test_annotations, orient="index")
    df.to_csv(os.path.join(out_dir, f"branch_classifier_{name}_{date}_test_labels.csv"))

    return


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