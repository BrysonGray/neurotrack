#!/usr/bin/env python

""" make simulated neuron images """
import argparse
import json
import numpy as np
import os
import scipy
import torch
from typing import Tuple
from tqdm import tqdm

import load_data


# def draw_path(img, path, width, binary):
#     if isinstance(path, list):
#         path = torch.tensor(path)
#     elif isinstance(path, np.ndarray):
#         path = torch.from_numpy(path)

#     segments = torch.stack((path[:-1],path[1:]), dim=1)
#     for s in segments:
#         img.draw_line_segment(s, width=width, binary=binary, channel=0)

#     return img

# def make_neuron_img(size: Tuple[int,...],
#                     length: int,
#                     step_size: float = 1.0,
#                     width: float = 3.0,
#                     kappa: float = 20.0,
#                     noise: float = 0.05,
#                     uniform_len: bool = False,
#                     random_start: bool = True,
#                     rng=None,
#                     binary: bool = False,
#                     num_branches: int=0) -> dict:
#     """ Make simulated neuron 3D image. 

#     Parameters
#     ----------
#     size: tuple of int
#         Size of the image
#     length: int
#         Number of segments used to draw the neuron.
#     step_size: float, optional
#         Length of each path segment in pixels. Default is 2.0.
#     width: float, optional
#         Width of the neuron in pixels.
#     noise: float, optional
#         Standard deviation of Gaussian random noise relative to the maximum signal value. Default is 0.05.
#     uniform_len: bool, optional
#         Whether the neuron length is fixed or sampled from a distribution with mean equal to length.

#     Returns
#     -------
#     images: tuple of image.Image
#         The neuron image, a binary mask of the neuron, and a mask used to define out-of-bounds for tracking training.

#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     img = torch.zeros((1,)+size)
#     img = image.Image(img)
#     canvas = torch.zeros_like(img.data)
#     canvas = image.Image(canvas)

#     start = tuple([x//2 for x in size]) # start in the center
#     boundary = np.array([[0,0,0],
#                          [size[0]-1, size[1]-1, size[2]-1]])
#     path = get_path(start, boundary=boundary, kappa=kappa, rng=rng, length=length, step_size=step_size, uniform_len=uniform_len,
#                     random_start=random_start)
#     paths = [path]
#     branch_points = []
#     for i in range(num_branches):
#         start_idx = np.random.randint(len(path))
#         branch_start = paths[0][start_idx]
#         branch_points.append(branch_start)
#         branch_start = tuple(int(np.round(t)) for t in branch_start)
#         new_path = get_path(branch_start, boundary=boundary, kappa=kappa, rng=rng, length=length, step_size=step_size, uniform_len=uniform_len,
#                     random_start=True)
#         paths.append(new_path)

#     for i,path in enumerate(paths):
#         img = draw_path(img, path, width=width, binary=False, )
#     img_data = torch.cat((img.data, img.data, img.data), dim=0)
#     sigma = img_data.amax() * noise
#     img_data = img_data + torch.randn(img_data.shape)*sigma # add noise
#     img_data = (img_data - img_data.amin()) / (img_data.amax() - img_data.amin()) # rescale to [0,1]
#     img = image.Image(img_data)

#     for path in paths:
#         neuron_density = draw_path(canvas, path, width=width, binary=binary)
#     neuron_label = image.Image((neuron_density.data > np.exp(-1)).to(dtype=int))
#     for point in branch_points:
#         neuron_label.draw_point(torch.from_numpy(point), radius=3, binary=True, value=2)

#     return {"image": img.data, "neuron": neuron_density.data, "neuron_mask": neuron_label.data}


def get_next_point(q0: np.ndarray, q1: np.ndarray, kappa: float, step_size: float = 1.0, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    last_step = (q1 - q0) / step_size
    vmf = scipy.stats.vonmises_fisher(last_step, kappa)
    step = vmf.rvs(1, random_state=rng)[0]
    # step[0] = 0.0 # for paths constrained to a 2d slice
    step = step/(np.linalg.norm(step) + np.finfo(float).eps)
    next_point = q1 + step * step_size

    return next_point


# compute neuron segment end points 
def get_path(start,
             boundary,
             kappa=20.0,
             rng=None,
             length=100,
             step_size=1.0,
             uniform_len=False,
             random_start=True,
             branch=False):
    """
    Get the neuron segment endpoints starting at a seed point,
    and ending if the path exits the boundary or reaches the path length.

    Parameters
    ----------
    start : Array or list of length 3.
        Path starting coordinate.
    boundary : (2,3) array
        Image boundaries. Two vertices marking the minimum and maximum
        values along each dimension.
    kappa : float, optional
        Concentration parameter for step direction distribution.
    rng : numpy.random._generator.Generator, optional
    length : int, optional
        Path length in number of segments. This is the expected path length
        if uniform_len is set to False. The minimum length is 10 if uniform_len is False.
        Default is 100.
    step_size : float, optional
        Length of each path segment in pixels. Default is 1.0
    uniform_len : bool
        If false, the path length will be a normally distributed random number
        with the expected value set by the "length" argument. Otherwise the
        path length will be equal to "length".
    
    Returns
    -------
    path : (N, 3) array

    """
    if rng is None:
        rng = np.random.default_rng()

    if not uniform_len:
        sigma = length // 5
        length = length + rng.standard_normal(1)*sigma
        length = int(round(length.item()))
        length = length if length > 10 else 10

    # first step
    if random_start:
        step = rng.normal(0.0, 1.0, 3)
        step[0] = 0.0
        step = step / sum(step**2)**0.5
    else:
        step = np.array([0.0,0.0,1.0])
    q1 = start + step * step_size
    path = [start, q1]

    for i in range(length):
        next_point = get_next_point(path[-2], path[-1], kappa=kappa, step_size=step_size, rng=rng)
        if any(next_point > boundary.max(axis=0)) or any(next_point < boundary.min(axis=0)):
            break
        path.append(next_point)
    
    path = np.array(path)

    return path


def make_swc_list(size: Tuple[int,...],
                    length: int,
                    step_size: float = 1.0,
                    kappa: float = 20.0,
                    uniform_len: bool = False,
                    random_start: bool = True,
                    rng=None,
                    num_branches: int=0) -> list:
    """ Make simulated neuron tree in the format of an swc list. 

    Parameters
    ----------
    size: tuple of int
        Size of the image
    length: int
        Number of segments used to draw the neuron.
    step_size: float, optional
        Length of each path segment in pixels. Default is 2.0.
    width: float, optional
        Width of the neuron in pixels.
    noise: float, optional
        Standard deviation of Gaussian random noise relative to the maximum signal value. Default is 0.05.
    uniform_len: bool, optional
        Whether the neuron length is fixed or sampled from a distribution with mean equal to length.

    Returns
    -------
    images: tuple of image.Image
        The neuron image, a binary mask of the neuron, and a mask used to define out-of-bounds for tracking training.

    """
    if rng is None:
        rng = np.random.default_rng()

    start = tuple([x//2 for x in size]) # start in the center
    boundary = np.array([[0,0,0],
                         [size[0]-1, size[1]-1, size[2]-1]])
    path = get_path(start, boundary=boundary, kappa=kappa, rng=rng, length=length, step_size=step_size, uniform_len=uniform_len,
                    random_start=random_start)
    graph = [[i+1, i] for i in range(len(path))]
    paths = [path]
    branch_points = []
    for i in range(num_branches):
        start_idx = rng.integers(0, len(path)-1)
        branch_start = paths[0][start_idx]
        branch_points.append(branch_start)
        branch_start = tuple(int(np.round(t)) for t in branch_start)
        new_path = get_path(branch_start, boundary=boundary, kappa=kappa, rng=rng, length=length, step_size=step_size, uniform_len=uniform_len,
                    random_start=True)
        graph.append([graph[-1][0]+1, start_idx+1])
        for i in np.arange(graph[-1][0], graph[-1][0] + len(new_path)-1):
            graph.append([i+1, i])
        paths.append(new_path)
    paths = np.concatenate(paths)

    swc_list = [[graph[i][0], 0]+list(paths[i])+[3.0, graph[i][1]] for i in range(len(graph))]
    
    return swc_list


#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--json', type=str, help='Path to input parameters json file.')
    args = parser.parse_args()
    parameters_file = args.json
    with open(parameters_file) as f:
        parameters = json.load(f)

    out = parameters["out"]
    count = parameters["count"]
    size = (parameters["size"],)*3
    length = parameters["length"]
    stepsize = parameters["stepsize"]
    width = parameters["width"]
    noise = parameters["noise"]
    uniform_len = parameters["uniform_len"]
    kappa = parameters["kappa"]
    random_start = parameters["random_start"]
    binary = parameters["binary"]
    branches = parameters["branches"]
    random_contrast = parameters["random_contrast"]
    dropout = parameters["dropout"]
    random_brightness = parameters["random_brightness"]
    rng = np.random.default_rng(parameters["seed"])

    if not os.path.exists(out):
        os.makedirs(out)

    print(f"size: {size}\n\
    length: {length}\n\
    step size: {stepsize}\n\
    width: {width}\n\
    noise: {noise}\n\
    uniform_len: {uniform_len}\n\
    kappa: {kappa}\n\
    random_start: {random_start}\n\
    binary: {binary}\n\
    branches: {branches}\n\
    random contrast: {random_contrast}\n\
    signal dropout: {dropout}\n\
    random brightness: {random_brightness}\n\
    seed: {parameters["seed"]}")

    for i in tqdm(range(count)):
        swc_list = make_swc_list(size,
                                 length=length,
                                 step_size=stepsize,
                                 kappa=kappa,
                                 uniform_len=uniform_len,
                                 random_start=random_start,
                                 rng=rng,
                                 num_branches=branches) # make simulated neuron paths.
        color = np.array([1.0, 1.0, 1.0])
        background = np.array([0., 0., 0.])
        if random_contrast:
            color = np.random.rand(3)
            color /= np.linalg.norm(color)
            background = np.random.rand(3)
            background = background / np.linalg.norm(background) * 0.01
        swc_data = load_data.draw_neuron_from_swc(swc_list,
                                                  width=width,
                                                  noise=noise,
                                                  adjust=False,
                                                  neuron_color=color,
                                                  background_color=background,
                                                  random_brightness=random_brightness,
                                                  dropout=dropout,
                                                  binary=binary) # Use simulated paths to draw the image.
        
        torch.save(swc_data, os.path.join(out, f"img_{i}.pt"))
