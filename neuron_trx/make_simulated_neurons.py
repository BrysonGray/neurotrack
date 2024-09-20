#!/usr/bin/env python

""" make simulated neuron images """
import numpy as np
import torch
import scipy
import sys
import os
import argparse
from typing import Tuple
from tqdm import tqdm
from skimage.morphology import binary_dilation, cube

sys.path.append(os.path.dirname(sys.path[0]))
import image


def get_next_point(q0: np.ndarray, q1: np.ndarray, kappa: float, step_size: float = 1.0, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    last_step = (q1 - q0) / step_size
    vmf = scipy.stats.vonmises_fisher(last_step, kappa)
    step = vmf.rvs(1, random_state=rng)[0]
    step[0] = 0.0
    step = step/(np.linalg.norm(step) + np.finfo(float).eps)
    next_point = q1 + step * step_size

    return next_point


# compute neuron segment end points 
def get_path(start, boundary, kappa=20.0, rng=None, length=100, step_size=1.0, uniform_len=False, random_start=True):
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


def draw_path(img, path, width, binary):
    if isinstance(path, list):
        path = torch.tensor(path)
    elif isinstance(path, np.ndarray):
        path = torch.from_numpy(path)

    segments = torch.stack((path[:-1],path[1:]), dim=1)
    for s in segments:
        img.draw_line_segment(s, width=width, binary=binary, channel=0)

    return img


def make_neuron_img(size: Tuple[int,...],
                    length: int,
                    step_size: float = 1.0,
                    width: float = 3.0,
                    kappa: float = 20.0,
                    noise: float = 0.05,
                    uniform_len: bool = False,
                    random_start: bool = True,
                    binary: bool = False,
                    rng=None) -> dict:
    """ Make simulated neuron 3D image. 

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

    img = torch.zeros((1,)+size)
    img = image.Image(img)
    mask = torch.zeros_like(img.data)
    mask = image.Image(mask)

    start = tuple([x//2 for x in size]) # start in the center
    boundary = np.array([[0,0,0],
                         [size[0]-1, size[1]-1, size[2]-1]])
    path = get_path(start, boundary=boundary, kappa=kappa, rng=rng, length=length, step_size=step_size, uniform_len=uniform_len,
                    random_start=random_start)

    img = draw_path(img, path, width=width, binary=False, )
    img_data = torch.cat((img.data, img.data, img.data), dim=0)
    sigma = img_data.amax() * noise
    img_data = img_data + torch.randn(img_data.shape)*sigma # add noise
    img_data = (img_data - img_data.amin()) / (img_data.amax() - img_data.amin()) # rescale to [0,1]
    img = image.Image(img_data)

    neuron_mask = draw_path(mask, path, width=width, binary=binary)
    if binary:
        tracking_mask = neuron_mask.data[0] > np.exp(-1)
    tracking_mask = binary_dilation(neuron_mask.data[0], cube(18))
    tracking_mask = image.Image(tracking_mask[None])

    return {"image": img.data, "neuron_mask": neuron_mask.data, "tracking_mask": tracking_mask.data}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, help='Output directory.')
    parser.add_argument('-c', '--count', help="Number of images to create.", type=int, default=100, required=False)
    parser.add_argument('-s', '--size', type=int, default="101 101 101", nargs='+', required=False, help="Size along each image dimension (the image is a cube).")
    parser.add_argument('-t', '--stepsize', type=float, default=2.0, required=False, help="Length of each path segment in pixels.")
    parser.add_argument('-l', '--length', type=int, default=100, required=False, help="Expected length of each neuron in number of segments.")
    parser.add_argument('-w', '--width', type=int, default=3, required=False, help="Width of a neuron in pixels.")
    parser.add_argument('-n', '--noise', type=float, default=0.05, required=False, help="Standard deviation of Gaussian random noise relative to the maximum signal value.")
    parser.add_argument('-u', '--uniform_len', action="store_true", required=False, help="Choose whether all simulated neurons have the same number of segments or a random distribution of lengths.")
    parser.add_argument('--seed', type=int, default=0, required=False, help="Random seed used for creating random paths.")
    parser.add_argument('-k', '--kappa', type=float, default=20.0, required=False, help="Concentration parameter for step direction distribution function.")
    parser.add_argument('-r', '--random_start', action="store_true", required=False, help="Whether the starting step is random. If false, set to the +x direction")
    parser.add_argument('-b', '--binary', action="store_true", required=False, help="Whether the neuron label is binary or a line with Gaussian filter applied.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    # size = (args.size,)*3
    size = tuple(args.size)

    print(f"size: {size}\n\
    length: {args.length}\n\
    step size: {args.stepsize}\n\
    width: {args.width}\n\
    noise: {args.noise}\n\
    uniform_len: {args.uniform_len}\n\
    kappa: {args.kappa}\n\
    random_start: {args.random_start}\n\
    binary: {args.binary}\n")

    for i in tqdm(range(args.count)):    
        neuron = make_neuron_img(size,
                                length=args.length,
                                step_size=args.stepsize,
                                width=args.width,
                                noise=args.noise,
                                uniform_len=args.uniform_len,
                                kappa=args.kappa,
                                random_start=args.random_start,
                                rng=rng)
        torch.save(neuron, os.path.join(args.out, f"img_{i}.pt"))