from glob import glob
import numpy as np
import os
from pathlib import Path
import scipy
import sys
import torch
from typing import Tuple

sys.path.append(str(Path(__file__).parent))
import draw
import load

def get_next_point(q0: np.ndarray, q1: np.ndarray, kappa: float, step_size: float=1.0, rng=None) -> np.ndarray:
    """
    Generate the next point in a path using a von Mises-Fisher distribution.
    
    Parameters
    ----------
    q0 : np.ndarray
        The starting point of the previous step.
    q1 : np.ndarray
        The ending point of the previous step.
    kappa : float
        The concentration parameter of the von Mises-Fisher distribution.
    step_size : float, optional
        The step size for the next point, by default 1.0.
    rng : np.random.Generator, optional
        A random number generator instance, by default None.
        
    Returns
    -------
    np.ndarray
        The next point in the tractography path.
    """
    
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
             random_start=True):
    """
    Get the neuron segment endpoints starting at a seed point,
    and ending if the path exits the boundary or reaches the path length.

    Parameters
    ----------
    start : Array or list of length 3.
        Path starting coordinate.
    boundary : np.ndarray of shape (2, 3)
        Image boundaries. Two vertices marking the minimum and maximum
        values along each dimension.
    kappa : float, optional
        Concentration parameter for step direction distribution.
    rng : np.random.Generator, optional
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
    random_start : bool, optional
        Whether to start the path with a random direction. Default is True.
    
    Returns
    -------
    path : np.ndarray of shape (N,3)

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
    """
    Generate a list of SWC formatted data representing a path with optional branches.
    
    Parameters
    ----------
    size : Tuple[int, ...]
        The dimensions of the 3D space.
    length : int
        The length of the path.
    step_size : float, optional
        The step size for each move in the path, by default 1.0.
    kappa : float, optional
        The concentration parameter for the von Mises-Fisher distribution, by default 20.0.
    uniform_len : bool, optional
        If True, the path length will be uniform, by default False.
    random_start : bool, optional
        If True, the path will start at a random position, by default True.
    rng : numpy.random.Generator, optional
        A random number generator instance, by default None.
    num_branches : int, optional
        The number of branches to generate, by default 0.
        
    Returns
    -------
    list
        A list of SWC formatted data representing the generated path and branches.
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


def save_images_from_swc(labels_dir, outdir, sync=True, random_contrast=False, rng=None):
    """
    Save images generated from SWC files to the specified output directory.
    
    Parameters
    ----------
    labels_dir : str
        Directory containing the SWC files.
    outdir : str
        Directory where the output images will be saved.
    sync : bool, optional
        If True, only save images for SWC files that do not have corresponding output files in the output directory (default is True).
    random_contrast : bool, optional
        If True, apply random contrast to the neuron images (default is False).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created (default is None).
        
    Returns
    -------
    None
    """

    if rng is None:
        rng = np.random.default_rng()

    files = [f for x in os.walk(labels_dir) for f in glob(os.path.join(x[0], '*.swc'))]
    if sync:
        outdir_fnames = [f for x in os.walk(outdir) for f in glob(os.path.join(x[0], '*.pt'))]
        files = [f for f in files if not os.path.splitext(f.split('/')[-1])[0] in outdir_fnames]

    for labels_file in files:
        swc_list = load.swc(labels_file)

        color = np.array([1.0, 1.0, 1.0])
        background = np.array([0., 0., 0.])
        if random_contrast:
            color = rng.uniform(size=3)
            color /= np.linalg.norm(color)
            background = rng.uniform(size=3)
            background = background / np.linalg.norm(background) * 0.01
            
        swc_data = draw.neuron_from_swc(swc_list,
                                        width=3,
                                        noise=0.0,
                                        dropout=False,
                                        adjust=True,
                                        background_color=background,
                                        neuron_color=color,
                                        random_brightness=False,
                                        binary=False,
                                        rng=rng)
        scale = swc_data.pop("scale")
        name = os.path.splitext(labels_file.split('/')[-1])[0]
        torch.save(swc_data, os.path.join(outdir, f"{name}_scale_{scale}x.pt"))
        
    return