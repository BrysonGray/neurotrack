from glob import glob
import numpy as np
import os
from pathlib import Path
import scipy
import sys
import torch
from typing import Tuple, Union

sys.path.append(str(Path(__file__).parent))
import draw
from draw import NeuronRenderer, DrawingConfig
import load


def rvmf(n, mu, k, rng=None):
    """Random sampling from the von Mises-Fisher distribution
    
    Parameters:
    -----------
    n : int
        Sample size
    mu : numpy.ndarray
        Mean direction
    k : float
        Concentration parameter
        
    Returns:
    --------
    numpy.ndarray
        Matrix of random samples
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(mu)
    
    if k > 0:
        mu = mu / np.sqrt(np.sum(mu**2))
        ini = np.zeros(d)
        ini[-1] = 1.0
        d1 = d - 1
        
        v1 = rng.normal(0, 1, (n, d1))
        v_rows_norm = np.sqrt(np.sum(v1**2, axis=1))
        v = v1 / v_rows_norm[:, np.newaxis]
        
        b = (-2 * k + np.sqrt(4 * k**2 + d1**2)) / d1
        x0 = (1 - b) / (1 + b)
        m = 0.5 * d1
        ca = k * x0 + (d - 1) * np.log(1 - x0**2)
        
        # Modified rejection sampling
        w = np.zeros(n)
        for i in range(n):
            accepted = False
            while not accepted:
                z = rng.beta(m, m)
                u = rng.uniform(0, 1)
                w_i = (1 - (1 + b) * z) / (1 - (1 - b) * z)
                ta = k * w_i + d1 * np.log(1 - x0 * w_i)
                if ta - ca >= np.log(u):
                    w[i] = w_i
                    accepted = True
                    
        S = np.column_stack((np.sqrt(1 - w**2)[:, np.newaxis] * v, w))
        
        # Rotation logic from ini to mu
        if np.allclose(ini, mu):
            x = S
        elif np.allclose(-ini, mu):
            x = -S
        else:
            # Implement rotation matrix calculation
            # This is a simplified version - would need proper implementation
            a = ini
            b = mu
            ab = np.sum(a * b)
            ca = a - b * ab
            ca = ca / np.sqrt(np.sum(ca**2))
            A = np.outer(b, ca) - np.outer(ca, b)
            theta = np.arccos(ab)
            rotation_matrix = np.eye(d) + np.sin(theta) * A + (np.cos(theta) - 1) * (np.outer(b, b) + np.outer(ca, ca))
            
            x = S @ rotation_matrix.T
    else:
        # Uniform distribution on sphere
        x1 = rng.normal(0,1,(n,d))
        x = x1 / np.sqrt(np.sum(x1**2, axis=1))[:, np.newaxis]
    
    return x


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
    last_step = (q1[:3] - q0[:3]) / step_size
    # vmf = scipy.stats.vonmises_fisher(last_step, kappa)
    # step = vmf.rvs(1, random_state=rng)[0]
    step = rvmf(1, last_step, kappa, rng)[0]
    # step[0] = 0.0 # for paths constrained to a 2d slice
    step = step/(np.linalg.norm(step) + np.finfo(float).eps)
    next_point = q1[:3] + step * step_size

    return next_point


def clipped_normal(mu, sigma, low, high, rng=None):
    """
    Generate a random number from a truncated normal distribution.
    
    Parameters
    ----------
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution.
    low : float
        Lower bound for truncation.
    high : float
        Upper bound for truncation.
    rng : np.random.Generator, optional
        Random number generator instance, by default None.
        
    Returns
    -------
    float
        A random number from the truncated normal distribution.
    """
    
    if rng is None:
        rng = np.random.default_rng()
    x = rng.normal(mu, sigma)
    x = np.clip(x, low, high)
    return x


def path_length(path: Union[np.ndarray, list]) -> float:
    """
    Calculate the length of a path defined by a sequence of 3D points.

    Parameters
    ----------
    path : Union[np.ndarray, list]
        A sequence of 3D points defining the path.
        
    Returns
    -------
    float
        The total length of the path.
    """
    
    if isinstance(path, list):
        path = np.array(path)
    if not (isinstance(path, np.ndarray) and path.ndim == 2 and path.shape[1] >= 3):
        raise ValueError("Input 'path' must be a 2D array with at least 3 columns (x, y, z coordinates).")
    diffs = np.diff(path[:,:3], axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    total_length = np.sum(seg_lengths)
    
    return total_length


# compute neuron segment end points 
def get_path(start,
             boundary,
             kappa=20.0,
             rng=None,
             length=500,
             step_size=1.0,
             width=3.0,
             random_len=True,
             random_width=False,
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
        Path length in pixels. This is the expected path length
        if uniform_len is set to False. The minimum length is 50 if uniform_len is False.
        Default is 500.
    step_size : float, optional
        Length of each path segment in pixels. Default is 1.0
    random_len : bool
        If True, the path length will be sampled from a normal distribution with mean given by length, by default True.
    random_width : bool, optional
        If True, the width of the path segments will be randomly sampled from a truncated normal distribution
    random_start : bool, optional
        Whether to start the path with a random direction. Default is True.
    
    Returns
    -------
    path : np.ndarray of shape (N,3)

    """
    if rng is None:
        rng = np.random.default_rng()

    if random_len:
        sigma = length / 5
        length = length + rng.standard_normal(1)*sigma
        length = max(length, 50)

    # first step
    if random_start:
        step = rng.normal(0.0, 1.0, 3)
        step = step / sum(step**2)**0.5
    else:
        step = np.array([0.0,0.0,1.0])
    q1 = start + step * step_size
    if random_width:
        w0 = clipped_normal(width, 0.5, 2.0, 12.0, rng=rng)
        w1 = clipped_normal(w0, 0.5, 2.0, 12.0, rng=rng)
    else:
        w0 = width
        w1 = width
    start = np.concatenate((start, [w0]))
    q1 = np.concatenate((q1, [w1]))
    path = [start, q1]

    while path_length(path) < length: # length in pixels
        next_point = get_next_point(path[-2], path[-1], kappa=kappa, step_size=step_size, rng=rng)
        if any(next_point > boundary.max(axis=0)) or any(next_point < boundary.min(axis=0)):
            break
        if random_width:
            w = clipped_normal(path[-1][3], 0.5, 2.0, 12.0, rng=rng)
        else:
            w = width
        next_point = np.concatenate((next_point, [w]))
        path.append(next_point)
    
    path = np.array(path)

    return path
 

def make_swc_list(size: Tuple[int,...],
                length: float,
                step_size: float = 1.0,
                kappa: float = 20.0,
                random_len: bool = True,
                random_start: bool = True,
                random_width: bool = False,
                rng=None,
                num_branches: int=0) -> list:
    """
    Generate a list of SWC formatted data representing a path with optional branches.
    
    Parameters
    ----------
    size : Tuple[int, ...]
        The dimensions of the 3D space.
    length : float
        The length of the path in pixels.
    step_size : float, optional
        The step size for each move in the path, by default 1.0.
    kappa : float, optional
        The concentration parameter for the von Mises-Fisher distribution, by default 20.0.
    random_len : bool, optional
        If True, the path length will be sampled from a normal distribution with mean given by length, by default True.
    random_start : bool, optional
        If True, the path will start at a random position, by default True.
    random_width : bool, optional
        If True, the width of the path segments will be randomly sampled, by default False.
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
    path = get_path(start, boundary=boundary, kappa=kappa, rng=rng, length=length, step_size=step_size, random_len=random_len,
                    random_width=random_width, random_start=random_start)
    graph = [[i+1, i] for i in range(len(path))]
    graph[0][1] = -1
    paths = [path]
    branch_points = []
    for i in range(num_branches):
        start_idx = rng.integers(0, len(path)-1)
        branch_start = paths[0][start_idx][:3]
        branch_points.append(branch_start)
        branch_start = tuple(int(np.round(t)) for t in branch_start)
        new_path = get_path(branch_start, boundary=boundary, kappa=kappa, rng=rng, length=length, step_size=step_size, random_len=random_len,
                    random_width=random_width, random_start=True)
        graph.append([graph[-1][0]+1, start_idx+1])
        for i in np.arange(graph[-1][0], graph[-1][0] + len(new_path)-1):
            graph.append([i+1, i])
        paths.append(new_path)
    paths = np.concatenate(paths)

    swc_list = [[graph[i][0], 0]+list(paths[i][:3])+[paths[i][3], graph[i][1]] for i in range(len(graph))]
    
    return swc_list


def save_images_from_swc(labels_dir, outdir, sync=True, random_contrast=False, rng=None):
    """
    Saves images from SWC files using the improved NeuronRenderer API.

    Parameters
    ----------
    labels_dir : str
        Path to directory containing SWC files.
    outdir : str
        Path to output directory.
    sync : bool, optional
        Whether to skip files that already have outputs, by default True.
    random_contrast : bool, optional
        Whether to use random contrast, by default False.
    rng : numpy.random.Generator, optional
        Random number generator, by default None.

    Returns
    -------
    None
    """

    if rng is None:
        rng = np.random.default_rng()

    # Initialize the renderer once for better performance
    renderer = NeuronRenderer(rng=rng)

    files = [f for x in os.walk(labels_dir) for f in glob(os.path.join(x[0], '*.swc'))]
    if sync:
        outdir_fnames = [f for x in os.walk(outdir) for f in glob(os.path.join(x[0], '*.pt'))]
        files = [f for f in files if not os.path.splitext(f.split('/')[-1])[0] in outdir_fnames]

    for labels_file in files:
        swc_list = load.swc(labels_file)

        # Configure colors
        color = np.array([1.0, 1.0, 1.0])
        background = np.array([0., 0., 0.])
        if random_contrast:
            color = rng.uniform(size=3)
            color /= np.linalg.norm(color)
            background = rng.uniform(size=3)
            background = background / np.linalg.norm(background) * 0.01
            
        # Create clean configuration object
        config = DrawingConfig(
            width=3,
            rgb=True,
            neuron_color=tuple(color),
            background_color=tuple(background)
        )
        
        # Use the new cleaner API
        swc_data = renderer.neuron_from_swc(
            swc_list, 
            config=config,
            dropout=False, 
            adjust=True
        )
        scale = swc_data.pop("scale")
        name = os.path.splitext(labels_file.split('/')[-1])[0]
        torch.save(swc_data, os.path.join(outdir, f"{name}_scale_{scale}x.pt"))
        
    return


if __name__ == "__main__":
    pass