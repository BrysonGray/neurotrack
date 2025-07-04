import numpy as np
from pathlib import Path
from skimage.filters import gaussian 
import sys
import torch
import imageio

sys.path.append(str(Path(__file__).parent))
from image import Image
import load


def draw_neuron_density(segments, shape, width=None, scale=1.0):
    """
    Draws neuron density on an image based on given segments.
    
    Parameters
    ----------
    segments : array-like or torch.Tensor
        A list or tensor of neuron segments, where each segment is represented by a set of points.
    shape : tuple of int
        The shape of the output density image (height, width, depth).
    width : int, optional
        The width of the line segments to be drawn, by default 3.
    scale : float
        The pixel size
        
    Returns
    -------
    Image
        An image object with the neuron density drawn on it.
    """
    
    # create density image
    density = Image(torch.zeros((1,)+shape))

    if not isinstance(segments, torch.Tensor):
        segments = torch.tensor(segments)
    if width is None and segments.shape[2] == 4: # segments include width in the last column
        for s in segments:
            width = ((s[0,3]+s[1,3])/2).item() / scale # convert from real coordinates to pixels
            density.draw_line_segment(s[:,:3], width=width, channel=0)
    else:
        if width is None:
            raise ValueError("Width must be specified when segments don't include width information")
        for s in segments:
            density.draw_line_segment(s[:,:3], width=width, channel=0)
    
    return density


def draw_neuron_mask(density, threshold=1.0):
    """ Create a binary mask from the neuron density image.
    
    Parameters
    ----------
    density : torch.Tensor
        Neuron density image.
    
    threshold : float
        Threshold value for classifying a voxel in the neuron density image as inside the neuron.
        The threshold value is relative to the width of the neuron. Specifically, the mask will label
        as neuron voxels within one standard deviation from the peak neuron value, where the neuron
        intensities are assumed to be normally distributed around the centerline.
    
    Returns
    -------
    mask : torch.Tensor
        A binary mask of the neuron.
    """

    peak = density.data.amax()
    mask = torch.zeros_like(density.data, dtype=torch.bool)
    mask[density.data > peak * np.exp(-0.5 * threshold)] = True

    return mask


def draw_section_labels(sections, shape, width=3):
    """
    Draws discrete labels for each section on an image.
    
    Parameters
    ----------
    sections : dict
        A dictionary where keys are section labels and values are lists of segments.
        Each segment is a numpy array with shape (n, 3) representing the coordinates.
    shape : tuple
        The shape of the output image (height, width, depth).
    width : int, optional
        The width of the line segments to be drawn, by default 3.
        
    Returns
    -------
    Image
        An image object with the drawn sections labeled.
    """
    
    # create discrete labels for each section
    labels = Image(torch.zeros((1,)+shape))
    for i, section in sections.items():
        for segment in section:
            labels.draw_line_segment(segment[:,:3], width=width, channel=0, binary=True, value=i)
    
    return labels


def draw_path(img, path, width, binary):
    """
    Draws a path on the given image.
    
    Parameters
    ----------
    img : object
        The image object on which the path will be drawn. It should have a method `draw_line_segment`.
    path : list or numpy.ndarray or torch.Tensor
        The path to be drawn. It can be a list of coordinates, a numpy array, or a torch tensor.
    width : int
        The width of the line segments to be drawn.
    binary : bool
        If True, the line segments will be drawn in binary mode.
        
    Returns
    -------
    object
        The image object with the path drawn on it.
    """
    
    if isinstance(path, list):
        path = torch.tensor(path)
    elif isinstance(path, np.ndarray):
        path = torch.from_numpy(path)

    segments = torch.stack((path[:-1],path[1:]), dim=1)
    for s in segments:
        img.draw_line_segment(s[:,:3], width=width, binary=binary, channel=0)

    return img


def draw_neuron(segments, shape, width, noise, neuron_color=None, background_color=None, random_brightness=False, binary=False, rng=None, save_gif=False, gif_path=None, gif_axis=0):
    """
    Draws a neuron image based on provided segments and parameters.
    Optionally saves a gif animating the drawing process.
    
    Parameters
    ----------
    segments : list of ndarray
        List of segments where each segment is an ndarray of shape (N, 3) representing the coordinates of the neuron segments.
    shape : tuple of int
        Shape of the output image (height, width).
    width : int
        Width of the neuron lines to be drawn.
    noise : float
        Standard deviation of the Gaussian noise to be added to the image.
    neuron_color : tuple of float, optional
        RGB color of the neuron lines. Each value should be in the range [0, 1]. Default is (1.0, 1.0, 1.0).
    background_color : tuple of float, optional
        RGB color of the background. Each value should be in the range [0, 1]. Default is None.
    random_brightness : bool, optional
        If True, random brightness will be applied to each segment. Default is False.
    binary : bool, optional
        If True, the image will be binary. Default is False.
    rng : numpy.random.Generator, optional
        Random number generator instance. Default is None, which uses numpy's default_rng.
    save_gif : bool, optional
        If True, saves a gif of the drawing process. Default is False.
    gif_path : str, optional
        Output path for the gif. Required if save_gif is True.
    gif_axis : int, optional
        Axis for max intensity projection (0, 1, or 2). Default is 0.
        
    Returns
    -------
    Image
        An Image object containing the drawn neuron.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = Image(torch.zeros((1,)+shape))
    value =  1.0
    n_segments = len(segments)
    gif_frames = []
    gif_steps = set(np.linspace(0, n_segments-1, 100, dtype=int)) if save_gif else set()
    for idx, s in enumerate(segments):
        if random_brightness:
            y0 = 0.5
            value = y0 + (1.0 - y0) * rng.uniform(0.0, 1.0, size=1).item()
        img.draw_line_segment(s[:,:3], width=width, binary=binary, channel=0, value=value)
        if save_gif and idx in gif_steps:
            arr = img.data[3].cpu().numpy()
            mip = arr.max(axis=gif_axis)
            # Normalize to 0-255 uint8
            mip = ((mip - mip.min()) / (mip.ptp() + 1e-8) * 255).astype(np.uint8)
            gif_frames.append(mip)
    if neuron_color is None:
        neuron_color = (1.0, 1.0, 1.0)

    img_data = torch.cat((neuron_color[0]*img.data, neuron_color[1]*img.data, neuron_color[2]*img.data), dim=0)
    if background_color is not None:
        img_data = img_data + torch.ones_like(img_data) * background_color[:,None,None,None]
        img_data /= img_data.amax()
    sigma = img_data.amax() * noise
    img_data = img_data + torch.from_numpy(rng.normal(size=img_data.shape))*sigma # add noise
    img_data = (img_data - img_data.amin()) / (img_data.amax() - img_data.amin()) # rescale to [0,1]
    img = Image(img_data)

    if save_gif and gif_path is not None and gif_frames:
        imageio.mimsave(gif_path, gif_frames, duration=0.05)

    return img


def neuron_from_swc(swc_list, width=3, noise=0.05, dropout=True, adjust=False, background_color=None, neuron_color=None, random_brightness=False, binary=False, rng=None):
    """
    Generate a neuron image from an SWC list.
    
    Parameters
    ----------
    swc_list : list
        List of SWC data representing neuron structure.
    width : int, optional
        Width of the neuron lines, by default 3.
    noise : float, optional
        Amount of noise to add to the neuron image, by default 0.05.
    dropout : bool, optional
        Whether to add random signal dropout, by default True.
    adjust : bool, optional
        Whether to adjust the SWC data, by default True.
    background_color : optional
        Background color of the neuron image, by default None.
    neuron_color : optional
        Color of the neuron, by default None.
    random_brightness : bool, optional
        Whether to apply random brightness to the neuron image, by default False.
    binary : bool, optional
        Whether to generate a binary image, by default False.
    rng : numpy.random.Generator, optional
        Random number generator, by default None.
        
    Returns
    -------
    dict
        Dictionary containing the following keys:
        - "image": torch.Tensor
            The generated neuron image.
        - "neuron_density": torch.Tensor
            The density map of the neuron.
        - "section_labels": torch.Tensor
            The section labels of the neuron.
        - "branch_mask": torch.Tensor
            The branch mask of the neuron.
        - "seeds": list
            List of seed points.
        - "scale": float
            Scale of the neuron.
        - "graph": dict
            Graph representation of the neuron.
    """
    
    if rng is None:
        rng = np.random.default_rng()

    # sections, graph, branches, terminals, scale = load.parse_swc_list(swc_list, adjust=adjust)
    sections, graph = load.parse_swc(swc_list)
    branches, terminals = load.get_critical_points(swc_list, sections)
    scale = 1.0
    if adjust:
        sections, branches, terminals, scale = load.adjust_neuron_coords(sections, branches, terminals)
    segments = []
    for section in sections.values():
        segments.append(section)
    segments = np.concatenate(segments)

    shape = np.ceil(np.max(segments[...,:3], axis=(0,1)))
    shape = shape.astype(np.uint16)
    shape = shape + np.array([10, 10, 10])  # type: ignore
    shape = tuple(shape.tolist())

    img = draw_neuron(segments, shape=shape, width=width, noise=noise, neuron_color=neuron_color,
                      background_color=background_color, random_brightness=random_brightness,
                      binary=binary, rng=rng)

    density = draw_neuron_density(segments, shape, width=width)
    section_labels = draw_section_labels(sections, shape, width=2*width)
    mask = draw_neuron_mask(density, threshold=5.0)

    if dropout: # add random signal dropout (subtract gaussian blobs)
        neuron_coords = torch.nonzero(section_labels.data)
        dropout_density = 0.001
        size = int(dropout_density * len(neuron_coords))
        if size > 0:
            rand_ints = rng.integers(0, len(neuron_coords), size=(size,))
            dropout_points = neuron_coords[rand_ints]
            dropout_points = dropout_points[:,1:].T
            dropout_img = torch.zeros_like(img.data)
            dropout_img[:, dropout_points[0], dropout_points[1], dropout_points[2]] = 1.0
            dropout_img = gaussian(dropout_img, sigma=0.5*width)
            dropout_img /= dropout_img.max()
            img.data = img.data - dropout_img
            img.data = torch.where(img.data < 0, 0.0, img.data)

    branch_mask = Image(torch.zeros_like(mask))
    for point in branches:
        branch_mask.draw_point(point[:3], radius=width/2, binary=True, value=1, channel=0)
    # set branch_mask.data to zero where mask is zero
    branch_mask.data = branch_mask.data * mask.data
    root_key = min(sections.keys())
    seed = sections[root_key][0,0,:3].round().astype(np.uint16).tolist() # type: ignore

    swc_data = {"image": img.data,
                "neuron_density": density.data,
                "section_labels": section_labels.data,
                "branch_mask": branch_mask.data,
                "seeds": [seed],
                "scale": scale,
                "graph": graph}

    return swc_data

if __name__ == "__main__":
    pass