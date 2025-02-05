import numpy as np
from pathlib import Path
from skimage.filters import gaussian 
import sys
import torch

sys.path.append(str(Path(__file__).parent))
from image import Image
import load


def draw_neuron_density(segments, shape, width=3):
    # create density image
    density = Image(torch.zeros((1,)+shape))

    if not isinstance(segments, torch.Tensor):
        segments = torch.tensor(segments)

    for s in segments:
        density.draw_line_segment(s[:,:3], width=width, channel=0)
    
    return density


def draw_neuron_mask(density, threshold=1.0):
    """ Create a binary mask from the neuron density image.
    Parameters
    ----------
    density: torch.Tensor
        Neuron density image.
    
    threshold: float
        Threshold value for classifying a voxel in the neuron density image as inside the neuron.
        The threshold value is relative to the width of the neuron. Specifically, the mask will label
        as neuron voxels within one standard deviation from the peak neuron value, where the neuron
        intensities are assumed to be normally distributed around the centerline.
    """

    peak = density.data.amax()
    mask = torch.zeros_like(density.data)
    mask[density.data > peak * np.exp(-0.5 * threshold)] = 1.0

    return mask


def draw_section_labels(sections, shape, width=3):
    # create discrete labels for each section
    labels = Image(torch.zeros((1,)+shape))
    for i, section in sections.items():
        for segment in section:
            labels.draw_line_segment(segment[:,:3], width=width, channel=0, binary=True, value=i)
    
    return labels


def draw_path(img, path, width, binary):
    if isinstance(path, list):
        path = torch.tensor(path)
    elif isinstance(path, np.ndarray):
        path = torch.from_numpy(path)

    segments = torch.stack((path[:-1],path[1:]), dim=1)
    for s in segments:
        img.draw_line_segment(s[:,:3], width=width, binary=binary, channel=0)

    return img


def draw_neuron(segments,
                shape,
                width,
                noise,
                neuron_color=None,
                background_color=None,
                random_brightness=False,
                binary=False,
                rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    img = Image(torch.zeros((1,)+shape))
    value =  1.0
    for s in segments:
        if random_brightness:
            y0 = 0.5
            value = y0 + (1.0 - y0) * rng.uniform(0.0, 1.0, size=1).item()
        img.draw_line_segment(s[:,:3], width=width, binary=binary, channel=0, value=value)
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

    return img


def neuron_from_swc(swc_list,
                         width=3,
                         noise=0.05,
                         dropout=True,
                         adjust=True,
                         background_color=None,
                         neuron_color=None,
                         random_brightness=False,
                         binary=False,
                         rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    sections, graph, branches, terminals, scale = load.parse_swc_list(swc_list, adjust=adjust)

    segments = []
    for section in sections.values():
        segments.append(section)
    segments = torch.concatenate(segments)

    shape = torch.ceil(torch.amax(segments, dim=(0,1)))
    shape = shape.to(torch.int)
    shape = shape + torch.tensor([10, 10, 10])  # type: ignore
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
        rand_ints = rng.integers(0, len(neuron_coords), size=(size,))
        dropout_points = neuron_coords[rand_ints]
        dropout_img = torch.zeros_like(img.data)
        dropout_img[:, *dropout_points[:,1:].T] = 1.0
        dropout_img = gaussian(dropout_img, sigma=0.5*width)
        dropout_img /= dropout_img.max()
        img.data = img.data - dropout_img
        img.data = torch.where(img.data < 0, 0.0, img.data)

    branch_mask = Image(torch.zeros_like(mask))
    for point in branches:
        branch_mask.draw_point(point, radius=3, binary=True, value=1, channel=0)
    # set branch_mask.data to zero where mask is zero
    branch_mask.data = branch_mask.data * mask.data

    seed = sections[1][0,0].round().to(int).tolist() # type: ignore

    swc_data = {"image": img.data,
                "neuron_density": density.data,
                "section_labels": section_labels.data,
                "branch_mask": branch_mask.data,
                "seeds": [seed],
                "scale": scale,
                "graph": graph}

    return swc_data