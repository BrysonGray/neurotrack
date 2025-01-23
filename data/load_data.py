#!/usr/bin/env python

"""
Deep Q-network tractography data loading functions.

Author: Bryson Gray
2024
"""
#%%

import random
import tifffile as tf
import os
import numpy as np
import torch
from data.data_utils import interp
import sys
from neurom.io.utils import load_morphology
from data.image import Image
from skimage.filters import gaussian 
from skimage.morphology import dilation, cube



def parse_labels(file, scale=1.0):
    # load label
    label = load_morphology(file)

    segments = []
    branch_points = []
    terminals = []
    ids = []
    for i in range(len(label.sections)):
        points = label.sections[i].points
        starts = []
        ends = []
        for ii in range(len(points)-1):
            if np.all(points[ii,:3] == points[ii+1,:3]):
                continue
            else:
                starts.append(points[ii])
                ends.append(points[ii+1])
        segments_ = np.stack((starts, ends), axis=1)
        r = (segments_[:,0,-1]+segments_[:,1,-1])/2
        segments_[:,:,-1] = np.stack((r,r), axis=-1)

        segments.append(segments_)
        if len(label.sections[i].children) > 0:
            branch_points.append(segments_[-1,1,:3])
        else:
            terminals.append(segments_[-1,1,:3])

    segments = np.concatenate(segments, axis=0)

    # rescale points
    segments = np.stack((segments[:,:,2], segments[:,:,1]/scale, segments[:,:,0]/scale, segments[...,-1]), axis=-1)
    branch_points = torch.from_numpy(np.array(branch_points))
    branch_points = torch.stack((branch_points[:,2], branch_points[:,1]/scale, branch_points[:,0]/scale), dim=-1)

    terminals = torch.from_numpy(np.array(terminals))
    terminals = torch.stack((terminals[:,2], terminals[:,1]/scale, terminals[:,0]/scale), dim=-1)

    return segments, branch_points, terminals


def load_img_stack(img_dir, pixelsize=[1.0,1.0,1.0], downsample_factor=1.0, inverse=False):
    # load image stack
    files = os.listdir(img_dir)
    stack = []
    
    # load first image and initialize interp coordinates
    img = tf.imread(os.path.join(img_dir,files[0])).transpose(2,0,1).astype(np.float32) # channels are in the last dim
    
    # downsample in x-y by stepping in intervals of dz*downsample_factor so that if downsampling is zero
    # this will set the image to isotropic pixel size 
    x = [torch.arange(x)*d for x,d in zip(img.shape[1:], pixelsize[1:])]
    scale = pixelsize[0]*downsample_factor
    x_ = [torch.arange(start=0.0, end=x*d, step=scale) for x,d in zip(img.shape[1:], pixelsize[1:])]
    phii = torch.stack(torch.meshgrid(x_, indexing='ij'))

    img = interp(x, img, phii, interp2d=True) # channels along the first axis
    stack.append(img)
    # now do the rest
    for i in range(len(files)-1):
        img = tf.imread(os.path.join(img_dir,files[i+1])).transpose(2,0,1).astype(np.float32)
        # downsample x,y first to reduce memory
        # img = img[::downsample_factor, ::downsample_factor]
        img = interp(x, img, phii, interp2d=True) # channels along the first axis
        stack.append(img)
    stack = torch.tensor(np.array(stack))
    stack = torch.permute(stack, (1,0,2,3)) # reshape to c x h x w x d
    stack = stack / stack.amax(dim=(1,2,3))[:,None,None,None] # rescale to [0,1]. Each channel separately

    if inverse:
        stack = 1.0 - stack
    
    return stack


def make_neuron_density(segments, shape, width=3):
    # create density image
    density = Image(torch.zeros((1,)+shape))

    if not isinstance(segments, torch.Tensor):
        segments = torch.tensor(segments)

    for s in segments:
        density.draw_line_segment(s[:,:3], width=width, channel=0)
    
    return density


def make_neuron_mask(density, threshold=1.0):
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

def make_section_labels(sections, shape, width=3):
    # create discrete labels for each section
    labels = Image(torch.zeros((1,)+shape))
    for i, section in sections.items():
        for segment in section:
            labels.draw_line_segment(segment[:,:3], width=width, channel=0, binary=True, value=i)
    
    return labels


def load_data(img_dir, label_file, pixelsize=[1.0,1.0,1.0], downsample_factor=1.0, inverse=False):
   
    stack = load_img_stack(img_dir, pixelsize=pixelsize, downsample_factor=downsample_factor, inverse=inverse)

    segments, branch_points, terminals = parse_labels(label_file, scale=pixelsize[0]*downsample_factor)

    # create density image
    density = make_neuron_density(segments, stack.shape[1:])

    mask = make_neuron_mask(density, threshold=1.0)
    boundary = torch.tensor(dilation(mask, cube(10)))

    return stack, mask, boundary, branch_points, terminals


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
                binary=False):

    img = Image(torch.zeros((1,)+shape))
    value =  1.0
    for s in segments:
        if random_brightness:
            y0 = 0.5
            value = y0 + (1.0 - y0) * np.random.rand(1).item()
        img.draw_line_segment(s[:,:3], width=width, binary=binary, channel=0, value=value)
    if neuron_color is None:
        neuron_color = (1.0, 1.0, 1.0)

    img_data = torch.cat((neuron_color[0]*img.data, neuron_color[1]*img.data, neuron_color[2]*img.data), dim=0)
    if background_color is not None:
        img_data = img_data + torch.ones_like(img_data) * background_color[:,None,None,None]
        img_data /= img_data.amax()
    sigma = img_data.amax() * noise
    img_data = img_data + torch.randn(img_data.shape)*sigma # add noise
    img_data = (img_data - img_data.amin()) / (img_data.amax() - img_data.amin()) # rescale to [0,1]
    img = Image(img_data)

    return img


def read_swc(labels_file):
    print(f"loading file: {labels_file}")
    try:
        with open(labels_file, 'r') as f:
            lines = f.readlines()
    except:
        with open(labels_file, 'r', encoding="latin1") as f:
            lines = f.readlines()

    lines = [line for line in lines if not line.startswith('#') and line.strip()]
    lines = [line.split() for line in lines]
    swc_list = [list(map(int, line[:2])) + list(map(float, line[2:6])) + [int(line[6])] for line in lines]

    return swc_list


def parse_swc_list(swc_list, adjust=True):

    graph = {}
    for parent in swc_list:
        children = []
        for child in swc_list:
            if child[6] == parent[0]:
                children.append(child[0])
        graph[parent[0]] = children

    sections = {1:[]}
    section_graph = {1:[]}
    i = 1
    section_id = 1
    for key, value in graph.items():
        if len(value) == 0:
            sections[i] = np.array(sections[i]) # type: ignore #
            sections[i] = np.stack((sections[i][...,2], sections[i][...,1], sections[i][...,0]), axis=2) #type: ignore #
            i = key+1 # go to the section whose first segment corresponds to the next key
            section_id = key+1
            section_graph[section_id] = []
        elif len(value) == 1:
            sections[i].append([swc_list[key-1][2:5], swc_list[value[0]-1][2:5]])
        else:
            for child in value:
                if child == key + 1:
                    sections[i].append([swc_list[key-1][2:5], swc_list[child-1][2:5]])
                else:
                    sections[child] = []
                    sections[child].append([swc_list[key-1][2:5], swc_list[child-1][2:5]])
                    section_graph[section_id].append(child)

    # get average segment length
    lengths = []
    for section in sections.values():
        for segment in section:
            lengths.append(np.linalg.norm(segment[1] - segment[0]))
    avg_length = np.median(lengths)

    branches = []
    terminals = []
    for key, value in graph.items():
        if len(value) == 0:
            terminals.append(swc_list[key-1][2:5])
        elif len(value) > 1:
            # check the remaining length of each section starting from the current key
            lengths = []
            for child in value:
                # get the last position of each section starting from the current key
                # length = 0
                i = child
                while len(graph[i]) > 0:
                    i += 1
                    # length += 1
                lengths.append(np.linalg.norm(np.array(swc_list[i-1][2:5]) - np.array(swc_list[key-1][2:5])))
                # lengths.append(length)
            # if at least two sections have avg_lengths greater than 1, then the current key is a branch
            if key == 1:
                if sum([l > 2*avg_length for l in lengths]) > 2:
                    branches.append(swc_list[key-1][2:5])
            else:
                if sum([l > 2*avg_length for l in lengths]) > 1:
                    branches.append(swc_list[key-1][2:5])

    branches = np.array(branches)
    if len(branches) > 0:
        branches = np.stack((branches[:,2], branches[:,1], branches[:,0]), axis=1)
    terminals = np.array(terminals)
    terminals = np.stack((terminals[:,2], terminals[:,1], terminals[:,0]), axis=1)

    scale = 1
    if adjust:
        # scale and shift coordinates
        max = np.array([-1e6, -1e6, -1e6])
        min = np.array([1e6, 1e6, 1e6])
        for id, section in sections.items():
            max = np.maximum(max, section.max(axis=(0,1))) # type: ignore #
            min = np.minimum(min, section.min(axis=(0,1))) # type: ignore #
        vol = np.prod(max - min)
        scale = np.round((5e7 / vol)**(1/3)) # scale depends on the volume

        for id, section in sections.items():
            section = (section - min) * scale + np.array([10.0, 10.0, 10.0])
        branches = (branches - min) * scale + np.array([10.0, 10.0, 10.0])
        terminals = (terminals - min) * scale + np.array([10.0, 10.0, 10.0])
    
    for id, section in sections.items():
        sections[id] = torch.from_numpy(section) # type: ignore #

    return sections, section_graph, branches, terminals, scale


def draw_neuron_from_swc(swc_list,
                         width=3,
                         noise=0.05,
                         dropout=True,
                         adjust=True,
                         background_color=None,
                         neuron_color=None,
                         random_brightness=False,
                         binary=False):

    sections, graph, branches, terminals, scale = parse_swc_list(swc_list, adjust=adjust)

    segments = []
    for section in sections.values():
        segments.append(section)
    segments = torch.concatenate(segments)

    shape = torch.ceil(torch.amax(segments, dim=(0,1)))
    shape = shape.to(int) + torch.tensor([10, 10, 10])  # type: ignore
    shape = tuple(shape.tolist())

    img = draw_neuron(segments, shape=shape, width=width, noise=noise, neuron_color=neuron_color,
                      background_color=background_color, random_brightness=random_brightness,
                      binary=binary)

    density = make_neuron_density(segments, shape, width=width)
    section_labels = make_section_labels(sections, shape, width=2*width)
    mask = make_neuron_mask(density, threshold=5.0)

    if dropout: # add random signal dropout (subtract gaussian blobs)
        neuron_coords = torch.nonzero(section_labels.data)
        dropout_density = 0.001
        size = int(dropout_density * len(neuron_coords))
        rand_ints = torch.randint(0, len(neuron_coords), size=(size,))
        dropout_points = neuron_coords[rand_ints]
        dropout_img = torch.zeros_like(img.data)
        dropout_img[:, *dropout_points[:,1:].T] = 1.0
        dropout_img = gaussian(dropout_img, sigma=0.5*width)
        dropout_img /= dropout_img.max()
        img.data = img.data - dropout_img
        img.data = torch.where(img.data < 0, 0.0, img.data)

    branch_mask = Image(torch.zeros_like(mask))
    for point in branches:
        branch_mask.draw_point(torch.from_numpy(point), radius=3, binary=True, value=1, channel=0)
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