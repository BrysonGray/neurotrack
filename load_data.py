#!/usr/bin/env python

"""
Deep Q-network tractography data loading functions.

Author: Bryson Gray
2024
"""

import tifffile as tf
import os
import numpy as np
import torch
import scipy
from neurom.io.utils import load_morphology
import sys
sys.path.append('/home/brysongray/tractography')
from image import Image


def load_data(img_dir, label_file, downsample_factor, binary=False):
    # load image stack
    files = os.listdir(img_dir)
    stack = []
    for i in range(len(files)):
        img = tf.imread(os.path.join(img_dir,files[i])) # channels are in the last dim
        img = img[::downsample_factor, ::downsample_factor]
        stack.append(img)
    stack = torch.tensor(np.array(stack))
    stack = torch.permute(stack, (-1,0,1,2))
    stack = stack / stack.amax(dim=(0,1,2,3))

    # load label
    label = load_morphology(label_file)

    segments = []
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
    segments = np.concatenate(segments, axis=0)
    segments = np.stack((segments[:,:,2], segments[:,:,1]/downsample_factor, segments[:,:,0]/downsample_factor, segments[...,-1]), axis=-1)

    density = Image(torch.zeros((1,)+stack.shape[1:]), dx=[0.88, 1.0, 1.0])

    for s in segments:
        s = torch.tensor(s)
        density.draw_line_segment(s[:,:3], width=s[0,-1].item(), binary=binary)

    mask = torch.zeros_like(density.data)
    mask[density.data>0.68] = 1.0 

    # # get points
    # points = label.points[:,:3]
    # points = torch.Tensor(points)
    # points = torch.stack([points[:,2], points[:,1]//downsample_factor, points[:,0]//downsample_factor])
    # points = points.to(int)
    
    # # convert points to density map
    # density = torch.zeros(stack.shape[1:])
    # density[*points] = 1.0
    # scipy.ndimage.gaussian_filter(density, sigma=2, output=density)
    # density = density / density.max()
    # mask = torch.zeros_like(density)
    # mask[density>0.04] = 1.0
    # density = density.unsqueeze(0)
    # mask = mask.unsqueeze(0)

    return stack, density.data, mask