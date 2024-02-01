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


def load_data(img_dir, label_file, downsample_factor):
    # load image stack
    files = os.listdir(img_dir)
    stack = []
    for i in range(len(files)):
        img = tf.imread(os.path.join(img_dir,files[i])) # channels are in the last dim
        img = img[::downsample_factor, ::downsample_factor]
        stack.append(img)
    stack = torch.Tensor(np.array(stack))
    stack = torch.permute(stack, (-1,0,1,2))

    # load label
    label = load_morphology(label_file)

    # get points
    points = label.points[:,:3]
    points = torch.Tensor(points)
    points = torch.stack([points[:,2], points[:,1]//downsample_factor, points[:,0]//downsample_factor])
    points = points.to(int)
    
    # convert points to density map
    density = torch.zeros(stack.shape[1:])
    density[*points] = 1.0
    scipy.ndimage.gaussian_filter(density, sigma=2, output=density)
    density = density / density.max()
    mask = torch.zeros_like(density)
    mask[density>0.04] = 1.0
    density = density.unsqueeze(0)
    mask = mask.unsqueeze(0)

    return stack, density, mask