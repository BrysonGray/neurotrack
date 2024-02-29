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
import utils
import sys
from neurom.io.utils import load_morphology
sys.path.append('/home/brysongray/tractography')
from image import Image
from skimage.morphology import dilation, cube


def load_data(img_dir, label_file, pixelsize=[1.0,1.0,1.0], downsample_factor=1.0, inverse=False):
    # load image stack
    files = os.listdir(img_dir)
    stack = []
    
    # load first image and initialize interp coordinates
    img = tf.imread(os.path.join(img_dir,files[0])).transpose(2,0,1).astype(np.float32) # channels are in the last dim
    
    # downsample in x-y by stepping in intervals of dz*downsample_factor so that if downsampling is zero,
    # this will set the image to isotropic pixel size 
    x = [torch.arange(x)*d for x,d in zip(img.shape[1:], pixelsize[1:])]
    scale = pixelsize[0]*downsample_factor
    x_ = [torch.arange(start=0.0, end=x*d, step=scale) for x,d in zip(img.shape[1:], pixelsize[1:])]
    phii = torch.stack(torch.meshgrid(x_, indexing='ij'))

    img = utils.interp(x, img, phii, interp2d=True) # channels along the first axis
    stack.append(img)
    # now do the rest
    for i in range(len(files)-1):
        img = tf.imread(os.path.join(img_dir,files[i+1])).transpose(2,0,1).astype(np.float32)
        # downsample x,y first to reduce memory
        # img = img[::downsample_factor, ::downsample_factor]
        img = utils.interp(x, img, phii, interp2d=True) # channels along the first axis
        stack.append(img)
    stack = torch.tensor(np.array(stack))
    stack = torch.permute(stack, (1,0,2,3)) # reshape to c x h x w x d
    stack = stack / stack.amax(dim=(1,2,3))[:,None,None,None] # rescale to [0,1]. Each channel separately

    if inverse:
        stack = 1.0 - stack

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

    # rescale segments
    segments = np.stack((segments[:,:,2], segments[:,:,1]/scale, segments[:,:,0]/scale, segments[...,-1]), axis=-1)

    density = Image(torch.zeros((1,)+stack.shape[1:]))

    for s in segments:
        s = torch.tensor(s)
        density.draw_line_segment(s[:,:3], width=s[0,-1].item()/2)

    mask = torch.zeros_like(density.data)
    mask[density.data>np.exp(-3)] = 1.0 
    mask = torch.tensor(dilation(mask, cube(10)[None]))


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