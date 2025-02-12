#!/usr/bin/env python

"""Utils.py : Helper functions."""

import torch
from torch.nn.functional import grid_sample

def interp(x, I, phii, interp2d=False, **kwargs):
    '''
    Interpolate an image with specified regular voxel locations at specified sample points.
    
    Interpolate the image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D or 4D arrays with first channel storing component)
    
    Parameters
    ----------
    x : list of numpy arrays
        x[i] is a numpy array storing the pixel locations of imaging data along the i-th axis.
        Note that this MUST be regularly spaced, only the first and last values are queried.
    I : array
        Numpy array or torch tensor storing 2D or 3D imaging data.  In the 3D case, I is a 4D array with 
        channels along the first axis and spatial dimensions along the last 3. For 2D, I is a 3D array with
        spatial dimensions along the last 2.
    phii : array
        Numpy array or torch tensor storing positions of the sample points. phii is a 3D or 4D array
        with components along the first axis (e.g. x0,x1,x1) and spatial dimensions 
        along the last axes.
    interp2d : bool, optional
        If True, interpolates a 2D image, otherwise 3D. Default is False (expects a 3D image).
    kwargs : dict
        keword arguments to be passed to the grid sample function. For example
        to specify interpolation type like nearest.  See pytorch grid_sample documentation.
    
    Returns
    -------
    out : torch tensor
        Array storing an image with channels stored along the first axis. 
        This is the input image resampled at the points stored in phii.


    '''
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    ndim = 2 if interp2d==True else 3
    for i in range(ndim):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done

    # NOTE I should check that I can reproduce identity
    # note that phii must now store x,y,z along last axis
    # is this the right order?
    # I need to put batch (none) along first axis
    # what order do the other 3 need to be in?    
    # feb 2022
    if 'padding_mode' not in kwargs:
        kwargs['padding_mode'] = 'border' # note that default is zero, but we switchthe default to border
    if interp2d==True:
        phii = phii.flip(0).permute((1,2,0))[None]
    else:
        phii = phii.flip(0).permute((1,2,3,0))[None]
    out = grid_sample(I[None], phii, align_corners=True, **kwargs)

    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension
    out = out[0]
    return out

if __name__ == "__main__":
    pass