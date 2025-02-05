#!/usr/bin/env python

"""Utils.py : Helper functions."""

import numpy as np
import struct
import torch
from torch.nn.functional import grid_sample
import xml.etree.ElementTree as ET

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


def dft_read(dft_filename, verbose=False):
    """Simple reader for BrainSuite's .dft files.
    This reads a .dft file and outputs a the vertices corresponding to the
    curves and the color of each curve.
    Note: if using Python 2, you must do
    "from __future__ import print_function" before calling this function.
    EXAMPLE USAGE
        list_curves, list_colors = dft_read("subj1_curves.dft")
    INPUT:
        > dft_filename: string of the file to be read.
        > verbose: Boolean, set to True if you want verbose output
    OUTPUT:
        > list_curves: list with each element representing one curve using a
            (N x 3) NumPy array where each row is a vertex.
        > list_color: list where n-th element is the a 3-element list of ints
            in the form [R,G,B] corresponding to color of the n-th curve"""

    # list of curves to return,
    list_curves = []

    # if verbose is set, we'll time how long it takes to finish
    if verbose:
        import time
        tic = time.time()

    # open the file to be read in binary mode
    with open(dft_filename, "rb") as fo:

        # for now, I just discard the first 8 bytes which are the text label
        # for the file version
        _ = fo.read(8)[0]
        # discard the next 4 bytes corresponding to the version code.
        _ = fo.read(4)[0]
        # read in as an integer the header size  (4 bytes)
        hdrsize = struct.unpack('i', fo.read(4))[0]
        # start of data of the curve vertices
        dataStart = struct.unpack('i', fo.read(4))[0]
        # start of XML data which gives the color of each curve
        mdoffset = struct.unpack('i', fo.read(4))[0]
        # Discard the next 4 bytes ("pdoffset") since I'm not sure what they do
        _ = struct.unpack('i', fo.read(4))[0]
        # Number of curves (read in as an unsigned int32)
        nContours = struct.unpack('I', fo.read(4))[0]

        if verbose:
            print("the number of curves is: ", nContours)

        # move current file reading position to start of xml block
        fo.seek(mdoffset)

        # calculate size of the XML block
        xml_block_size = dataStart - mdoffset
        # read in XML block
        xml_block = fo.read(xml_block_size)

        # get root element of XML block
        root = ET.fromstring(xml_block)
        # list of all the colors.  Each color is represented by a
        # list of 3 elements corresponding to RGB.
        list_colors = []

        for child in root:
            temp_color = child.attrib['color']
            # convert to float
            temp_color = [float(x) for x in temp_color.split(" ")]
            list_colors.append(temp_color)

        # move current file reading position to start of curve vertex data
        fo.seek(dataStart)

        # loop through every curve
        for curve in range(nContours):
            # number of points in current curve
            num_points = struct.unpack('i', fo.read(4))[0]

            if verbose:
                print("Number of points in current curve:", num_points)

            # in order to read off all the points of a curve in one fell swoop,
            # we need to know the number of (4 byte) floats to read off
            num_floats = num_points*3
            points = struct.unpack('f'*num_floats, fo.read(4*num_floats))

            # make NumPy array from the points.  We reshape it to be a
            # Nx3 array.
            points_arr = np.array(points).reshape((-1, 3))

            # add to the list of curves
            list_curves.append(points_arr)

        if verbose:
            toc = time.time()
            print("Finished processing", nContours,
                  "curves in", toc-tic, "seconds.")

        return list_curves, list_colors