#!/usr/bin/env python

'''

Functions for setting up neuron images and ground truth data in swc format for neuron tracing.

Author: Bryson Gray
2024

'''
import torch
import numpy as np
from skimage.draw import line_nd
from skimage.filters import gaussian
from skimage.morphology import dilation, footprint_rectangle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_line_segment(segment, width, binary=False, value=1.0):
    """ Generate an image of a line segment with width.

    Parameters
    ----------
    segment: torch.Tensor
        array with two three dimensional points (shape: 2x3)
    width: scalar
        segment width
    binary: bool
        Make a line mask rather than a blurred idealized line.
    value: float
        If binary is set to True, set the line brightness to this value. Default is 1.0.
    
    Returns
    -------
    X : torch.Tensor
        A patch with the new line segment starting at its center.
    """
    
    segment = segment[1] - segment[0]
    segment_length = torch.sqrt(torch.sum(segment**2))

    # the patch should contain both line end points plus some blur
    L = int(torch.ceil(segment_length)) + 1 # The radius of the patch is the whole line length since the line starts at patch center.
    overhang = int(2*width) # include space beyond the end of the line
    patch_radius = L + overhang

    patch_size = 2*patch_radius + 1
    X = torch.zeros((patch_size,patch_size,patch_size))
    # get endpoints
    start_ = torch.tensor([patch_radius]*3) # the patch center is the start point rounded to the nearest pixel
    # start = torch.round(segment_length*direction + c).to(int)
    end = torch.round(start_ + segment.cpu()).to(dtype=torch.int)
    line = line_nd(start_, end, endpoint=True)
    X[line] = float(value)

    # if width is 0, don't blur
    if width > 0:
        if binary:
            X = torch.tensor(dilation(X, footprint_rectangle((int(width),)*3)))
        else:
            sigma = width/2
            X = torch.tensor(gaussian(X, sigma=sigma))
            X = X / torch.amax(X) * value
    
    return X.to(device=segment.device)


class Image:

    """ 
    Image class for tracking environment image data     
    """
    def __init__(self, data):
        """
        Initialize the Image object with data.
        
        Parameters
        ----------
        data : numpy.ndarray or torch.Tensor
            The input data to be stored in the Image object. If the input data is 
            not a torch.Tensor, it will be converted to one.
            
        Attributes
        ----------
        data : torch.Tensor
            The data stored in the Image object as a torch.Tensor.
        """
    
        self.data = data
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.from_numpy(self.data)


    def crop(self, center, radius, interp=False, padding_mode="zeros", pad=True, value=0.0):


        """ Crop an image around a center point (rounded to the nearest pixel center).
            The cropped image will be smaller than the given radius if it overlaps with the image boundary unless pad is set to True.
            Interpolation is not currently implemented.

            Parameters
            ----------
            center : list or tuple
                The center of the cropped image in slice-row-col coordinates. This will be rounded to the nearest pixel index.
            radius : int
                The radius of the cropped image. The total width is 2*radius + 1  in each dimension assuming it doesn't intersect with a boundary.
            interp : bool, optional
                If True, interpolation will be applied (currently not implemented). Default is False.
            padding_mode : str, optional
                The mode to use for padding if `pad` is True. Default is "zeros". Not currently implemented.
            pad : bool, optional
                If True, the cropped image will be padded to the size specified by the radius. Default is True.
            value : float, optional
                The value to use for padding if `pad` is True. Default is 0.0.
            
            Returns
            -------
            patch  : ndarray
                Cropped image
            padding : ndarray
                Length that patch overlaps with image boundaries on each end of each dimension.
        """
        i,j,k = [int(torch.round(x)) for x in center]
        shape = self.data.shape[1:]

        # get amount of padding for each face
        zpad_top = zpad_btm = ypad_front = ypad_back = xpad_left = xpad_right = 0

        if (i + radius) > shape[0]-1:
            zpad_btm = i + radius - (shape[0]-1)
        if (i - radius) < 0:
            zpad_top = radius - i
        if (j + radius) > shape[1]-1: # back is the max y idx
            ypad_back = j + radius - (shape[1]-1) # number of zeros to append in the y dim
        if (j - radius) < 0: # front is zeroth idx
            ypad_front = radius - j
        if (k + radius) > shape[2]-1:
            xpad_right = k + radius - (shape[2]-1) # number of zeros to append in the x dim
        if (k - radius) < 0:
            xpad_left = radius - k
        padding = np.array([zpad_top, zpad_btm, ypad_front, ypad_back, xpad_left, xpad_right])
        # get remainder for each face (patch radius minus padding) 
        remainder = np.array([radius]*6) - padding # zrmd_top, zrmd_btm, yrmd_front, yrmd_back, xrmd_left, xrmd_right
        # patch is data cropped around center. Note: slicing img creates a view (not a copy of img)
        patch = self.data[:, i-remainder[0]:i+remainder[1]+1, j-remainder[2]:j+remainder[3]+1, k-remainder[4]:k+remainder[5]+1]

        if interp:
            raise NotImplementedError("Interpolation is not currently implemented.")
        #     center = center.numpy().astype(np.float32)
        #     remainder = remainder.reshape(3,2)
        #     x = [np.arange(x-r[0], x+r[1]+1).astype(np.float32) for x,r in zip(np.round(center), remainder)]
        #     x_ = [np.arange(x-r[0], x+r[1]+1).astype(np.float32) for x,r in zip(center, remainder)]
        #     phii = np.stack(np.meshgrid(*x_, indexing='ij'))
        #     patch = utils.interp(x, patch, phii, padding_mode=padding_mode) # after interp patch is a copy (not a view of data)

        if pad:
            patch_size = 2*radius+1
            patch_ = torch.ones((self.data.shape[0], patch_size, patch_size, patch_size), device=self.data.device) * value
            patch_[:, zpad_top:patch_size - zpad_btm, ypad_front:patch_size - ypad_back, xpad_left:patch_size - xpad_right] = patch
            patch = patch_

        return patch, padding
    

    def draw_line_segment(self, segment, width, channel=3, value=1.0, binary=False):

        """ Add an image patch with the new line segment to the existing image in the specified channel.

        Parameters
        ----------
        segment : array_like
            array with two three dimensional points (shape: 2x3)
        width : scalar
            segment width
        channel : int, optional
            The channel in which to draw the line segment (default is 3).
        value : float, optional
            The value to assign to the line segment (default is 1.0).
        binary : bool, optional
            If True, the line segment will be added in a binary fashion (default is False).
            
        Returns
        -------
        old_patch : torch.Tensor
            The original patch of the image before the line segment was added.
        new_patch : torch.Tensor
            The new patch of the image after the line segment was added.
        """
        
        # create the patch with the new line segment starting at its center.
        X = draw_line_segment(segment, width, binary, value)

        # get the patch centered on the new segment start point from the current image.
        center = torch.round(segment[0]).to(torch.int)
        patch_radius = int((X.shape[0] - 1)/2)
        patch, padding = self.crop(center, patch_radius, interp=False, pad=False) # patch is a view of self.data (c x h x w x d)
        old_patch = patch[channel].clone()
        # if the patch overlaps with the image boundary, it must be cropped to fit
        X = X[padding[0]:X.shape[0]-padding[1], padding[2]:X.shape[1]-padding[3], padding[4]:X.shape[2]-padding[5]]

        # add segment to patch
        if binary:
            # set the new patch to the minimum values between arrays X excluding zeros.
            patch[channel] = torch.where(X*patch[channel] > 0, torch.minimum(X,patch[channel]), torch.maximum(X,patch[channel]))
        else:
            patch[channel] = torch.maximum(X, patch[channel])
        new_patch = patch[channel].clone()

        return old_patch, new_patch
    
    
    def draw_point(self, point: torch.Tensor, radius: float = 3.0, channel: int = -1, binary: bool = False, value: int = 1):
        """
        Draw a point on the image data with a specified radius and value.
        
        Parameters
        ----------
        point : torch.Tensor
            The coordinates of the point to be drawn.
        radius : float, optional
            The radius of the point to be drawn. Default is 3.0.
        channel : int, optional
            The channel on which to draw the point. Default is -1.
        binary : bool, optional
            If True, the point will be drawn as a binary value. Default is False.
        value : int, optional
            The value to assign to the point. Default is 1.
            
        Returns
        -------
        None
        """
        
        c = round(radius)
        patch_size = 2*c+1
        if binary:
            X = torch.ones((patch_size,patch_size,patch_size)) * value
        else:
            X = torch.zeros((patch_size,patch_size,patch_size))
            X[c,c,c] = 1.0
            X = torch.tensor(gaussian(X, sigma=radius))
            X = (X / torch.amax(X)) * value
        patch, padding = self.crop(point, radius=c, interp=False, pad=False)
        new_patch = X[padding[0]:X.shape[0]-padding[1], padding[2]:X.shape[1]-padding[3], padding[4]:X.shape[2]-padding[5]]
        patch[channel] = torch.maximum(new_patch.to(device=patch.device), patch[channel])

        return
    
if __name__ == "__main__":
    pass