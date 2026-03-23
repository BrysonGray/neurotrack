#!/usr/bin/env python

"""Utils.py : Helper functions."""

import numpy as np
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


def inhomogeneity_correction(image, background_threshold=None):
    # piecewise intensity normalization
    patch_size = 21

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        image = image.cpu().numpy()
    elif not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array or a PyTorch tensor.")
    # Convert to float32 for processing
    image = image.astype(np.float32)
    # first do global normalization
    global_max = image.max()
    if global_max > 0:
        image = image / global_max
    else:
        image = np.zeros_like(image)
    
    background_threshold = np.quantile(image, 0.9995) if background_threshold is None else background_threshold

    # Get image dimensions
    d, h, w = image.shape
    
    # Calculate padding needed to ensure all patches fit
    pad_d = (patch_size - d % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    # Pad the image
    padded_image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
    
    # Initialize normalized image with same size as padded
    normalized_image = np.zeros_like(padded_image)
    
    # Process each patch
    for z in range(0, padded_image.shape[0], patch_size):
        for y in range(0, padded_image.shape[1], patch_size):
            for x in range(0, padded_image.shape[2], patch_size):
                # Extract patch
                patch = padded_image[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                
                # Create mask for foreground pixels (above background threshold)
                foreground_mask = patch > background_threshold
                
                # Only normalize if there are foreground pixels
                if np.any(foreground_mask):
                    # Find maximum value among foreground pixels only
                    max_val = patch[foreground_mask].max()
                    
                    # Normalize patch (avoid division by zero)
                    if max_val > 0:
                        normalized_patch = patch / max_val
                    else:
                        normalized_patch = patch
                else:
                    # If no foreground pixels, keep original values
                    normalized_patch = patch
                
                # Store normalized patch
                normalized_image[z:z+patch_size, y:y+patch_size, x:x+patch_size] = normalized_patch
    
    # Remove padding to return to original size
    normalized_image = normalized_image[:d, :h, :w]
    
    return normalized_image


def kmeans(image, k=2, tolerance=1e-3, init_means=None, max_iter=100, verbose=False):
    """
    Perform k-means clustering on the input image.
    
    Parameters
    ----------
    image: Array
        3D NumPy array representing the scalar-valued image.
    k: int
        Number of clusters for k-means.
    tolerance: float
        Tolerance value used to determine convergence. Tolerance is relative to the range of values in the image.
    verbose: bool
        Run in verbose mode.

    Returns
    -------
    means: Array
        Mean values for k clusters. 
    """

    if init_means is not None:
        if not isinstance(init_means, np.ndarray) or init_means.ndim != 1 or init_means.size != k:
            raise ValueError("init_means must be a 1D NumPy array of size k.")
        mu = init_means
    else:
        # Initialize means randomly within the range of the image values
        mu = np.random.uniform(size=(k,), high=image.max(), low=image.min())
    
    tol = (image.max() - image.min()) * tolerance
    mu_ = np.zeros_like(mu)
    
    # Flatten image for faster computation
    image_flat = image.flatten()
    
    n = 1
    while True:
        if n > max_iter:
            if verbose:
                print(f"Maximum iterations {max_iter} reached without convergence.")
            break
        if verbose:
            print(f'it: {n}')
        # Compute distances more efficiently using broadcasting
        distances = np.abs(image_flat[:, None] - mu[None, :])
        clusters = np.argmin(distances, axis=1)
        
        # Update means more efficiently
        for i in range(k):
            mask = clusters == i
            if np.any(mask):
                mu_[i] = np.mean(image_flat[mask])
            else:
                mu_[i] = mu[i]  # Keep old value if no points assigned
                
        if np.allclose(mu_, mu, atol=tol):
            break

        mu = mu_.copy()
        n += 1
    
    sigmas = np.zeros((k,))
    for i in range(k):
        mask = clusters == i
        if np.any(mask):
            sigmas[i] = np.std(image_flat[mask])
        else:
            sigmas[i] = sigmas[i]  # Keep old value if no points assigned
    
    # order means and sigmas by means
    order = np.argsort(mu)
    mu = mu[order]
    sigmas = sigmas[order]

    return mu, sigmas

if __name__ == "__main__":
    pass