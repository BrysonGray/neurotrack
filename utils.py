#!/usr/bin/env python

"""Utils.py : Helper functions."""

import struct
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.nn.functional import grid_sample
import matplotlib.pyplot as plt
import os
import re
import glob
import json
import tifffile as tf
import nibabel as nib
from warnings import warn

def draw(J,xJ=None,fig=None,n_slices=5,vmin=None,vmax=None,disp=True,**kwargs):    
    """ Draw 3D imaging data.
    
    Images are shown by sampling slices along 3 orthogonal axes.
    Color or grayscale data can be shown.
    
    Parameters
    ----------
    J : array like (torch tensor or numpy array)
        A 3D image with C channels should be size (C x nslice x nrow x ncol)
        Note grayscale images should have C=1, but still be a 4D array.
    xJ : list
        A list of 3 numpy arrays.  xJ[i] contains the positions of voxels
        along axis i.  Note these are assumed to be uniformly spaced. The default
        is voxels of size 1.0.
    fig : matplotlib figure
        A figure in which to draw pictures. Contents of the figure will be cleared.
        Default is None, which creates a new figure.
    n_slices : int
        An integer denoting how many slices to draw along each axis. Default 5.
    vmin
        A minimum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.001 quantile on each channel.
    vmax
        A maximum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.999 quantile on each channel.
    disp : bool
        Figure display toggle
    kwargs : dict
        Other keywords will be passed on to the matplotlib imshow function. For example
        include cmap='gray' for a gray colormap

    Returns
    -------
    fig : matplotlib figure
        The matplotlib figure variable with data.
    axs : array of matplotlib axes
        An array of matplotlib subplot axes containing each image.


    Example
    -------
    Here is an example::

       >>> example test


    """
    if type(J) == torch.Tensor:
        J = J.detach().clone().cpu()
    J = np.array(J)
    if xJ is None:
        nJ = J.shape[-3:]
        xJ = [np.arange(n) - (n-1)/2.0 for n in nJ] 
    if type(xJ[0]) == torch.Tensor:
        xJ = [np.array(x.detach().clone().cpu()) for x in xJ]
    xJ = [np.array(x) for x in xJ]
    
    if fig is None:
        fig = plt.figure()
    fig.clf()    
    if vmin is None:
        vmin = np.quantile(J,0.001,axis=(-1,-2,-3))
    if vmax is None:
        vmax = np.quantile(J,0.999,axis=(-1,-2,-3))
    vmin = np.array(vmin)
    vmax = np.array(vmax)    
    # I will normalize data with vmin, and display in 0,1
    if vmin.ndim == 0:
        vmin = np.repeat(vmin,J.shape[0])
    if vmax.ndim == 0:
        vmax = np.repeat(vmax,J.shape[0])
    if len(vmax) >= 2 and len(vmin) >= 2:
        # for rgb I'll scale it, otherwise I won't, so I can use colorbars
        J -= vmin[:,None,None,None]
        J /= (vmax[:,None,None,None] - vmin[:,None,None,None])
        J[J<0] = 0
        J[J>1] = 1
        vmin = 0.0
        vmax = 1.0
    # I will only show the first 3 channels
    if J.shape[0]>3:
        J = J[:3]
    if J.shape[0]==2:
        J = np.stack((J[0],J[1],J[0]))
    
    
    axs = []
    axsi = []
    # ax0
    slices = np.round(np.linspace(0,J.shape[1]-1,n_slices+2)[1:-1]).astype(int)        
    # for origin upper (default), extent is x (small to big), then y reversed (big to small)
    extent = (xJ[2][0],xJ[2][-1],xJ[1][-1],xJ[1][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1)
        toshow = J[:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax1
    slices = np.round(np.linspace(0,J.shape[2]-1,n_slices+2)[1:-1]).astype(int)    
    extent = (xJ[2][0],xJ[2][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1+n_slices)      
        toshow = J[:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax2
    slices = np.round(np.linspace(0,J.shape[3]-1,n_slices+2)[1:-1]).astype(int)        
    extent = (xJ[1][0],xJ[1][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):        
        ax = fig.add_subplot(3,n_slices,i+1+n_slices*2)
        toshow = J[:,:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    
    fig.subplots_adjust(wspace=0,hspace=0)
    if not disp:
        plt.close(fig)

    return fig,axs
    
    
def load_slices(target_name, xJ=None):
    """ Load a slice dataset.
    
    Load a slice dataset for histology registration. Slice datasets include pairs
    of images and json sidecar files, as well as one tsv file explaining the dataset.
    Note this code creates a 3D array by padding.
    
    Parameters
    ----------
    target_name : string
        Name of a directory containing slice dataset.
    xJ : list, optional
        list of numpy arrays containing voxel positions along each axis.
        Images will be resampled by interpolation on this 3D grid.

    Returns
    -------
    xJ : list of numpy arrays
        Location of v
    J : numpy array
        Numpy array of size C x nslices x nrows x ncols where C is the number of channels
        e.g. C=3 for RGB.
    W0 : numpy array
        A nslices x nrows x ncols numpy array containing weights.  Weights are 0 where there 
        was padding
    
    References
    ----------
    Document describing dataset format here: TODO XXXXX
    documented XXXX
    
    Raises
    ------
    Exception
        If the first image is not present in the image series.

    """
    print('loading target images')
    fig,ax = plt.subplots()
    ax = [ax]
    # current limitation
    # requires the word 'present'
    # requires the first image to be present
    # expects data type to be in 0,1
    # assumes space directions are diagonal
    # todo: origin
    # we will need more control over the size, and we will need to maintain the origin of each slice
    # right now we have a heuristic for taking 99th percentile and expanding by 1%
    
    data = []
    # load the one tsv file
    tsv_name = os.path.join(target_name, 'samples.tsv' )
    with open(tsv_name,'rt') as f:
        for count,line in enumerate(f):
            line = line.strip()
            key = '\t' if '\t' in line else '    '
            if count == 0:
                headings = re.split(key,line)                
                continue
            data.append(re.split(key,line))
    data_ = np.zeros((len(data),len(data[0])),dtype=object)
    for i in range(data_.shape[0]):
        for j in range(data_.shape[1]):
            try:
                data_[i,j] = data[i][j]
            except:
                data_[i,j] = ''
    data = data_
    
    # now we will loop through the files and get the sizes 
    nJ_ = np.zeros((data.shape[0],3),dtype=int)
    origin = np.zeros((data.shape[0],3),dtype=float)
    slice_status = data[:,3]
    J_ = []
    for i in range(data.shape[0]):
        if not slice_status[i] == 'present':
            # if i == 0:
            #     raise Exception('First image is not present')
            # J_.append(np.array([[[0.0,0.0,0.0]]]))
            continue
        namekey = data[i,0]
        searchstring = os.path.join(target_name,'*'+os.path.splitext(namekey)[0]+'*.json')
        jsonfile = glob.glob(searchstring)
        with open(jsonfile[0]) as f:
            jsondata = json.load(f)
        #nJ_[i] = np.array(jsondata['Sizes'])


        # this should contain an image and a json    
        image_name = jsondata['DataFile']
        _, ext = os.path.splitext(image_name)
        if ext == '.tif':
            J__ = tf.imread(os.path.join(target_name, image_name))
        else:
            J__ = plt.imread(os.path.join(target_name,image_name))

        if J__.dtype == np.uint8:
            J__ = J__.astype(float)/255.0
            J__ = J__[...,:3] # no alpha
        else:
            J__ = J__[...,:3].astype(float)
            J__ = J__ / np.mean(np.abs(J__.reshape(-1, J__.shape[-1])), axis=0)

        if not i%20:
            ax[0].cla()
            toshow = (J__- np.min(J__)) / (np.max(J__)-np.min(J__))
            ax[0].imshow(toshow)
            fig.suptitle(f'slice {i} of {data.shape[0]}: {image_name}')
            fig.canvas.draw()    

        nJ_[i] = np.array(J__.shape)

        J_.append(J__)


        # the domain
        # if this is the first file we want to set up a 3D volume
        if 'dJ' not in locals():
            dJ = np.diag(np.array(jsondata['SpaceDirections'][1:]))[::-1]
        # note the order needs to be reversed
        origin[i] = np.array(jsondata['SpaceOrigin'])
        x0 = origin[:,2] # z coordinates of slices
    if xJ == None:
        # build 3D coordinate grid
        nJ0 = np.array(int((np.max(x0) - np.min(x0))//dJ[0]) + 1) # length of z axis on the grid (there may be missing slices)
        nJm = np.max(nJ_,0)
        nJm = (np.quantile(nJ_,0.95,axis=0)*1.01).astype(int) # this will look for outliers when there are a small number, really there just shouldn't be outliers
        nJ = np.concatenate(([nJ0],nJm[:-1]))
        # get the minimum coordinate on each axis
        xJmin = [-(n-1)*d/2.0 for n,d in zip(nJ[1:],dJ[1:])]
        xJmin.insert(0,np.min(x0))
        xJ = [(np.arange(n)*d + o) for n,d,o in zip(nJ,dJ,xJmin)]
    XJ = np.stack(np.meshgrid(*xJ, indexing='ij'))

    # resample slices on 3D grid
    J = np.zeros(XJ.shape[1:] + tuple([3]))
    W0 = np.zeros(XJ.shape[1:])
    i = 0
    for j in range(XJ.shape[1]):
        # if slice_status[j] == False:
        if slice_status[j] == 'missing':
            continue
        xJ_ = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(J_[i].shape[:-1], dJ[1:])]
        J[j] = np.transpose(interp(xJ_, J_[i].transpose(2,0,1), XJ[1:,0], interp2d=True, padding_mode="border"), (1,2,0))
        W0_ = np.zeros(W0.shape[1:])
        W0_[J[i,...,0] > 0.0] = 1.0
        W0[i] = W0_
        i += 1
    J = np.transpose(J,(3,0,1,2))

    return xJ,J,W0


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
        kwargs['padding_mode'] = 'border' # note that default is zero
    if interp2d==True:
        phii = phii.flip(0).permute((1,2,0))[None]
    else:
        phii = phii.flip(0).permute((1,2,3,0))[None]
    out = grid_sample(I[None], phii, align_corners=True, **kwargs)

    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension
    out = out[0]
    return out


dtypes_reverse = {
    'float':np.dtype('float32'),
    'double':np.dtype('float64'),
    'unsigned_char':np.dtype('uint8'),
    'unsigned_short':np.dtype('uint16'),
    'unsigned_int':np.dtype('uint32'),
    'unsigned_long':np.dtype('uint64'),
    'char':np.dtype('int8'),
    'short':np.dtype('int16'),
    'int':np.dtype('int32'),
    'long':np.dtype('int64'),
}
def read_vtk_data(fname,endian='b'):
    '''
    Read vtk structured points legacy format data.
    
    Note endian should always be big, but we support little as well.
    
    Parameters
    ----------
    fname : str
        Name of .vtk file to read.
    endian : str
        Endian of data, with 'b' for big (default and only officially supported format)
        or 'l' for little (for compatibility if necessary).
        
    Returns
    -------
    x : list of numpy arrays
        Location of voxels along each spatial axis (last 3 axes)
    images : numpy array
        Image with last three axes corresponding to spatial dimensions.  If 4D,
        first axis is channel.  If 5D, first axis is time, and second is xyz 
        component of vector field.

    Raises
    ------
    Exception
        The first line should include vtk DataFile Version X.X
    Exception
        If the file contains data type other than BINARY.
    Exception
        If the dataset type is not STRUCTURED_POINTS.
    Exception
        If the dataset does not have either 3 or 4 axes.
    Exception
        If dataset does not contain POINT_DATA
    Exception
        If the file does not contain scalars or vectors.
    
    Warns
    -----
    If data not written in big endian
        Note (b) symbol not in data name {name}, you should check that it was written big endian. Specify endian="l" if you want little
    '''
    # TODO support skipping blank lines
    big = not (endian=='l')
    
    verbose = True
    verbose = False
    with open(fname,'rb') as f:        
        # first line should say vtk version
        line = f.readline().decode().strip()
        if verbose: print(line)
        if 'vtk datafile' not in line.lower():
            raise Exception('first line should include vtk DataFile Version X.X')
        # second line says title    
        line = f.readline().decode().strip()
        if verbose: print(line)
        title = line

        # third line should say type of data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if not line.upper() == 'BINARY':
            raise Exception(f'Only BINARY data type supported, but this file contains {line}')
        data_format = line

        # next line says type of data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if not line.upper() == 'DATASET STRUCTURED_POINTS':
            raise Exception(f'Only STRUCTURED_POINTS dataset supported, but this file contains {line}')
        geometry = line

        # next line says dimensions    
        # "ordered with x increasing fastest, theny,thenz"
        # this is the same as nrrd (fastest to slowest)
        # however our convention in python that we use channel z y x order
        # i.e. the first is channel
        line = f.readline().decode().strip()
        if verbose: print(line)
        dimensions = np.array([int(n) for n in line.split()[1:]])
        if len(dimensions) not in [3,4]:
            raise Exception(f'Only datasets with 3 or 4 axes supported, but this file contains {dimensions}')

        # next says origin
        line = f.readline().decode().strip()
        if verbose: print(line)
        origin = np.array([float(n) for n in line.split()[1:]])

        # next says spacing
        line = f.readline().decode().strip()
        if verbose: print(line)
        spacing = np.array([float(n) for n in line.split()[1:]])

        # now I can build axes
        # note I have to reverse the order for python
        x = [np.arange(n)*d+o for n,d,o in zip(dimensions[::-1],spacing[::-1],origin[::-1])]

        # next line must contain point_data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if 'POINT_DATA' not in line:
            raise Exception(f'only POINT_DATA supported but this file contains {line}')                          
        N = int(line.split()[-1])

        # now we will loop over available datasets
        names = []
        images = []
        count = 0
        while True:
            
            # first line contains data type (scalar or vector), name, and format
            # it could be a blank line
            line = f.readline().decode()
            if line == '\n':
                line = f.readline().decode()        
            line = line.strip()
            
            if line is None or not line: # check if we've reached the end of the file
                break
                
            if verbose: print(f'starting to load dataset {count}')
                
            if verbose: print(line)            
            S_V = line.split()[0]
            name = line.split()[1]
            dtype = line.split()[2]
            names.append(name)

            if S_V.upper() not in ['SCALARS','VECTORS']:
                raise Exception(f'Only scalars or vectors supported but this file contains {S_V}')        
            
            if '(b)' not in name and big: 
                warn(f'Note (b) symbol not in data name {name}, you should check that it was written big endian. Specify endian="l" if you want little')
                            
            dtype_numpy = dtypes_reverse[dtype]
            if big:
                dtype_numpy_big = dtype_numpy.newbyteorder('>') # > means big endian
            else:
                dtype_numpy_big = dtype_numpy
            #
            # read the data
            if S_V == 'SCALARS':
                # there should be a line with lookup table
                line = f.readline().decode()
                if verbose: print(line)
                data = np.fromfile(f,dtype_numpy_big,N).astype(dtype_numpy)
                # shape it
                data = data.reshape(dimensions[::-1])
                # axis order is already correct because of slowest to fastest convention in numpy

            elif S_V == 'VECTORS':            
                data = np.fromfile(f,dtype_numpy_big,N*3).astype(dtype_numpy)
                # shape it
                data = data.reshape((dimensions[-1],dimensions[-2],dimensions[-3],3))
                # move vector components first
                data = data.transpose((3,0,1,2))
                # with vector data we should flip xyz (file) to zyx (python) (added march 29)
                data = np.copy(data[::-1])
            images.append(data)
            count += 1
        images = np.stack(images) # stack on axis 0
        images = images / np.mean(np.abs(images)) # nomarlize
    return x,images,title,names


def read_matrix_data(fname):
    '''
    Read rigid transforms as matrix text file.
    
    Parameters
    ----------
    fname : str
    
    Returns
    -------
    A : array
        matrix in xyz order
    '''
    A = np.zeros((3,3))
    with open(fname,'rt') as f:
        i = 0
        for line in f:            
            if ',' in line:
                # we expect this to be a csv
                for j,num in enumerate(line.split(',')):
                    A[i,j] = float(num)
            else:
                # it may be separated by spaces
                for j,num in enumerate(line.split(' ')):
                    A[i,j] = float(num)
            i += 1
    
    return A


def read_data(fname, x=None, **kwargs):
    '''
    Read array data from several file types.
    
    This function will read array based data of several types
    and output x,images,title,names. Note we prefer vtk legacy format, 
    but accept some other formats as read by nibabel.
    
    Parameters
    ----------
    fname : str
        Filename (full path or relative) of array data to load. Can be .vtk or 
        nibabel supported formats (e.g. .nii)
    x : list of arrays, optional
        Coordinates for 2D series space
    **kwargs : dict
        Keyword parameters that are passed on to the loader function
    
    Returns
    -------
    
    x : list of numpy arrays
        Pixel locations where each element of the list identifies pixel
        locations in corresponding axis.
    images : numpy array
        Imaging data of size channels x slices x rows x cols, or of size
        time x 3 x slices x rows x cols for velocity fields
    title : str
        Title of the dataset (read from vtk files)        
    names : list of str
        Names of each dataset (channel or time point)
    
    Raises
    ------
    Exception
        If file type is nrrd.
    Exception
        If data is a single slice, json reader does not support it.
    Exception
        If opening with Nibabel and the affine matrix is not diagonal.
    
    '''
    # find the extension
    # if no extension use slice reader
    # if vtk use our reader
    # if nrrd use nrrd
    # otherwise try nibabel
    base,ext = os.path.splitext(fname)
    if ext == '.gz':
        base,ext_ = os.path.splitext(base)
        ext = ext_+ext
    #print(f'Found extension {ext}')
    
    if ext == '':
        x,J,W0 = load_slices(fname, xJ=x)
        images = np.concatenate((J,W0[None]))
        # set the names, I will separate out mask later
        names = ['red','green','blue','mask']
        title = 'slice_dataset'
    elif ext == '.vtk':
        x,images,title,names = read_vtk_data(fname,**kwargs)
    elif ext == '.nrrd':
        print('opening with nrrd')
        raise Exception('NRRD not currently supported')
    elif ext in ['.tif','.tiff','.jpg','.jpeg','.png']:
        # 2D image file, I can specify dx and ox
        # or I can search for a companion file
        print('opening 2D image file')
        if 'dx' not in kwargs and 'ox' not in kwargs:
            print('No geometry information provided')
            print('Searching for geometry information files')
            json_name = fname.replace(ext,'.json')
            geometry_name = os.path.join(os.path.split(fname)[0],'geometry.csv')
            if os.path.exists(json_name):
                print('Found json sidecar')
                raise Exception('json reader for single slice not implemented yet')            
            elif os.path.exists(geometry_name):                
                print('Found legacy geometry file')
                with open(geometry_name,'rt') as f:
                    for line in f:
                        if os.path.split(fname)[-1] in line:
                            #print(line)
                            parts = line.split(',')
                            # filename, nx,ny,nz,dx,dy,dz,ox,oy,oz
                            nx = np.array([int(p) for p in parts[1:4]])
                            #print(nx)
                            dx = np.array([float(p) for p in parts[4:7]])
                            #print(dx)
                            ox = np.array([float(p) for p in parts[7:10]])
                            #print(ox)
                            # change xyz to zyx
                            nx = nx[::-1]
                            dx = dx[::-1]
                            ox = ox[::-1]
                            kwargs['dx'] = dx
                            kwargs['ox'] = ox                                            
            else:
                print('did not found geomtry info, using some defaults')
        if 'dx' not in kwargs:
            warn('Voxel size dx not in keywords, using (1,1,1)')
            dx = np.array([1.0,1.0,1.0])
        if 'ox' not in kwargs:
            warn('Origin not in keywords, using 0 for z, and image center for xy')
            ox = [0.0,None,None]
        if ext in ['.tif','.tiff']:
            images = tf.imread(fname)
        else:
            images = plt.imread(fname)
        # convert to float
        if images.dtype == np.uint8:
            images = images.astype(float)/255.0
        else:
            images = images.astype(float) # this may do nothing if it is already float
            images = images / np.mean(np.abs(images.reshape(-1, images.shape[-1])), axis=0) # normalize by the mean of each channel
        # add leading dimensions and reshape, note offset may be none in dims 1 and 2.
        images = images[None].transpose(-1,0,1,2)
        nI = images.shape[1:]
        x0 = np.arange(nI[0])*dx[0] + ox[0]
        x1 = np.arange(nI[1])*dx[1]
        if ox[1] is None:
            x1 -= np.mean(x1)
        else:
            x1 += ox[1]
        x2 = np.arange(nI[2])*dx[2]
        if ox[2] is None:
            x2 -= np.mean(x2)
        else:
            x2 += ox[2]
        x = [x0,x1,x2]
        title = ''
        names = ['']            
        
        
    else:
        print('Opening with nibabel, note only 3D images supported')
        vol = nib.load(fname,**kwargs)
        print(vol.header)
        images = np.array(vol.get_fdata())
        if images.ndim == 3:
            images = images[None]
        A = vol.header.get_base_affine()
        if not np.allclose(np.diag(np.diag(A[:3,:3])),A[:3,:3]):
            raise Exception('Only support diagonal affine matrix with nibabel')
        x = [ A[i,-1] + np.arange(images.shape[i+1])*A[i,i] for i in range(3)]
        for i in range(3):
            if A[i,i] < 0:
                x[i] = x[i][::-1]
                images = np.array(np.flip(images,axis=i+1))
        images = images / np.mean(np.abs(images)) # normalize
            
        title = ''
        names = ['']
        
    return x,images,title,names


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


def get_xy_angle(list_curves):
    """input: a list of curves as Nx3 lists of coordinates
        output: list of Nx1 orientations as angles from the positive x direction in the xy plane"""

    list_angles = []
    for i in range(len(list_curves)):
        vectors = []
        for j in range(list_curves[i].shape[0] - 1):
            vectors.append(list_curves[i][j+1,:] - list_curves[i][j,:])
        list_angles.append(np.arctan(vectors[:,1] / vectors[:,0]))
    
    return list_angles