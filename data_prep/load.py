import numpy as np
import os
from pathlib import Path
import sys
import tifffile as tf
import torch

sys.path.append(str(Path(__file__).parent))
from data_utils import interp


# Rotation matrices
def Rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

def Ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def tiff(img_dir, pixelsize=[1.0,1.0,1.0], downsample_factor=1.0, inverse=False):
    """
    Load a stack of TIFF images from a directory, downsample them, and optionally invert the pixel values.
    
    Parameters
    ----------
    img_dir : str
        Directory containing the TIFF images.
    pixelsize : list of float, optional
        Pixel size in each dimension (z, y, x). Default is [1.0, 1.0, 1.0].
    downsample_factor : float, optional
        Factor by which to downsample the images. Default is 1.0.
    inverse : bool, optional
        If True, invert the pixel values. Default is False.
        
    Returns
    -------
    torch.Tensor
        A tensor containing the processed image stack with shape (channels, height, width, depth).
    """
    
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



def swc(labels_file, rotate=False):
    """
    Load and parse an SWC file.
    
    Parameters
    ----------
    labels_file : str
        Path to the SWC file to be loaded.
        
    Returns
    -------
    list of list
        A list of parsed SWC data, where each sublist contains:
        [index, type, x, y, z, radius, parent_index].
        
    Notes
    -----
    The function attempts to read the file with the default encoding first.
    If it fails, it retries with 'latin1' encoding.
    Lines starting with '#' or empty lines are ignored.
    """
    
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

    if rotate:

        choices = [0, 90, 180, 270]
        angle = [np.random.choice(choices), np.random.choice(choices), np.random.choice(choices)]
        rotation = np.eye(7)
        rotation[2:5,2:5] = np.matmul(np.matmul(Rx(angle[0]), Ry(angle[1])), Rz(angle[2]))

        swc_list = np.matmul(swc_list, rotation.T)
        swc_list = swc_list.tolist()
        
    return swc_list


def parse_swc_list(swc_list, adjust=True, transpose=True):
    """
    Parses a list of SWC data and constructs sections, section graph, branches, and terminals.

    Parameters
    ----------
    swc_list : list
        A list of SWC data where each element is a list representing a node in the SWC format.
    adjust : bool, optional
        If True, scales and shifts the coordinates (default is True).
    transpose : bool, optional
        If True, transposes the coordinates (default is True).

    Returns
    -------
    sections : dict
        A dictionary where keys are section IDs and values are tensors of section coordinates.
    section_graph : dict
        A dictionary representing the graph of sections.
    branches : torch.Tensor
        A tensor of branch points.
    terminals : torch.Tensor
        A tensor of terminal points.
    scale : float
        The scaling factor applied to the coordinates.
    """

    graph = {}
    for parent in swc_list:
        children = []
        for child in swc_list:
            if int(child[6]) == int(parent[0]):
                children.append(int(child[0]))
        graph[int(parent[0])] = children

    sections = {1:[]}
    section_graph = {1:[]}
    i = 1
    section_id = 1
    for key, value in graph.items():
        if len(value) == 0:
            sections[i] = torch.tensor(sections[i]) # type: ignore #
            if transpose:
                sections[i] = torch.stack((sections[i][...,2], sections[i][...,1], sections[i][...,0]), dim=2) #type: ignore #
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
            lengths.append(torch.linalg.norm(segment[1] - segment[0]))
    avg_length = torch.median(torch.tensor(lengths))

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

    branches = torch.tensor(branches)
    if transpose and len(branches) > 0:
        branches = torch.stack((branches[:,2], branches[:,1], branches[:,0]), dim=1)
    terminals = torch.tensor(terminals)
    if transpose:
        terminals = torch.stack((terminals[:,2], terminals[:,1], terminals[:,0]), dim=1)

    scale = 1
    if adjust:
        # scale and shift coordinates
        max = torch.tensor([-1e6, -1e6, -1e6])
        min = torch.tensor([1e6, 1e6, 1e6])
        for id, section in sections.items():
            max = torch.maximum(max, section.amax(dim=(0,1))) # type: ignore #
            min = torch.minimum(min, section.amin(dim=(0,1))) # type: ignore #
        vol = torch.prod(max - min)
        scale = torch.round((5e7 / vol)**(1/3)) # scale depends on the volume

        for id, section in sections.items():
            section = (section - min) * scale + torch.tensor([10.0, 10.0, 10.0])
            sections[id] = section # type: ignore #
        if len(branches) > 0:
            branches = (branches - min) * scale + torch.tensor([10.0, 10.0, 10.0])
        terminals = (terminals - min) * scale + torch.tensor([10.0, 10.0, 10.0])

    return sections, section_graph, branches, terminals, scale

if __name__ == "__main__":
    pass