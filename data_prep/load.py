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



def swc(labels_file, rotate=False, verbose=True):
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
    if verbose:
        print(f"loading file: {labels_file}")
    try:
        with open(labels_file, 'r') as f:
            lines = f.readlines()
    except:
        with open(labels_file, 'r', encoding="latin1") as f:
            lines = f.readlines()

    lines = [line.split() for line in lines if not line.startswith('#') and line.strip()]
    swc_list = [list(map(int, line[:2])) + list(map(float, line[2:6])) + [int(line[6])] for line in lines]

    if rotate:

        choices = [0, 90, 180, 270]
        angle = [np.random.choice(choices), np.random.choice(choices), np.random.choice(choices)]
        rotation = np.eye(7)
        rotation[2:5,2:5] = np.matmul(np.matmul(Rx(angle[0]), Ry(angle[1])), Rz(angle[2]))

        swc_list = np.matmul(swc_list, rotation.T)
        swc_list = swc_list.tolist()
        
    return swc_list


def parse_swc_list_original(swc_list, transpose=True):
    graph = {}
    for parent in swc_list:
        children = []
        for child in swc_list:
            if int(child[6]) == int(parent[0]):
                children.append(int(child[0]))
        graph[int(parent[0])] = children

    sections = {1:[]}
    section_graph = {1:[]}
    section_id = 1
    for key, value in graph.items():
        if len(value) == 0: 
            sections[section_id] = torch.tensor(sections[section_id]) # type: ignore #
            if transpose:
                sections[section_id] = torch.stack((sections[section_id][...,2], sections[section_id][...,1], sections[section_id][...,0], sections[section_id][...,3]), dim=2) #type: ignore #
            section_id = key+1 # go to the section whose first segment corresponds to the next key
            section_graph[section_id] = []
        elif len(value) == 1:
            sections[section_id].append([swc_list[key-1][2:6], swc_list[value[0]-1][2:6]])
        else:
            # Edit 2/5/25: Every branch spawns new sections and terminates the parent section
            if len(sections[section_id]) == 0: # The section is empty. This might happen if the root node is also a branch.
                # In this case do not terminate the section
                for child in value:
                    if child == key + 1:
                        # add segment to the same section
                        sections[section_id].append([swc_list[key-1][2:6], swc_list[child-1][2:6]])
                    else:
                        # add segment to a new section
                        sections[child] = [[swc_list[key-1][2:6], swc_list[child-1][2:6]]]
                        section_graph[section_id].append(child)
            else:  
                for child in value:
                    sections[child] = [[swc_list[key-1][2:6], swc_list[child-1][2:6]]]
                    section_graph[section_id].append(child)
                # end the section and go to the next one
                sections[section_id] = torch.tensor(sections[section_id]) # type: ignore #
                if transpose:
                    sections[section_id] = torch.stack((sections[section_id][...,2], sections[section_id][...,1], sections[section_id][...,0], sections[section_id][...,3]), dim=2) #type: ignore #
                section_id = key+1 # go to the section whose first segment corresponds to the next key
                section_graph[section_id] = []
    return sections, section_graph


def undirected_edge_list(swc_list):
    """
    Parse a list of SWC lines into an undirected edge list.

    """
    edge_list = {}
    graph = []
    # swc_list -> (id, type, x,y,z, radius, parent)
    for line in swc_list:
        if line[6] == -1:
            continue
        graph.append([int(line[0]), int(line[6])])
    graph = np.array(graph)
    # Make graph undirected
    graph = np.vstack([graph, graph[:, ::-1]])
    # for node id in the first column, find all connected nodes in the second column
    ids = np.unique(graph[:, 0])
    for i in ids:
        adjacents = graph[graph[:, 0] == i, 1]
        edge_list[i.item()] = adjacents.tolist()

    return edge_list


def parse_swc(swc_list, transpose=True):
    """
    Parse a list of SWC lines and return an directed adjacency list of neuron sections,
    and a dictionary of sections with their corresponding coordinates. Sections are defined
    as the nodes between and including branching points or terminal points.

    Parameters
    ----------
    swc_list : list of lists
        The list of SWC lines
    transpose : bool, optional
        Whether to transpose coordinates (i.e. (x,y,z)->(z,y,x)). (default is True)
    
    Returns
    -------
    sections : dict
        The dictionary of sections with their corresponding coordinates.
    sections_graph : dict
        The directed adjacency list of neuron sections.
    """

    # Compute undirected edge list
    edge_list = undirected_edge_list(swc_list)
    # Make list of branching nodes
    branchings = [i for i in edge_list.keys() if len(edge_list[i]) > 2]
    sections = {}
    section_ends = {}
    while len(edge_list) > 1:
        # Make list of terminal nodes
        terminals = [i for i in edge_list.keys() if len(edge_list[i]) == 1]
        # from each terminal node walk along the tree until you reach a branching node
        # or another terminal node
        for terminal in terminals:
            if edge_list[terminal] == []: # if terminal node is the only node left
                break
            section = []
            node = terminal
            while True:
                next_node = edge_list[node][0]
                section.append([swc_list[node-1][2:6], swc_list[next_node-1][2:6]])
                edge_list.pop(node)
                edge_list[next_node].remove(node)
                node = next_node

                if node in branchings or node in terminals:
                    # complete the section
                    section = np.array(section)
                    if transpose:
                        section = np.concatenate((section[:,:,:3][:,:,::-1], section[:,:,3,None]), axis=-1)
                    sections[terminal] = section
                    section_ends[terminal] = node

                    # repeat with a new list of terminals until the edge list only contains one final node
                    break
    
    # make undirected sections graph
    sections_graph = {}
    for section in section_ends:
        sections_graph[section] = []
        # find all sections that have the same end node
        for other_section in section_ends:
            if other_section == section:
                continue
            if section_ends[section] == section_ends[other_section]:
                sections_graph[section].append(other_section)

    return sections, sections_graph
    

def get_critical_points(swc_list, sections, transpose=True):
    # filter branches
    # get average segment length
    lengths = []
    for section in sections.values():
        for segment in section:
            lengths.append(np.linalg.norm(segment[1,:3] - segment[0,:3]))
    avg_length = np.median(np.array(lengths))

    edge_list = undirected_edge_list(swc_list)
    branches = [i for i in edge_list.keys() if len(edge_list[i]) > 2]
    # for each branch, walk along each section to which it connects,
    branches_to_remove = []
    for branch in branches:
        # check the remaining length of each section starting from the current branch
        num_long_sections = 0
        for child in edge_list[branch]:
            node = child
            prev = branch
            l = np.linalg.norm(np.array(swc_list[node-1][2:5]) - np.array(swc_list[prev-1][2:5]))
            while len(edge_list[node]) >= 2:
                next_node = [n for n in edge_list[node] if n != prev][0]
                prev = node
                node = next_node
                l += np.linalg.norm(np.array(swc_list[node-1][2:5]) - np.array(swc_list[prev-1][2:5]))
                if l > 2*avg_length:
                    num_long_sections += 1
                    break
            # if the length of at least two sections is greater than 2*avg_length, continue
            if num_long_sections > 2:
                break
        # else, mark the branch for removal
        if num_long_sections <= 2:
            branches_to_remove.append(branch)
    for branch in branches_to_remove:
        branches.remove(branch)

    branches = [swc_list[i-1][2:6] for i in branches]
    terminals = [swc_list[i-1][2:6] for i in edge_list.keys() if len(edge_list[i]) == 1]

    branches = np.array(branches)
    if transpose and len(branches) > 0:
        branches = np.concatenate((branches[:,:3][:,::-1], branches[:,3,None]), axis=-1)
    terminals = np.array(terminals)
    if transpose:
        terminals = np.concatenate((terminals[:,:3][:,::-1], terminals[:,3,None]), axis=-1)


    return branches, terminals


def adjust_neuron_coords(sections, branches, terminals):
    scale = 1
    # scale and shift coordinates
    max = torch.tensor([-1e6, -1e6, -1e6])
    min = torch.tensor([1e6, 1e6, 1e6])
    for id, section in sections.items():
        max = torch.maximum(max, section.max(axis=(0,1))) # type: ignore #
        min = torch.minimum(min, section.max(axis=(0,1))) # type: ignore #
    vol = torch.prod(max - min)
    scale = torch.round((5e7 / vol)**(1/3)) # scale depends on the volume

    for id, section in sections.items():
        section = (section - min) * scale + torch.tensor([10.0, 10.0, 10.0])
        sections[id] = section # type: ignore #
    if len(branches) > 0:
        branches = (branches - min) * scale + torch.tensor([10.0, 10.0, 10.0])
    terminals = (terminals - min) * scale + torch.tensor([10.0, 10.0, 10.0])

    return sections, branches, terminals, scale


if __name__ == "__main__":
    pass