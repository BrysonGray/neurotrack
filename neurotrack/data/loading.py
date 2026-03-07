import numpy as np
import os
from pathlib import Path
from scipy.ndimage import zoom
import tifffile as tf
import torch

from neurotrack.data.data_utils import interp


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


def swc(labels_file, rotate=False, verbose=False):
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
    # Columns are: index, type, x, y, z, radius, parent_index
    # Index, type, and parent_index should be integers, while x, y, z, and radius are floats.
    # Handle cases where all columns are saved as a decimal number.
    idx_type = 'int' if lines[0][0].isdigit() else 'float'
    if idx_type == 'int':
        swc_list = [list(map(int, line[:2])) + list(map(float, line[2:6])) + [int(line[6])] for line in lines]
    else:
        swc_list = [list(map(int, map(float, line[:2]))) + list(map(float, line[2:6])) + [int(float(line[6]))] for line in lines]

    if rotate:

        choices = [0, 90, 180, 270]
        angle = [np.random.choice(choices), np.random.choice(choices), np.random.choice(choices)]
        rotation = np.eye(7)
        rotation[2:5,2:5] = np.matmul(np.matmul(Rx(angle[0]), Ry(angle[1])), Rz(angle[2]))

        swc_list = np.matmul(swc_list, rotation.T)
        swc_list = swc_list.tolist()
        
    return swc_list


def map_tiff_to_swc(image_root, swc_root, use_fixed=False, verbose=True,
                    fixed_suffix="_FIXED_PARENT_CONNECTIONS.swc"):
    """
    Build a mapping from TIFF files to matching SWC files. Specific to Gold166 challenge data file naming conventions. 

    Parameters
    ----------
    image_root : str or Path
        Root directory containing TIFF files.
    swc_root : str or Path
        Root directory containing SWC files.
    use_fixed : bool, optional
        If True, prefer SWC files ending with ``fixed_suffix``. If False,
        exclude SWC files ending with ``fixed_suffix``.
    verbose : bool, optional
        Print matching diagnostics.
    fixed_suffix : str, optional
        SWC filename suffix used to identify cleaned/fixed files.

    Returns
    -------
    dict
        Dictionary mapping ``{tiff_path: swc_path}`` as Path objects.
    """
    image_root = Path(image_root)
    swc_root = Path(swc_root)

    tif_files = sorted([f for f in image_root.rglob("*.tif") if f.is_file()])
    swc_files = [f for f in swc_root.rglob("*.swc") if f.is_file()]

    img_to_swc_map = {}
    n_matched = 0

    for tiff_path in tif_files:
        tiff_stem = str(tiff_path.stem)

        # 1) Try split on '.v3dpbd' first
        # 2) Only if no matches, try split on '.v3draw'
        matching_swcs = [swc for swc in swc_files if tiff_stem == str(swc.name).split(".v3dpbd")[0]]
        if len(matching_swcs) == 0:
            matching_swcs = [swc for swc in swc_files if tiff_stem == str(swc.name).split(".v3draw")[0]]
        
        # if still no matches, try matching the stem
        if len(matching_swcs) == 0:
            matching_swcs = [swc for swc in swc_files if tiff_stem == str(swc.stem)]

        if len(matching_swcs) == 0:
            if verbose:
                print(f"    No matching SWC file found for {tiff_path.name}, skipping.")
            continue

        if use_fixed:
            fixed_matches = [swc for swc in matching_swcs if str(swc.name).endswith(fixed_suffix)]
            if fixed_matches:
                matching_swcs = fixed_matches
        else:
            matching_swcs = [swc for swc in matching_swcs if not str(swc.name).endswith(fixed_suffix)]
            if len(matching_swcs) == 0:
                if verbose:
                    print(f"    No non-fixed SWC file found for {tiff_path.name}, skipping.")
                continue

        matching_swcs = sorted(matching_swcs)
        if len(matching_swcs) > 1 and verbose:
            print(
                f"    Warning: Multiple matching SWC files found for {tiff_path.name}, "
                f"using first: {matching_swcs[0].name}"
            )

        swc_file = matching_swcs[0]
        if verbose:
            print(f"    Matched with SWC: {swc_file.name}")
        img_to_swc_map[tiff_path] = swc_file
        n_matched += 1

    if verbose:
        print(f"Matched {n_matched}/{len(tif_files)} TIFF files to SWC files.")

    return img_to_swc_map


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


# def undirected_edge_list(swc_list):
def adjacency_dict(swc_list):
    """
    Parse a list of SWC lines into an undirected edge list.
    Only includes edges between nodes that actually exist in the swc_list.

    """
    # First, build a set of valid node IDs for O(1) lookup
    swc_array = np.array(swc_list) if not isinstance(swc_list, np.ndarray) else swc_list
    valid_node_ids = set(swc_array[:, 0].astype(int))
    
    adj_dict = {}
    for node in swc_list:
        node_id = int(node[0])
        parent_id = int(node[6])
        if parent_id != -1:  # Ignore the root node which has no parent
            # Only add edges if both nodes exist
            if parent_id in valid_node_ids:
                adj_dict.setdefault(node_id, []).append(parent_id)
                adj_dict.setdefault(parent_id, []).append(node_id)
            else:
                # Parent doesn't exist - treat this node as a root
                adj_dict.setdefault(node_id, [])
        else:
            adj_dict.setdefault(node_id, [])  # Ensure the root node is in the adjacency dict
    return adj_dict


def parse_swc(swc_list, transpose=True, verbose=False):
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
    swc_list = np.array(swc_list)
    if swc_list.size == 0:
        return {}, {}
    
    # Compute undirected edge list
    adj_dict = adjacency_dict(swc_list)
    # Create node lookup for validation
    node_lookup = {int(row[0]): row for row in swc_list}
    
    # Make list of branching nodes
    branchings = [i for i in adj_dict.keys() if len(adj_dict[i]) > 2]
    sections = {}
    section_ends = {}
    while len(adj_dict) > 1:
        terminals_ = None
        if 'terminals' in locals():
            terminals_ = terminals
        # Make list of terminal nodes
        terminals = [i for i in adj_dict.keys() if len(adj_dict[i]) == 1]
        # check if the list of terminals has changed
        if terminals == terminals_:
            if verbose:
                print(f"Warning: The neuron tree has disconnected sections.")
            break
        # from each terminal node walk along the tree until you reach a branching node
        # or another terminal node
        for terminal in terminals:
            if adj_dict[terminal] == []: # if terminal node is the only node left
                break
            section = []
            node = terminal
            while True:
                if len(adj_dict[node]) == 0:
                    # if verbose:
                    print(f"Warning: Node {node} has no neighbors in adjacency dict")
                    # Clean up the isolated node
                    adj_dict.pop(node, None)
                    break
                next_node = adj_dict[node][0]
                
                # Defensive checks for data integrity
                if node not in node_lookup:
                    # if verbose:
                    print(f"Warning: Node {node} not found in swc_list. Skipping section starting at {terminal}.")
                    # Clean up adjacency dict for this broken path
                    adj_dict.pop(node, None)
                    if next_node in adj_dict:
                        adj_dict[next_node] = [n for n in adj_dict[next_node] if n != node]
                    break
                if next_node not in node_lookup:
                    # if verbose:
                    print(f"Warning: Node {next_node} not found in swc_list. Skipping section starting at {terminal}.")
                    # Clean up adjacency dict for this broken path
                    adj_dict.pop(node, None)
                    if next_node in adj_dict:
                        adj_dict[next_node] = [n for n in adj_dict[next_node] if n != node]
                    break
                
                section.append([node_lookup[node][2:6], node_lookup[next_node][2:6]])
                adj_dict.pop(node)
                adj_dict[next_node].remove(node)
                node = next_node

                if node in branchings or node in terminals:
                    # complete the section
                    section = np.array(section)
                    if transpose:
                        section = np.concatenate((section[:,:,:3][:,:,::-1], section[:,:,3,None]), axis=-1)
                    sections[terminal] = section
                    section_ends[terminal] = [terminal, node]

                    # repeat with a new list of terminals until the edge list only contains one final node
                    break
    
    # make directed sections graph
    sections_graph = {}
    for section in section_ends:
        sections_graph[section] = []
        # find all sections that share an end
        for other_section in section_ends:
            if other_section == section:
                continue
            if any(x in section_ends[other_section] for x in section_ends[section]):
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

    adj_dict = adjacency_dict(swc_list)
    branches = [i for i in adj_dict.keys() if len(adj_dict[i]) > 2]
    # for each branch, walk along each section to which it connects,
    # branches_to_remove = []
    # for branch in branches:
    #     # check the remaining length of each section starting from the current branch
    #     num_long_sections = 0
    #     for child in adj_dict[branch]:
    #         node = child
    #         prev = branch
    #         l = np.linalg.norm(np.array(swc_list[node-1][2:5]) - np.array(swc_list[prev-1][2:5]))
    #         if l >= avg_length:
    #             num_long_sections += 1
    #         while len(adj_dict[node]) >= 2:
    #             next_node = [n for n in adj_dict[node] if n != prev][0]
    #             prev = node
    #             node = next_node
    #             l += np.linalg.norm(np.array(swc_list[node-1][2:5]) - np.array(swc_list[prev-1][2:5]))
    #             if l >= avg_length:
    #                 num_long_sections += 1
    #                 break
    #         # if the length of at least two sections is greater than 2*avg_length, continue
    #         if num_long_sections > 2:
    #             break
    #     # else, mark the branch for removal
    #     if num_long_sections <= 2:
    #         branches_to_remove.append(branch)
    # # print(f'removing {len(branches_to_remove)} branches')
    # for branch in branches_to_remove:
    #     branches.remove(branch)
    swc_list = np.array(swc_list)
    branches = [swc_list[swc_list[:, 0] == i][0][2:6] for i in branches]
    terminals = [swc_list[swc_list[:, 0] == i][0][2:6] for i in adj_dict.keys() if len(adj_dict[i]) == 1]

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
    max = np.array([-1e6, -1e6, -1e6])
    min = np.array([1e6, 1e6, 1e6])
    for id, section in sections.items():
        max = np.maximum(max, section.max(axis=(0,1))[:3]) # type: ignore #
        min = np.minimum(min, section.min(axis=(0,1))[:3]) # type: ignore #
    vol = np.prod(max - min)
    scale = np.round((5e7 / vol)**(1/3)) # scale depends on the volume

    for id, section in sections.items():
        section[...,:3] = (section[...,:3] - min) * scale + np.array([10.0, 10.0, 10.0])
        sections[id] = section # type: ignore #
    if len(branches) > 0:
        branches[...,:3] = (branches[...,:3] - min) * scale + np.array([10.0, 10.0, 10.0])
    terminals[...,:3] = (terminals[...,:3] - min) * scale + np.array([10.0, 10.0, 10.0])

    return sections, branches, terminals, scale


def _resolve_spatial_axes(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Normalize TIFF-like arrays and return the axes corresponding to (z, y, x).

    Supports common 3D and 4D layouts while preserving non-spatial axes
    (e.g. RGB channels or time/channel dimensions).
    """
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim < 3:
        raise ValueError(f"Expected image with at least 3 dimensions, found shape {image.shape}")

    if image.ndim == 3:
        return image, (0, 1, 2)

    if image.ndim == 4:
        channel_like_axes = [axis for axis, size in enumerate(image.shape) if size in (3, 4)]
        if len(channel_like_axes) == 1:
            channel_axis = channel_like_axes[0]
            spatial_axes = tuple(axis for axis in range(4) if axis != channel_axis)
            return image, (spatial_axes[0], spatial_axes[1], spatial_axes[2])

        # Fallback for generic 4D stacks (e.g., t-z-y-x): treat last three axes as spatial.
        return image, (1, 2, 3)

    raise ValueError(
        f"Unsupported image shape {image.shape}. Expected a 3D volume or 4D volume with one extra axis."
    )


def crop_and_adjust_coords(image, swc_list):
    image, spatial_axes = _resolve_spatial_axes(image)
    # Get the bounding box from the SWC file
    swc_list = np.array(swc_list)
    max_coord = np.max(swc_list[:, 2:5], axis=0)
    min_coord = np.min(swc_list[:, 2:5], axis=0)
    # Calculate the crop size
    crop_max = np.ceil(max_coord + 11).astype(int)  # add 10 pixels margin plus one to include the max coordinate
    # ensure we don't exceed image dimensions
    # remember swc coordinates are in x-y-z not slice-row-column order
    spatial_shape = np.array([image.shape[axis] for axis in spatial_axes])  # (z, y, x)
    crop_max = np.minimum(crop_max, spatial_shape[::-1])  # compare in (x, y, z)
    crop_min = np.floor(min_coord - 10).astype(int)  # subtract 10 pixels margin
    crop_min = crop_min.clip(min=0)  # ensure we don't go below zero
    # Crop the image
    slices = [slice(None)] * image.ndim
    z_axis, y_axis, x_axis = spatial_axes
    slices[z_axis] = slice(crop_min[2], crop_max[2])
    slices[y_axis] = slice(crop_min[1], crop_max[1])
    slices[x_axis] = slice(crop_min[0], crop_max[0])
    cropped_image = image[tuple(slices)]
    # Adjust the coordinates in swc_list
    adjusted_swc_list = swc_list.copy()
    adjusted_swc_list[:, 2:5] -= (min_coord - 10).clip(min=0.0)  # adjust coordinates to the new origin

    return cropped_image, adjusted_swc_list.tolist()


def scale_and_adjust_coords(image, swc_list, scale, anisotropy_factor=None):
    swc_list_corrected = np.array(swc_list).copy()
    image, spatial_axes = _resolve_spatial_axes(image)
    img_scaled = image.copy()

    zoom_factors = [1.0] * image.ndim
    z_axis, y_axis, x_axis = spatial_axes

    if anisotropy_factor is not None:
        zoom_factors[z_axis] = anisotropy_factor
        img_scaled = zoom(img_scaled, tuple(zoom_factors))
        swc_list_corrected[:, 4] *= anisotropy_factor

    zoom_factors = [1.0] * img_scaled.ndim
    z_axis, y_axis, x_axis = _resolve_spatial_axes(img_scaled)[1]
    zoom_factors[z_axis] = scale
    zoom_factors[y_axis] = scale
    zoom_factors[x_axis] = scale
    img_scaled = zoom(img_scaled, tuple(zoom_factors))
    swc_list_corrected[:, 2:5] *= scale

    return img_scaled, swc_list_corrected.tolist()


if __name__ == "__main__":
    pass