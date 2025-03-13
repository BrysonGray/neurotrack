import numpy as np
import os

def paths_to_swc(paths):
    """
    Convert paths list to swc list.

    Parameters
    ----------
    paths : list
        A list of lists of 3D points.
    
    Returns
    -------
    swc_list : list
        A list of lists in swc data format. 
    """

    id = 1
    swc_list = []
    for i,path in enumerate(paths):
        if len(path) < 2:
            continue
        # set first point
        point = path[0]
        if i == 0:
            parent_id = -1
        else:
            # The first point from a child section should match a point on the parent section.
            match = 0
            for k in swc_list:
                if k[2:5] == list(point):
                    parent_id = k[0]
                    match = 1
            # If not, move to the next path
            if not match:
                continue
        swc_list.append([id, 0, point[0].item(), point[1].item(), point[2].item(), 1.0, parent_id])
        parent_id = id
        id += 1
        # now the rest
        for point in path[1:]:
            swc_list.append([id, 0, point[0].item(), point[1].item(), point[2].item(), 1.0, parent_id])
            parent_id = id
            id += 1

    return swc_list


def write_swc(swc_list, out):
    if os.path.exists(out):
        raise FileExistsError(f"The file {out} already exists.")
    if not os.path.exists(os.path.split(out)[0]):
        os.makedirs(os.path.split(out)[0], exist_ok=True)
    with open(out, mode='x') as f:
        for line in swc_list:
            f.write(' '.join(map(str,line)) + '\n')