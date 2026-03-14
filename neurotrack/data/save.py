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
    # Map (x, y, z) coordinates to their SWC node IDs for quick lookup
    coord_to_id = {}
    
    for path in paths:
        if len(path) < 2:
            continue
        
        point = path[0]
        point_tuple = (float(point[0]), float(point[1]), float(point[2]))
        
        # Check if starting point already exists (branch point)
        if point_tuple in coord_to_id:
            parent_id = coord_to_id[point_tuple]
        else:
            parent_id = -1
        
        # Add first point if not a duplicate
        if point_tuple not in coord_to_id:
            swc_list.append([id, 0, point[0].item(), point[1].item(), point[2].item(), 1.0, parent_id])
            coord_to_id[point_tuple] = id
            parent_id = id
            id += 1
        
        # Add remaining points
        for point in path[1:]:
            point_tuple = (float(point[0]), float(point[1]), float(point[2]))
            
            # Skip if duplicate
            if point_tuple in coord_to_id:
                parent_id = coord_to_id[point_tuple]
                continue
            
            swc_list.append([id, 0, point[0].item(), point[1].item(), point[2].item(), 1.0, parent_id])
            coord_to_id[point_tuple] = id
            parent_id = id
            id += 1

    return swc_list


def write_swc(swc_list, out, exist_ok=True):
    if os.path.exists(out) and not exist_ok:
        raise FileExistsError(f"The file {out} already exists.")
    if not os.path.exists(os.path.split(out)[0]):
        os.makedirs(os.path.split(out)[0], exist_ok=True)
    with open(out, mode='w') as f:
        for line in swc_list:
            if isinstance(line[0], float):
                line = [int(line[0]), int(line[1])] + list(line[2:6]) + [int(line[6])]
            f.write(' '.join(map(str,line)) + '\n')