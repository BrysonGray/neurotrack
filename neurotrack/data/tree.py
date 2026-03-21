import numpy as np
import torch
from typing import Union 
from scipy.spatial import KDTree
from neurotrack.data import loading as load

def split_swc_into_sections(swc_list):
    """
    Splits SWC formatted list into sections.
    Each section is a torch tensor of 3D points defined as a segment of the tree between endpoints or branch points.
    """
    adj_dict = load.adjacency_dict(swc_list)
    # Convert to NumPy array of float32 to ensure consistent numeric type and catch malformed input
    swc_array = torch.from_numpy(np.array(swc_list, dtype=np.float32))
    id_to_index = {int(row[0].item()): i for i, row in enumerate(swc_array)}
    sections = {}
    visited = set()
    all_node_ids = set(id_to_index.keys())

    while len(visited) < len(all_node_ids):
        # Find a start node for the next component
        start_node = None
        unvisited = all_node_ids - visited
        
        # Priority 1: Unvisited isolated nodes (degree 0)
        for node_id in unvisited:
            if node_id not in adj_dict:
                start_node = node_id
                break
        
        # Priority 2: Unvisited endpoints (degree 1)
        if start_node is None:
            for node_id in unvisited:
                if len(adj_dict[node_id]) == 1:
                    start_node = node_id
                    break
        
        # Priority 3: Any unvisited node (e.g. cycle)
        if start_node is None:
            start_node = next(iter(unvisited))

        # Handle isolated node
        if start_node not in adj_dict:
            section_id = len(sections) + 1
            sections[section_id] = swc_array[[id_to_index[start_node]], 2:5]
            visited.add(start_node)
            continue

        current_section_id = len(sections) + 1
        sections[current_section_id] = [start_node]
        unvisited_sections = {current_section_id}
        visited.add(start_node)

        # walk along edges until reaching another endpoint or branch point
        while unvisited_sections:
            current_section_id = unvisited_sections.pop()
            current_node = sections[current_section_id][-1]
            while True:
                neighbors = adj_dict[current_node]
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                if len(unvisited_neighbors) == 1:
                    # continue along the section
                    next_node = unvisited_neighbors[0]
                    sections[current_section_id].append(next_node)
                    visited.add(next_node)
                    current_node = next_node
                elif len(unvisited_neighbors) > 1:
                    # add new section for each branch beginning with the current node
                    for branch_node in unvisited_neighbors:
                        section_id = len(sections) + 1
                        sections[section_id] = [current_node, branch_node]
                        visited.add(branch_node)
                        unvisited_sections.add(section_id)
                    # complete current section
                    section_nodes = sections[current_section_id]
                    sections[current_section_id] = swc_array[[id_to_index[n] for n in section_nodes], 2:5]
                    break
                elif len(unvisited_neighbors) == 0:
                    # dead end
                    section_nodes = sections[current_section_id]
                    sections[current_section_id] = swc_array[[id_to_index[n] for n in section_nodes], 2:5]
                    break

    return sections


def split_paths_into_sections(paths):
    """
    Splits paths into sections based on their origins.
    Each section is defined as a segment of the path between intersections with other paths' origins.
    """
    path_origins = np.array([p[0][:3] for p in paths])
    sections = {}
    # divide each path wherever it intersects with the origin of other paths
    # create a new section for each segment of the path
    for path in paths:
        path_array = np.array(path) if not isinstance(path, np.ndarray) else path
        intersections = [0] + [i+1 for i, point in enumerate(path_array[1:]) if point[:3] in path_origins]
        if len(intersections) > 1:
            sections |= {len(sections) + i+1: torch.from_numpy(path_array[intersections[i]:intersections[i+1]+1].astype(np.float32)) for i in range(len(intersections)-1)}
            if intersections[-1] != len(path_array) - 1:
                sections[len(sections)+1] = torch.from_numpy(path_array[intersections[-1]:].astype(np.float32))  # Add the last segment
        else:
            sections[len(sections)+1] = torch.from_numpy(path_array.astype(np.float32))

    return sections


def get_single_stream_length(section_id, section_graph, sections, visited=None):
    """
    Recursively calculates the total length of a section and its downstream sections.
    Prevents infinite recursion by tracking visited sections.
    """
    if visited is None:
        visited = set()
    if section_id in visited:
        return 0
    visited.add(section_id)
    # total_length = len(sections[section_id])
    # total_length = np.linalg.norm(sections[section_id][0,:3] - sections[section_id][-1,:3])
    total_length = (torch.sum((sections[section_id][1:, :3] - sections[section_id][:-1, :3])**2, dim=1)**0.5).sum()
    for connected_section in section_graph.get(section_id, []):
        total_length += get_single_stream_length(connected_section, section_graph, sections, visited)
    return total_length


def get_all_stream_lengths(sections):
    """
    Calculate the lengths of each section.
    """
    # Make section adjacency graph
    section_graph = {}
    for id,section in sections.items():
        section_graph[id] = []
        for other_id,other_section in sections.items():
            if id != other_id and all(section[-1] == other_section[0]):
                section_graph[id].append(other_id)
    # Calculate total length of each section and its downstream sections
    stream_lengths = {}
    for section_id in sections.keys():
        stream_lengths[section_id] = get_single_stream_length(section_id, section_graph, sections)

    return stream_lengths, section_graph


def get_longest_stream(section_id, section_graph, section_lengths):
    """
    Get longest stream, starting at the given section id, where a stream is a list of section ids of consecutive sections.
    """
    stream = [section_id]
    while section_id in section_graph:
        next_section = max(section_graph[section_id], key=lambda x: section_lengths.get(x, 0), default=None)
        if next_section is not None:
            stream.append(next_section)
            section_id = next_section
        else:
            break
    return stream


def get_hierarchical_streams(stream_lengths, section_graph):

    hierarchical_streams = {}
    # list all sections in descending order of their stream lengths
    section_ids_sorted = sorted(stream_lengths.keys(), key=lambda x: stream_lengths[x], reverse=True)
    # start with the longest stream
    while section_ids_sorted:
        hierarchical_streams[section_ids_sorted[0]] = get_longest_stream(section_ids_sorted[0], section_graph, stream_lengths)
        # remove the sections that have been collected into a stream
        for section in hierarchical_streams[section_ids_sorted[0]]:
            if section in section_ids_sorted:
                section_ids_sorted.remove(section)
    
    return hierarchical_streams


def restitch_sections(hierarchical_streams, sections):
    restitched_sections = {}
    for i in hierarchical_streams.keys():
        stream_ids = hierarchical_streams[i]
        if not stream_ids:
            continue
            
        # Start with the first section
        stitched = [sections[stream_ids[0]]]
        
        # For subsequent sections, exclude the first point (which duplicates the last point of previous section)
        for section_id in stream_ids[1:]:
            stitched.append(sections[section_id][1:])
            
        restitched_sections[i] = torch.cat(stitched)
    
    return restitched_sections


def restructure_neuron_tree(input: list, input_type='paths'):
    """
    Restructure neuron tree by splitting paths into sections, calculating stream lengths,
    and restitching sections based on hierarchical streams.
    """
    
    # Split paths into sections
    if input_type == 'swc':
        sections = split_swc_into_sections(input)
    elif input_type == 'paths':
        sections = split_paths_into_sections(input)
    else:
        raise ValueError(f"Invalid input_type '{input_type}'. Must be 'swc' or 'paths'.")
    stream_lengths, section_graph = get_all_stream_lengths(sections)
    hierarchical_streams = get_hierarchical_streams(stream_lengths, section_graph)
    restitched_sections = restitch_sections(hierarchical_streams, sections)

    return restitched_sections


def remove_soma(swc_list, max_radius=7.0, verbose=False):
    """Remove soma from neuron tree based on radius threshold.
    
    Parameters
    ----------
    swc_list : list
        SWC neuron tree data in list format.
    max_radius : float
        Maximum radius to consider for soma removal.
    
    Returns
    -------
    swc_list : list
        Updated SWC data with soma removed.
    seeds : np.ndarray
        Coordinates of the seeds after soma removal.
    """
    buffer = []
    buffer_size = 5
    adj_dict = load.adjacency_dict(swc_list)
    nodes_to_remove = []
    visited = [1]
    branch_beginnings = []
    temp_branches = []
    seeds = []
    swc_list = np.array(swc_list)
    # walk along adj_dict
    # assuming first node is 1 and is the root node
    node = 1
    temp_branches = [n for n in adj_dict[node] if n not in [node - 1, node + 1] and n not in visited]
    branch_beginnings += temp_branches
    while True:
        radius = swc_list[swc_list[:, 0] == node][0, 5]  # Assuming radius is at index 5
        if radius >= max_radius:
            nodes_to_remove.append(node)
            nodes_to_remove += buffer
            buffer = []
            branch_beginnings += temp_branches
            temp_branches = []
        else:
            buffer.append(node)
            if len(buffer) > buffer_size:
                seeds.append(buffer[0])
                # If buffer exceeds size, move to next branch beginning
                node = branch_beginnings.pop(0) if branch_beginnings else None
                if node is None:
                    if verbose:
                        print("Done")
                    break
                else:
                    buffer = []
                    # reset temp_branches
                    temp_branches = [n for n in adj_dict[node] if n not in [node - 1, node + 1] and n not in visited]
                    branch_beginnings += temp_branches
                    visited.append(node)
                    continue

        next_node = node + 1
        if next_node not in adj_dict[node]:
            branch_beginnings += temp_branches
            temp_branches = []
            node = branch_beginnings.pop(0) if branch_beginnings else None
            if node is None:
                if verbose:
                    print(f"Reached end of the tree without stopping. Adjust max_radius to prevent this.")
                break
            buffer = []
            temp_branches += [n for n in adj_dict[node] if n not in [node - 1, node + 1] and n not in visited]
            branch_beginnings += temp_branches
            visited.append(node)
        else:
            node = next_node
            temp_branches += [n for n in adj_dict[node] if n not in [node - 1, node + 1] and n not in visited]
            visited.append(node)

    seeds = np.array([swc_list[swc_list[:, 0] == seed][0, 2:5] for seed in seeds if seed not in nodes_to_remove])
    # remove nodes from swc_list
    for node in nodes_to_remove:
        swc_list = np.delete(swc_list, swc_list[:, 0] == node, axis=0)
    
    return swc_list.tolist(), seeds


def resample_tree(paths, step_size=1.0):
    """
    Resample the neuron tree to have points at regular intervals defined by step_size.
    
    Connection points and endpoints are preserved
    at their exact locations. Only the segments between these fixed points are resampled.

    Parameters
    ----------
    paths : list or array_like
        List of paths, each an Nx3 array of points representing the neuron tree.
    step_size : float
        The distance between consecutive points after resampling.

    Returns
    -------
    list
        Resampled list of paths with preserved connection points and endpoints.
    """
    # Convert all paths to numpy arrays
    paths_np = [np.asarray(path) if not isinstance(path, np.ndarray) else path for path in paths]
    
    # Collect all path start points
    path_start_points = set()
    for path in paths_np:
        if len(path) > 0:
            point_tuple = tuple(path[0][:3] if len(path[0]) >= 3 else path[0])
            path_start_points.add(point_tuple)
    
    # Resample each path, preserving fixed points
    new_paths = []
    for path_idx, path in enumerate(paths_np):
        if len(path) == 0:
            new_paths.append(path)
            continue
            
        # Find indices of fixed points in this path
        fixed_indices = []
        for point_idx, point in enumerate(path):
            point_tuple = tuple(point[:3] if len(point) >= 3 else point)
            if point_idx == 0 or point_idx == len(path) - 1 or point_tuple in path_start_points:
                fixed_indices.append(point_idx)
        
        # If no fixed points or only one, resample the entire path normally
        if len(fixed_indices) <= 1:
            distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
            new_distances = np.arange(0, cumulative_distances[-1], step_size)
            new_points = []
            
            for d in new_distances:
                idx = np.searchsorted(cumulative_distances, d)
                if idx == 0:
                    new_points.append(path[0])
                elif idx >= len(cumulative_distances):
                    new_points.append(path[-1])
                else:
                    t = (d - cumulative_distances[idx - 1]) / (cumulative_distances[idx] - cumulative_distances[idx - 1])
                    new_point = (1 - t) * path[idx - 1] + t * path[idx]
                    new_points.append(new_point)
            
            new_paths.append(np.array(new_points))
            continue
        
        # Resample segments between fixed points
        resampled_path = []
        for i in range(len(fixed_indices)):
            start_idx = fixed_indices[i]
            # Include the next fixed point in the segment (hence +1)
            if i + 1 < len(fixed_indices):
                end_idx = fixed_indices[i + 1] + 1  # +1 to include the next fixed point
            else:
                end_idx = len(path)  # Last segment goes to the end
            
            # Extract segment (now includes both start and end fixed points)
            segment = path[start_idx:end_idx]
            
            if len(segment) <= 1:
                # Single point segment, just add it
                if i == 0:  # Only add if first segment
                    resampled_path.append(segment[0])
                continue
            
            # Resample this segment
            distances = np.linalg.norm(np.diff(segment, axis=0), axis=1)
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
            segment_length = cumulative_distances[-1]
            
            # Sample points between start and end
            # Always include both start and end points to preserve branch points
            if segment_length < step_size:
                # Segment is shorter than step_size, just keep start and end
                segment_points = [segment[0], segment[-1]]
            else:
                new_distances = np.arange(0, segment_length, step_size)
                segment_points = []
                
                for d in new_distances:
                    idx = np.searchsorted(cumulative_distances, d)
                    if idx == 0:
                        segment_points.append(segment[0])
                    elif idx >= len(cumulative_distances):
                        segment_points.append(segment[-1])
                    else:
                        t = (d - cumulative_distances[idx - 1]) / (cumulative_distances[idx] - cumulative_distances[idx - 1])
                        new_point = (1 - t) * segment[idx - 1] + t * segment[idx]
                        segment_points.append(new_point)
                
                # Ensure the end point is always included (critical for preserving branch points)
                if not np.allclose(segment_points[-1], segment[-1]):
                    segment_points.append(segment[-1])
            
            # Add points, avoiding duplicates at connection points
            if i == 0:
                resampled_path.extend(segment_points)
            else:
                # Skip first point since it duplicates the last point of previous segment
                if len(segment_points) > 1:
                    resampled_path.extend(segment_points[1:])
                elif len(segment_points) == 1 and not np.allclose(segment_points[0], resampled_path[-1]):
                    resampled_path.append(segment_points[0])
        
        new_paths.append(np.array(resampled_path))
    
    return new_paths