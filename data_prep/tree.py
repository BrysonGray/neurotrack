import numpy as np
from pathlib import Path
import torch
from typing import Union 
from scipy.spatial import KDTree
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_prep import load

def split_swc_into_sections(swc_list):
    """
    Splits SWC formatted list into sections.
    Each section is a torch tensor of 3D points defined as a segment of the tree between endpoints or branch points.
    """
    edge_list = load.undirected_edge_list(swc_list)
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
            if node_id not in edge_list:
                start_node = node_id
                break
        
        # Priority 2: Unvisited endpoints (degree 1)
        if start_node is None:
            for node_id in unvisited:
                if len(edge_list[node_id]) == 1:
                    start_node = node_id
                    break
        
        # Priority 3: Any unvisited node (e.g. cycle)
        if start_node is None:
            start_node = next(iter(unvisited))

        # Handle isolated node
        if start_node not in edge_list:
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
                neighbors = edge_list[current_node]
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
        intersections = [0] + [i+1 for i, point in enumerate(path[1:]) if point[:3] in path_origins]
        if len(intersections) > 1:
            sections |= {len(sections) + i+1: path[intersections[i]:intersections[i+1]+1] for i in range(len(intersections)-1)}
            if intersections[-1] != len(path) - 1:
                sections[len(sections)+1] = path[intersections[-1]:]  # Add the last segment
        else:
            sections[len(sections)+1] = path

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
    edge_list = load.undirected_edge_list(swc_list)
    nodes_to_remove = []
    visited = [1]
    branch_beginnings = []
    temp_branches = []
    seeds = []
    swc_list = np.array(swc_list)
    # walk along edge_list
    # assuming first node is 1 and is the root node
    node = 1
    temp_branches = [n for n in edge_list[node] if n not in [node - 1, node + 1] and n not in visited]
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
                    temp_branches = [n for n in edge_list[node] if n not in [node - 1, node + 1] and n not in visited]
                    branch_beginnings += temp_branches
                    visited.append(node)
                    continue

        next_node = node + 1
        if next_node not in edge_list[node]:
            branch_beginnings += temp_branches
            temp_branches = []
            node = branch_beginnings.pop(0) if branch_beginnings else None
            if node is None:
                if verbose:
                    print(f"Reached end of the tree without stopping. Adjust max_radius to prevent this.")
                break
            buffer = []
            temp_branches += [n for n in edge_list[node] if n not in [node - 1, node + 1] and n not in visited]
            branch_beginnings += temp_branches
            visited.append(node)
        else:
            node = next_node
            temp_branches += [n for n in edge_list[node] if n not in [node - 1, node + 1] and n not in visited]
            visited.append(node)

    seeds = np.array([swc_list[swc_list[:, 0] == seed][0, 2:5] for seed in seeds if seed not in nodes_to_remove])
    # remove nodes from swc_list
    for node in nodes_to_remove:
        swc_list = np.delete(swc_list, swc_list[:, 0] == node, axis=0)
    
    return swc_list.tolist(), seeds


def resample_tree(paths, step_size=1.0):
    """
    Resample the neuron tree to have points at regular intervals defined by step_size.

    Parameters
    ----------
    paths : list or array_like
        Nx3 list or array of points representing the neuron tree.
    step_size : float
        The distance between consecutive points after resampling.

    Returns
    -------
    list
        Resampled SWC formatted list of points.
    """

    new_paths = []
    for path in paths:
        # Calculate cumulative distances along the path
        path = np.asarray(path)
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Determine new sample points
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

        # Replace original path points with resampled points
        new_paths.append(np.array(new_points))
    
    return new_paths


def directed_divergence(tree_a, tree_b, threshold=0.0):
    """
    Calculate the directed divergence from tree_a to tree_b.
    For each point in tree_a, find the nearest point in tree_b and compute the average distance.

    Parameters
    ----------
    tree_a : list or array_like
        Nx7 SWC formatted list or array of points representing the first tree.
    tree_b : list or array_like
        Mx7 SWC formatted list or array of points representing the second tree.
    threshold : float
        Distance threshold to consider for divergence calculation.

    Returns
    -------
    float
        The average distance from each point in tree_a to the nearest point in tree_b.
    float
        The number of points in tree_a that are within the threshold distance to tree_b.
    """
    tree_a = np.asarray(tree_a)
    tree_b = np.asarray(tree_b)
    kdtree_b = KDTree(tree_b[:, 2:5])  # Use only x, y, z coordinates
    distances, _ = kdtree_b.query(tree_a[:, 2:5])
    avg_distance = np.mean(distances)
    n_substantial = np.sum(distances >= threshold)
    return avg_distance, n_substantial
       


def spatial_distance(tree_a, tree_b, threshold=0.0):
    """
    Calculate the average of the directed divergence from A to B and from B to A. 

    Parameters
    ----------
    tree_a : list or array_like
        Nx7 SWC formatted list or array of points representing the first tree.
    tree_b : list or array_like
        Mx7 SWC formatted list or array of points representing the second tree.

    Returns
    -------
    float
        The average bi-directional distance.
    float
        The proportion of points within the threshold distance in both directions.
    """

    divergence_a_to_b, n_substantial_a = directed_divergence(tree_a, tree_b, threshold)
    divergence_b_to_a, n_substantial_b = directed_divergence(tree_b, tree_a, threshold)
    avg_divergence = (divergence_a_to_b + divergence_b_to_a) / 2
    total_points = len(tree_a) + len(tree_b)
    proportion_substantial = (n_substantial_a + n_substantial_b) / total_points if total_points > 0 else 0

    return avg_divergence, proportion_substantial