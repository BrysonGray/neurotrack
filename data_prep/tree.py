import numpy as np
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_prep import load


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
        restitched_section = np.concatenate([sections[section_id] for section_id in hierarchical_streams[i]])
        # remove duplicate points, keep order of first occurrence
        restitched_section, indices = np.unique(restitched_section, axis=0, return_index=True)
        order = np.argsort(indices)
        restitched_section = restitched_section[order]
        restitched_sections[i] = restitched_section
    
    return restitched_sections


def restructure_neuron_tree(paths):
    """
    Restructure neuron tree by splitting paths into sections, calculating stream lengths,
    and restitching sections based on hierarchical streams.
    """
    
    # Split paths into sections
    sections = split_paths_into_sections(paths)
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