"""Functions for neuron tree based reward calculations."""
from collections import deque
import numpy as np
import os
from pathlib import Path
import sys
import torch
from typing import Union, Tuple, Dict

script_path = Path(os.path.abspath(__file__))
parent_dir = script_path.parent.parent
sys.path.append(str(parent_dir))
from data_prep import load


def ensure_tensor(x, dtype=torch.float32, device=None, copy=False) -> torch.Tensor:
    """
    Ensure input is a torch.Tensor with given dtype/device.
    - Uses as_tensor for zero-copy when possible unless copy=True.
    - Preserves device when not provided; otherwise moves to device.
    """
    if torch.is_tensor(x):
        t = x.clone() if copy else x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device != torch.device(device):
            t = t.to(device)
        return t
    return torch.tensor(x, dtype=dtype, device=device) if copy else torch.as_tensor(x, dtype=dtype, device=device)


def get_section_nodes(swc_list, verbose=False):
    """
    Given an SWC list, return the sections as a dictionary of adjacency lists. Each adjacency list is a dict representing the undirected graph of nodes.
    Also return a directed graph of sections where sections are represented by their starting node id.
    Sections are defined as the nodes between and including branching points or terminal points.

    Parameters
    ----------
    swc_list : list or np.ndarray
        List or array of SWC entries. Each entry should be a list or array of the form [id, type, x, y, z, radius, parent].
    verbose : bool, optional
    
    Returns
    -------
    sections : dict
        Dictionary where keys are the starting node of each section and values are lists of nodes in that section.
    sections_graph : dict
         The directed adjacency list of neuron sections.
    """

    swc_list = np.array(swc_list)
    # Compute undirected edge list
    edge_list = load.undirected_edge_list(swc_list)
    # Make list of branching nodes
    branchings = [i for i in edge_list.keys() if len(edge_list[i]) > 2]
    sections = {}
    section_ends = {}
    while len(edge_list) > 1:
        terminals_ = None
        if 'terminals' in locals():
            terminals_ = terminals
        # Make list of terminal nodes
        terminals = [i for i in edge_list.keys() if len(edge_list[i]) == 1]
        # check if the list of terminals has changed
        if terminals == terminals_:
            if verbose:
                print(f"Warning: The neuron tree has disconnected sections.")
            break
        # from each terminal node walk along the tree until you reach a branching node
        # or another terminal node
        for terminal in terminals:
            if edge_list[terminal] == []: # if terminal node is the only node left
                break
            section = {}
            node = terminal
            while True:
                next_node = edge_list[node][0]
                section[node] = next_node
                edge_list.pop(node)
                edge_list[next_node].remove(node)
                node = next_node

                if node in branchings or node in terminals:
                    # complete the section                
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


def _graph_distance(start_point: np.ndarray, end_point: np.ndarray, swc_list: Union[np.ndarray, list], edge_list: Dict[int, list], id_to_idx: Dict[int, int]) -> float:
    """
    Compute the graph distance, the distance traversed along the neuron tree between two points.
    
    Note: the order of point coordinates (x-y-z or z-y-x) must match the order in the swc_list.
    
    Parameters:
    -----------
    start_point : np.ndarray
        The starting point (x, y, z).
    end_point : np.ndarray
        The ending point (x, y, z).
    swc_list : Union[np.ndarray, list]
        The SWC list representing the neuron.
    edge_list : dict
        Undirected edge list from load.undirected_edge_list()
        
    Returns:
    --------
    float
        The traversal distance between the two points along the neuron structure.
        Returns np.inf if no path is found.
    """
    swc_list = np.array(swc_list)
    # find nearest nodes to start and end points
    start_node = _get_nearest_node(start_point, swc_list, current_section_id=None)
    end_node = _get_nearest_node(end_point, swc_list, current_section_id=None)
    
    # find path between start and end nodes
    path = _find_path_between_nodes(start_node, end_node, edge_list)
    
    if not path:
        return np.inf  # no path found
    
    # compute distance along path
    distance = 0.0
    for i in range(len(path) - 1):
        node_a = swc_list[id_to_idx[path[i]], 2:5]
        node_b = swc_list[id_to_idx[path[i + 1]], 2:5]
        distance += np.linalg.norm(node_b - node_a)
    
    # add distance from start_point to start_node and end_point to end_node
    start_node_pos = swc_list[id_to_idx[start_node], 2:5]
    end_node_pos = swc_list[id_to_idx[end_node], 2:5]
    distance += np.linalg.norm(start_point - start_node_pos)
    distance += np.linalg.norm(end_point - end_node_pos)
    
    return distance


def _distance_points_to_segment(points: Union[torch.Tensor, np.ndarray],
                                           segment_start: Union[torch.Tensor, np.ndarray],
                                           segment_end: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distances from multiple points to a line segment.
    
    Parameters
    ----------
    points: array-like
        (N,3) or (3,)
    segment_start: array-like
        (3,)
    segment_end: array-like
        (3,)

    Returns
    -------
    (dists, closest_points):
      dists -> (N,) torch.float32
      closest_points -> (N,3) torch.float32
    """
    # Coerce to torch and common device
    device = None
    if torch.is_tensor(points):
        device = points.device
    elif torch.is_tensor(segment_start):
        device = segment_start.device
    elif torch.is_tensor(segment_end):
        device = segment_end.device

    points = ensure_tensor(points, dtype=torch.float32, device=device)
    seg_start = ensure_tensor(segment_start, dtype=torch.float32, device=points.device)
    seg_end = ensure_tensor(segment_end, dtype=torch.float32, device=points.device)

    points = points.view(-1, 3)
    seg_vec = seg_end - seg_start
    point_vecs = points - seg_start.unsqueeze(0)

    seg_len_sq = torch.dot(seg_vec, seg_vec)
    if seg_len_sq == 0.0:
        t = torch.tensor([1.0])
    else:
        t = (point_vecs @ seg_vec) / seg_len_sq
        t = torch.clamp(t, 0.0, 1.0)

    closest_points = seg_start.unsqueeze(0) + t.unsqueeze(1) * seg_vec.unsqueeze(0)
    dists = torch.linalg.norm(points - closest_points, dim=1)

    return dists, closest_points


def _area_spanned_single_edge(edge1_start: np.ndarray, edge1_end: np.ndarray, edge2_start: np.ndarray, edge2_end: np.ndarray) -> float:
    """
    Compute the area spanned between a single edge of the neuron and a single edge of the path. Ideally edge1 is the shorter edge
    since it will be discretized. The area is computed by integrating the distance from points along edge1 to edge2.

    Note: the order of point coordinates (x-y-z or z-y-x) must match for every point.
    
    Parameters:
    -----------
    edge1_start : np.ndarray
        The first point of the first edge.
    edge1_end : np.ndarray
        The second point of the first edge.
    edge2_start : np.ndarray
        The first point of the second edge.
    edge2_end : np.ndarray
        The second point of the second edge.
        
    Returns:
    --------
    float
        The area spanned between the two edges.
    """

    # discretize the path segment
    dt = 1
    edge1_vec = edge1_end - edge1_start
    edge1_length = np.linalg.norm(edge1_vec)
    if edge1_length < 1e-6:
        return 0.0
    edge1_dir = edge1_vec / edge1_length
    num_steps = max(int(edge1_length / dt), 1)
    edge1_points = np.array([edge1_start + i * dt * edge1_dir for i in range(num_steps + 1)])
    # compute distances from edge1 points to edge2 segment
    dists, _closest_points = _distance_points_to_segment(edge1_points, edge2_start, edge2_end)
    # integrate distances along edge1 segment to get area
    # dists is torch tensor; convert to numpy for trapz
    area = np.trapz(dists.cpu().numpy(), dx=dt)
    return area


def _area_spanned_path(edge: np.ndarray, neuron_path: np.ndarray, visited: Dict = None) -> float:
    """
    """
    # for each segment in the neuron path compute the area spanned with the edge
    area = 0.0
    for i in range(len(neuron_path) - 1):
        area += _area_spanned_single_edge(neuron_path[i], neuron_path[i + 1], edge[0], edge[1])

    return area


def _get_section_id(node: int, sections: Dict[int, Dict[int, int]]) -> Union[int, None]:
    """
    Get the section id of the given node.
    
    Parameters:
    -----------
    node : int
        The node ID.
    sections : dict
        Sections dictionary from get_section_nodes()
        
    Returns:
    --------
    int or None
        The section ID if the node is found, otherwise None.
    """
    for section_id, section_nodes in sections.items():
        if node in section_nodes:
            return section_id
    return None


def _get_nearest_node(point: Union[torch.Tensor, np.ndarray], swc_list: Union[torch.Tensor, np.ndarray], current_section_id: int = None) -> int:
    """
    Get the node nearest to the given point that is in the same section or an adjacent section.
    If current_section_id is None, return the nearest node in the entire SWC.

    Parameters
    ----------
    point : (3,) torch.Tensor or array
        The point to find the nearest node to.
    current_section_id : int or None
        The id of the current section. If None, search the entire SWC.
    swc_list : torch.Tensor or array
        The SWC list representing the neuron.

    Returns
    -------
    int
        The id of the nearest node.

    """
    # Work in torch for consistency
    swc_t = ensure_tensor(swc_list, dtype=torch.float32)
    if not swc_t.any():
        return None
    pt = ensure_tensor(point, dtype=torch.float32, device=swc_t.device)

    if current_section_id is None:
        valid_nodes = swc_t[:, 0].to(torch.int64)
        node_coords = swc_t[:, 2:5]
        nearest_idx = torch.argmin(torch.linalg.norm(node_coords - pt.unsqueeze(0), dim=1))
    else:
        # Fallback to previous numpy implementation for sections if needed
        section_nodes, sections_graph = get_section_nodes(swc_t.cpu().numpy(), verbose=True)
        valid_nodes = torch.tensor(list(section_nodes[current_section_id].keys()), dtype=torch.int64, device=swc_t.device)
        for adjacent_section in sections_graph[current_section_id]:
            more = torch.tensor(list(section_nodes[adjacent_section].keys()), dtype=torch.int64, device=swc_t.device)
            valid_nodes = torch.cat([valid_nodes, more])
        # Coordinates of valid nodes
        # Build mask and gather
        mask = (swc_t[:, 0].to(torch.int64).unsqueeze(1) == valid_nodes.unsqueeze(0)).any(dim=1)
        node_coords = swc_t[mask][:, 2:5]
        # Compute distances
        dists = torch.linalg.norm(node_coords - pt.unsqueeze(0), dim=1)
        # Map back to node ids
        nearest_idx = torch.argmin(dists)

    nearest_node = valid_nodes[nearest_idx].item()
    return nearest_node


def _get_nearest_point(point: Union[torch.Tensor, np.ndarray], swc_list: Union[torch.Tensor, np.ndarray], id_to_idx: Dict[int, int], current_section_id: int = None, edge_list: Dict[int, list] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Get the nearest point on the neuron structure to the given point that is in the same section or an adjacent section.
    If current_section_id is None, return the nearest point in the entire SWC.

    Parameters
    ----------
    point: (3,) torch.Tensor or array
        The point to find the nearest point on the neuron to.
    current_section_id : int or None
        The id of the current section. If None, search the entire SWC.
    swc_list : torch.Tensor or array
        The SWC list representing the neuron.
    edge_list : dict, optional
        Undirected edge list from load.undirected_edge_list(). If None, will be computed.
    id_to_idx : dict
        Mapping from node IDs to their indices in the SWC list.

    Returns
    -------
    torch.Tensor
        (3,) nearest point on the neuron structure.
    tuple
        edge (node1_id, node2_id) defining the segment where the nearest point lies.
    """

    swc_t = ensure_tensor(swc_list, dtype=torch.float32)
    if not swc_t.any():
        return None, (None, None)
    pt = ensure_tensor(point, dtype=torch.float32, device=swc_t.device)

    if edge_list is None:
        edge_list = load.undirected_edge_list(swc_t)
    nearest_node = _get_nearest_node(pt, swc_t, current_section_id=current_section_id)

    # Positions
    nearest_node_pos = swc_t[id_to_idx[nearest_node], 2:5]
    neighbors = edge_list[nearest_node]
    neighbor_idx = torch.tensor([id_to_idx[n] for n in neighbors])
    neighbor_positions = swc_t[neighbor_idx, 2:5]

    distances = []
    closest_points = []
    for neighbor_pos in neighbor_positions:
        dist, closest_point = _distance_points_to_segment(pt, nearest_node_pos, neighbor_pos)
        distances.append(dist[0])
        closest_points.append(closest_point[0])

    # Select best
    d_stack = torch.stack(distances) if isinstance(distances[0], torch.Tensor) else torch.tensor(distances, dtype=torch.float32, device=swc_t.device)
    min_idx = torch.argmin(d_stack).item()
    return closest_points[min_idx], (nearest_node, neighbors[min_idx])


def _get_termination_nodes(current_node: int, edge_list: Dict[int, list], visited=None, include_start=False) -> list:
    """
    Recursively find all termination nodes from a section.
    
    Parameters:
    -----------
    current_node : int
        The current node ID.
    edge_list : dict
        Undirected edge list from load.undirected_edge_list()
    sections : dict
        Sections dictionary from get_section_nodes()
    visited : set, optional
        Set of visited nodes to avoid cycles. Defaults to None.
    include_start : bool, optional
        Whether to include the start node if it is a termination node. Defaults to False.
        
    Returns:
    --------
    list
        List of termination nodes
    """
    termination_nodes = []
    if visited is None:
        visited = set()
    visited.add(current_node)
    neighbors = edge_list[current_node]
    neighbors = [n for n in neighbors if n not in visited]
    if include_start and len(neighbors) == 1:
        termination_nodes.append(current_node)
        termination_nodes.extend(_get_termination_nodes(neighbors[0], edge_list, visited))
    elif not neighbors: # this is a termination node
        termination_nodes.append(current_node)
    else:
        for neighbor in neighbors:
            termination_nodes.extend(_get_termination_nodes(neighbor, edge_list, visited))
    
    return termination_nodes


def _compute_target_point(current_pos: Union[torch.Tensor, np.ndarray], swc_list: Union[torch.Tensor, list], step_size: float, id_to_idx: Dict[int, int], edge_list: Dict[int, list] = None) -> torch.Tensor:
    """
    Compute target points, usually one but potentially multiple possible target points, along the neuron structure at a specified distance from the nearest
    neuron point to the current position. Note: This function does not use the visited dictionary. It assumes the given swc_list has been pruned to remove
    visited edges using _get_swc_sans_visited(). Otherwise the previous target point may be in the middle of the neuron and the point "ahead" of it will be ambiguous.
    
    Parameters
    ----------
    current_pos : (3,) torch.Tensor
        The current position (x, y, z).
    swc_list : torch.Tensor or list
        The SWC list representing the neuron.
    step_size : float
        The target step distance.
    edge_list : dict, optional
        Undirected edge list from load.undirected_edge_list(). If None, will be computed.
        
    Returns
    -------
    torch.Tensor
        N x 3 tensor of target point coordinates (x, y, z). Returns empty (0,3) if none.
    """
    swc_t = ensure_tensor(swc_list, dtype=torch.float32)
    if swc_t.ndim == 0 or swc_t.shape[0] < 2:
        return torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)

    pt = ensure_tensor(current_pos, dtype=torch.float32, device=swc_t.device)
    if edge_list is None:
        edge_list = load.undirected_edge_list(swc_t)

    # identify the nearest point on the neuron to the current position
    nearest_point, nearest_edge = _get_nearest_point(pt, swc_t, edge_list=edge_list, id_to_idx=id_to_idx)

    # check if the nearest point is close to either end of the edge
    edge_start_pos = swc_t[id_to_idx[int(nearest_edge[0])], 2:5]
    edge_end_pos = swc_t[id_to_idx[int(nearest_edge[1])], 2:5]

    dist_to_start = torch.linalg.norm(nearest_point - edge_start_pos).item()
    dist_to_end = torch.linalg.norm(nearest_point - edge_end_pos).item()

    visited_nodes = set()
    queue = deque()  # (path_nodes: list[int], dist_to_end: float)
    if dist_to_start < 1.0:
        queue.append(([nearest_edge[1]], dist_to_end))
        visited_nodes.add(nearest_edge[0])
    elif dist_to_end < 1.0:
        queue.append(([nearest_edge[0]], dist_to_start))
        visited_nodes.add(nearest_edge[1])
    else:
        queue.append(([nearest_edge[0]], dist_to_start))
        queue.append(([nearest_edge[1]], dist_to_end))

    targets = []
    while queue:
        current_path, dist_to_edge_end = queue.popleft()
        if current_path[-1] in visited_nodes:
            continue
        visited_nodes.add(current_path[-1])
        current_node = current_path[-1]
        current_node_pos = swc_t[id_to_idx[int(current_node)], 2:5]
        if dist_to_edge_end >= step_size:
            if len(current_path) == 1:
                prev_pos = nearest_point
            else:
                prev_node = current_path[-2]
                prev_pos = swc_t[id_to_idx[int(prev_node)], 2:5]
            direction = (current_node_pos - prev_pos)
            seg_len = torch.linalg.norm(direction).item()
            if seg_len < 1e-8:
                continue
            direction = direction / seg_len
            # step remaining distance along the segment
            tgt = prev_pos + direction * (step_size - (dist_to_edge_end - seg_len))
            targets.append(tgt)
        else:
            neighbors = [n for n in edge_list[current_node] if n not in visited_nodes and n != -1]
            if not neighbors:
                targets.append(current_node_pos)
            for neighbor in neighbors:
                nb_pos = swc_t[id_to_idx[int(neighbor)], 2:5]
                step = torch.linalg.norm(nb_pos - current_node_pos).item()
                queue.append((current_path + [neighbor], dist_to_edge_end + step))

    if len(targets) == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
    return torch.stack(targets)


def _distance_reward(current_position: Union[torch.Tensor, np.ndarray], target_position: Union[torch.Tensor, np.ndarray], max_distance: float = None) -> float:
    """
    Compute a reward based on the distance between the current position and the target position.

    Parameters
    ----------
    current_position : (3,) torch.Tensor or array
        The current position (x, y, z).
    target_position : (N,3) torch.Tensor or array
        The target positions (x, y, z).
    max_distance : float, optional
        If provided, clamp the distance to at most this value before squaring.

    Returns
    -------
    float
        The computed reward (negative squared distance).
    """
    tp = ensure_tensor(target_position, dtype=torch.float32)
    cp = ensure_tensor(current_position, dtype=torch.float32, device=tp.device)

    if tp.numel() == 0:
        # Assume a distance of patch radius away
        return torch.tensor([-289.0], dtype=torch.float32)  # -17^2
    tp = tp.view(-1, 3)
    dist_sq = (tp - cp.unsqueeze(0) ** 2).sum(dim=1).min()
    if max_distance is not None:
        dist_sq = torch.minimum(dist_sq, torch.tensor(max_distance ** 2, dtype=torch.float32, device=dist_sq.device))
    reward = -dist_sq
    return reward.unsqueeze(0)


def _find_path_between_nodes(start_node: int, end_node: int, edge_list: Dict[int, list]) -> list:
    """
    Find a path between two nodes in the edge list using BFS.
    
    Parameters:
    -----------
    start_node : int
        The starting node ID.
    end_node : int
        The ending node ID.
    edge_list : dict
        Undirected edge list from load.undirected_edge_list()
        
    Returns:
    --------
    list
        List of node IDs representing the path from start_node to end_node (inclusive).
        Returns an empty list if no path is found.
    """
    visited = set()
    queue = deque([(start_node, [start_node])])

    while queue:
        current_node, path = queue.popleft()
        if current_node == end_node:
            return path
        visited.add(current_node)
        for neighbor in edge_list.get(current_node, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return []


def _find_path_between_points(start_point: np.ndarray, end_point: np.ndarray, swc_list: Union[np.ndarray, list], id_to_idx: Dict[int, int]) -> np.ndarray:
    """
    Find a path between two points along the neuron structure.
    
    Parameters:
    -----------
    start_point : np.ndarray
        The starting point (x, y, z).
    end_point : np.ndarray
        The ending point (x, y, z).
    swc_list : Union[np.ndarray, list]
        The SWC list representing the neuron.
        
    Returns:
    --------
    np.ndarray
        Array of shape (N, 3) representing the path points from start_point to end_point (inclusive).
        Returns an empty array if no path is found.
    """
    swc_list = np.array(swc_list)
    edge_list = load.undirected_edge_list(swc_list)
    
    # find nearest nodes to start and end points
    start_node = _get_nearest_node(start_point, swc_list, current_section_id=None)
    end_node = _get_nearest_node(end_point, swc_list, current_section_id=None)
    
    # find path between start and end nodes
    node_path = _find_path_between_nodes(start_node, end_node, edge_list)
    
    if not node_path:
        return np.array([])  # no path found
    
    # construct full path including start and end points
    path_points = [start_point]
    for node in node_path:
        point = swc_list[id_to_idx[node], 2:5]
        path_points.append(point)
    path_points.append(end_point)
    
    return np.array(path_points)


def _init_visited(swc_list) -> Dict[tuple[int, int], float]:
    """
    Given an SWC list, return a dictionary with edges as keys (node1, node2) and initial coverage values.
    Edges are unordered, so each pair is listed only once.

    Parameters
    ----------
    swc_list : list or np.ndarray
        List or array of SWC entries. Each entry should be a list or array of the form [id, type, x, y, z, radius, parent].
    Returns
    -------
    Dict[Tuple[int, int], float]
        Dictionary with edges as keys and initial coverage values as floats.
    """

    swc_list = np.array(swc_list)
    edge_list = load.undirected_edge_list(swc_list)
    visited = {}
    checked = set()
    for node, neighbors in edge_list.items():
        checked.add(node)
        neighbors = [n for n in neighbors if n not in checked]
        for neighbor in neighbors:
            edge = tuple((node, neighbor))
            visited[edge] = 0.0  # initial coverage value
    return visited


def _add_to_visited(start_point: Union[torch.Tensor, np.ndarray], end_point: Union[torch.Tensor, np.ndarray], swc_list: Union[torch.Tensor, list], visited: Dict[Tuple[int, int], float], id_to_idx: Dict[int, int], edge_list: Dict[int, list] = None) -> Dict[Tuple[int, int], float]:
    """
    Given a start and end point, update the visited dictionary with the length spanned by the path between each of the nodes in the SWC list.

    Don't assume start and end points are on the neuron tree.
    
    Parameters
    ----------
    start_point : torch.Tensor or array
        The starting point (x, y, z).
    end_point : torch.Tensor or array
        The ending point (x, y, z).
    swc_list : torch.Tensor or list
        The SWC list representing the neuron.
    visited : dict
        Dictionary with edges as keys and coverage values as floats.
    edge_list : dict, optional
        Undirected edge list from load.undirected_edge_list(). If None, will be computed.
        
    Returns
    -------
    dict
        Updated visited dictionary.
    """

    swc_t = ensure_tensor(swc_list, dtype=torch.float32)
    start_pt = ensure_tensor(start_point, dtype=torch.float32, device=swc_t.device)
    end_pt = ensure_tensor(end_point, dtype=torch.float32, device=swc_t.device)
    if swc_t.numel() == 0:
        return visited
    
    if edge_list is None:
        edge_list = load.undirected_edge_list(swc_t)

    neuron_start_point, start_edge = _get_nearest_point(start_pt, swc_t, edge_list=edge_list, id_to_idx=id_to_idx)
    neuron_end_point, end_edge = _get_nearest_point(end_pt, swc_t, edge_list=edge_list, id_to_idx=id_to_idx)

    # find nearest nodes to start and end points
    start_node = start_edge[0]
    end_node = end_edge[0]

    node_ids = swc_t[:, 0].to(torch.int64)

    # Check if start and end points are on the same edge.
    if start_edge == end_edge or start_edge == (end_edge[1], end_edge[0]):
        edge = start_edge
        start_node_pos = swc_t[id_to_idx[int(edge[0])], 2:5]
        end_node_pos = swc_t[id_to_idx[int(edge[1])], 2:5]
        edge_vec = end_node_pos - start_node_pos
        edge_length = torch.linalg.norm(edge_vec).item()
        if edge_length > 0.0:
            # Determine direction: which node is closer to the points
            # Start from node nearest to any path point and go from that node to the farthest path point from it.
            dists_sq = torch.tensor([
                torch.sum((start_node_pos - neuron_start_point) ** 2),
                torch.sum((start_node_pos - neuron_end_point) ** 2),
                torch.sum((end_node_pos - neuron_start_point) ** 2),
                torch.sum((end_node_pos - neuron_end_point) ** 2)
            ], dtype=torch.float32)

            nearest_node_index = int(torch.argmin(dists_sq).item())
            if nearest_node_index in [2, 3]: # nearest to end_node
                # reverse edge order so coverage is positive from edge[1] toward edge[0].
                edge = (edge[1], edge[0])
                t = torch.sqrt(dists_sq[2:].max()).item() / edge_length
            else: # nearest to start_node
                t = torch.sqrt(dists_sq[:2].max()).item() / edge_length

        else:
            t = 1.0

        if visited.get(edge) is None:
            edge = (edge[1], edge[0])
            t = -t
        visited[edge] = float(t)

    else:
        # find path between start and end nodes
        node_path = _find_path_between_nodes(start_node, end_node, edge_list)
        # If start_edge[1] is in the node_path, then remove start_edge[0]. Do the same for end edge.
        # This ensures node_path only includes nodes between the start point and end point.
        if start_edge[1] in node_path:
            node_path = node_path[node_path.index(start_edge[1]):]
        if end_edge[1] in node_path:
            node_path = node_path[:node_path.index(end_edge[1]) + 1]

        if not node_path:
            # The edges are disconnected.
            # Find the node at the end of the same section as the start point that is nearest to the start point.
            start_section_terminals = _get_termination_nodes(start_edge[0], edge_list, include_start=True)
            start_section_terminals_positions = torch.stack([
                swc_t[id_to_idx[int(node)], 2:5] for node in start_section_terminals
            ])
            dists_start_section = torch.linalg.norm(start_section_terminals_positions - start_pt.unsqueeze(0), dim=1)
            # Mark the start point to that node as visited.
            visited = _add_to_visited(start_pt, start_section_terminals_positions[int(torch.argmin(dists_start_section))], swc_t, visited, id_to_idx=id_to_idx, edge_list=edge_list)
            # Do the same for the end point.
            end_section_terminals = _get_termination_nodes(end_edge[0], edge_list, include_start=True)
            end_section_terminals_positions = torch.stack([
                swc_t[id_to_idx[int(node)], 2:5] for node in end_section_terminals
            ])
            dists_end_section = torch.linalg.norm(end_section_terminals_positions - end_pt.unsqueeze(0), dim=1)
            visited = _add_to_visited(end_section_terminals_positions[int(torch.argmin(dists_end_section))], end_pt, swc_t, visited, id_to_idx=id_to_idx, edge_list=edge_list)

        else: # path found
            # first deal with the edges from end points to end nodes
            # how far along the edge is the nearest point?
            for point, edge in zip([neuron_start_point, neuron_end_point], [start_edge, end_edge]):
                if abs(visited.get(edge, 0.0)) == 1.0 or abs(visited.get((edge[1], edge[0]), 0.0)) == 1.0:
                    continue  # edge already fully visited
                # One of the edge nodes should match one of the ends of node_path
                node = next(n for n in edge if n in node_path)
                node_pos = swc_t[id_to_idx[int(node)], 2:5]
                # vector from node to nearest point
                node_to_point = point - node_pos
                other_edge_node = edge[1] if edge[0] == node else edge[0]
                edge_vec = swc_t[id_to_idx[int(other_edge_node)], 2:5] - node_pos
                edge_length = torch.linalg.norm(edge_vec).item()
                if edge_length > 0.0:
                    t = torch.linalg.norm(node_to_point).item() / edge_length
                else:
                    t = 1.0
                # order the edge so the node inside the path comes second
                edge_ord = (node, other_edge_node)
                if visited.get(edge_ord) is None:
                    edge_ord = (edge_ord[1], edge_ord[0])
                    t = -t  # fraction visited; negative means opposite direction
                visited[edge_ord] = float(t)

            for i in range(len(node_path) - 1):
                edge = (node_path[i], node_path[i + 1])
                if visited.get(edge) is None:
                    edge = (node_path[i + 1], node_path[i])
                visited[edge] = 1.0
    
    return visited


def remove_visited(swc_list: Union[np.ndarray, list], visited: Dict[Tuple[int, int], float], edge_list: Dict[Tuple[int, int], float], id_to_idx: Dict[int, int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:

    swc_list = np.array(swc_list)
    if not swc_list.any():
        return swc_list, visited, edge_list
    
    # Keep all nodes that are not only part of fully visited edges
    nodes_to_keep = set()
    for edge, coverage in visited.copy().items():
        if abs(coverage) < 1e-3:
            nodes_to_keep.update(edge)
        # for partially visited edges, identify the node to keep
        elif 1e-3 < abs(coverage) < 0.999:
            # if coverage is positive keep the second node, otherwise keep the first node
            to_keep = int(coverage > 0)
            keep_node_id = edge[to_keep]
            other_node_id = edge[1 - to_keep]
            nodes_to_keep.add(keep_node_id)

            keep_idx = id_to_idx[keep_node_id]
            other_idx = id_to_idx[other_node_id]          
            # create a new node at the appropriate position along the edge
            node1_pos = np.array(swc_list[id_to_idx[edge[0]], 2:5])
            node2_pos = np.array(swc_list[id_to_idx[edge[1]], 2:5])
            # if coverage is positive, new node is at coverage fraction along the edge from node1 to node2
            # if coverage is negative, new node is at (1 + coverage) fraction along the edge from node1 to node2
            frac = coverage if coverage > 0 else (1 + coverage)
            new_node_pos = node1_pos + (node2_pos - node1_pos) * frac

            new_node_id = max(swc_list[:, 0]) + 1 # every new node gets a new unique id
            # Its parent is either the node to keep or -1 if the node to keep is its child.
            if swc_list[keep_idx, 6] == other_node_id:
                new_node_parent = -1
                swc_list[keep_idx, 6] = new_node_id 
            else:
                new_node_parent = keep_node_id
                if other_idx in nodes_to_keep:
                    swc_list[other_idx, 6] = -1  # other node becomes a root if it was kept
            new_node = np.array([new_node_id,
                                 swc_list[other_idx, 1], # type
                                 new_node_pos[0],
                                 new_node_pos[1],
                                 new_node_pos[2],
                                 swc_list[other_idx, 5], # radius
                                 new_node_parent])
            
            # replace the node in the swc_list
            swc_list = np.concatenate([swc_list, new_node.reshape(1, -1)])
            nodes_to_keep.add(new_node_id)
            # update the visited dictionary to reflect the new edge
            new_edge = (edge[to_keep], new_node_id)
            visited[new_edge] = 0.0  # new edge starts unvisited
            del visited[edge]
            # update the edge_list
            edge_list[keep_node_id].remove(other_node_id)
            edge_list[other_node_id].remove(keep_node_id)
            edge_list[keep_node_id].append(new_node_id)
            edge_list[new_node_id] = [keep_node_id]
            if not edge_list[other_node_id]:
                del edge_list[other_node_id]

    # remove all edges that are fully visited from the visited dictionary and edge_list
    for edge, coverage in list(visited.items()):
        if abs(coverage) > 0.999:
            del visited[edge]
            edge_list[edge[0]].remove(edge[1])
            edge_list[edge[1]].remove(edge[0])
            if not edge_list[edge[0]]:
                del edge_list[edge[0]]
            if not edge_list[edge[1]]:
                del edge_list[edge[1]]
    # remove nodes from swc_list
    new_swc_list = np.array([node for node in swc_list if node[0] in nodes_to_keep])

    return new_swc_list, visited, edge_list