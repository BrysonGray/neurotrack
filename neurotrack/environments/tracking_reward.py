"""Functions for neuron tree based reward calculations."""
from collections import deque
import numpy as np
import torch
from typing import List, Union, Tuple, Dict

from neurotrack.data import loading as load


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
    adj_dict = load.adjacency_dict(swc_list)
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
            section = {}
            node = terminal
            while True:
                next_node = adj_dict[node][0]
                section[node] = next_node
                adj_dict.pop(node)
                adj_dict[next_node].remove(node)
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


def _graph_distance(start_point: np.ndarray, end_point: np.ndarray, swc_list: Union[np.ndarray, list], adj_dict: Dict[int, list], id_to_idx: Dict[int, int]) -> float:
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
    adj_dict : dict
        Undirected edge list from load.adjacency_dict()
        
    Returns:
    --------
    float
        The traversal distance between the two points along the neuron structure.
        Returns np.inf if no path is found.
    """
    swc_list = np.array(swc_list)
    # find nearest nodes to start and end points
    start_node = _get_nearest_node(start_point, swc_list, id_to_idx=id_to_idx, valid_nodes=None)
    end_node = _get_nearest_node(end_point, swc_list, id_to_idx=id_to_idx, valid_nodes=None)
    
    # find path between start and end nodes
    path = _find_path_between_nodes(start_node, end_node, adj_dict)
    
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


def _get_connected_nodes(
    node: int,
    adj_dict: Dict[int, list],
    max_dist=None,
    swc_list=None,
    id_to_idx=None,
    neuron_root_ids=None,
) -> list:
    """
    Get all the nodes on the same tree as the given node.

    Parameters:
    -----------
    node : int
        The node ID.
    adj_dict : dict
        Undirected edge list from load.adjacency_dict()
    max_dist : float, optional
        Maximum distance to traverse in coordinate space. If None, traverse until end point is reached. Requires swc_list and id_to_idx if provided. Defaults to None.
    swc_list : np.ndarray, optional
        The SWC list representing the neuron. Required if max_dist is provided. Defaults to None.
    id_to_idx : dict, optional
        Mapping from node ID to index in swc_list. Required if max_dist is provided. Defaults to None.
    neuron_root_ids : set, optional
        Cached neuron root node IDs (where parent is -1). If None, roots are derived from swc_list.

    Returns:
    --------
    connected_nodes : list
        List of connected node IDs.
    terminals : list
        List of terminal node IDs found during traversal.
    """
    if max_dist is not None and (swc_list is None or id_to_idx is None):
        raise ValueError("swc_list and id_to_idx must be provided if max_dist is specified.")
    if max_dist is not None:
        swc_array = np.array(swc_list)
    visited = set()
    stack = [(node, 0.0)]  # (node_id, cumulative_distance_from_start)
    terminals = []
    if neuron_root_ids is None:
        root_nodes = {int(row[0]) for row in swc_list if row[6] == -1}
    else:
        root_nodes = {int(n) for n in neuron_root_ids}
    while stack:
        n, dist = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        if len(adj_dict.get(n, [])) == 1 and n != node and n not in root_nodes:  # terminal node (but not the starting node or a root node)
            terminals.append(n)
        
        # Check if we've exceeded max_dist. If so, don't explore neighbors
        if max_dist is not None and dist > max_dist:
            continue
            
        for nb in adj_dict.get(n, []):
            if nb not in visited:
                if max_dist is not None:
                    edge_dist = sum((swc_array[id_to_idx[nb], 2:5] - swc_array[id_to_idx[n], 2:5]) ** 2) ** 0.5
                    new_dist = dist + edge_dist
                    stack.append((nb, new_dist))
                else:
                    stack.append((nb, 0.0))
            
    connected_nodes = list(visited)

    return connected_nodes, terminals


def _get_nearest_node(point: Union[torch.Tensor, np.ndarray], swc_list: Union[torch.Tensor, np.ndarray], id_to_idx: Dict[int, int], valid_nodes: set = None) -> int:
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

    if valid_nodes is None:
        node_coords = swc_t[:, 2:5]
        dists2 = torch.sum((node_coords - pt.unsqueeze(0)) ** 2, dim=1)
        nearest_idx = torch.argmin(dists2)
        nearest_node = swc_t[nearest_idx, 0].item()
    else:
        # Get nearest node from only valid nodes
        valid_indices = [id_to_idx[int(v)] for v in valid_nodes if int(v) in id_to_idx]  # O(M)
        if not valid_indices:
            return None
        swc_filtered = swc_t[valid_indices]
        node_coords = swc_filtered[:, 2:5]  # shape (M,3)
        # Compute distances
        dists2 = torch.sum((node_coords - pt.unsqueeze(0)) ** 2, dim=1)
        # Map back to node ids
        nearest_idx = torch.argmin(dists2)
        nearest_node = swc_filtered[nearest_idx, 0].item()

    return nearest_node


def _get_nearest_point(point: Union[torch.Tensor, np.ndarray], swc_list: Union[torch.Tensor, np.ndarray], id_to_idx: Dict[int, int], valid_nodes: set = None, adj_dict: Dict[int, list] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
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
    adj_dict : dict, optional
        Undirected edge list from load.adjacency_dict(). If None, will be computed.
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

    if adj_dict is None:
        adj_dict = load.adjacency_dict(swc_t)
    else:
        adj_dict = adj_dict
    nearest_node = _get_nearest_node(pt, swc_t, id_to_idx=id_to_idx, valid_nodes=valid_nodes)

    if nearest_node is None:
        return None, (None, None)

    nearest_node = int(nearest_node)
    neighbors = [int(n) for n in adj_dict.get(nearest_node, []) if int(n) in id_to_idx]

    # Positions
    nearest_node_pos = swc_t[id_to_idx[nearest_node], 2:5]
    neighbor_idx = torch.tensor(
        [id_to_idx[n] for n in neighbors],
        dtype=torch.long,
        device=swc_t.device,
    )
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


# def _get_termination_nodes(current_node: int, adj_dict: Dict[int, list], visited=None, include_start=False) -> list:
#     """
#     Recursively find all termination nodes from a section.
    
#     Parameters:
#     -----------
#     current_node : int
#         The current node ID.
#     adj_dict : dict
#         Undirected edge list from load.adjacency_dict()
#     sections : dict
#         Sections dictionary from get_section_nodes()
#     visited : set, optional
#         Set of visited nodes to avoid cycles. Defaults to None.
#     include_start : bool, optional
#         Whether to include the start node if it is a termination node. Defaults to False.
        
#     Returns:
#     --------
#     list
#         List of termination nodes
#     """
#     termination_nodes = []
#     if visited is None:
#         visited = set()
#     visited.add(current_node)
#     neighbors = adj_dict[current_node]
#     neighbors = [n for n in neighbors if n not in visited]
#     if include_start and len(neighbors) == 1:
#         termination_nodes.append(current_node)
#         termination_nodes.extend(_get_termination_nodes(neighbors[0], adj_dict, visited))
#     elif not neighbors: # this is a termination node
#         termination_nodes.append(current_node)
#     else:
#         for neighbor in neighbors:
#             termination_nodes.extend(_get_termination_nodes(neighbor, adj_dict, visited))
    
#     return termination_nodes

def _get_termination_nodes(current_node: int, adj_dict: Dict[int, list], visited=None, include_start: bool = False, max_depth: int = 1000000) -> list:
    """
    Find termination nodes from a section using an iterative DFS with a recursion depth limit.

    
    Parameters:
    -----------
    current_node : int
        The current node ID.
    adj_dict : dict
        Undirected edge list from load.adjacency_dict()
    visited : set, optional
        Set of visited nodes to avoid cycles. Defaults to None.
    include_start : bool, optional
        Whether to include the start node if it is a termination node. Defaults to False.
    max_depth : int, optional
        Maximum search depth to prevent unbounded recursion. Defaults to 1000000.

    Returns:
    --------
    list
        List of termination nodes.
    """
    if visited is None:
        visited = set()
    else:
        visited = set(visited)

    termination_nodes = []
    seen_terms = set()

    # stack entries: (node, depth)
    stack = [(current_node, 0)]

    while stack:
        node, depth = stack.pop()
        if node in visited:
            continue
        # mark visited for traversal
        visited.add(node)

        neighbors = [n for n in adj_dict.get(node, []) if n not in visited]

        # enforce depth limit: treat node as termination if limit reached
        if depth >= max_depth:
            if node not in seen_terms:
                seen_terms.add(node)
                termination_nodes.append(node)
            continue

        # handle include_start case similarly to original recursive behaviour
        if node == current_node and include_start and len(adj_dict.get(node, [])) == 1:
            if node not in seen_terms:
                seen_terms.add(node)
                termination_nodes.append(node)
            # still explore the single neighbor
            if adj_dict.get(node):
                nb = adj_dict[node][0]
                if nb not in visited:
                    stack.append((nb, depth + 1))
            continue

        # if no unvisited neighbors -> termination
        if not neighbors:
            if node not in seen_terms:
                seen_terms.add(node)
                termination_nodes.append(node)
            continue

        # otherwise push neighbors for further exploration
        for nb in neighbors:
            stack.append((nb, depth + 1))

    return termination_nodes

def _get_points_at_distance(
    current_pos: Union[torch.Tensor, np.ndarray],
    swc_list: Union[torch.Tensor, np.ndarray],
    step_size: float,
    id_to_idx: Dict[int, int],
    adj_dict: Dict[int, list] = None,
    valid_nodes: set = None) -> torch.Tensor:
    """
    Compute the points along the neuron structure at a specified euclidean distance away from the current position.

    Parameters
    ----------
    current_pos : (3,) torch.Tensor
        The current position (x, y, z).
    swc_list : torch.Tensor or list
        The SWC list representing the neuron.
    step_size : float
        The target step distance.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3) with absolute coordinates (x, y, z). Returns
        empty tensor of shape (0, 3) if no points are found.
    """
    swc_t = ensure_tensor(swc_list, dtype=torch.float32)
    empty_points = torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
    if swc_t.numel() == 0:
        return empty_points

    step_size = float(step_size)
    if step_size < 0:
        raise ValueError("step_size must be non-negative.")

    current_pos_t = ensure_tensor(current_pos, dtype=torch.float32, device=swc_t.device)
    nearest_node = _get_nearest_node(current_pos_t, swc_t, id_to_idx=id_to_idx, valid_nodes=valid_nodes)
    if nearest_node is None:
        return empty_points
    nearest_node = int(nearest_node)

    if adj_dict is None:
        adj_dict = load.adjacency_dict(swc_t)

    points = []
    sq_step_size = step_size ** 2
    boundary_pairs = []  # (node inside step size, node outside step size)
    queue = deque([(nearest_node, None)])  # (current_node, previous_node)
    visited = set()

    while queue:
        node, prev_node = queue.popleft()
        node = int(node)
        if node in visited:
            continue
        if node not in id_to_idx:
            continue
        visited.add(node)

        node_pos = swc_t[id_to_idx[node], 2:5]
        sq_dist = torch.sum((node_pos - current_pos_t) ** 2).item()
        if sq_dist >= sq_step_size:
            if prev_node is not None:
                boundary_pairs.append((int(prev_node), node))
            continue

        for neighbor in adj_dict.get(node, []):
            neighbor = int(neighbor)
            if neighbor not in visited:
                queue.append((neighbor, node))

    if not boundary_pairs:
        return empty_points

    # for each pair of nodes where one is inside the step size and the other is outside,
    # compute the point along the edge between them that is exactly step_size away from current_pos
    for node_in, node_out in boundary_pairs:
        if node_in not in id_to_idx or node_out not in id_to_idx:
            continue

        a = swc_t[id_to_idx[node_in], 2:5] - current_pos_t
        b = swc_t[id_to_idx[node_out], 2:5] - current_pos_t
        # solve step_size = ||a + t(b - a)||
        ab = b - a
        # quadratic formula coefficients for ||a + t*ab||^2 = step_size^2
        A = torch.dot(ab, ab)
        if float(A.item()) <= 1e-12:
            continue

        B = 2 * torch.dot(a, ab)
        C = torch.dot(a, a) - sq_step_size
        discriminant = B ** 2 - 4 * A * C
        discriminant_val = float(discriminant.item())
        if discriminant_val < -1e-8:
            continue  # no solution, should not happen if one node is inside and the other is outside

        discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_disc = torch.sqrt(discriminant)
        denom = 2 * A
        t1 = (-B + sqrt_disc) / denom
        t2 = (-B - sqrt_disc) / denom

        # Choose the solution that gives a point on the edge between the two nodes
        for t in (t1, t2):
            t_val = float(t.item())
            if 0.0 <= t_val <= 1.0:
                points.append(a + t * ab + current_pos_t)

    if not points:
        return empty_points

    points_t = torch.stack(points, dim=0)

    # De-duplicate intersections from tangent or numerically identical roots.
    keep_indices = []
    tol_sq = 1e-10
    for i in range(points_t.shape[0]):
        if not keep_indices:
            keep_indices.append(i)
            continue
        sq_diffs = torch.sum((points_t[keep_indices] - points_t[i]) ** 2, dim=1)
        if torch.all(sq_diffs > tol_sq):
            keep_indices.append(i)

    return points_t[keep_indices]


def _target_vectors_from_points(
    position: torch.Tensor,
    target_points: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert absolute target points to relative direction vectors."""
    position_t = ensure_tensor(position, dtype=torch.float32, device=device).view(3)
    if target_points is None:
        return torch.zeros((1, 3), dtype=torch.float32, device=device)
    target_points_t = ensure_tensor(target_points, dtype=torch.float32, device=device).view(-1, 3)
    if target_points_t.numel() == 0:
        return torch.zeros((1, 3), dtype=torch.float32, device=device)
    return target_points_t - position_t.unsqueeze(0)


def _compute_target_action(
    current_pos: Union[torch.Tensor, np.ndarray],
    swc_list: Union[torch.Tensor, list],
    step_size: float,
    id_to_idx: Dict[int, int],
    adj_dict: Dict[int, list] = None,
    terminal_points: torch.Tensor = None,
    valid_nodes: set = None,
    valid_dist2: float = 49.0,
    stop_distance: float = None,
) -> Tuple[torch.Tensor, bool]:
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
    adj_dict : dict, optional
        Undirected edge list from load.adjacency_dict(). If None, will be computed.
    id_to_idx : dict
        Mapping from node IDs to their indices in the SWC list.
    terminal_points : (N,3) torch.Tensor, optional
        Terminal points in the neuron structure.
    valid_nodes : set, optional
        Optional node IDs to restrict nearest-point and target search.
        
    Returns
    -------
    tuple
        (target_vectors, stop_target)
        target_vectors : (K, 3) tensor of relative action vectors.
        stop_target : bool indicating if current_pos is close enough to a terminal point.
    """
    swc_t = ensure_tensor(swc_list, dtype=torch.float32)
    pt = ensure_tensor(current_pos, dtype=torch.float32, device=swc_t.device)
    step_size = float(step_size)
    if step_size < 0:
        raise ValueError("step_size must be non-negative.")
    if stop_distance is None:
        stop_distance = step_size
    stop_distance = float(stop_distance)
    if stop_distance < 0:
        raise ValueError("stop_distance must be non-negative.")

    terminals_t = None
    terminals_exist = False
    nearest_valid_terminal = None
    if terminal_points is not None:
        terminals_t = ensure_tensor(terminal_points, dtype=torch.float32, device=swc_t.device).view(-1, 3)
        terminals_exist = terminals_t.numel() > 0

    stop_target = False
    if terminals_exist:
        sq_dists_to_terminals = torch.sum((terminals_t - pt.unsqueeze(0)) ** 2, dim=1)
        nearest_terminal = terminals_t[torch.argmin(sq_dists_to_terminals)]
        sq_dist_to_nearest_terminal = torch.min(sq_dists_to_terminals).item()
        nearest_valid_terminal = nearest_terminal if sq_dist_to_nearest_terminal <= valid_dist2 else None
        stop_target = sq_dist_to_nearest_terminal <= (stop_distance ** 2)

    # if the swc_list is empty or has only one point, return the nearest terminal if within step size, otherwise return empty
    if swc_t.ndim < 2 or swc_t.shape[0] < 2:
        if nearest_valid_terminal is not None:
            targets = nearest_valid_terminal.unsqueeze(0)
        else:
            targets = torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
            stop_target = True  # if there are no points in the neuron and no nearby terminals, signal to stop
    else:
        if adj_dict is None:
            adj_dict = load.adjacency_dict(swc_t)

        # Prefer a nearby terminal when it is reachable within one target step.
        if nearest_valid_terminal is not None and sq_dist_to_nearest_terminal <= step_size ** 2:
            target_points = nearest_valid_terminal.unsqueeze(0)
            target_vectors = _target_vectors_from_points(pt, target_points, device=swc_t.device)
            return target_vectors, bool(stop_target)

        nearest_point, _nearest_edge = _get_nearest_point(
            pt,
            swc_t,
            adj_dict=adj_dict,
            id_to_idx=id_to_idx,
            valid_nodes=valid_nodes,
        )

        if nearest_point is None:
            if nearest_valid_terminal is not None:
                target_points = nearest_valid_terminal.unsqueeze(0)
                target_vectors = _target_vectors_from_points(pt, target_points, device=swc_t.device)
                return target_vectors, bool(stop_target)
            target_points = torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
            target_vectors = _target_vectors_from_points(pt, target_points, device=swc_t.device)
            stop_target = True  # if we can't find a nearest point on the neuron and there are no nearby terminals, signal to stop
            return target_vectors, bool(stop_target)

        sq_dist_to_nearest_point = torch.sum((nearest_point - pt) ** 2).item()
        if sq_dist_to_nearest_point > step_size ** 2:
            if sq_dist_to_nearest_point <= valid_dist2:
                target_points = nearest_point.unsqueeze(0)
                target_vectors = _target_vectors_from_points(pt, target_points, device=swc_t.device)
                return target_vectors, bool(stop_target)
            else:
                if nearest_valid_terminal is not None:
                    target_points = nearest_valid_terminal.unsqueeze(0)
                    target_vectors = _target_vectors_from_points(pt, target_points, device=swc_t.device)
                    return target_vectors, bool(stop_target)
                else:
                    target_points = torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
                    target_vectors = _target_vectors_from_points(pt, target_points, device=swc_t.device)
                    stop_target = True  # if we're too far from the neuron and there are no nearby terminals, signal to stop
                    return target_vectors, bool(stop_target)
        
        targets = _get_points_at_distance(
            current_pos=pt,
            swc_list=swc_t,
            step_size=step_size,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            valid_nodes=valid_nodes
        )

        # If no points are found at step_size, use the farthest node away from the current position.
        # This should never happen since the farthest node should be a terminal if there are nodes within
        # step_size but no edges cross the step_size boundary.
        if targets.numel() == 0:
            if valid_nodes is None:
                node_coords = swc_t[:, 2:5]
            else:
                valid_indices = [id_to_idx[int(v)] for v in valid_nodes if int(v) in id_to_idx]
                node_coords = swc_t[valid_indices, 2:5] if valid_indices else torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)

            if node_coords.numel() > 0:
                dists2 = torch.sum((node_coords - pt.unsqueeze(0)) ** 2, dim=1)
                farthest_idx = torch.argmax(dists2)
                targets = node_coords[farthest_idx].unsqueeze(0)
            else:
                targets = nearest_point.unsqueeze(0)

    target_vectors = _target_vectors_from_points(pt, targets, device=swc_t.device)
    if targets.numel() == 0:
        stop_target = True  # if we couldn't find any target points, signal to stop
    return target_vectors, bool(stop_target)


# def _compute_target_point(
#     current_pos: Union[torch.Tensor, np.ndarray],
#     swc_list: Union[torch.Tensor, list],
#     step_size: float,
#     id_to_idx: Dict[int, int],
#     adj_dict: Dict[int, list] = None,
#     terminal_points: torch.Tensor = None,
#     valid_nodes: set = None,
# ) -> torch.Tensor:
#     """
#     Compute target points, usually one but potentially multiple possible target points, along the neuron structure at a specified distance from the nearest
#     neuron point to the current position. Note: This function does not use the visited dictionary. It assumes the given swc_list has been pruned to remove
#     visited edges using _get_swc_sans_visited(). Otherwise the previous target point may be in the middle of the neuron and the point "ahead" of it will be ambiguous.
    
#     Parameters
#     ----------
#     current_pos : (3,) torch.Tensor
#         The current position (x, y, z).
#     swc_list : torch.Tensor or list
#         The SWC list representing the neuron.
#     step_size : float
#         The target step distance.
#     adj_dict : dict, optional
#         Undirected edge list from load.adjacency_dict(). If None, will be computed.
#     id_to_idx : dict
#         Mapping from node IDs to their indices in the SWC list.
#     terminal_points : (N,3) torch.Tensor, optional
#         Terminal points in the neuron structure.
#     current_section_filter : bool, optional
#         Whether to apply filtering based on the current section. Defaults to True.
        
#     Returns
#     -------
#     torch.Tensor
#         N x 3 tensor of target point coordinates (x, y, z). Returns empty (0,3) if none.
#     """
#     swc_t = ensure_tensor(swc_list, dtype=torch.float32)
#     pt = ensure_tensor(current_pos, dtype=torch.float32, device=swc_t.device)
#     terminals_exist = False
#     if terminal_points is not None:
#         terminals_exist = terminal_points.numel() > 0
#     if terminals_exist:
#         sq_dists_to_terminals = torch.sum((terminal_points - pt.unsqueeze(0)) ** 2, dim=1)
#         nearest_terminal = terminal_points[torch.argmin(sq_dists_to_terminals)]
#         sq_dist_to_nearest_terminal = torch.min(sq_dists_to_terminals).item()
#     # if the swc_list is empty or has only one point, return the nearest terminal if available
#     if swc_t.ndim == 0 or swc_t.shape[0] < 2:
#         if terminals_exist:
#             targets = nearest_terminal.unsqueeze(0)
#         else:
#             targets = torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
#     # Otherwise identify the nearest point on the neuron to the current position.
#     else:
#         if adj_dict is None:
#             adj_dict = load.adjacency_dict(swc_t)

#         nearest_point, nearest_edge = _get_nearest_point(pt, swc_t, adj_dict=adj_dict, id_to_idx=id_to_idx, valid_nodes=valid_nodes)
#         # If the nearest terminal point is closer than the nearest point on the neuron, set the target to the nearest terminal point.
#         sq_dist_to_nearest_point = 0.0
#         if nearest_point is None:
#             if terminals_exist:
#                 targets = nearest_terminal.unsqueeze(0)
#             else:
#                 targets = torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
#         else:
#             sq_dist_to_nearest_point = torch.sum((nearest_point - pt) ** 2).item()
#             if "sq_dist_to_nearest_terminal" in locals() and sq_dist_to_nearest_terminal < sq_dist_to_nearest_point:
#                 targets = nearest_terminal.unsqueeze(0)
#             # Otherwise, step along the neuron structure from the nearest point to find the target point(s).
#             else:
#                 # check if the nearest point is close to either end of the edge
#                 edge_start_pos = swc_t[id_to_idx[int(nearest_edge[0])], 2:5]
#                 edge_end_pos = swc_t[id_to_idx[int(nearest_edge[1])], 2:5]

#                 dist_to_start = torch.linalg.norm(nearest_point - edge_start_pos).item()
#                 dist_to_end = torch.linalg.norm(nearest_point - edge_end_pos).item()

#                 visited_nodes = set()
#                 queue = deque()  # (path_nodes: list[int], dist_to_end: float)
#                 if dist_to_start < 1.0:
#                     queue.append(([nearest_edge[1]], dist_to_end))
#                     visited_nodes.add(nearest_edge[0])
#                 elif dist_to_end < 1.0:
#                     queue.append(([nearest_edge[0]], dist_to_start))
#                     visited_nodes.add(nearest_edge[1])
#                 else:
#                     queue.append(([nearest_edge[0]], dist_to_start))
#                     queue.append(([nearest_edge[1]], dist_to_end))

#                 targets = []
#                 while queue:
#                     current_path, dist_to_edge_end = queue.popleft()
#                     if current_path[-1] in visited_nodes:
#                         continue
#                     visited_nodes.add(current_path[-1])
#                     current_node = current_path[-1]
#                     current_node_pos = swc_t[id_to_idx[int(current_node)], 2:5]
#                     if dist_to_edge_end >= step_size:
#                         if len(current_path) == 1:
#                             prev_pos = nearest_point
#                         else:
#                             prev_node = current_path[-2]
#                             prev_pos = swc_t[id_to_idx[int(prev_node)], 2:5]
#                         direction = (current_node_pos - prev_pos)
#                         seg_len = torch.linalg.norm(direction).item()
#                         if seg_len < 1e-8:
#                             continue
#                         direction = direction / seg_len
#                         # step remaining distance along the segment
#                         tgt = prev_pos + direction * (step_size - (dist_to_edge_end - seg_len))
#                         targets.append(tgt)
#                     else:
#                         neighbors = [n for n in adj_dict[current_node] if n not in visited_nodes and n != -1]
#                         if not neighbors:
#                             targets.append(current_node_pos)
#                         for neighbor in neighbors:
#                             nb_pos = swc_t[id_to_idx[int(neighbor)], 2:5]
#                             step = torch.linalg.norm(nb_pos - current_node_pos).item()
#                             queue.append((current_path + [neighbor], dist_to_edge_end + step))

#                 if len(targets) == 0:
#                     targets = torch.empty((0, 3), dtype=torch.float32, device=swc_t.device)
#                 else:
#                     targets = torch.stack(targets)
#     return targets


# def _distance_reward(current_position: Union[torch.Tensor, np.ndarray], target_position: Union[torch.Tensor, np.ndarray], max_distance: float = None, terminated: bool = False, gamma: float = 0.99) -> float:
#     """
#     Compute a reward based on the distance between the current position and the target position.

#     Parameters
#     ----------
#     current_position : (3,) torch.Tensor or array
#         The current position (x, y, z).
#     target_position : (N,3) torch.Tensor or array
#         The target positions (x, y, z).
#     max_distance : float, optional
#         If provided, clamp the distance to at most this value before squaring.
#     terminated : bool, optional
#         Whether the episode has terminated. If True, scale the reward by 1 / (1 - gamma).
#     gamma : float, optional
#         The discount factor used if terminated is True.

#     Returns
#     -------
#     reward : float
#         The computed reward (negative squared distance).
#     target_vector : (1,3) torch.Tensor
#         The vector from current position to the closest target position, returns nans if no targets.
#     """

#     tp = ensure_tensor(target_position, dtype=torch.float32)
#     cp = ensure_tensor(current_position, dtype=torch.float32, device=tp.device)
#     if tp.numel() == 0:
#         # Assume a distance of patch radius away
#         return torch.tensor([-289.0], dtype=torch.float32), torch.full((1, 3), float('nan'), dtype=torch.float32)  # -17^2

#     tp = tp.view(-1, 3)
#     # Vectorized min squared distance over termination points for this path
#     error_vecs = tp - cp.unsqueeze(0)
#     dist_sq = (target_vecs * target_vecs).sum(dim=1)
#     min_idx = torch.argmin(dist_sq)
#     target_vector = target_vecs[min_idx]
#     target_vector = target_vector.unsqueeze(0)
#     d2_min = dist_sq[min_idx]
#     if max_distance is not None:
#         d2_min = torch.minimum(d2_min, torch.tensor(max_distance ** 2, dtype=torch.float32, device=d2_min.device))
#     reward = -d2_min.to(dtype=torch.float32).unsqueeze(0)
#     if terminated:
#         reward = reward * 1 / (1 - gamma)

#     return reward, target_vector


def _prepare_target_candidates(
    step: torch.Tensor,
    target_step: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize target candidates and masks for nearest-target reward selection."""
    step_t = torch.as_tensor(step, dtype=torch.float32)
    if step_t.ndim == 1:
        step_t = step_t.unsqueeze(0)
    step_t = step_t.view(-1, 3)

    target_t = torch.as_tensor(target_step, dtype=torch.float32, device=step_t.device)
    if target_t.ndim == 1:
        target_t = target_t.view(1, 1, 3)
    elif target_t.ndim == 2:
        if target_t.shape[-1] != 3:
            raise ValueError(f"target_step must have trailing dimension 3, got {tuple(target_t.shape)}")
        if step_t.shape[0] == 1:
            target_t = target_t.unsqueeze(0)
        elif target_t.shape[0] == step_t.shape[0]:
            target_t = target_t.unsqueeze(1)
        else:
            raise ValueError(
                "target_step with shape (K, 3) is only valid for a single step or one target per batch element. "
                f"Got step batch {step_t.shape[0]} and target shape {tuple(target_t.shape)}."
            )
    elif target_t.ndim == 3:
        if target_t.shape[0] != step_t.shape[0] or target_t.shape[2] != 3:
            raise ValueError(
                f"target_step must have shape (B, K, 3) to match steps. Got {tuple(target_t.shape)} for batch {step_t.shape[0]}."
            )
    else:
        raise ValueError(f"target_step must have 1, 2, or 3 dimensions, got {target_t.ndim}")

    if valid_mask is None:
        mask_t = torch.ones(target_t.shape[:2], dtype=torch.bool, device=step_t.device)
    else:
        mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=step_t.device)
        if mask_t.ndim == 1:
            if step_t.shape[0] == 1 and mask_t.shape[0] == target_t.shape[1]:
                mask_t = mask_t.unsqueeze(0)
            elif target_t.shape[1] == 1 and mask_t.shape[0] == step_t.shape[0]:
                mask_t = mask_t.view(step_t.shape[0], 1)
            else:
                raise ValueError(
                    f"valid_mask with shape {tuple(mask_t.shape)} is incompatible with target candidates {tuple(target_t.shape[:2])}."
                )
        elif mask_t.ndim == 2:
            if tuple(mask_t.shape) != tuple(target_t.shape[:2]):
                raise ValueError(
                    f"valid_mask shape {tuple(mask_t.shape)} must match target candidates {tuple(target_t.shape[:2])}."
                )
        else:
            raise ValueError(f"valid_mask must have 1 or 2 dimensions, got {mask_t.ndim}")

    return step_t, target_t, mask_t


def distance_reward(
    step: torch.Tensor,
    target_step: torch.Tensor,
    terminated: Union[torch.Tensor, bool] = None,
    gamma: float = 0.99,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute a reward based on the difference between the current step and the target step.

    Parameters
    ----------
    step : (3,) or (N,3) torch.Tensor
        The current step (x, y, z).
    target_step : (3,), (N,3), or (N,K,3) torch.Tensor
        The target step candidate vector(s) (x, y, z).
    terminated : torch.Tensor or bool, optional
        Whether the episode has terminated. If True, scale the reward by 1 / (1 - gamma).
        If the input is a tensor, it should be of shape (N,).
    gamma : float, optional
        The discount factor used if terminated is True.
    valid_mask : torch.Tensor, optional
        Boolean mask of valid target candidates. Shape should match the
        leading target dimensions `(N, K)`.
    Returns
    -------
    reward: (N,) torch.Tensor
        The computed reward (negative squared distance to the nearest valid candidate).
    """
    step_t, target_t, mask_t = _prepare_target_candidates(step, target_step, valid_mask=valid_mask)
    if terminated is None:
        terminated = torch.zeros((step_t.shape[0],), dtype=torch.bool, device=step_t.device)
    elif isinstance(terminated, bool):
        terminated = torch.full((step_t.shape[0],), terminated, dtype=torch.bool, device=step_t.device)

    sq_dist = torch.sum((target_t - step_t.unsqueeze(1)) ** 2, dim=-1)
    sq_dist = sq_dist.masked_fill(~mask_t, torch.inf)
    has_valid_target = mask_t.any(dim=1)
    min_sq_dist = sq_dist.min(dim=1).values
    min_sq_dist = torch.where(has_valid_target, min_sq_dist, torch.zeros_like(min_sq_dist))

    reward = -min_sq_dist.to(dtype=torch.float32)
    terminated = terminated.view(-1)
    reward[terminated] = reward[terminated] * 1 / (1 - gamma)

    return reward


def _find_path_between_nodes(start_node: int, end_node: int, adj_dict: Dict[int, list]) -> list:
    """
    Find a path between two nodes in the edge list using BFS.
    
    Parameters:
    -----------
    start_node : int
        The starting node ID.
    end_node : int
        The ending node ID.
    adj_dict : dict
        Undirected edge list from load.adjacency_dict()
        
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
        for neighbor in adj_dict.get(current_node, []):
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
        The SWC list representing the neuron.n        
    Returns:
    --------
    np.ndarray
        Array of shape (N, 3) representing the path points from start_point to end_point (inclusive).
        Returns an empty array if no path is found.
    """
    swc_list = np.array(swc_list)
    adj_dict = load.adjacency_dict(swc_list)
    
    # find nearest nodes to start and end points
    start_node = _get_nearest_node(start_point, swc_list, id_to_idx=id_to_idx, valid_nodes=None)
    end_node = _get_nearest_node(end_point, swc_list, id_to_idx=id_to_idx, valid_nodes=None)
    
    # find path between start and end nodes
    node_path = _find_path_between_nodes(start_node, end_node, adj_dict)
    
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
    adj_dict = load.adjacency_dict(swc_list)
    visited = {}
    checked = set()
    for node, neighbors in adj_dict.items():
        checked.add(node)
        neighbors = [n for n in neighbors if n not in checked]
        for neighbor in neighbors:
            edge = tuple((node, neighbor))
            visited[edge] = 0.0  # initial coverage value
    return visited


def _add_to_visited(start_point: Union[torch.Tensor, np.ndarray],
                    end_point: Union[torch.Tensor, np.ndarray],
                    swc_list: Union[torch.Tensor, list],
                    visited: Dict[Tuple[int, int], float],
                    id_to_idx: Dict[int, int],
                    adj_dict: Dict[int, list] = None,
                    valid_nodes=None,
                    valid_dist2: float = 49.0) -> Dict[Tuple[int, int], float]:
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
    adj_dict : dict, optional
        Undirected edge list from load.adjacency_dict(). If None, will be computed.
    id_to_idx : dict
        Mapping from node IDs to their indices in the SWC list.
    valid_nodes : set, optional
        Optional node IDs to restrict nearest-point and path search.
    valid_dist2 : float, optional
        Maximum squared distance from the end point to the neuron for coverage update to occur. Defaults to 7^2 = 49.0.
        
    Returns
    -------
    visited : dict
        Updated visited dictionary.
    neuron_end_point : torch.Tensor or None
        The nearest point on the neuron to the end point.
    """

    swc_t = ensure_tensor(swc_list, dtype=torch.float32)
    start_pt = ensure_tensor(start_point, dtype=torch.float32, device=swc_t.device)
    end_pt = ensure_tensor(end_point, dtype=torch.float32, device=swc_t.device)
    if swc_t.numel() == 0:
        return visited, None
    
    if adj_dict is None:
        adj_dict = load.adjacency_dict(swc_t)
    else:
        adj_dict = adj_dict

    neuron_start_point, start_edge = _get_nearest_point(start_pt, swc_t, adj_dict=adj_dict, id_to_idx=id_to_idx, valid_nodes=valid_nodes)
    neuron_end_point, end_edge = _get_nearest_point(end_pt, swc_t, adj_dict=adj_dict, id_to_idx=id_to_idx, valid_nodes=valid_nodes)

    dist_to_neuron_end = torch.sum((neuron_end_point - end_pt) ** 2).item() if neuron_end_point is not None else float('inf')
    if neuron_start_point is None or neuron_end_point is None or dist_to_neuron_end > valid_dist2:
        return visited, neuron_end_point

    # Edge-based coverage update requires both projected points to lie on valid segments.
    if start_edge[0] is None or start_edge[1] is None or end_edge[0] is None or end_edge[1] is None:
        return visited, neuron_end_point

    # find nearest nodes to start and end points
    start_node = start_edge[0]
    end_node = end_edge[0]

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
        node_path = _find_path_between_nodes(start_node, end_node, adj_dict)
        # If start_edge[1] is in the node_path, then remove start_edge[0]. Do the same for end edge.
        # This ensures node_path only includes nodes between the start point and end point.
        if start_edge[1] in node_path:
            node_path = node_path[node_path.index(start_edge[1]):]
        if end_edge[1] in node_path:
            node_path = node_path[:node_path.index(end_edge[1]) + 1]

        if not node_path:
            # The edges are disconnected.
            # # Find the node at the end of the same section as the start point that is nearest to the start point.
            # start_section_terminals = _get_termination_nodes(start_edge[0], adj_dict, include_start=True)
            # start_section_terminals_positions = torch.stack([
            #     swc_t[id_to_idx[int(node)], 2:5] for node in start_section_terminals
            # ])
            # dists_start_section = torch.linalg.norm(start_section_terminals_positions - start_pt.unsqueeze(0), dim=1)
            # # Mark the start point to that node as visited.
            # visited, _ = _add_to_visited(start_pt, start_section_terminals_positions[int(torch.argmin(dists_start_section))], swc_t, visited, id_to_idx=id_to_idx, adj_dict=adj_dict, valid_nodes=None)
            # # Do the same for the end point.
            # end_section_terminals = _get_termination_nodes(end_edge[0], adj_dict, include_start=True)
            # end_section_terminals_positions = torch.stack([
            #     swc_t[id_to_idx[int(node)], 2:5] for node in end_section_terminals
            # ])
            # dists_end_section = torch.linalg.norm(end_section_terminals_positions - end_pt.unsqueeze(0), dim=1)
            # visited, _ = _add_to_visited(end_section_terminals_positions[int(torch.argmin(dists_end_section))], end_pt, swc_t, visited, id_to_idx=id_to_idx, adj_dict=adj_dict, valid_nodes=None)

            pass  # no path found, do nothing

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
    
    return visited, neuron_end_point


def remove_visited(swc_list: Union[np.ndarray, torch.Tensor],
                   visited: Dict[Tuple[int, int], float],
                   adj_dict: Dict[Tuple[int, int], float],
                   id_to_idx: Dict[int, int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], Dict[int, list], list]:

    swc_list = ensure_tensor(swc_list, dtype=torch.float32)
    if not swc_list.any():
        return swc_list, visited, adj_dict, []
    
    # Keep all nodes that are not only part of fully visited edges
    nodes_to_keep = set()
    cut_ends = set()
    # Compute max node ID once; incremented per new node to avoid O(n) scan per partial edge.
    next_node_id = int(swc_list[:, 0].max().item()) + 1
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
            node1_pos = swc_list[id_to_idx[edge[0]], 2:5]
            node2_pos = swc_list[id_to_idx[edge[1]], 2:5]
            # if coverage is positive, new node is at coverage fraction along the edge from node1 to node2
            # if coverage is negative, new node is at (1 + coverage) fraction along the edge from node1 to node2
            frac = coverage if coverage > 0 else (1 + coverage)
            new_node_pos = node1_pos + (node2_pos - node1_pos) * frac

            new_node_id = next_node_id
            next_node_id += 1
            # Its parent is either the node to keep or -1 if the node to keep is its child.
            if swc_list[keep_idx, 6] == other_node_id:
                new_node_parent = -1
                swc_list[keep_idx, 6] = new_node_id 
            else:
                new_node_parent = keep_node_id
                if other_idx in nodes_to_keep:
                    swc_list[other_idx, 6] = -1  # other node becomes a root if it was kept
            new_node = torch.tensor([new_node_id,
                                 swc_list[other_idx, 1], # type
                                 new_node_pos[0],
                                 new_node_pos[1],
                                 new_node_pos[2],
                                 swc_list[other_idx, 5], # radius
                                 new_node_parent])
            
            # replace the node in the swc_list
            swc_list = torch.cat([swc_list, new_node.reshape(1, -1)])
            nodes_to_keep.add(new_node_id)
            
            # Only add the new node (it always has degree 1)
            cut_ends.add(new_node_id)

            # update the visited dictionary to reflect the new edge
            new_edge = (edge[to_keep], new_node_id)
            visited[new_edge] = 0.0  # new edge starts unvisited
            del visited[edge]
            # update the adj_dict
            adj_dict[keep_node_id].remove(other_node_id)
            adj_dict[other_node_id].remove(keep_node_id)
            adj_dict[keep_node_id].append(new_node_id)
            adj_dict[new_node_id] = [keep_node_id]
            if not adj_dict[other_node_id]:
                del adj_dict[other_node_id]

    # remove all edges that are fully visited from the visited dictionary and adj_dict
    for edge, coverage in list(visited.items()):
        if abs(coverage) > 0.999:
            del visited[edge]
            # Check if nodes still exist in adj_dict before accessing
            if edge[0] in adj_dict:
                adj_dict[edge[0]].remove(edge[1])
                if not adj_dict[edge[0]]:
                    del adj_dict[edge[0]]
                else:
                    cut_ends.add(edge[0])
            if edge[1] in adj_dict:
                adj_dict[edge[1]].remove(edge[0])
                if not adj_dict[edge[1]]:
                    del adj_dict[edge[1]]
                else:
                    cut_ends.add(edge[1])
    # Keep only nodes that are still part of at least one remaining edge.
    # This keeps swc_list and adj_dict synchronized and avoids isolated nodes.
    active_nodes = set()
    for edge in visited.keys():
        active_nodes.update([int(edge[0]), int(edge[1])])

    if not active_nodes:
        return (
            torch.empty((0, swc_list.shape[1]), dtype=swc_list.dtype, device=swc_list.device),
            {},
            {},
            [],
        )

    # prune adjacency to active nodes only
    pruned_adj = {}
    for node, neighbors in adj_dict.items():
        node_i = int(node)
        if node_i not in active_nodes:
            continue
        pruned_neighbors = [int(nb) for nb in neighbors if int(nb) in active_nodes]
        if pruned_neighbors:
            pruned_adj[node_i] = pruned_neighbors
    adj_dict = pruned_adj

    active_nodes = set(adj_dict.keys())
    visited = {
        (int(edge[0]), int(edge[1])): float(coverage)
        for edge, coverage in visited.items()
        if int(edge[0]) in active_nodes and int(edge[1]) in active_nodes
    }
    cut_ends = {int(n) for n in cut_ends if int(n) in active_nodes}

    # remove nodes from swc_list
    new_swc_list = [node for node in swc_list if int(node[0]) in nodes_to_keep and int(node[0]) in active_nodes]
    if new_swc_list:
        new_swc_list = torch.stack(new_swc_list)
    else:
        new_swc_list = torch.empty((0, swc_list.shape[1]), dtype=swc_list.dtype, device=swc_list.device)
    # new_swc_list = torch.stack([swc_list[id_to_idx[n]] for n in nodes_to_keep])

    return new_swc_list, visited, adj_dict, list(cut_ends)


def update_visited_edges(
        prev_position,
        new_position,
        section_nodes,
        visited,
        unvisited_tree,
        id_to_idx,
        adj_dict,
        cut_ends,
        valid_dist2=49.0):
    """
    Updates the visited edges based on the path from the previous position to the new position. If section_nodes is not None, only
    consider paths that go through those nodes. Update cut_ends to reflect any new cut ends created by removing visited edges.
    Also updates the unvisited_tree, adj_dict, and id_to_idx to reflect any removed nodes.

    Parameters
    ----------
    prev_position : torch.Tensor
        The previous position (x, y, z).
    new_position : torch.Tensor
        The new position (x, y, z).
    section_nodes : list of int or None
        List of node IDs that are part of the current section, or None if all nodes are considered.
    unvisited_tree : torch.Tensor
        The current unvisited tree as an SWC tensor.
    id_to_idx : dict
        Mapping from node ID to index in the unvisited_tree tensor.
    adj_dict : dict
        Undirected edge list from load.adjacency_dict() for the current unvisited tree.
    cut_ends : list of int
        List of node IDs that are the current cut ends of the unvisited tree.
    valid_dist2 : float, optional
        The squared distance threshold for considering cut ends as part of the current section. Default is 49.0 (7 pixels).

    Returns
    -------
    visited : dict
        Updated dictionary of visited edges.
    unvisited_tree : torch.Tensor
        Updated unvisited tree with removed nodes.
    adj_dict : dict
        Updated adjacency dictionary with removed nodes.
    cut_ends : list of int
        Updated list of cut ends.
    id_to_idx : dict
        Updated mapping from node ID to index in the unvisited_tree tensor.
    """

    if section_nodes is None:
        return visited, unvisited_tree, adj_dict, cut_ends, id_to_idx

    # update visited edges
    if cut_ends:
        # use nearest cut end as starting point
        id_lookup = id_to_idx.get
        cut_ends_indices = [idx for v in cut_ends for idx in (id_lookup(int(v)),) if idx is not None]
        if cut_ends_indices:
            swc_filtered = unvisited_tree[cut_ends_indices]
            node_coords = swc_filtered[:, 2:5]  # shape (M,3)
            # Compute distances
            dists2 = torch.sum((node_coords - new_position.unsqueeze(0)) ** 2, dim=1)
            start_pos = node_coords[torch.argmin(dists2)]
        else:
            start_pos = prev_position
    else:
        start_pos = prev_position

    visited, neuron_end_point = _add_to_visited(
        start_pos,
        new_position,
        unvisited_tree,
        visited,
        adj_dict=adj_dict,
        id_to_idx=id_to_idx,
        valid_nodes=section_nodes,
        valid_dist2=valid_dist2,
    )
    if neuron_end_point is not None:
        unvisited_tree, visited, adj_dict, changed_nodes = remove_visited(unvisited_tree, visited, adj_dict, id_to_idx=id_to_idx)
    
        # Update id_to_idx mapping
        if unvisited_tree.shape[0] > 0:
            id_to_idx = {int(node_id): idx for idx, node_id in enumerate(unvisited_tree[:, 0].tolist())}
        else:
            id_to_idx = {}

    
        cut_ends.extend(changed_nodes)
        # remove nodes from cut ends that no longer exist
        adj_get = adj_dict.get
        cut_ends = list({
            ce_int
            for ce in cut_ends
            for ce_int in (int(ce),)
            if ce_int in id_to_idx
            and adj_get(ce_int)
        })

    return visited, unvisited_tree, adj_dict, cut_ends, id_to_idx


def update_current_section(
    new_position,
    section_nodes,
    unvisited_tree,
    terminal_points,
    cut_ends,
    adj_dict,
    id_to_idx,
    valid_dist2=49.0,
    neuron_root_ids=None,
):
    """
    Updates the current section nodes based on the proximity of cut ends to the new position. If no cut ends are within
    valid_dist2 pixels of the new position, the section nodes remain unchanged. For each cut end within
    valid_dist2 pixels of the new position, add descendants of the cut end, up to a maximum geodesic distance,to the
    current section nodes. Add terminal points encountered in the new section to the terminal points list. Remove
    terminal points not in the new section and farther than valid_dist2 pixels from the new position from the terminal
    points list.

    Parameters
    ----------
    new_position : torch.Tensor
        The new position (x, y, z) to compare against cut ends.
    section_nodes : list of int or None
        List of node IDs that are part of the current section, or None if all nodes are considered.
    unvisited_tree : torch.Tensor
        The current unvisited tree as an SWC tensor.
    terminal_points : torch.Tensor or None
        The current terminal points as a tensor of shape (N, 3), or None if there are no terminal points.
    cut_ends : list of int
        List of node IDs that are the current cut ends of the unvisited tree.
    adj_dict : dict
        Undirected edge list from load.adjacency_dict() for the current unvisited tree.
    id_to_idx : dict
        Mapping from node ID to index in the unvisited_tree tensor.
    valid_dist2 : float, optional
        The squared distance threshold for considering cut ends as part of the current section. Default is 49.0 (7 pixels).
    neuron_root_ids : set, optional
        Cached neuron root node IDs (where parent is -1). If None, roots are derived from unvisited_tree.

    Returns
    -------
    section_nodes : list of int or None
        List of node IDs that are part of the current section based on proximity to cut ends, or None if there are no section
        nodes.
    terminal_points : torch.Tensor or None
        Updated tensor of terminal points including new terminals from the current section and excluding those farther than
        valid_dist2 from the new position.
    """
    terminal_nodes = None
    # Update section nodes based on proximity to the new position. Use any cut ends within close_dist pixels.
    if unvisited_tree.shape[0] == 0 or not cut_ends:
        return section_nodes, terminal_points

    id_lookup = id_to_idx.get
    cut_ends_indices = [idx for v in cut_ends for idx in (id_lookup(int(v)),) if idx is not None]
    if not cut_ends_indices:
        return section_nodes, terminal_points

    pos_row = new_position.unsqueeze(0)
    swc_filtered = unvisited_tree[cut_ends_indices]
    cut_ends_coords = swc_filtered[:, 2:5]  # shape (M,3)
    # Compute distances
    dists2 = torch.sum((cut_ends_coords - pos_row) ** 2, dim=1)
    # get cut ends within close_dist pixels
    close_mask = dists2 <= valid_dist2
    if torch.any(close_mask):
        close_cut_ends = [int(node_id) for node_id in swc_filtered[close_mask][:, 0].tolist()]
        section_nodes = []
        terminal_nodes = []
        for node in close_cut_ends:
            connected_nodes, terminals = _get_connected_nodes(
                int(node),
                adj_dict=adj_dict,
                max_dist=12.0,
                swc_list=unvisited_tree,
                id_to_idx=id_to_idx,
                neuron_root_ids=neuron_root_ids,
            )
            section_nodes.extend(connected_nodes)
            terminal_nodes.extend(terminals)

    # Don't remove terminal points
    # # remove terminal points that are not part of the section and farther than valid_dist from the new position
    # if terminal_points is not None and terminal_points.shape[0] > 0:
    #     dists2 = torch.sum((terminal_points - pos_row) ** 2, dim=1)
    #     valid_mask = dists2 <= valid_dist2
    #     terminal_points = terminal_points[valid_mask]
    # and add new terminals to terminal points
    if terminal_nodes:
        new_terminal_points = [unvisited_tree[id_to_idx[int(t)], 2:5] for t in terminal_nodes]
        new_terminal_points_t = torch.stack(new_terminal_points)
        if terminal_points is None:
            terminal_points = new_terminal_points_t
        else:
            terminal_points = torch.cat([terminal_points, new_terminal_points_t])


    return section_nodes, terminal_points