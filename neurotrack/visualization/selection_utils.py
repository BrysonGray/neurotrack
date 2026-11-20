from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from neurotrack.visualization.editor_state import AnnotationGraph


ViewName = str
SeedPointZYX = Tuple[int, int, int]
CursorZYX = Tuple[int, int, int]


def seed_visible_in_view(
    seed: SeedPointZYX,
    view: ViewName,
    projection_mode: str,
    cursor_zyx: CursorZYX,
) -> bool:
    if projection_mode == "mip":
        return True
    z, y, x = seed
    current_z, current_y, current_x = cursor_zyx
    if view == "xy":
        return z == current_z
    if view == "xz":
        return y == current_y
    return x == current_x


def seed_plot_coords(seed: SeedPointZYX, view: ViewName) -> Tuple[float, float]:
    z, y, x = seed
    if view == "xy":
        return float(x), float(y)
    if view == "xz":
        return float(x), float(z)
    return float(y), float(z)


def visible_seed_indices(
    seeds: Sequence[SeedPointZYX],
    view: ViewName,
    projection_mode: str,
    cursor_zyx: CursorZYX,
) -> List[int]:
    if projection_mode == "mip":
        return list(range(len(seeds)))
    return [
        idx for idx, seed in enumerate(seeds)
        if seed_visible_in_view(seed, view, projection_mode, cursor_zyx)
    ]


def hit_test_seed(
    seeds: Sequence[SeedPointZYX],
    view: ViewName,
    xdata: float,
    ydata: float,
    projection_mode: str,
    cursor_zyx: CursorZYX,
    tolerance: float = 4.0,
) -> Optional[int]:
    if not seeds:
        return None
    best_idx: Optional[int] = None
    best_dist_sq = float(tolerance) * float(tolerance)
    for idx in visible_seed_indices(seeds, view, projection_mode, cursor_zyx):
        sx, sy = seed_plot_coords(seeds[idx], view)
        dist_sq = (sx - xdata) ** 2 + (sy - ydata) ** 2
        if dist_sq <= best_dist_sq:
            best_idx = idx
            best_dist_sq = dist_sq
    return best_idx


def annotation_node_visible_in_view(
    xyz: Sequence[float],
    view: ViewName,
    projection_mode: str,
    cursor_zyx: CursorZYX,
    atol: float = 0.5,
) -> bool:
    if projection_mode == "mip":
        return True
    current_z, current_y, current_x = cursor_zyx
    if view == "xy":
        return bool(np.isclose(float(xyz[2]), float(current_z), atol=atol))
    if view == "xz":
        return bool(np.isclose(float(xyz[1]), float(current_y), atol=atol))
    return bool(np.isclose(float(xyz[0]), float(current_x), atol=atol))


def annotation_plot_coords(xyz: Sequence[float], view: ViewName) -> Tuple[float, float]:
    if view == "xy":
        return float(xyz[0]), float(xyz[1])
    if view == "xz":
        return float(xyz[0]), float(xyz[2])
    return float(xyz[1]), float(xyz[2])


def hit_test_annotation_node(
    annotation: AnnotationGraph,
    view: ViewName,
    xdata: float,
    ydata: float,
    projection_mode: str,
    cursor_zyx: CursorZYX,
    tolerance: float = 4.0,
) -> Optional[int]:
    if not annotation.nodes_by_id:
        return None
    best_idx: Optional[int] = None
    best_dist_sq = float(tolerance) * float(tolerance)
    for node_id, node in annotation.nodes_by_id.items():
        if not annotation_node_visible_in_view(node.xyz, view, projection_mode, cursor_zyx):
            continue
        sx, sy = annotation_plot_coords(node.xyz, view)
        dist_sq = (sx - xdata) ** 2 + (sy - ydata) ** 2
        if dist_sq <= best_dist_sq:
            best_idx = node_id
            best_dist_sq = dist_sq
    return best_idx