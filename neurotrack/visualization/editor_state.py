from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from neurotrack.data import save as data_save


class AnnotationTarget(str, Enum):
    REFERENCE = "reference"
    PREDICTION = "prediction"


@dataclass
class AnnotationNode:
    node_id: int
    xyz: Tuple[float, float, float]
    radius: float = 1.0
    parent_id: int = -1
    child_ids: List[int] = field(default_factory=list)


@dataclass
class AnnotationGraph:
    nodes_by_id: Dict[int, AnnotationNode] = field(default_factory=dict)
    root_ids: List[int] = field(default_factory=list)

    @classmethod
    def from_swc_rows(cls, swc_rows: Optional[object]) -> "AnnotationGraph":
        arr = _normalize_swc_rows(swc_rows)
        nodes_by_id: Dict[int, AnnotationNode] = {}
        for row in arr:
            node_id = int(row[0])
            nodes_by_id[node_id] = AnnotationNode(
                node_id=node_id,
                xyz=(float(row[2]), float(row[3]), float(row[4])),
                radius=float(row[5]),
                parent_id=int(row[6]),
            )

        for node in nodes_by_id.values():
            if node.parent_id in nodes_by_id:
                nodes_by_id[node.parent_id].child_ids.append(node.node_id)
            else:
                node.parent_id = -1

        for node in nodes_by_id.values():
            node.child_ids.sort()

        root_ids = sorted(node_id for node_id, node in nodes_by_id.items() if node.parent_id == -1)
        return cls(nodes_by_id=nodes_by_id, root_ids=root_ids)

    @classmethod
    def from_paths(cls, paths: Optional[Sequence[Sequence[Sequence[float]]]]) -> "AnnotationGraph":
        normalized_paths: List[np.ndarray] = []
        for path in paths or []:
            path_np = np.asarray(path, dtype=np.float32)
            if path_np.ndim == 2 and path_np.shape[1] >= 3 and path_np.shape[0] > 0:
                normalized_paths.append(path_np[:, :3].copy())
        swc_rows = data_save.paths_to_swc(normalized_paths)
        return cls.from_swc_rows(swc_rows)

    def copy(self) -> "AnnotationGraph":
        return AnnotationGraph.from_swc_rows(self.to_swc_rows())

    def to_swc_rows(self) -> np.ndarray:
        if not self.nodes_by_id:
            return np.empty((0, 7), dtype=np.float32)

        rows: List[List[float]] = []
        for node_id in sorted(self.nodes_by_id.keys()):
            node = self.nodes_by_id[node_id]
            rows.append([
                float(node.node_id),
                0.0,
                float(node.xyz[0]),
                float(node.xyz[1]),
                float(node.xyz[2]),
                float(node.radius),
                float(node.parent_id),
            ])
        return np.asarray(rows, dtype=np.float32)

    def to_paths(self) -> List[np.ndarray]:
        paths: List[np.ndarray] = []
        if not self.nodes_by_id:
            return paths

        def _walk(node_id: int, prefix: List[Tuple[float, float, float]]) -> None:
            node = self.nodes_by_id[node_id]
            next_prefix = prefix + [node.xyz]
            if len(node.child_ids) == 0:
                paths.append(np.asarray(next_prefix, dtype=np.float32))
                return
            for child_id in node.child_ids:
                _walk(child_id, list(next_prefix))

        for root_id in self.root_ids:
            if root_id in self.nodes_by_id:
                _walk(root_id, [])

        if not paths:
            for root_id in self.root_ids:
                node = self.nodes_by_id.get(root_id)
                if node is not None:
                    paths.append(np.asarray([node.xyz], dtype=np.float32))
        return paths

    def descendant_ids_including(self, node_id: int) -> Set[int]:
        if node_id not in self.nodes_by_id:
            return set()

        stack = [node_id]
        descendants: Set[int] = set()
        while stack:
            current = stack.pop()
            if current in descendants:
                continue
            descendants.add(current)
            stack.extend(self.nodes_by_id[current].child_ids)
        return descendants

    def remove_nodes(self, node_ids: Iterable[int]) -> None:
        removed = {int(node_id) for node_id in node_ids}
        if not removed:
            return

        remaining: Dict[int, AnnotationNode] = {}
        for node_id, node in self.nodes_by_id.items():
            if node_id in removed:
                continue
            remaining[node_id] = AnnotationNode(
                node_id=node.node_id,
                xyz=node.xyz,
                radius=node.radius,
                parent_id=node.parent_id,
                child_ids=[child_id for child_id in node.child_ids if child_id not in removed],
            )

        for node in remaining.values():
            if node.parent_id not in remaining:
                node.parent_id = -1

        self.nodes_by_id = remaining
        self.root_ids = sorted(node_id for node_id, node in remaining.items() if node.parent_id == -1)

    def add_child(self, parent_id: int, xyz: Sequence[float], radius: float = 1.0) -> int:
        if parent_id not in self.nodes_by_id:
            raise KeyError(f"Unknown parent node id: {parent_id}")

        next_node_id = max(self.nodes_by_id.keys(), default=0) + 1
        parent = self.nodes_by_id[parent_id]
        child = AnnotationNode(
            node_id=next_node_id,
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            radius=float(radius),
            parent_id=parent_id,
        )
        self.nodes_by_id[next_node_id] = child
        parent.child_ids.append(next_node_id)
        parent.child_ids.sort()
        return next_node_id


@dataclass
class EditorSelectionState:
    selected_seed_index: Optional[int] = None
    selected_annotation_node_ids: Set[int] = field(default_factory=set)
    clip_preview_node_ids: Set[int] = field(default_factory=set)
    pending_branch_seed_xyz: Optional[Tuple[float, float, float]] = None
    selection_view: Optional[str] = None

    def clear(self) -> None:
        self.selected_seed_index = None
        self.selected_annotation_node_ids.clear()
        self.clip_preview_node_ids.clear()
        self.pending_branch_seed_xyz = None
        self.selection_view = None


@dataclass
class ViewerSessionState:
    reference_annotation: AnnotationGraph = field(default_factory=AnnotationGraph)
    prediction_annotation: AnnotationGraph = field(default_factory=AnnotationGraph)
    active_annotation: str = AnnotationTarget.REFERENCE.value
    active_tool: str = "zoom"
    show_reference_overlay: bool = False
    show_prediction_overlay: bool = True
    selection: EditorSelectionState = field(default_factory=EditorSelectionState)

    def active_annotation_graph(self) -> AnnotationGraph:
        if self.active_annotation == AnnotationTarget.PREDICTION.value:
            return self.prediction_annotation
        return self.reference_annotation

    def set_reference_swc_rows(self, swc_rows: Optional[object]) -> None:
        self.reference_annotation = AnnotationGraph.from_swc_rows(swc_rows)

    def set_prediction_paths(self, paths: Optional[Sequence[Sequence[Sequence[float]]]]) -> None:
        self.prediction_annotation = AnnotationGraph.from_paths(paths)

    def clear_transient_selection(self) -> None:
        self.selection.clear()


def _normalize_swc_rows(swc_rows: Optional[object]) -> np.ndarray:
    if swc_rows is None:
        return np.empty((0, 7), dtype=np.float32)
    arr = np.asarray(swc_rows, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7:
        return np.empty((0, 7), dtype=np.float32)
    return arr[:, :7].copy()