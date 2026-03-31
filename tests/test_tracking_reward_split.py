import unittest

import torch

from neurotrack.environments.tracking_reward import (
    _init_visited,
    update_current_section,
    update_visited_edges,
)


class TrackingRewardMidEdgeSplitTests(unittest.TestCase):
    def _make_single_edge_swc(self) -> torch.Tensor:
        # id, type, x, y, z, radius, parent
        return torch.tensor(
            [
                [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
                [2, 3, 10.0, 0.0, 0.0, 1.0, 1],
            ],
            dtype=torch.float32,
        )

    def test_first_mid_edge_step_keeps_both_unvisited_sides(self):
        unvisited_tree = self._make_single_edge_swc()
        id_to_idx = {1: 0, 2: 1}
        adj_dict = {1: [2], 2: [1]}
        visited = _init_visited(unvisited_tree)

        prev_position = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float32)
        new_position = torch.tensor([8.0, 0.0, 0.0], dtype=torch.float32)

        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes={1, 2},
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[],
            valid_dist2=1e6,
        )

        self.assertEqual(len(cut_ends), 2)
        self.assertEqual(unvisited_tree.shape[0], 4)
        self.assertEqual(len(visited), 2)

        active_nodes = {int(n) for n in unvisited_tree[:, 0].tolist()}
        self.assertIn(1, active_nodes)
        self.assertIn(2, active_nodes)

    def test_update_current_section_includes_both_sides_after_split(self):
        unvisited_tree = self._make_single_edge_swc()
        id_to_idx = {1: 0, 2: 1}
        adj_dict = {1: [2], 2: [1]}
        visited = _init_visited(unvisited_tree)

        prev_position = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float32)
        new_position = torch.tensor([8.0, 0.0, 0.0], dtype=torch.float32)

        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes={1, 2},
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[],
            valid_dist2=1e6,
        )

        section_nodes, terminal_points = update_current_section(
            new_position=new_position,
            section_nodes=None,
            unvisited_tree=unvisited_tree,
            terminal_points=None,
            cut_ends=cut_ends,
            adj_dict=adj_dict,
            id_to_idx=id_to_idx,
            valid_dist2=1e6,
            neuron_root_ids={1},
        )

        self.assertIsNotNone(section_nodes)
        section_node_set = {int(n) for n in section_nodes}
        self.assertIn(1, section_node_set)
        self.assertIn(2, section_node_set)
        self.assertIsNotNone(terminal_points)

    def test_update_visited_edges_fallback_projects_prev_to_end_component(self):
        # Two disconnected components:
        # component A: 1--2 (cut-end anchored start lands here)
        # component B: 3--4 (actual motion occurs here)
        unvisited_tree = torch.tensor(
            [
                [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
                [2, 3, 10.0, 0.0, 0.0, 1.0, 1],
                [3, 3, 100.0, 0.0, 0.0, 1.0, -1],
                [4, 3, 110.0, 0.0, 0.0, 1.0, 3],
            ],
            dtype=torch.float32,
        )
        id_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        adj_dict = {1: [2], 2: [1], 3: [4], 4: [3]}
        visited = _init_visited(unvisited_tree)

        prev_position = torch.tensor([103.0, 0.0, 0.0], dtype=torch.float32)
        new_position = torch.tensor([107.0, 0.0, 0.0], dtype=torch.float32)

        # Non-empty cut_ends forces first anchor to component A (node 1),
        # but the movement is on component B. Fallback should recover.
        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes={1, 2, 3, 4},
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[1],
            valid_dist2=1e6,
        )

        # Regression expectation: component B edge gets updated and split,
        # so we should retain both original component roots and get new cut ends.
        active_nodes = {int(n) for n in unvisited_tree[:, 0].tolist()}
        self.assertIn(1, active_nodes)
        self.assertIn(2, active_nodes)
        self.assertIn(3, active_nodes)
        self.assertIn(4, active_nodes)
        self.assertGreaterEqual(unvisited_tree.shape[0], 6)
        self.assertGreaterEqual(len(cut_ends), 2)
        self.assertGreaterEqual(len(visited), 3)


if __name__ == "__main__":
    unittest.main()
