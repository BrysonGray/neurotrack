import unittest

import torch

from neurotrack.environments.tracking_reward import (
    _init_visited,
    _compute_target_action,
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

        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx, terminal_nodes = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes={1, 2},
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[],
            terminal_nodes=set(),
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

        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx, terminal_nodes = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes={1, 2},
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[],
            terminal_nodes=set(),
            valid_dist2=1e6,
        )

        section_nodes, terminal_nodes = update_current_section(
            new_position=new_position,
            section_nodes=None,
            unvisited_tree=unvisited_tree,
            terminal_nodes=set(),
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
        self.assertIsNotNone(terminal_nodes)

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
        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx, terminal_nodes = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes={1, 2, 3, 4},
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[1],
            terminal_nodes=set(),
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

    def test_update_visited_edges_fallback_still_makes_progress(self):
        # Construct a simple section with cut-end anchoring away from motion,
        # then verify fallback logic still updates visited coverage.
        unvisited_tree = torch.tensor(
            [
                [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
                [2, 3, 10.0, 0.0, 0.0, 1.0, 1],
                [3, 3, 20.0, 0.0, 0.0, 1.0, 2],
            ],
            dtype=torch.float32,
        )
        id_to_idx = {1: 0, 2: 1, 3: 2}
        adj_dict = {1: [2], 2: [1, 3], 3: [2]}
        visited = _init_visited(unvisited_tree)

        # Put cut_end near node 1 while movement occurs near node 3.
        prev_position = torch.tensor([18.0, 0.0, 0.0], dtype=torch.float32)
        new_position = torch.tensor([19.5, 0.0, 0.0], dtype=torch.float32)

        visited_before = dict(visited)
        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx, terminal_nodes = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes={1, 2, 3},
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[1],
            terminal_nodes=set(),
            valid_dist2=1e6,
        )

        # Ensure some progress happened versus original visited state.
        self.assertNotEqual(len(visited_before), len(visited))
        self.assertLess(unvisited_tree.shape[0], 3)
        self.assertGreaterEqual(len(cut_ends), 1)

    def test_update_visited_edges_last_resort_unrestricted_retry(self):
        # section_nodes can become stale/disconnected and contain no valid edge,
        # which blocks section-constrained projections. Last-resort retry should
        # still make progress by allowing full-graph projection.
        unvisited_tree = torch.tensor(
            [
                [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
                [2, 3, 10.0, 0.0, 0.0, 1.0, 1],
                [3, 3, 20.0, 0.0, 0.0, 1.0, 2],
            ],
            dtype=torch.float32,
        )
        id_to_idx = {1: 0, 2: 1, 3: 2}
        adj_dict = {1: [2], 2: [1, 3], 3: [2]}
        visited = _init_visited(unvisited_tree)

        prev_position = torch.tensor([18.0, 0.0, 0.0], dtype=torch.float32)
        new_position = torch.tensor([19.0, 0.0, 0.0], dtype=torch.float32)

        # Stale/disconnected section filter: no edge exists between nodes {1, 3}.
        section_nodes = {1, 3}
        visited_before = dict(visited)
        visited, unvisited_tree, adj_dict, cut_ends, id_to_idx, terminal_nodes = update_visited_edges(
            prev_position=prev_position,
            new_position=new_position,
            section_nodes=section_nodes,
            visited=visited,
            unvisited_tree=unvisited_tree,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            cut_ends=[1],
            terminal_nodes=set(),
            valid_dist2=1e6,
        )

        self.assertNotEqual(visited_before, visited)
        self.assertNotEqual(unvisited_tree.shape[0], 3)
        self.assertGreaterEqual(len(cut_ends), 1)

    def test_compute_target_action_falls_back_to_nearest_point_when_no_node_is_inside_radius(self):
        # A long edge can have its nearest point inside the step radius while both
        # endpoints are outside. The target computation should return the nearest
        # projected point instead of failing.
        swc_list = torch.tensor(
            [
                [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
                [2, 3, 100.0, 0.0, 0.0, 1.0, 1],
            ],
            dtype=torch.float32,
        )
        id_to_idx = {1: 0, 2: 1}
        adj_dict = {1: [2], 2: [1]}
        current_pos = torch.tensor([60.0, 10.0, 0.0], dtype=torch.float32)

        target_vectors, stop_target, nearest_dist2 = _compute_target_action(
            current_pos=current_pos,
            swc_list=swc_list,
            step_size=20.0,
            id_to_idx=id_to_idx,
            adj_dict=adj_dict,
            terminal_nodes=set(),
            valid_nodes=None,
            valid_dist2=1e6,
        )

        self.assertFalse(stop_target)
        # The geometry can yield one or two intersection targets; accept either.
        self.assertIn(target_vectors.shape[0], (1, 2))
        self.assertEqual(target_vectors.shape[1], 3)
        self.assertAlmostEqual(float(nearest_dist2), 100.0, places=4)

        # Each returned target vector should be at distance ~= step_size from current_pos
        norms = torch.linalg.norm(target_vectors, dim=1)
        expected = torch.full_like(norms, 20.0)
        self.assertTrue(torch.allclose(norms, expected, atol=1e-3, rtol=1e-3))

        # Y component should be approximately -10 for all targets (projection onto x-axis)
        y_comp = target_vectors[:, 1]
        self.assertTrue(torch.allclose(y_comp, torch.full_like(y_comp, -10.0), atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
