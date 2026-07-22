import math
import unittest

from neurotrack.evaluation.metrics import evaluate_reconstruction


def _make_y_tree(branch_x=1.0, leaf_x=2.0):
    return [
        [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
        [2, 3, branch_x, 0.0, 0.0, 1.0, 1],
        [3, 3, leaf_x, 1.0, 0.0, 1.0, 2],
        [4, 3, leaf_x, -1.0, 0.0, 1.0, 2],
    ]


def _make_chain():
    return [
        [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
        [2, 3, 1.0, 0.0, 0.0, 1.0, 1],
        [3, 3, 2.0, 0.0, 0.0, 1.0, 2],
    ]


class ReconstructionSpecialNodeMetricTests(unittest.TestCase):
    def test_identical_tree_has_zero_special_node_errors(self):
        gt_swc = _make_y_tree()

        metrics = evaluate_reconstruction(gt_swc, gt_swc, threshold=0.5)

        self.assertEqual(metrics["endpoint_count_error"], 0)
        self.assertEqual(metrics["branchpoint_count_error"], 0)
        self.assertAlmostEqual(metrics["endpoint_localization_error"], 0.0)
        self.assertAlmostEqual(metrics["branchpoint_localization_error"], 0.0)

    def test_shifted_tree_reports_expected_localization_errors(self):
        gt_swc = _make_y_tree(branch_x=1.0, leaf_x=2.0)
        pred_swc = _make_y_tree(branch_x=2.0, leaf_x=3.0)

        metrics = evaluate_reconstruction(pred_swc, gt_swc, threshold=0.5)

        self.assertEqual(metrics["endpoint_count_error"], 0)
        self.assertEqual(metrics["branchpoint_count_error"], 0)
        self.assertAlmostEqual(metrics["endpoint_localization_error"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["branchpoint_localization_error"], 1.0)

    def test_missing_branch_reports_count_error_and_nan_branchpoint_localization(self):
        gt_swc = _make_y_tree()
        pred_swc = [
            [1, 3, 0.0, 0.0, 0.0, 1.0, -1],
            [2, 3, 1.0, 0.0, 0.0, 1.0, 1],
            [3, 3, 2.0, 1.0, 0.0, 1.0, 2],
        ]

        metrics = evaluate_reconstruction(pred_swc, gt_swc, threshold=0.5)

        self.assertEqual(metrics["endpoint_count_error"], 1)
        self.assertEqual(metrics["branchpoint_count_error"], 1)
        self.assertAlmostEqual(metrics["endpoint_localization_error"], 1.0 / 3.0)
        self.assertTrue(math.isnan(metrics["branchpoint_localization_error"]))

    def test_trees_without_branchpoints_report_zero_branchpoint_localization_error(self):
        chain = _make_chain()

        metrics = evaluate_reconstruction(chain, chain, threshold=0.5)

        self.assertEqual(metrics["branchpoint_count_error"], 0)
        self.assertAlmostEqual(metrics["branchpoint_localization_error"], 0.0)


if __name__ == "__main__":
    unittest.main()