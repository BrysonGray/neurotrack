import tempfile
import unittest
from pathlib import Path

import pandas as pd

from neurotrack.evaluation.io import evaluate_postprocessed_results, save_evaluation_results


def _write_swc(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                "{} {} {:.1f} {:.1f} {:.1f} {:.1f} {}\n".format(*row)
            )


def _make_chain_rows():
    return [
        (1, 3, 0.0, 0.0, 0.0, 1.0, -1),
        (2, 3, 1.0, 0.0, 0.0, 1.0, 1),
        (3, 3, 2.0, 0.0, 0.0, 1.0, 2),
    ]


class EvaluationIOTests(unittest.TestCase):
    def test_evaluate_postprocessed_results_includes_l_measures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            swc_dir = root / "gt_swc"
            swc_dir.mkdir()

            gt_rows = _make_chain_rows()
            _write_swc(swc_dir / "sample.swc", gt_rows)

            postprocessed_results = [
                {
                    "neuron_name": "sample.tif",
                    "swc_list": [list(row) for row in gt_rows],
                    "n_raw_paths": 1,
                    "n_processed_paths": 1,
                }
            ]

            results = evaluate_postprocessed_results(
                postprocessed_results,
                swc_dir=str(swc_dir),
                distance_threshold=0.5,
            )

            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].get("skipped", True))
            self.assertIn("num_bifurcations_pred", results[0])
            self.assertIn("num_bifurcations_gt", results[0])
            self.assertIn("percent_different_structure_average", results[0])

    def test_save_evaluation_results_persists_l_measure_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "metrics.csv"
            summary_path = root / "summary.json"

            rows = [
                {
                    "neuron_name": "sample.tif",
                    "skipped": False,
                    "precision": 1.0,
                    "num_bifurcations_pred": 0,
                    "num_bifurcations_gt": 0,
                    "percent_different_structure_average": 0.0,
                }
            ]

            save_evaluation_results(rows, str(csv_path), summary_path=str(summary_path))

            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertIn("num_bifurcations_pred", df.columns)
            self.assertIn("num_bifurcations_gt", df.columns)
            self.assertIn("percent_different_structure_average", df.columns)
            self.assertEqual(int(df.loc[0, "num_bifurcations_pred"]), 0)


if __name__ == "__main__":
    unittest.main()
