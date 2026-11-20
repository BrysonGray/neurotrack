"""Summarize round-based inference results into a CSV table.

This module scans inference output directories shaped like:

    /home/brysongray/data/neurotrack_data/inference_outputs/bc_test_inference/
        test_bc_inference_round_19_2026-05-11_11-55-48/test_bc_inference_round_19_summary.json

and writes one CSV row per round with the core summary metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_BASE_DIR = Path("/home/brysongray/data/neurotrack_data/inference_outputs/bc_test_inference")
DEFAULT_OUTPUT_CSV = Path("/home/brysongray/neurotrack/outputs/inference_outputs/dagger_soma_removed_inference_results.csv")
DEFAULT_ROUNDS = range(1, 21)
SUMMARY_FIELDS = [
    "n_neurons",
    "n_valid",
    "n_skipped",
    "mean_bidirectional_distance",
    "mean_directed_pred_to_gt",
    "mean_directed_gt_to_pred",
    "mean_endpoint_localization_error",
    "mean_branchpoint_localization_error",
]


def _find_summary_path(base_dir: Path, round_number: int) -> Optional[Path]:
    pattern = str(base_dir / f"test_bc_inference_round_{round_number}_*" / "*_summary.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[-1])


def _load_summary(summary_path: Path) -> Dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(base_dir: Path, rounds: Iterable[int] = DEFAULT_ROUNDS) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_number in rounds:
        summary_path = _find_summary_path(base_dir, round_number)
        if summary_path is None:
            continue

        summary = _load_summary(summary_path)
        row: Dict[str, Any] = {"round": round_number}
        for field in SUMMARY_FIELDS:
            row[field] = summary.get(field)
        row["summary_path"] = str(summary_path)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    fieldnames = ["round", *SUMMARY_FIELDS, "summary_path"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize inference round outputs into a CSV file.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Directory containing round output folders.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Destination CSV file path.")
    args = parser.parse_args()

    rows = build_rows(args.base_dir)
    write_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()