"""Evaluation IO and dataset-level aggregation utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from neurotrack.data import loading as load
from neurotrack.evaluation.metrics import evaluate_reconstruction


def evaluate_postprocessed_results(
    postprocessed_results: List[Dict[str, Any]],
    swc_dir: str,
    distance_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    evaluation_results: List[Dict[str, Any]] = []

    gt_swc_dir = Path(swc_dir)
    gt_swc_files = {f.stem: f for f in gt_swc_dir.glob("*.swc")}

    for result in postprocessed_results:
        neuron_name = result["neuron_name"]
        pred_swc = result["swc_list"]
        neuron_basename = Path(neuron_name).stem

        if "error" in result:
            print(f"[eval SKIP] '{neuron_basename}': {result['error']}")
            evaluation_results.append(
                {
                    "neuron_name": neuron_name,
                    "error": result["error"],
                    "skipped": True,
                }
            )
            continue

        gt_file = None
        if neuron_basename in gt_swc_files:
            gt_file = gt_swc_files[neuron_basename]
        else:
            for gt_name, gt_path in gt_swc_files.items():
                if neuron_basename in gt_name or gt_name in neuron_basename:
                    gt_file = gt_path
                    break

        if gt_file is None:
            print(f"[eval SKIP] '{neuron_basename}': no ground truth SWC file found in {gt_swc_dir}")
            evaluation_results.append(
                {
                    "neuron_name": neuron_name,
                    "error": "No ground truth file found",
                    "skipped": True,
                }
            )
            continue

        try:
            gt_swc = load.swc(str(gt_file), verbose=False)

            if len(pred_swc) == 0:
                print(f"[eval SKIP] '{neuron_basename}': empty prediction (0 SWC nodes)")
                evaluation_results.append(
                    {
                        "neuron_name": neuron_name,
                        "n_nodes_pred": 0,
                        "n_nodes_gt": len(gt_swc),
                        "error": "Empty prediction",
                        "skipped": True,
                    }
                )
                continue

            metrics = evaluate_reconstruction(
                pred_swc,
                gt_swc,
                threshold=distance_threshold,
                return_l_measures=True,
            )
            metrics["neuron_name"] = neuron_name
            metrics["gt_file"] = str(gt_file)
            metrics["skipped"] = False
            metrics["n_raw_paths"] = result.get("n_raw_paths", 0)
            metrics["n_processed_paths"] = result.get("n_processed_paths", 0)

            evaluation_results.append(metrics)
        except Exception as exc:
            print(f"[eval ERROR] '{neuron_basename}': {exc}")
            evaluation_results.append(
                {
                    "neuron_name": neuron_name,
                    "error": str(exc),
                    "skipped": True,
                }
            )

    return evaluation_results


def save_evaluation_results(
    results: List[Dict[str, Any]],
    output_path: str,
    summary_path: Optional[str] = None,
    write_summary: bool = True,
) -> None:
    df = upsert_evaluation_results_csv(results, output_path)

    if not write_summary:
        return

    if summary_path is None:
        summary_path = output_path.replace(".csv", "_summary.json")

    write_evaluation_summary_from_csv(output_path, summary_path=summary_path)


def upsert_evaluation_results_csv(
    results: List[Dict[str, Any]],
    output_path: str,
    key_column: str = "neuron_name",
) -> pd.DataFrame:
    output_file = Path(output_path)
    new_df = pd.DataFrame(results)

    if output_file.exists():
        existing_df = pd.read_csv(output_file)
    else:
        existing_df = pd.DataFrame()

    if len(new_df) == 0 and len(existing_df) == 0:
        empty_df = pd.DataFrame(columns=[key_column])
        empty_df.to_csv(output_file, index=False)
        return empty_df

    if key_column not in new_df.columns:
        raise ValueError(f"'{key_column}' column is required to upsert evaluation results.")

    if len(existing_df) == 0:
        combined_df = new_df.copy()
    else:
        if key_column not in existing_df.columns:
            raise ValueError(f"Existing evaluation CSV is missing '{key_column}' column: {output_file}")
        combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)

    combined_df = combined_df.drop_duplicates(subset=[key_column], keep="last")
    combined_df.to_csv(output_file, index=False)
    return combined_df


def write_evaluation_summary_from_csv(
    csv_path: str | Path,
    summary_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    csv_file = Path(csv_path)
    if summary_path is None:
        summary_path = csv_file.with_name(f"{csv_file.stem}_summary.json")
    summary_file = Path(summary_path)

    df = pd.read_csv(csv_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary: Dict[str, Any] = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "median": float(df[col].median()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    summary["n_neurons"] = int(len(df))

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def compute_pipeline_summary(
    postprocessed_results: List[Dict[str, Any]],
    evaluation_results: List[Dict[str, Any]],
    has_ground_truth: bool,
) -> Dict[str, Any]:
    def _finite_stats(values: List[Any]) -> Dict[str, Optional[float]]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return {"mean": None, "std": None}
        return {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
        }

    if not has_ground_truth:
        n_neurons = len(postprocessed_results)
        skipped_neurons = [
            {
                "neuron_name": r.get("neuron_name", "unknown"),
                "reason": r.get("error", "postprocessing_error"),
            }
            for r in postprocessed_results
            if "error" in r
        ]
        n_failed = len(skipped_neurons)
        processed_paths = [result.get("n_processed_paths", 0) for result in postprocessed_results]
        swc_nodes = [result.get("n_swc_nodes", 0) for result in postprocessed_results]
        return {
            "n_neurons": n_neurons,
            "n_valid": n_neurons - n_failed,
            "n_skipped": n_failed,
            "skipped_neurons": skipped_neurons,
            "mean_processed_paths": float(np.mean(processed_paths)) if len(processed_paths) > 0 else 0.0,
            "std_processed_paths": float(np.std(processed_paths)) if len(processed_paths) > 0 else 0.0,
            "mean_swc_nodes": float(np.mean(swc_nodes)) if len(swc_nodes) > 0 else 0.0,
            "std_swc_nodes": float(np.std(swc_nodes)) if len(swc_nodes) > 0 else 0.0,
        }

    valid_results = [result for result in evaluation_results if not result.get("skipped", False)]
    skipped_neurons = [
        {
            "neuron_name": r.get("neuron_name", "unknown"),
            "reason": r.get("error", "unknown"),
        }
        for r in evaluation_results
        if r.get("skipped", False)
    ]
    if len(valid_results) == 0:
        return {
            "n_neurons": len(evaluation_results),
            "n_valid": 0,
            "n_skipped": len(evaluation_results),
            "skipped_neurons": skipped_neurons,
            "mean_bidirectional_distance": 0.0,
            "mean_directed_pred_to_gt": 0.0,
            "mean_directed_gt_to_pred": 0.0,
        }

    bidirectional_dists = [result["bidirectional_distance"] for result in valid_results]
    directed_pred_to_gt = [result["directed_div_pred_to_gt"] for result in valid_results]
    directed_gt_to_pred = [result["directed_div_gt_to_pred"] for result in valid_results]
    endpoint_loc_stats = _finite_stats([result.get("endpoint_localization_error") for result in valid_results])
    branchpoint_loc_stats = _finite_stats([result.get("branchpoint_localization_error") for result in valid_results])
    endpoint_count_stats = _finite_stats([result.get("endpoint_count_error") for result in valid_results])
    branchpoint_count_stats = _finite_stats([result.get("branchpoint_count_error") for result in valid_results])

    return {
        "n_neurons": len(evaluation_results),
        "n_valid": len(valid_results),
        "n_skipped": len(evaluation_results) - len(valid_results),
        "skipped_neurons": skipped_neurons,
        "mean_bidirectional_distance": float(np.mean(bidirectional_dists)),
        "std_bidirectional_distance": float(np.std(bidirectional_dists)),
        "median_bidirectional_distance": float(np.median(bidirectional_dists)),
        "mean_directed_pred_to_gt": float(np.mean(directed_pred_to_gt)),
        "std_directed_pred_to_gt": float(np.std(directed_pred_to_gt)),
        "mean_directed_gt_to_pred": float(np.mean(directed_gt_to_pred)),
        "std_directed_gt_to_pred": float(np.std(directed_gt_to_pred)),
        "mean_endpoint_localization_error": endpoint_loc_stats["mean"],
        "std_endpoint_localization_error": endpoint_loc_stats["std"],
        "mean_branchpoint_localization_error": branchpoint_loc_stats["mean"],
        "std_branchpoint_localization_error": branchpoint_loc_stats["std"],
        "mean_endpoint_count_error": endpoint_count_stats["mean"],
        "std_endpoint_count_error": endpoint_count_stats["std"],
        "mean_branchpoint_count_error": branchpoint_count_stats["mean"],
        "std_branchpoint_count_error": branchpoint_count_stats["std"],
    }
