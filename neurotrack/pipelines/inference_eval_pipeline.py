"""Thin pipeline orchestrator for inference -> postprocess -> evaluation."""

from datetime import datetime
import json
import numpy as np
import os
from pathlib import Path
import torch
from typing import Any, Dict


from neurotrack.core.pipeline_config import PostprocessConfig, load_pipeline_config
from neurotrack.evaluation.io import (
    compute_pipeline_summary,
    evaluate_postprocessed_results,
    save_evaluation_results,
)
from neurotrack.inference.postprocess import process_results, write_processed_swc
from neurotrack.inference.runtime import run_inference

date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

_INFERENCE_PIPELINE_DEFAULTS: Dict[str, Any] = {
    "step_width": 2.0,
    "repeat_starts": False,
    "rng_seed": 1,
    "n_trials":1,
    "seeds_path": None,
    "auto_seed_selection_mode": "remote_endnode",
    "review_before_next": False,
    "sync": False,            # Skip images whose *_trace.json already exists
    "run_postprocessing": True,
    "run_evaluation": None,   # None → infer from swc_dir
    "min_branch_length": 5.0,
    "resampling_step_size": 4.0,
    "smoothing_window": 5,
    "overlap_threshold": 0.5,
    "overlap_distance_threshold": 5.0,
    "eval_distance_threshold": None,
    "distance_threshold": 5.0,
    "scales_path": None,
    "swc_dir": None,
}


class InferenceEvaluationPipeline:
    """Orchestrate inference, post-processing, and optional evaluation."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_pipeline_config(self.config_path, _INFERENCE_PIPELINE_DEFAULTS)
        self._validate_config()

        seed = self.config.get("rng_seed")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def _validate_config(self) -> None:
        required = ["img_dir", "out_dir", "test_name", "sac_weights"]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

        paths_to_check = {
            "img_dir": self.config["img_dir"],
            "sac_weights": self.config["sac_weights"],
        }
        for key in ("swc_dir", "seeds_path", "scales_path"):
            if self.config.get(key) is not None:
                paths_to_check[key] = self.config[key]

        for name, path in paths_to_check.items():
            if not os.path.exists(path):
                raise ValueError(f"{name} does not exist: {path}")

    def run(
        self,
        run_postprocessing: bool | None = None,
        run_evaluation: bool | None = None,
    ) -> Dict[str, Any]:
        run_out_dir = Path(self.config["out_dir"]) / (self.config["test_name"] + "_" + date_time)
        run_out_dir.mkdir(parents=True, exist_ok=True)

        should_postprocess = (
            bool(self.config.get("run_postprocessing", True))
            if run_postprocessing is None
            else bool(run_postprocessing)
        )
        # Priority: caller kwarg > explicit JSON flag > infer from swc_dir.
        if run_evaluation is not None:
            should_evaluate = bool(run_evaluation)
        elif self.config.get("run_evaluation") is not None:
            should_evaluate = bool(self.config["run_evaluation"])
        else:
            should_evaluate = self.config.get("swc_dir") is not None

        if should_evaluate and self.config.get("swc_dir") is None:
            raise ValueError("Evaluation requested but 'swc_dir' is not configured.")
        if should_evaluate and not should_postprocess:
            raise ValueError("Evaluation requires post-processing. Enable post-processing first.")

        postprocess_config = PostprocessConfig.from_config(self.config)

        inference_payload = run_inference(self.config, run_out_dir)
        inference_results = list(inference_payload["results"])

        # When sync=True, run_inference skips already-traced images.  Load
        # their saved *_trace.json files and merge them in so that
        # post-processing and evaluation cover the full image set.
        if bool(self.config.get("sync", False)):
            tracing_results_dir = Path(inference_payload["tracing_results_dir"])
            traced_stems = {
                Path(r.get("neuron_name", "")).stem for r in inference_results
            }
            for json_path in sorted(tracing_results_dir.glob("*_trace.json")):
                # Strip the trailing "_trace" suffix to recover the image stem.
                img_stem = json_path.stem[: -len("_trace")]
                if img_stem not in traced_stems:
                    with json_path.open("r", encoding="utf-8") as fh:
                        cached = json.load(fh)
                    inference_results.append(cached)
                    traced_stems.add(img_stem)
                    print(f"[sync] Loaded cached inference result: {json_path.name}")

        postprocessed_results = []
        postprocess_payload = None
        if should_postprocess:
            # Process each image with its own scale-aware params.
            postprocessed_results = [
                r
                for result in inference_results
                for r in process_results(
                    [result],
                    postprocess_config.scaled_params_for_image(
                        result.get("neuron_name", "")
                    ),
                )
            ]
            postprocess_payload = write_processed_swc(postprocessed_results, run_out_dir)

        evaluation_results = []
        if should_evaluate and len(postprocessed_results) > 0:
            evaluation_results = evaluate_postprocessed_results(
                postprocessed_results,
                swc_dir=self.config["swc_dir"],
                distance_threshold=float(self.config.get("distance_threshold", 2.0)),
            )
            metrics_csv = run_out_dir / f"{self.config['test_name']}_metrics.csv"
            summary_json = run_out_dir / f"{self.config['test_name']}_summary.json"
            save_evaluation_results(
                evaluation_results,
                str(metrics_csv),
                summary_path=str(summary_json),
            )

        pipeline_summary = compute_pipeline_summary(
            postprocessed_results=postprocessed_results,
            evaluation_results=evaluation_results,
            has_ground_truth=should_evaluate,
        )

        pipeline_summary_path = run_out_dir / "pipeline_summary.json"
        with open(pipeline_summary_path, "w") as handle:
            json.dump(pipeline_summary, handle, indent=2)

        skipped = pipeline_summary.get("skipped_neurons", [])
        if skipped:
            print(f"\n[pipeline] {len(skipped)} neuron(s) skipped:")
            for entry in skipped:
                print(f"  - {Path(entry['neuron_name']).stem}: {entry['reason']}")
        else:
            print(f"\n[pipeline] All {pipeline_summary.get('n_neurons', 0)} neurons processed successfully.")

        return {
            "summary": pipeline_summary,
            "summary_path": pipeline_summary_path,
            "run_out_dir": run_out_dir,
            "mode": {
                "run_postprocessing": should_postprocess,
                "run_evaluation": should_evaluate,
            },
            "inference": {
                "tracing_results_dir": str(inference_payload["tracing_results_dir"]),
            },
            "postprocess": None
            if postprocess_payload is None
            else {
                "swc_out_dir": str(postprocess_payload["swc_out_dir"]),
            },
        }


def run_inference_eval_pipeline(
    config_path: str,
    run_postprocessing: bool | None = None,
    run_evaluation: bool | None = None,
) -> Dict[str, Any]:
    pipeline = InferenceEvaluationPipeline(config_path=config_path)
    return pipeline.run(
        run_postprocessing=run_postprocessing,
        run_evaluation=run_evaluation,
    )
