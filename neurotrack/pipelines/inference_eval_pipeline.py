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


class InferenceEvaluationPipeline:
    """Orchestrate inference, post-processing, and optional evaluation."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_pipeline_config(self.config_path)
        self._validate_config()

        seed = self.config.get("rng_seed")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def _validate_config(self) -> None:
        required = ["img_dir", "out_dir", "name", "sac_weights"]
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
        run_evaluation: bool | None = None,
    ) -> Dict[str, Any]:
        run_out_dir = Path(self.config["out_dir"]) / (self.config["name"] + "_" + date_time)
        run_out_dir.mkdir(parents=True, exist_ok=True)

        postprocess_config = PostprocessConfig.from_config(self.config)

        # Determine if any postprocessing step is enabled.
        should_postprocess = (
            postprocess_config.filter_branches_by_length
            or postprocess_config.resample
            or postprocess_config.smooth_paths
            or postprocess_config.merge_paths
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

        inference_payload = run_inference(self.config, run_out_dir)
        inference_results = list(inference_payload["results"])

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
            metrics_csv = run_out_dir / f"{self.config['name']}_metrics.csv"
            save_evaluation_results(
                evaluation_results,
                str(metrics_csv),
                write_summary=False,
            )

        pipeline_summary = compute_pipeline_summary(
            postprocessed_results=postprocessed_results,
            evaluation_results=evaluation_results,
            has_ground_truth=should_evaluate,
        )

        pipeline_summary_path = run_out_dir / f"{self.config['name']}_summary.json"
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
            "postprocess": None
            if postprocess_payload is None
            else {
                "swc_out_dir": str(postprocess_payload["swc_out_dir"]),
            },
        }


def run_inference_eval_pipeline(
    config_path: str,
    run_evaluation: bool | None = None,
) -> Dict[str, Any]:
    pipeline = InferenceEvaluationPipeline(config_path=config_path)
    return pipeline.run(run_evaluation=run_evaluation)
