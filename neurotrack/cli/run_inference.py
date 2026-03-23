"""
This is the single CLI entrypoint for inference with optional
post-processing and optional evaluation controlled by JSON config.
"""

import argparse

from neurotrack.pipelines import run_inference_eval_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with optional post-processing/evaluation from JSON config.",
    )
    parser.add_argument("-i", "--json", type=str, required=True, help="Path to inference config JSON file.")
    args = parser.parse_args()

    result = run_inference_eval_pipeline(
        config_path=args.json,
        run_postprocessing=None,
        run_evaluation=None,
    )

    print("Inference complete.")
    print(f"Outputs saved to: {result['run_out_dir']}")
    if result["mode"]["run_postprocessing"]:
        print(f"Processed SWC files saved to: {result['postprocess']['swc_out_dir']}")
    if result["mode"]["run_evaluation"]:
        print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
