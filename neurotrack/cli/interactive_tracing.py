"""
CLI entrypoint for the interactive tracing pipeline: seed selection, tracing,
post-processing, and evaluation, with a JSON config for I/O specification.
"""

import argparse
from neurotrack.pipelines import run_interactive_tracing_session

def main():

    parser = argparse.ArgumentParser(
        description="Run an interactive tracing session (seed selection, tracing, post-processing, evaluation)."
    )
    parser.add_argument('-c', '--config', type=str, required=False, help='Optional path to session config JSON.')
    parser.add_argument('-i', '--img_dir', type=str, required=False, help='Directory containing input images (TIFF format).')
    parser.add_argument('-s', '--seeds_input', type=str, required=False, help='Optional existing seeds JSON file.')
    parser.add_argument('-o', '--seeds_output', type=str, required=False, help='Output seeds JSON file path.')
    args = parser.parse_args()

    run_interactive_tracing_session(
        image_dir=args.img_dir,
        seeds_output_path=args.seeds_output,
        seeds_input_path=args.seeds_input,
        config_path=args.config,
    )

    print("Interactive tracing session complete.")
    return

if __name__ == "__main__":
    main()
