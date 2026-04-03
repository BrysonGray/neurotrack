from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import tifffile as tiff
from tqdm import tqdm
from neurotrack.data.image import to_uint8


from neurotrack.data import DrawingComplexityConfig, rendering, loading


def parse_args():
    parser = ArgumentParser(description="Simulate neuron images from SWC files.")
    parser.add_argument("--swc_dir", type=str, required=True, help="Path to the SWC directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--complexity", type=float, default=0.5, help="Drawing complexity for simulated neurons (0.0 to 1.0).")
    parser.add_argument("--rng_seed", type=int, default=1, help="Optional random seed for reproducible simulation output.")

    return parser.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)

    complexity_config = DrawingComplexityConfig().interpolate_config(args.complexity)
    renderer = rendering.NeuronRenderer(rng)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    swc_dir = Path(args.swc_dir)
    swc_files = [f for f in swc_dir.rglob("*.swc") if f.is_file()]
    for swc_file in tqdm(swc_files, desc="Simulating neurons"):
        swc = loading.swc(swc_file)
        sections, _ = loading.parse_swc(swc)
        img = renderer.draw_neuron(sections, config=complexity_config)
        img_data = to_uint8(img.data)
        img_path = output_dir / (swc_file.stem + ".tif")
        tiff.imwrite(img_path, np.asarray(img_data))


if __name__ == "__main__":
    main()

