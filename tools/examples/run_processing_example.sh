#!/bin/bash

# Example usage of process_neuron_data.py script
# This script demonstrates how to call the Python script with typical arguments

# Set your paths here
TIFS_PATH="/home/brysongray/data/neurotrack_data/gold166/gold166_tifs_scaled/"
SWC_PATH="/home/brysongray/data/neurotrack_data/gold166/gold166_swc_scaled/"
TIFS_OUT="/home/brysongray/data/neurotrack_data/gold166/gold166_tifs_processed"
SWC_OUT="/home/brysongray/data/neurotrack_data/gold166/gold166_swc_processed"
SCALING_DICT="/home/brysongray/data/neurotrack_data/gold166/scaling_dict.npy"
SCALES_DF="/home/brysongray/data/neurotrack_data/gold166/scaling_gold.csv"

# Run the processing script
python -m neurotrack.cli.process_neuron_data \
    --tifs_path "$TIFS_PATH" \
    --swc_path "$SWC_PATH" \
    --tifs_out "$TIFS_OUT" \
    --swc_out "$SWC_OUT" \
    --scaling_dict "$SCALING_DICT" \
    --scales_df "$SCALES_DF" \
    --correct_inhomogeneity \
    --sync \
    --save_out

# Alternative usage with plotting enabled
# python -m neurotrack.cli.process_neuron_data \
#     --tifs_path "$TIFS_PATH" \
#     --swc_path "$SWC_PATH" \
#     --tifs_out "$TIFS_OUT" \
#     --swc_out "$SWC_OUT" \
#     --scaling_dict "$SCALING_DICT" \
#     --scales_df "$SCALES_DF" \
#     --plot \
#     --correct_inhomogeneity \
#     --sync \
#     --save_out

echo "Processing complete!"
