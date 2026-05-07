#!/bin/bash

# Slurm batch script to launch one BC training process per allocated node.
# Each node gets its own temporary config with a unique rng_seed, name, and outdir.

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-$BASE_DIR/configs/training/train_bc_example.json}"
SEED_START="${SEED_START:-1}"
PYTHON_BIN="${PYTHON_BIN:-/ifshome/bgray/anaconda3/envs/intel_env/bin/python}"
export PYTHON_BIN

JOB_NODE_COUNT="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"

echo "Launching ${JOB_NODE_COUNT} BC training instance(s) across Slurm nodes."

srun \
    --ntasks="$JOB_NODE_COUNT" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    /bin/bash -c '
        set -euo pipefail
        node_id="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
        base_config="$1"
        python_bin="$2"
        seed_start="$3"
        config_dir="${SLURM_TMPDIR:-/tmp}/neurotrack_bc_${SLURM_JOB_ID:-local}_${node_id}"
        mkdir -p "$config_dir"
        cfg="${config_dir}/node_$(printf "%02d" "$node_id").json"
        "$python_bin" - "$base_config" "$cfg" "$node_id" "$seed_start" <<"PY"
import json
import pathlib
import sys

base_config_path = pathlib.Path(sys.argv[1])
output_config_path = pathlib.Path(sys.argv[2])
node_id = int(sys.argv[3])
seed_start = int(sys.argv[4])

with base_config_path.open("r", encoding="utf-8") as handle:
    base_params = json.load(handle)

seed = seed_start + node_id
base_name = str(base_params.get("name", "bc_training"))
base_outdir = pathlib.Path(str(base_params["outdir"]))

params = dict(base_params)
params["rng_seed"] = seed
params["name"] = f"{base_name}_node{node_id:02d}_seed{seed}"
params["outdir"] = str(base_outdir / params["name"])

with output_config_path.open("w", encoding="utf-8") as handle:
    json.dump(params, handle, indent=4)
    handle.write("\n")
PY
        echo "[$node_id] Starting training with ${cfg}"
        "$python_bin" -m neurotrack.cli.run_bc_train -i "$cfg"
    ' _ "$BASE_CONFIG" "$PYTHON_BIN" "$SEED_START"