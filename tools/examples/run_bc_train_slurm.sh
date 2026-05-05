#!/bin/bash

# Slurm batch script to launch one BC training process per allocated node.
# Each node gets its own temporary config with a unique rng_seed, name, and outdir.

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-$BASE_DIR/configs/training/train_bc_example.json}"
SEED_START="${SEED_START:-1}"
TMP_ROOT="${TMP_ROOT:-${SLURM_TMPDIR:-/tmp}/neurotrack_bc_${SLURM_JOB_ID:-local}}"
CONFIG_DIR="$TMP_ROOT/configs"
PYTHON_BIN="${PYTHON_BIN:-/ifshome/bgray/anaconda3/envs/intel_env/bin/python}"

mkdir -p "$CONFIG_DIR"
export CONFIG_DIR
export PYTHON_BIN

JOB_NODE_COUNT="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"

"$PYTHON_BIN" - "$BASE_CONFIG" "$CONFIG_DIR" "$JOB_NODE_COUNT" "$SEED_START" <<'PY'
import json
import pathlib
import sys

base_config_path = pathlib.Path(sys.argv[1])
config_dir = pathlib.Path(sys.argv[2])
node_count = int(sys.argv[3])
seed_start = int(sys.argv[4])

with base_config_path.open("r", encoding="utf-8") as handle:
    base_params = json.load(handle)

base_name = str(base_params.get("name", "bc_training"))
base_outdir = pathlib.Path(str(base_params["outdir"]))

for node_id in range(node_count):
    seed = seed_start + node_id
    params = dict(base_params)
    params["rng_seed"] = seed
    params["name"] = f"{base_name}_node{node_id:02d}_seed{seed}"
    params["outdir"] = str(base_outdir / params["name"])

    out_path = config_dir / f"node_{node_id:02d}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=4)
        handle.write("\n")
PY

echo "Launching ${JOB_NODE_COUNT} BC training instance(s) across Slurm nodes."

srun \
    --ntasks="$JOB_NODE_COUNT" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    /bin/bash -c '
        set -euo pipefail
        node_id="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
        cfg="${CONFIG_DIR}/node_$(printf "%02d" "$node_id").json"
        echo "[$node_id] Starting training with ${cfg}"
        "${PYTHON_BIN}" -m neurotrack.cli.run_bc_train -i "$cfg"
    '