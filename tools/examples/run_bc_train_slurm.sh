#!/bin/bash
#SBATCH --job-name=fold5_all_dagger
#SBATCH --output=result_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=42G
#SBATCH --time=3-00:00:00

set -e

python_bin="/ifshome/bgray/anaconda3/envs/intel_env/bin/python"
cfg="/nafs/dtward/bryson/neurotrack/configs/training/train_dagger_example.json"
cd /nafs/dtward/bryson/neurotrack

"$python_bin" -m neurotrack.cli.run_bc_train -i "$cfg"