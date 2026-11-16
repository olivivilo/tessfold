#!/usr/bin/env bash
#SBATCH --job-name=tessfold-1075-control
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=artifacts/tessfold_acceleration_20261115/budget1075/controller-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -u -m scripts.tessfold_budget1075.controller control
