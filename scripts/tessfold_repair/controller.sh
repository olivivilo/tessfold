#!/usr/bin/env bash
#SBATCH --job-name=tessfold-repair
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/tessfold_system_repair_20261115/controller-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -u -m scripts.tessfold_repair.suite control
