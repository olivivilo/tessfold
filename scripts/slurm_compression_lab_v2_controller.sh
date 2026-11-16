#!/usr/bin/env bash
#SBATCH --job-name=v2-controller
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/compression_lab_v2/controller-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -u -m scripts.control_compression_lab_v2
