#!/usr/bin/env bash
#SBATCH --job-name=compression-supplement
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/compression_supplement_20261111/controller-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -u -m scripts.compression_supplement_suite control
