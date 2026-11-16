#!/usr/bin/env bash
#SBATCH --job-name=meta-strict-geometry-init
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/meta_compress_strict/geometry-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPYCACHEPREFIX="/tmp/meta_strict_geometry_${SLURM_JOB_ID}"
.venv/bin/python -u -m scripts.calibrate_meta_compress_strict_geometric "$@"
