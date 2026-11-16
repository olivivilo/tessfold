#!/usr/bin/env bash
#SBATCH --job-name=meta-strict-full-data
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/meta_compress_strict/full-data-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/meta_strict_full_data_${SLURM_JOB_ID}"
.venv/bin/python -u -m scripts.audit_meta_compress_strict_full_data
