#!/usr/bin/env bash
#SBATCH --job-name=meta-strict-controller
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=6-00:00:00
#SBATCH --output=artifacts/meta_compress_strict/controller-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/meta_strict_mpl
export PYTHONPYCACHEPREFIX="/tmp/meta_strict_controller_${SLURM_JOB_ID}"
.venv/bin/python -u -m scripts.orchestrate_meta_compress_strict "$@"
