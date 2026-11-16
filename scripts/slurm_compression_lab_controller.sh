#!/usr/bin/env bash
#SBATCH --job-name=lab-controller
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=artifacts/compression_lab/controller-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/compression_lab_controller_${SLURM_JOB_ID}"
.venv/bin/python -u -m scripts.control_compression_lab
