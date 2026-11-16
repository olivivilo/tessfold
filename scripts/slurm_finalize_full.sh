#!/usr/bin/env bash
#SBATCH --job-name=monkey-data-finalize
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/data-finalize-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$PWD/artifacts/python_cache/$SLURM_JOB_ID"
.venv/bin/python scripts/append_dtvqa.py
.venv/bin/python scripts/preflight_full_data.py
