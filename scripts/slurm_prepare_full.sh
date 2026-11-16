#!/usr/bin/env bash
#SBATCH --job-name=monkey-full-data
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/full-data-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/prepare_full_pipeline.py
