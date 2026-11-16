#!/usr/bin/env bash
#SBATCH --job-name=monkey-dt-download
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/dt-download-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/download_dtvqa.py
