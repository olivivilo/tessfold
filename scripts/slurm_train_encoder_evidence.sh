#!/usr/bin/env bash
#SBATCH --job-name=encoder-evidence-head
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/encoder-evidence-head-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.train_encoder_evidence
