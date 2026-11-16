#!/usr/bin/env bash
#SBATCH --job-name=encoder-OT-audit
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/encoder-OT-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.audit_encoder_transport --workers 8
