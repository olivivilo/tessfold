#!/usr/bin/env bash
#SBATCH --job-name=encoder-line-probe
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/encoder-line-probe-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.train_encoder_relation
.venv/bin/python -m scripts.eval_encoder_relation --workers 32
