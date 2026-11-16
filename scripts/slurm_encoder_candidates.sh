#!/usr/bin/env bash
#SBATCH --job-name=encoder-proxy-loss
#SBATCH --partition=a128m512u
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/encoder-proxy-loss-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.eval_encoder_candidates --workers 32 --shard "$SLURM_ARRAY_TASK_ID" --shards 4
