#!/usr/bin/env bash
#SBATCH --job-name=monkey-vqa-align
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/artifacts/hf_cache"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=16
CONFIG="${1:-configs/alignment.json}"
EXPERIMENT_NAME="$(basename "$CONFIG" .json)"
.venv/bin/python -m vqa.pipeline --config "$CONFIG" run > "artifacts/${EXPERIMENT_NAME}_pipeline.log" 2>&1
.venv/bin/python scripts/report.py --config "$CONFIG"
