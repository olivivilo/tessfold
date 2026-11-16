#!/usr/bin/env bash
#SBATCH --job-name=monkey-vqa-verify
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/verify-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/artifacts/hf_cache"
export OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/verify_pipeline.py --config configs/alignment.json
