#!/usr/bin/env bash
#SBATCH --job-name=meta-diagnosis
#SBATCH --partition=a128m512u
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/meta-diagnosis-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16
.venv/bin/python -u -m scripts.diagnose_meta_compress --variant "$SLURM_ARRAY_TASK_ID" --threads 16 --steps 50
