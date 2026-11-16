#!/usr/bin/env bash
#SBATCH --job-name=meta-search
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=18:00:00
#SBATCH --output=artifacts/meta-search-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
.venv/bin/python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m scripts.train_meta_compress_search --config "$1" --device cpu --threads 8
