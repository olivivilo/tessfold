#!/usr/bin/env bash
#SBATCH --job-name=meta-mean
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/meta-mean-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
.venv/bin/python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m scripts.baseline_meta_compress_search --ratio "$1" --cohort "${2:-search}" --threads 8
