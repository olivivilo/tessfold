#!/usr/bin/env bash
#SBATCH --job-name=meta-search-cache32
#SBATCH --partition=a128m512u
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/meta-search-cache32-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
srun --cpu-bind=none bash scripts/meta_compress_search_cache_node.sh
