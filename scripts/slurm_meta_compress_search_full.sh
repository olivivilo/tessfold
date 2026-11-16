#!/usr/bin/env bash
#SBATCH --job-name=meta-retrain-full
#SBATCH --partition=a128m512u
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=2-00:00:00
#SBATCH --output=artifacts/meta-retrain-full-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 20000))
export META_SEARCH_FULL_CONFIG="$1"
export META_SEARCH_PROCESSES=$((36 / SLURM_NNODES))
export META_SEARCH_THREADS=$((96 / META_SEARCH_PROCESSES))
if (( 36 % SLURM_NNODES != 0 || 96 % META_SEARCH_PROCESSES != 0 )); then
  echo 'Full retrain requires 36 ranks and an integer CPU allocation per rank' >&2;exit 2
fi
srun --cpu-bind=none bash scripts/meta_compress_search_full_node.sh
