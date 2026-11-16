#!/usr/bin/env bash
#SBATCH --job-name=meta-final-benchmark
#SBATCH --partition=a128m512u
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/meta-final-benchmark-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
export META_BENCH_CONFIG="$1" META_BENCH_CHECKPOINT="$2"
srun --cpu-bind=none bash scripts/meta_compress_search_benchmark_node.sh
