#!/usr/bin/env bash
#SBATCH --job-name=monkey-metacompress-cpu
#SBATCH --partition=a128m512u
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=artifacts/meta-compress-cpu-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16
export META_PROCESSES_PER_NODE=6 META_THREADS=16
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 20000))
export META_OUTPUT="${META_OUTPUT:-artifacts/meta_compress}"
srun --cpu-bind=none bash scripts/meta_compress_cpu_node.sh
