#!/usr/bin/env bash
#SBATCH --job-name=column-tree-full
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/compression_lab_v2/column_tree_full_20261108/controller-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -u -m scripts.column_tree_full_suite control
