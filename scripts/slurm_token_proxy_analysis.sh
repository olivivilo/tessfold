#!/usr/bin/env bash
#SBATCH --job-name=token-proxy-analysis
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/token-proxy-analysis-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv-und/bin/python -m scripts.audit_token_proxy "$@"
.venv/bin/python -m scripts.sensitivity_token_proxy --workers 4 "$@"
.venv/bin/python -m scripts.report_token_proxy "$@"
