#!/usr/bin/env bash
#SBATCH --job-name=proxy-20x-analysis
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/token-proxy-20x-analysis-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv-und/bin/python -m scripts.audit_token_proxy --label random_v1_20x
.venv/bin/python -m scripts.sensitivity_token_proxy --label random_v1_20x --workers 16
.venv/bin/python -m scripts.report_token_proxy --label random_v1_20x
.venv/bin/python -m scripts.compare_token_proxy_scale --label random_v1_20x
