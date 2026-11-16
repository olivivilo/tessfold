#!/usr/bin/env bash
#SBATCH --job-name=tessfold-training-audit
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/tessfold_acceleration_20261115/trainingaudit-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
.venv-und/bin/python -u -m scripts.tessfold_accel.training_audit_npu
