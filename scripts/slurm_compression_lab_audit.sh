#!/usr/bin/env bash
#SBATCH --job-name=lab-audit
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/compression_lab/audit-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="/tmp/compression_lab_audit_${SLURM_JOB_ID}"
.venv-und/bin/python -u -m scripts.audit_compression_lab
