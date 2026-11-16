#!/usr/bin/env bash
#SBATCH --job-name=lab-confirm
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/compression_lab/suite-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="/tmp/compression_lab_suite_${SLURM_JOB_ID}"
.venv-und/bin/python -u -m scripts.evaluate_compression_lab_suite --list "$1"
