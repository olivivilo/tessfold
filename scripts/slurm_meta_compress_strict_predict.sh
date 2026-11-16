#!/usr/bin/env bash
#SBATCH --job-name=meta-strict-predict
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/meta_compress_strict/predict-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/meta_strict_predict_${SLURM_JOB_ID}"
# The partition accepts only even NPU allocations; one image uses rank 0.
.venv-und/bin/python -u -m scripts.predict_meta_compress_strict "$@"
