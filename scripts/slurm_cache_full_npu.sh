#!/usr/bin/env bash
#SBATCH --job-name=monkey-vqa-cache
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/full-cache-npu-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HOME="$PWD/artifacts/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX="$PWD/artifacts/python_cache/$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=7200
test -f data/full/ALL_TASKS_READY
test -f artifacts/lm_backend_verification.json
CONFIG="${1:-configs/full.json}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
.venv-npu/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    -m vqa.full_train --config "$CONFIG" --phase cache
.venv/bin/python scripts/report_full.py
