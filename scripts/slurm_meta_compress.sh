#!/usr/bin/env bash
#SBATCH --job-name=monkey-meta-compress
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/meta-compress-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/artifacts/hf_cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=7200
MAX_ITEMS="${MAX_ITEMS:-20000}"
EPOCHS="${EPOCHS:-2}"
OUTPUT="${OUTPUT:-artifacts/meta_compress}"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 20000))
.venv-und/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m scripts.train_meta_compress --output "$OUTPUT" --max-items "$MAX_ITEMS" --epochs "$EPOCHS"
