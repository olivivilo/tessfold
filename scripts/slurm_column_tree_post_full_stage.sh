#!/usr/bin/env bash
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/compression_lab_v2/column_tree_post_full_20261108/stage-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=7200
export PYTHONPYCACHEPREFIX="/tmp/column_tree_post_full_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
.venv-und/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.run_column_tree_post_full_stage --stage "$1" --config "$2"
