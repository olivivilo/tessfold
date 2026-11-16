#!/usr/bin/env bash
#SBATCH --job-name=meta-strict-cache
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/meta_compress_strict/prefetch-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=7200
.venv-und/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m scripts.prefetch_meta_compress_strict "$@"
