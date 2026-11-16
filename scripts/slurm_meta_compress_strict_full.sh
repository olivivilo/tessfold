#!/usr/bin/env bash
#SBATCH --job-name=meta-strict-full
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/meta_compress_strict/full-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=7200
export PYTHONPYCACHEPREFIX="/tmp/meta_strict_full_${SLURM_JOB_ID}"
.venv-und/bin/python -u -m torch.distributed.run --standalone --nproc_per_node="${META_STRICT_TRAIN_NPROC:-8}" \
  -m scripts.train_meta_compress_strict_full --device npu --threads 8 "$@"
