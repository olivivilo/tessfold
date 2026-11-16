#!/usr/bin/env bash
#SBATCH --job-name=tessfold-8b-memory
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/tessfold_acceleration_20261115/qwen-memory-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="/tmp/tessfold_8b_mem_${SLURM_JOB_ID}"
.venv-und/bin/python -u -m torch.distributed.run --standalone --nproc_per_node="${TESSFOLD_NPROC:-4}" -m scripts.tessfold_accel.qwen_memory "$@"
