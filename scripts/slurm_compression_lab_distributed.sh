#!/usr/bin/env bash
#SBATCH --job-name=lab-expanded
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/compression_lab/distributed-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=7200
export PYTHONPYCACHEPREFIX="/tmp/compression_lab_dist_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
.venv-und/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
 -m scripts.train_compression_lab_distributed --config "$1"
