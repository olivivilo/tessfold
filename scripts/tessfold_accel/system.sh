#!/usr/bin/env bash
#SBATCH --job-name=tessfold-compiled-system
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/tessfold_acceleration_20261115/system-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=7200
export PYTHONPYCACHEPREFIX="/tmp/tessfold_compiled_system_${SLURM_JOB_ID}"
.venv-und/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.tessfold_accel.system_benchmark --mode single
.venv/bin/python -m scripts.tessfold_accel.system_report
