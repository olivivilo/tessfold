#!/usr/bin/env bash
#SBATCH --job-name=tessfold-tiled-vision
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/tessfold_acceleration_20261115/tiledvision-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="/tmp/tessfold_tile_${SLURM_JOB_ID}"
.venv-und/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=2 -m scripts.tessfold_accel.tiled_vision_probe "$@"
