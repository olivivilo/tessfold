#!/usr/bin/env bash
#SBATCH --job-name=encoder-evidence-cache
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/encoder-evidence-cache-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
PIDS=()
for ((I=0; I<4; I++)); do
 RANK="$I" WORLD_SIZE=4 LOCAL_RANK="$I" .venv-und/bin/python -m scripts.cache_encoder_probe &
 PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
exit "$FAILED"
