#!/usr/bin/env bash
#SBATCH --job-name=encoder-full-OT
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/encoder-full-OT-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
PIDS=()
TRANSPORT_WORLD=8
if [[ "${1:-}" == "--benchmark" ]]; then TRANSPORT_WORLD=4; fi
for ((I=0; I<TRANSPORT_WORLD; I++)); do
 RANK="$I" WORLD_SIZE="$TRANSPORT_WORLD" LOCAL_RANK="$((I%4))" .venv-und/bin/python -m scripts.exact_encoder_transport_npu "$@" &
 PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
exit "$FAILED"
