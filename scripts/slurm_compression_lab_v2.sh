#!/usr/bin/env bash
#SBATCH --job-name=compression-v2
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/compression_lab_v2/job-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="/tmp/compression_v2_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
V2_MODE=${1:-smoke}
V2_WORKERS=${2:-1}
V2_LIST=${3:-artifacts/compression_lab_v2/screen_configs.json}
V2_PIDS=()
trap 'kill "${V2_PIDS[@]}" 2>/dev/null || true' EXIT
for ((V2_LOCAL=0;V2_LOCAL<V2_WORKERS;V2_LOCAL++));do
 LOCAL_RANK=$V2_LOCAL .venv-und/bin/python -u -m scripts.train_compression_lab_v2_repaired --mode "$V2_MODE" --worker "$V2_LOCAL" --workers "$V2_WORKERS" --config-list "$V2_LIST" > "artifacts/compression_lab_v2/${V2_MODE}-${SLURM_JOB_ID}-worker${V2_LOCAL}.log" 2>&1 &
 V2_PIDS+=("$!")
done
V2_FAILED=0
for V2_PID in "${V2_PIDS[@]}";do wait "$V2_PID" || V2_FAILED=1;done
exit "$V2_FAILED"
