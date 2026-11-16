#!/usr/bin/env bash
#SBATCH --job-name=compression-lab
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/compression_lab/job-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="/tmp/compression_lab_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
LAB_MODE=${1:-smoke}
LAB_WORKERS=${2:-1}
LAB_LIST=${3:-artifacts/compression_lab/screen_configs.json}
LAB_MODULE=${4:-scripts.train_compression_lab}
LAB_PIDS=()
trap 'kill "${LAB_PIDS[@]}" 2>/dev/null || true' EXIT
for ((LAB_LOCAL=0;LAB_LOCAL<LAB_WORKERS;LAB_LOCAL++));do
 LOCAL_RANK=$LAB_LOCAL .venv-und/bin/python -u -m "$LAB_MODULE" --mode "$LAB_MODE" --worker "$LAB_LOCAL" --workers "$LAB_WORKERS" --config-list "$LAB_LIST" > "artifacts/compression_lab/${LAB_MODE}-${SLURM_JOB_ID}-worker${LAB_LOCAL}.log" 2>&1 &
 LAB_PIDS+=("$!")
done
LAB_FAILED=0
for LAB_PID in "${LAB_PIDS[@]}";do wait "$LAB_PID" || LAB_FAILED=1;done
exit "$LAB_FAILED"
