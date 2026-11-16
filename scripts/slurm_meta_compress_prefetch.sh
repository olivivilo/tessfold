#!/usr/bin/env bash
#SBATCH --job-name=meta-teacher-cache
#SBATCH --partition=a128m512u
#SBATCH --array=0-3
#SBATCH --cpus-per-task=112
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --output=artifacts/meta-compress-prefetch-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16
META_PIDS=()
trap 'kill "${META_PIDS[@]}" 2>/dev/null || true' EXIT
for ((META_LOCAL=0; META_LOCAL<7; META_LOCAL++)); do
  RANK=$((SLURM_ARRAY_TASK_ID*7+META_LOCAL)) WORLD_SIZE=28 .venv/bin/python -u -m scripts.prefetch_meta_compress &
  META_PIDS+=("$!")
done
META_FAILED=0
for META_PID in "${META_PIDS[@]}";do wait "$META_PID" || META_FAILED=1;done
exit "$META_FAILED"
