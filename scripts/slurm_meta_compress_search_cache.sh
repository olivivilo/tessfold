#!/usr/bin/env bash
#SBATCH --job-name=meta-search-cache
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/meta-search-cache-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
SEARCH_PIDS=()
trap 'kill "${SEARCH_PIDS[@]}" 2>/dev/null || true' EXIT
for ((SEARCH_LOCAL=0; SEARCH_LOCAL<8; SEARCH_LOCAL++));do
  RANK="$SEARCH_LOCAL" WORLD_SIZE=8 .venv/bin/python -u -m scripts.cache_meta_compress_search &
  SEARCH_PIDS+=("$!")
done
SEARCH_FAILED=0
for SEARCH_PID in "${SEARCH_PIDS[@]}";do wait "$SEARCH_PID" || SEARCH_FAILED=1;done
exit "$SEARCH_FAILED"
