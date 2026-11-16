#!/usr/bin/env bash
set -euo pipefail
SEARCH_PIDS=()
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
trap 'kill "${SEARCH_PIDS[@]}" 2>/dev/null || true' EXIT
for ((SEARCH_LOCAL=0; SEARCH_LOCAL<16; SEARCH_LOCAL++));do
  RANK=$((SLURM_NODEID*16+SEARCH_LOCAL)) WORLD_SIZE=32 .venv/bin/python -u -m scripts.cache_meta_compress_search &
  SEARCH_PIDS+=("$!")
done
SEARCH_FAILED=0
for SEARCH_PID in "${SEARCH_PIDS[@]}";do wait "$SEARCH_PID" || SEARCH_FAILED=1;done
exit "$SEARCH_FAILED"
