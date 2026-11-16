#!/usr/bin/env bash
set -euo pipefail
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
BENCH_PIDS=()
trap 'kill "${BENCH_PIDS[@]}" 2>/dev/null || true' EXIT
for ((BENCH_LOCAL=0; BENCH_LOCAL<16; BENCH_LOCAL++));do
  RANK=$((SLURM_NODEID*16+BENCH_LOCAL)) WORLD_SIZE=32 .venv/bin/python -u -m scripts.benchmark_meta_compress_search \
    --config "$META_BENCH_CONFIG" --checkpoint "$META_BENCH_CHECKPOINT" --threads 8 &
  BENCH_PIDS+=("$!")
done
BENCH_FAILED=0
for BENCH_PID in "${BENCH_PIDS[@]}";do wait "$BENCH_PID" || BENCH_FAILED=1;done
exit "$BENCH_FAILED"
