#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-status}" in
  status)
    squeue -u "$USER" -o '%.18i %.18j %.12T %.12M %.30R'
    .venv/bin/python -m scripts.report_meta_compress --allow-partial
    tail -n 3 artifacts/meta_compress/training.jsonl 2>/dev/null || true
    ;;
  report) .venv/bin/python -m scripts.report_meta_compress ;;
  train|resume) sbatch scripts/slurm_meta_compress_cpu.sh ;;
  eval) sbatch scripts/slurm_meta_compress_eval_cpu.sh ;;
  *) echo 'Usage: bash scripts/run_meta_compress.sh status|report|train|resume|eval'; exit 2 ;;
esac
