#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
case "${1:-help}" in
  prepare)
    .venv/bin/python -m scripts.prepare_token_proxy --label random_v1_20x --per-task 480 --cap-to-available
    ;;
  run|resume)
    sbatch scripts/slurm_token_proxy_20x.sh
    ;;
  status)
    .venv/bin/python -m scripts.status_token_proxy --label random_v1_20x
    squeue -u "$(id -un)"
    ;;
  report)
    sbatch scripts/slurm_token_proxy_20x_analysis.sh
    ;;
  *)
    echo 'Usage: bash scripts/run_token_proxy_20x.sh {prepare|run|resume|status|report}'
    ;;
esac
