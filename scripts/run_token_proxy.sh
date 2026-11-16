#!/usr/bin/env bash
# Explicit entry for the random-compression correlation experiment (no training).
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
PROXY_LABEL="${PROXY_LABEL:-random_v1}"
case "${1:-help}" in
  prepare)
    .venv/bin/python -m scripts.test_token_proxy
    .venv/bin/python -m scripts.prepare_token_proxy --label "$PROXY_LABEL" --per-task 24
    ;;
  smoke)
    .venv/bin/python -m scripts.prepare_token_proxy --label smoke_v1 --per-task 1 --smoke
    sbatch --gres=npu:2 --cpus-per-task=8 --mem=64G --export=ALL,PROXY_RANKS=2 scripts/slurm_token_proxy.sh --label smoke_v1
    ;;
  run|resume)
    test -f "artifacts/token_proxy/$PROXY_LABEL/protocol.json"
    sbatch scripts/slurm_token_proxy.sh --label "$PROXY_LABEL"
    ;;
  status)
    .venv/bin/python -m scripts.status_token_proxy --label "$PROXY_LABEL"
    squeue -u "$(id -un)"
    ;;
  report)
    .venv-und/bin/python -m scripts.audit_token_proxy --label "$PROXY_LABEL"
    .venv/bin/python -m scripts.report_token_proxy --label "$PROXY_LABEL"
    ;;
  sensitivity)
    sbatch scripts/slurm_token_proxy_analysis.sh --label "$PROXY_LABEL"
    ;;
  *)
    echo 'Usage: bash scripts/run_token_proxy.sh {prepare|smoke|run|resume|status|report|sensitivity}'
    ;;
esac
