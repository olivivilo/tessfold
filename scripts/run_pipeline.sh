#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/alignment.json}"
EXPERIMENT_NAME="$(basename "$CONFIG" .json)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HOME="$PWD/artifacts/hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p artifacts
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/verify_models.py
.venv/bin/python scripts/prepare_data.py --dataset-root "${DATASET_ROOT:-datasets/monkeydocv2}"
.venv/bin/python -m vqa.pipeline --config "$CONFIG" run 2>&1 | tee "artifacts/${EXPERIMENT_NAME}_pipeline.log"
.venv/bin/python scripts/report.py --config "$CONFIG"
