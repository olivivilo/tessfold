#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON_BIN:-python3.10}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
if [[ "${DEVICE:-cpu}" == "cpu" ]]; then
  .venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
elif [[ "${DEVICE}" == "npu" ]]; then
  echo 'Install matching CANN/torch/torch-npu wheels on this host before running this installer.' >&2
  .venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 torch-npu==2.8.0.post5
else
  echo 'CUDA: select a PyTorch 2.8.0 / torchvision 0.23.0 wheel matching your driver.' >&2
  .venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0
fi
.venv/bin/python -m pip install -e '.[test,analysis]'
.venv/bin/python -m tessfold --help
