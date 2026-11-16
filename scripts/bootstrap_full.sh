#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r requirements-full.txt
.venv/bin/python -m pip check
"$PYTHON_BIN" -m venv .venv-npu
# Keep torch_npu isolated so the CPU preparation runtime does not auto-load CANN.
.venv-npu/bin/python - <<'PY'
import sysconfig,subprocess
from pathlib import Path
cpu_site=subprocess.check_output(['.venv/bin/python','-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True).strip()
Path(sysconfig.get_paths()['purelib'],'project_runtime.pth').write_text(cpu_site+'\n')
PY
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv-npu/bin/python -m pip install --no-deps torch-npu==2.8.0.post5
