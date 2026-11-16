#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then
        PYTHON_BIN=python3.10
    else
        PYTHON_BIN=python3
    fi
fi
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
echo 'Environment ready. Entry: bash scripts/run_pipeline.sh'
