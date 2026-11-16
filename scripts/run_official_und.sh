#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -x .venv-und/bin/python || { echo 'Run bash scripts/bootstrap_official_und.sh first.' >&2; exit 1; }
.venv/bin/python -m scripts.download_official_und
.venv/bin/python -m scripts.prepare_official_und
.venv/bin/python -m scripts.submit_official_und
