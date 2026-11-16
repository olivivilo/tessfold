#!/usr/bin/env bash
# Explicit entrypoint. Existing diagnostics are preserved; partial runs fail closed.
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "${1:-status}" in
  status)
    .venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/meta_compress_diagnosis')
for root in [p,p/'factorial']:
    protocol=root/'protocol.json'
    if not protocol.exists():continue
    for v in json.loads(protocol.read_text())['variants']:
        d=root/v['name'];log=d/'training.jsonl'
        steps=len(log.read_text().splitlines()) if log.exists() else 0
        print(f"{v['name']}: {steps}/50, complete={(d/'complete.json').exists()}")
for name in ['controls','precision']:
    print(f"{name}: complete={(p/name/'complete.json').exists()}")
PY
    squeue -u "$USER"
    ;;
  report) PYTHONPATH=. .venv/bin/python scripts/report_meta_compress_diagnosis.py ;;
  train) sbatch scripts/slurm_meta_compress_diagnosis.sh ;;
  *) echo 'Usage: bash scripts/run_meta_compress_diagnosis.sh status|report|train' >&2; exit 2 ;;
esac
