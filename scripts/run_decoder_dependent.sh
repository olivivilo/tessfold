#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
submit_report() {
    sbatch --parsable "$@" --job-name=decoder-loss-report --partition=a128m512u \
      --cpus-per-task=8 --mem=48G --time=01:00:00 \
      --output=artifacts/decoder_dependent_loss/report-%j.log \
      --wrap='export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; .venv/bin/python -m scripts.report_decoder_dependent'
}
case "${1:-status}" in
  cpu)
    if [ -n "$(squeue -h -u "$USER" --name=decoder-loss-cpu)" ]; then
      echo 'A decoder-loss-cpu allocation is already active; use status.' >&2; exit 2
    fi
    # Only after every prior CPU allocation ended, release abandoned empty claims.
    .venv/bin/python - <<'PY'
from pathlib import Path
p=Path('artifacts/decoder_dependent_loss/claims')
if p.exists():
 for d in p.iterdir():d.rmdir()
PY
    LOSS_JOB=$(sbatch --parsable scripts/slurm_decoder_dependent_cpu.sh)
    echo "Compute job: $LOSS_JOB"
    echo "Report job: $(submit_report --dependency="afterok:$LOSS_JOB")" ;;
  npu)
    LOSS_JOB=$(sbatch --parsable scripts/slurm_decoder_dependent.sh)
    echo "Compute job: $LOSS_JOB"
    echo "Report job: $(submit_report --dependency="afterok:$LOSS_JOB")" ;;
  status)
    .venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/decoder_dependent_loss');files=list((p/'losses').glob('*.json'))
print(f'Completed image losses: {len(files)}/3601; interventions: {26*len(files)}/93626')
if (p/'audit.json').exists():
 d=json.loads((p/'audit.json').read_text());print('Report audit complete:',d['complete'])
print('Methods: docs/DECODER_DEPENDENT_LOSS_VALIDATION.md')
print('Results: artifacts/decoder_dependent_loss/RESULTS.md')
PY
    squeue -u "${USER}" ;;
  report)
    submit_report ;;
  *) echo 'Usage: bash scripts/run_decoder_dependent.sh {status|cpu|npu|report}' >&2; exit 2 ;;
esac
