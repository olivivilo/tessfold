#!/usr/bin/env bash
# Public entrypoint: status | report | resume-controller
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_cli_$$"
case "${1:-status}" in
status)
  python3 - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/meta_compress_search')
f=p/'controller.json'
if f.exists():
 state=json.loads(f.read_text());print('phase:',state.get('phase'));print('updated:',state.get('updated_at'))
 if state.get('error'):print('error:',state['error'])
 if state.get('selection'):print('selection:',json.dumps(state['selection'],ensure_ascii=False))
else:state={}
for registry in ['early_round2_jobs.json','early_round3_jobs.json']:
 path=p/registry
 if path.exists():
  for item in json.loads(path.read_text()).get('jobs',[]):
   if item['name'] not in state.get('jobs',{}):
    print('prelaunched candidate (see Slurm state below):',item['name'],'job',item.get('active_job',item['job']))
for run in sorted((p/'runs').glob('*')):
 if run.name=='smoke':continue
 path=run/'training.jsonl';rows=path.read_text().splitlines() if path.exists() else []
 last=None
 for line in reversed(rows):
  try:last=json.loads(line);break
  except json.JSONDecodeError:pass
 if last:print(run.name,'updates',str(last['step'])+'/'+str(last['total_steps']),'last_train_batch_KL',round(last['prediction'],5),'complete',(run/'complete.json').exists(),'early_stop',(run/'early_stop.json').exists())
print('Report: docs/METACOMPRESS_TRAINING_SEARCH.md')
print('Live: artifacts/meta_compress_search/RUNNING_STATUS.md')
PY
  squeue -u "$USER" -o '%.12i %.22j %.9T %.12M %.25R'
  ;;
report)
  .venv/bin/python -m scripts.report_meta_compress_search
  .venv/bin/python -m scripts.report_meta_compress_search --cohort confirm
  .venv/bin/python -m scripts.plot_meta_compress_search
  ;;
resume-controller)
  python3 - <<'PY'
import json,subprocess
from pathlib import Path
p=Path('artifacts/meta_compress_search/controller_job.json')
old=json.loads(p.read_text())['job'] if p.exists() else None
if old:
 active=subprocess.run(['squeue','-j',str(old),'--noheader','--format=%T'],text=True,capture_output=True).stdout.strip()
 if active:print('Controller already active:',old,active);raise SystemExit(0)
job=subprocess.check_output(['sbatch','--parsable','scripts/slurm_meta_compress_search_controller.sh'],text=True).strip()
p.write_text(json.dumps({'job':job},indent=2));print(job)
PY
  ;;
*) echo 'Usage: bash scripts/run_meta_compress_search.sh status|report|resume-controller' >&2;exit 2;;
esac
