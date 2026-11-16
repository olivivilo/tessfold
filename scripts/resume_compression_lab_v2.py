"""Resume interrupted jobs/checkpoints; never reopen a scientifically rejected stage."""
import argparse,fcntl,subprocess
from scripts.report_compression_lab_v2 import ROOT,BASE,load,save,refresh
from scripts.advance_compression_lab_v2 import submit,submit_full
from scripts.compression_lab_v2_jobs import active,named
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--no-controller',action='store_true');args=parser.parse_args()
 with (BASE/'advance.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  paths=load(BASE/'screen_configs.json');unfinished=[]
  for p in paths:
   cfg=load(ROOT/p);out=ROOT/cfg['output'];refresh(out)
   if not (out/'complete.json').exists():unfinished.append(p)
  if unfinished and not named('v2-screen'):
   job=subprocess.check_output(['sbatch','--parsable','--job-name=v2-screen','--gres=npu:14','--cpus-per-task=112','--mem=448G',
    'scripts/slurm_compression_lab_v2.sh','screen','14'],cwd=ROOT,text=True).strip();print('Resumed screen',job)
  state=load(BASE/'stages.json') or {'ratios':{}}
  for record in state['ratios'].values():
   stage=record.get('stage')
   if stage=='expanded_running':
    jobs=record.get('expanded_jobs',[None]*2)
    for i,p in enumerate(record['expanded_configs']):
     cfg=load(ROOT/p);out=ROOT/cfg['output'];refresh(out)
     if not (out/'complete.json').exists() and not active(jobs[i]):jobs[i]=submit('scripts/slurm_compression_lab_v2_distributed.sh',p);print('Resumed expansion',jobs[i])
    record['expanded_jobs']=jobs
   elif stage=='confirmation_running' and not active(record.get('confirmation_job')):
    cases=load(ROOT/record['confirmation_list'])
    for case in cases:refresh(ROOT/case['output'])
    if not all((ROOT/c['output']/f'confirmation.{c["stage"]}.summary.json').exists() for c in cases):
     record['confirmation_job']=subprocess.check_output(['sbatch','--parsable','--job-name=v2-confirm','--gres=npu:4','--cpus-per-task=32','--mem=128G',
      'scripts/slurm_compression_lab_v2.sh','suite','4',record['confirmation_list']],cwd=ROOT,text=True).strip().split(';')[0];print('Resumed confirmation',record['confirmation_job'])
   elif stage in ['full_running','benchmark_running']:
    cfg=load(ROOT/record['full_config']);out=ROOT/cfg['output']
    if stage=='benchmark_running':out=out/'benchmark'
    refresh(out);jobkey='full_job' if stage=='full_running' else 'benchmark_job'
    if not (out/'complete.json').exists() and not active(record.get(jobkey)):
     retries=record.setdefault('recovery_submissions',{}).get(jobkey,[])
     if len(retries)>=3:
      record['recovery_exhausted']=jobkey
      save(BASE/'stages.json',state)
      raise RuntimeError(f'Infrastructure recovery exhausted for {cfg["name"]}: {jobkey}')
     script='scripts/slurm_compression_lab_v2_distributed.sh' if stage=='full_running' else 'scripts/slurm_compression_lab_v2_benchmark.sh'
     previous=record.get(jobkey);record[jobkey]=submit_full(script,record['full_config'])
     retries.append({'previous_job':previous,'new_job':record[jobkey]});record['recovery_submissions'][jobkey]=retries
     save(BASE/'stages.json',state);print('Resumed',stage,record[jobkey])
  save(BASE/'stages.json',state)
 if not args.no_controller and not named('v2-controller'):print('Controller',submit('scripts/slurm_compression_lab_v2_controller.sh'))
if __name__=='__main__':main()
