"""Resume committed work, preserve active jobs; no gate or architecture changes."""
import fcntl,json,subprocess
from scripts.report_compression_lab import ROOT,BASE,load
from scripts.advance_compression_lab import save

def main():
 rows=subprocess.check_output(['squeue','-h','-u',subprocess.check_output(['id','-un'],text=True).strip(),'-o','%i|%j'],text=True).splitlines();active={r.split('|')[0] for r in rows};names={r.split('|')[1] for r in rows}
 def submit(*args):return subprocess.check_output(['sbatch','--parsable',*args],cwd=ROOT,text=True).strip().split(';')[0]
 with (BASE/'advance.lock').open('w') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  for listname,name,workers,module in [('screen_configs.json','lab-screen',8,'scripts.train_compression_lab'),('followup_configs.json','lab-followup',6,'scripts.train_compression_lab'),('pre_configs.json','lab-pre',4,'scripts.train_compression_lab_pre')]:
   configs=[load(ROOT/p) for p in load(BASE/listname)]
   if configs and not all((ROOT/c['output']/'complete.json').exists() for c in configs) and name not in names:
    job=submit('--job-name='+name,'--gres=npu:'+str(workers),'--cpus-per-task='+str(8*workers),'--mem='+str(32*workers)+'G','scripts/slurm_compression_lab.sh','screen',str(workers),'artifacts/compression_lab/'+listname,module);print(name,job)
  state=load(BASE/'stages.json')
  for ratio,record in state.get('ratios',{}).items():
   stage=record.get('stage')
   if stage=='expanded_running':
    for i,path in enumerate(record['expanded_configs']):
     cfg=load(ROOT/path)
     if not (ROOT/cfg['output']/'complete.json').exists() and record['expanded_jobs'][i] not in active:
      record['expanded_jobs'][i]=submit('scripts/slurm_compression_lab_distributed.sh',path)
   if stage in ['full_running','benchmark_running']:
    cfg=load(ROOT/record['full_config']);key='full_job' if stage=='full_running' else 'benchmark_job';complete=ROOT/cfg['output']/('complete.json' if stage=='full_running' else 'benchmark/complete.json')
    if not complete.exists() and record[key] not in active:record[key]=submit('scripts/slurm_compression_lab_distributed.sh' if stage=='full_running' else 'scripts/slurm_compression_lab_benchmark.sh',record['full_config'])
   if stage=='confirmation_running' and record['confirmation_job'] not in active:
    cases=load(ROOT/record['confirmation_list'])
    if not all((ROOT/c['output']/f'confirmation.{c["stage"]}.summary.json').exists() for c in cases):record['confirmation_job']=submit('scripts/slurm_compression_lab_suite.sh',record['confirmation_list'])
  if state:save(BASE/'stages.json',state)
 if 'lab-controller' not in names:
  complete=load(BASE/'controller_complete.json')
  if not complete:print('controller',submit('scripts/slurm_compression_lab_controller.sh'))
 print('Active jobs retained. Resumed runs keep their original code/config/data manifests.')
if __name__=='__main__':main()
