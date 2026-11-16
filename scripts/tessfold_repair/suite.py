"""Frozen no-retraining validation→single-request/multi-query efficiency suite."""
import argparse,collections,fcntl,os,subprocess,time
from scripts.tessfold_repair.common import *

def prepare():
 BASE.mkdir(exist_ok=True)
 if REG.exists():verify();return
 import copy,shutil
 from scripts.tessfold_system_common import verify as verify_parent,REG as PARENT_REG,BASE as PARENT_BASE
 reg=copy.deepcopy(verify_parent())
 for name in ['timing_rows.jsonl','multi_pages.json','cpu_equivalence.json','fp64_partition_audit.json']:
  if (PARENT_BASE/name).exists():shutil.copy2(PARENT_BASE/name,BASE/name)
 reg.update(registered_utc=now(),parent_registration=str(PARENT_REG.relative_to(ROOT)),parent_registration_sha256=sha(PARENT_REG),
  repair='Scoped real-ID restoration after HF embeddings prefill preparation; native UND source and all checkpoints unchanged.',
  source_sha256={p:sha(ROOT/p) for p in SOURCES},max_attempts=2)
 reg['data_sha256'].update({str((BASE/n).relative_to(ROOT)):sha(BASE/n) for n in ['timing_rows.jsonl','multi_pages.json']})
 reg['constraints']['optional_7B']='User20261115 authorizes newer similar-size Qwen as isolated optional extension; native UND regression remains Qwen3-1.7B. Never reuse old alignment weights as if compatible.'
 save(REG,reg);save(BASE/'state.json',dict(created=now(),stages={k:dict(status='registered',jobs=[]) for k in reg['stages']}))
 print('Registered zero-retraining system study; all historical science remains immutable.')
def queue():
 ss=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%R'],text=True)
 return {a[0]:dict(name=a[1],state=a[2],reason=a[3]) for l in ss.splitlines() if len(a:=l.split('|',3))==4}
def marker(stage):return BASE/('validation' if stage=='validate' else stage)/'complete.json'
def visible(p):
 for q in [BASE,p.parent]:
  try:list(q.iterdir())
  except FileNotFoundError:pass
 return p.exists()
def report(require=False):subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_repair.report']+(['--require-complete'] if require else []),cwd=ROOT,check=True)
def advance(reg,state):
 live=queue();active=0
 for stage,st in state['stages'].items():
  last=st['jobs'][-1] if st['jobs'] else None
  if last and last['job'] in live:st.update(status='active',scheduler=live[last['job']]);active+=1;continue
  if visible(marker(stage)):st['status']='complete';continue
  if last:
   st.pop('scheduler',None)
   if not last.get('exit_seen'):last['exit_seen']=now()
   if (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(last['exit_seen'])).total_seconds()<120:st['status']='waiting_commit';continue
   if len(st['jobs'])>=reg['max_attempts']:st['status']='failed';continue
  if stage!='validate' and state['stages']['validate']['status']=='failed':st['status']='blocked'
  else:st['status']='ready' if stage=='validate' or state['stages']['validate']['status']=='complete' else 'waiting'
 for stage,st in state['stages'].items():
  if active>=2:break
  if st['status']!='ready':continue
  cmd=['sbatch','--parsable','--job-name=tessfold-'+stage,'scripts/tessfold_repair/stage.sh',stage]
  job=subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0];st['jobs'].append(dict(job=job,time=now(),command=cmd));st.update(status='active');active+=1;save(BASE/'state.json',state);print(stage,job,flush=True)
 state['updated']=now();save(BASE/'state.json',state)
def control():
 with (BASE/'controller.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  reg=verify();save(BASE/'controller_runtime.json',dict(job=os.environ.get('SLURM_JOB_ID'),time=now()))
  while True:
   verify();state=load(BASE/'state.json');advance(reg,state);report()
   statuses=[v['status'] for v in state['stages'].values()]
   if all(s in ['complete','failed','blocked'] for s in statuses):
    if all(s=='complete' for s in statuses):report(True);save(BASE/'complete.json',dict(time=now(),no_retraining=True))
    else:save(BASE/'attention_required.json',state)
    return
   time.sleep(45)
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','resume','control','verify','status']);a=p.parse_args()
 if a.action=='prepare':prepare();report()
 elif a.action=='verify':verify();print('TessFold source/data/checkpoint signatures verified.')
 elif a.action=='resume':
  prepare()
  if any(v['name']=='tessfold-repair' for v in queue().values()):print('Existing controller retained');return
  job=subprocess.check_output(['sbatch','--parsable','scripts/tessfold_repair/controller.sh'],cwd=ROOT,text=True).strip().split(';')[0];save(BASE/'controller_submission.json',dict(job=job,time=now()));print(job)
 elif a.action=='control':control()
 else:report();print(json.dumps(load(BASE/'state.json'),ensure_ascii=False));print(json.dumps(queue()))
if __name__=='__main__':main()
