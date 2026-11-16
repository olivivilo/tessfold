"""Frozen no-retraining validation→single-request/multi-query efficiency suite."""
import argparse,collections,fcntl,os,subprocess,time
from scripts.tessfold_system_common import *

def prepare():
 BASE.mkdir(exist_ok=True)
 if REG.exists():verify();return
 from scripts.column_tree_full_suite import verify as a
 from scripts.column_tree_post_full_suite import verify as b
 from scripts.supplement_common import verify as c
 from scripts.seed_memory_common import verify as d
 old=a();b();c();d()
 models=[]
 for r in old['runs']:
  checkpoint=r['output']+'/epoch4.pt';models.append(dict(id=r['name'],kind=r['kind'],ratio=r['ratio'],output=r['output'],checkpoint=checkpoint,checkpoint_sha256=sha(ROOT/checkpoint)))
 models.sort(key=lambda m:(m['ratio'],m['kind']))
 timing=lines(ROOT/'artifacts/compression_supplement_20261111/timing_rows.jsonl')
 with (BASE/'timing_rows.jsonl').open('w') as f:
  for r in timing:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 formal=lines(ROOT/'artifacts/compression_supplement_20261111/same_page_rows.jsonl');pages=collections.defaultdict(list)
 for r in formal:pages[r['document_id']].append(r)
 tokens={r['document_id']:r['tokens'] for r in records(ROOT/models[0]['output']/'benchmark')}
 eligible={doc:rr for doc,rr in pages.items() if len(rr)>=10};selected=[]
 for task in sorted({rr[0]['task'] for rr in eligible.values()}):
  pool=[(doc,rr) for doc,rr in eligible.items() if rr[0]['task']==task]
  pool.sort(key=lambda x:hashlib.sha256(('tessfold-multi:'+x[0]).encode()).hexdigest())
  # One OCRBench page exists; distribute remainder in fixed hash order.
  selected+=pool[:min(6,len(pool))]
 used={x[0] for x in selected}
 extras=sorted([(doc,rr) for doc,rr in eligible.items() if doc not in used],key=lambda x:hashlib.sha256(('tessfold-multi:'+x[0]).encode()).hexdigest())
 selected=(selected+extras)[:24];assert len(selected)==24
 save(BASE/'multi_pages.json',[dict(document_id=doc,task=rr[0]['task'],questions=[dict(r,original_tokens=tokens[doc]) for r in sorted(rr,key=lambda r:r['id'])[:10]]) for doc,rr in selected])
 data=['artifacts/compression_lab_v2/validation.jsonl','artifacts/compression_supplement_20261111/same_page_rows.jsonl',str((BASE/'timing_rows.jsonl').relative_to(ROOT)),str((BASE/'multi_pages.json').relative_to(ROOT))]
 reg=dict(registered_utc=now(),models=models,source_sha256={p:sha(ROOT/p) for p in SOURCES},data_sha256={p:sha(ROOT/p) for p in data},
  constraints=dict(no_retraining=True,weights='fixed originalseed20261105 epoch4; four budgets2methods',bit_exact='Only strict_dense adopted; sparse and float64 partition are experimental unpromoted candidates. Validate448fixedval examples against original tensor+saved generation. Clean pipeline same tokens gate.',
   validation_requests=448,single_requests=9450,single_latency_requests=8100,single_memory_requests=1350,multi_pages=24,multi_queries_per_page=10,multi_repeats=3,multi_k=[1,2,5,10],
   multi_selection='24deterministic actual pages with>=10unique questions; only DocVQA/InfoVQA/WTQ/OCRBench qualify; not all3970pages, no repeated synthetic question cycling',
   methods=['REF','T_DENSE','T_STRICT','C_DENSE','C_STRICT','T_HARD','R','S'],rates=[.25,.5],
   single_modes=['natural_eos','fixed64'],latency_repeats=3,memory_separate_pass=True,
   main='Report actual measured outcomes without1ms/4xspeed targets. Full model receives same visual feature/prefix caching; clone cost counted; question-dependent Tree cannot share per-question compressed prefix.',
   gpu='ActualAscend910_9382; torch.npu.Event and synchronized wall, not untested CUDA code',
   optional_7B='Not run; would change backbone and violate the zero-retraining established-weight scope. No borrowed projected speedup.',
   limitations='FP32associativity/FP64cut changes can violate bitexact; long-stratum and cache superiority not assumed. No paperTeX source exists, generate separate evidence-based table/snippet; do not delete historical negative results.'),
  stages=['validate','single','multi'],max_attempts=3)
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
def report(require=False):subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.report_tessfold_system']+(['--require-complete'] if require else []),cwd=ROOT,check=True)
def advance(reg,state):
 live=queue();active=0
 for stage,st in state['stages'].items():
  last=st['jobs'][-1] if st['jobs'] else None
  if last and last['job'] in live:st.update(status='active',scheduler=live[last['job']]);active+=1;continue
  if visible(marker(stage)):st['status']='complete';continue
  if last:
   if not last.get('exit_seen'):last['exit_seen']=now()
   if (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(last['exit_seen'])).total_seconds()<120:st['status']='waiting_commit';continue
   if len(st['jobs'])>=3:st['status']='failed';continue
  if stage!='validate' and state['stages']['validate']['status']=='failed':st['status']='blocked'
  else:st['status']='ready' if stage=='validate' or state['stages']['validate']['status']=='complete' else 'waiting'
 for stage,st in state['stages'].items():
  if active>=2:break
  if st['status']!='ready':continue
  cmd=['sbatch','--parsable','--job-name=tessfold-'+stage,'scripts/slurm_tessfold_system_stage.sh',stage]
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
  if any(v['name']=='tessfold-system' for v in queue().values()):print('Existing controller retained');return
  job=subprocess.check_output(['sbatch','--parsable','scripts/slurm_tessfold_system_controller.sh'],cwd=ROOT,text=True).strip().split(';')[0];save(BASE/'controller_submission.json',dict(job=job,time=now()));print(job)
 elif a.action=='control':control()
 else:report();print(json.dumps(load(BASE/'state.json'),ensure_ascii=False));print(json.dumps(queue()))
if __name__=='__main__':main()
