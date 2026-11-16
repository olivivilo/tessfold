"""Durable controller: complete24 trained/evaluated models + phase memory."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,time
from scripts.seed_memory_common import *
CPU=ROOT/'.venv/bin/python'

def prepare():
 BASE.mkdir(parents=True,exist_ok=True)
 if REG.exists():verify();return
 from scripts.column_tree_full_suite import verify as v1
 from scripts.column_tree_post_full_suite import verify as v2
 from scripts.supplement_common import verify as vs
 old=v1();v2();vs()
 oldmap={(r['kind'],r['ratio']):r for r in old['runs']}
 template=load(ROOT/oldmap[('col_adaptive',.25)]['config'])
 models=[]
 for ratio in RATIOS:
  for kind in KINDS:
   for seed in SEEDS:
    ident=model_id(kind,ratio,seed);prior=None
    if seed==SEEDS[0]:prior=oldmap[(kind,ratio)]
    elif kind=='quadtree_question' and ratio==.25:
     prefix=f'artifacts/compression_supplement_20261111'
     prior=dict(config=f'{prefix}/configs/full_T_r0.25_seed{seed}.json',output=f'{prefix}/runs/full_T_r0.25_seed{seed}')
    if prior:
     cfgpath=prior['config'];output=prior['output'];cfg=load(ROOT/cfgpath)
     assert (ROOT/output/'benchmark/complete.json').exists()
     assert cfg['seed']==seed and cfg['ratio']==ratio and cfg['insertion']=='pre-merger'
    else:
     cfgpath=str((BASE/'configs'/f'{ident}.json').relative_to(ROOT));output=str((BASE/'runs'/ident).relative_to(ROOT))
     cfg=dict(template,name=ident,kind=kind,ratio=ratio,seed=seed,output=output,
       initialization=f'fresh generator seed{seed}; no warm start',selection='Fixed three paired seeds/four keep ratios; endpoint epoch4; no benchmark-based selection')
     assert not (ROOT/output).exists();save(ROOT/cfgpath,cfg)
    checkpoint=output+'/epoch4.pt'
    models.append(dict(id=ident,kind=kind,ratio=ratio,seed=seed,reuse=prior is not None,
      config=cfgpath,config_sha256=sha(ROOT/cfgpath),output=output,checkpoint=checkpoint,
      checkpoint_sha256=sha(ROOT/checkpoint) if prior else None))
 formal=lines(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl')
 timing=lines(ROOT/'artifacts/compression_supplement_20261111/timing_rows.jsonl')
 assert len(formal)==23565 and len(timing)==90
 save_lines(BASE/'formal_rows.jsonl',formal);save_lines(BASE/'timing_rows.jsonl',timing)
 smoke=[]
 for task in sorted({r['task'] for r in formal}):smoke.append(next(r for r in formal if r['task']==task))
 save_lines(BASE/'smoke_rows.jsonl',smoke)
 stages=[dict(id='technical_smoke',phase='smoke',model=None,depends=[])]
 for m in models:
  if m['reuse']:continue
  stages.extend([dict(id='train_'+m['id'],phase='train',model=m['id'],depends=['technical_smoke']),
                 dict(id='eval_'+m['id'],phase='eval',model=m['id'],depends=['train_'+m['id']])])
 stages.append(dict(id='phase_memory',phase='profile',model=None,depends=['technical_smoke']+[s['id'] for s in stages if s['phase']=='eval']))
 paths=[template[k] for k in ['train_rows','validation_rows','audit_rows']]+[str((BASE/x).relative_to(ROOT)) for x in ['formal_rows.jsonl','timing_rows.jsonl','smoke_rows.jsonl']]
 for p in ['data/full/val.jsonl','data/full/test.jsonl']:paths.append(p)
 protocol=dict(insertion='pre-merger',freeze='entire official UND encoder/merger/Qwen',keep_ratios=RATIOS,seeds=SEEDS,
  train_QA=20000,train_images=16731,validation_QA=56,benchmark_QA=23565,epochs=4,updates=2500,world=8,batch=32,
  optimizer='AdamW',lr=3e-4,weight_decay=0,clip=1,warmup=250,scheduler='cosine to10%peak',loss='KL-only teacher||student on common teacher greedy prefixes T=1',
  checkpoint='fixedepoch4, no seed/checkpoint selection',evaluation='same full benchmark, greedy cap256; new runs trained variant only, prior initial/controls retained',
  sampling_interval_seconds=.001,profile_requests=90,profile_repeats=3,profile_modes=['natural_eos','fixed64'],
  profile_models=25,profile_records=13500,profile_phase_names=['processor','encoder','native_merger','compression','qwen_prefill','generation'],
  phase_note='encoder ends at native merger prehook; native merger measured separately. compression includes P generation+PH+compressed official merger. Existing full-resolution features/input lifetimes preserved; no cleanup optimization. Correct fixed question remains Qwen input.',
  averages='time-weighted trapezoid on actual allocator samples at nominal1ms; not exact kernel-time average; save actual gaps and traces',
  peaks='backend allocated/reserved high-water within synchronized phase; average across requests and maximum across requests both reported; baseline plus incremental peak; no inference speed claim from instrumented timings',
  memory_context='batch1, weights resident, no cross-request visual/KV cache; modules one-at-a-time, all24 checkpoints plus uncompressedREF',
  statistics='per-seed scores + mean/sample SD; paired same-seed T-C; task-stratified image bootstrap10000 for three-seed per-question mean differences, separate seed differences SD, exploratory viewed benchmark',max_stage_attempts=3,max_live_jobs=2)
 save(REG,dict(registered_utc=now(),models=models,stages=stages,protocol=protocol,
  source_sha256={p:sha(ROOT/p) for p in OLD_SOURCES+NEW_SOURCES},data_sha256={p:sha(ROOT/p) for p in paths},
  old_registrations_sha256={p:sha(ROOT/p) for p in ['artifacts/compression_lab_v2/column_tree_full_20261108/registration.json','artifacts/compression_supplement_20261111/registration.json']},
  new_full_trainings=14,reused_models=10,request='自适应列划分共同三seed四预算与问题树比较，分阶段设备内存峰值与平均'))
 save(BASE/'state.json',dict(created_utc=now(),stages={s['id']:dict(status='registered',jobs=[]) for s in stages}))
 print('Registered24models:10reused+14fresh;30newstages incl technical smoke+memory.')

def queue():
 ss=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%R'],text=True)
 return {a[0]:dict(name=a[1],state=a[2],reason=a[3]) for line in ss.splitlines() if len(a:=line.split('|',3))==4}
def marker(stage,reg):
 if stage['phase']=='smoke':return BASE/'smoke/complete.json'
 if stage['phase']=='profile':return BASE/'memory/complete.json'
 m=next(m for m in reg['models'] if m['id']==stage['model'])
 return ROOT/m['output']/('complete.json' if stage['phase']=='train' else 'benchmark/complete.json')
def report(require=False):
 subprocess.run([str(CPU),'-m','scripts.report_seed_memory']+(['--require-complete'] if require else []),cwd=ROOT,check=True)
def advance(reg,state):
 live=queue();active=0
 for s in reg['stages']:
  st=state['stages'][s['id']];last=st['jobs'][-1] if st['jobs'] else None
  if last and last['job'] in live:st.update(status='active',scheduler=live[last['job']]);active+=1;continue
  if marker(s,reg).exists():st.update(status='complete',scheduler=None);continue
  if last:
   if not last.get('exit_seen'):last['exit_seen']=now()
   if (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(last['exit_seen'])).total_seconds()<90:st['status']='awaiting_artifacts';continue
   if len(st['jobs'])>=3:st.update(status='failed',error='No completion after3 attempts; inspect preserved stage logs.');continue
  if any(state['stages'][d]['status'] in ['failed','blocked'] for d in s['depends']):st['status']='blocked'
  else:st['status']='ready' if all(state['stages'][d]['status']=='complete' for d in s['depends']) else 'waiting'
 eligible=[s for s in reg['stages'] if state['stages'][s['id']]['status']=='ready']
 eligible.sort(key=lambda s: {'smoke':0,'eval':1,'train':2,'profile':3}[s['phase']])
 for s in eligible:
  if active>=2:break
  cmd=['sbatch','--parsable','--job-name=ctsm-'+s['phase'],'scripts/slurm_seed_memory_stage.sh',s['id']]
  job=subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0]
  st=state['stages'][s['id']];st['jobs'].append(dict(job=job,submitted=now(),command=cmd));st['status']='active';active+=1
  save(BASE/'state.json',state);print(json.dumps(dict(stage=s['id'],job=job)),flush=True)
 state['updated_utc']=now();save(BASE/'state.json',state)
def control():
 with (BASE/'controller.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  save(BASE/'controller_runtime.json',dict(job=os.environ.get('SLURM_JOB_ID'),started=now()))
  reg=verify()
  while True:
   verify();state=load(BASE/'state.json');advance(reg,state);report()
   statuses=[s['status'] for s in state['stages'].values()]
   if all(x in ['complete','failed','blocked'] for x in statuses):
    if all(x=='complete' for x in statuses):report(True);save(BASE/'complete.json',dict(completed_utc=now(),models=24,new_trainings=14,profile_records=13500))
    else:save(BASE/'attention_required.json',dict(time=now(),state=state))
    return
   time.sleep(45)
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','resume','status','verify','control']);a=p.parse_args()
 if a.action=='prepare':prepare();report()
 elif a.action=='verify':verify();print('New suite sources/data/config/prior checkpoint hashes verified.')
 elif a.action=='resume':
  prepare()
  if any(j['name']=='ct-seed-memory' for j in queue().values()):print('Existing controller retained.');return
  job=subprocess.check_output(['sbatch','--parsable','scripts/slurm_seed_memory_controller.sh'],cwd=ROOT,text=True).strip().split(';')[0];save(BASE/'controller_submission.json',dict(job=job,time=now()));print(job)
 elif a.action=='control':control()
 else:report();print(json.dumps(load(BASE/'state.json'),ensure_ascii=False,indent=2));print(json.dumps(queue(),ensure_ascii=False))
if __name__=='__main__':main()
