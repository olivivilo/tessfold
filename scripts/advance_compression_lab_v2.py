"""Advance v2 gates without revisiting confirmation for architecture selection."""
import argparse,fcntl,json,subprocess
from scripts.report_compression_lab_v2 import ROOT,BASE,load,save,gate,main as report,controls,refresh
from scripts.train_compression_lab_v2 import read
from scripts.report_meta_compress_strict import paired
from scripts.compression_lab_v2_jobs import active
TERMINAL={'no_pilot_pass','expanded_failed_no_large_training','confirmation_failed_no_large_training','benchmark_complete'}
def submit(script,*args):
 return subprocess.check_output(['sbatch','--parsable',script,*args],cwd=ROOT,text=True).strip().split(';')[0]
def submit_full(script,config):
 # Hardware-only repartition: global batch, data order, objective and fixed epochs stay unchanged.
 cfg=load(ROOT/config);workers=4 if 'benchmark' in script or cfg['ratio']==.25 else 8
 if 'benchmark' in script:
  workers=load(ROOT/cfg['output']/'benchmark_hardware.json').get('world',workers)
 return subprocess.check_output(['sbatch','--parsable',f'--gres=npu:{workers}',f'--cpus-per-task={workers*8}',
  f'--mem={workers*32}G',f'--export=ALL,V2_NPU_WORKERS={workers},V2_TRAIN_MODULE=scripts.compression_lab_v2_io_retry',script,config],cwd=ROOT,text=True).strip().split(';')[0]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args()
 with (BASE/'advance.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);report();summary=load(BASE/'summary.json');state=load(BASE/'stages.json') or {'ratios':{}}
  expanded=load(BASE/'expanded_configs.json') or []
  for ratio in [.1,.25,.5]:
   key=f'{ratio:g}';record=state['ratios'].setdefault(key,{'stage':'screen_running'})
   pilots=[r for r in summary['records'] if r['phase']=='v2_screen' and r['ratio']==ratio]
   if not record.get('selection_frozen'):
    if len(pilots)!=48 or not all(r['status']=='Completed' for r in pilots):continue
    eligible=[r for r in pilots if r['gate']];record['selection_frozen']=True
    if not eligible:record['stage']='no_pilot_pass';continue
    chosen=max(eligible,key=lambda r:(r['final_score'],r['kl_drop_percent'],r['name']))
    record['selected_pilot']=chosen;record['expanded_configs']=[];record['stage']='expanded_registered'
    for seed in [20261105,20261106]:
     name=f'expanded_{chosen["kind"]}_{chosen["insertion"]}_r{ratio:g}_seed{seed}'
     cfg=dict(name=name,kind=chosen['kind'],insertion=chosen['insertion'],ratio=ratio,seed=seed,
      epochs=8,lr=.0003,clip=1.,batch_size=32,phase='v2_expanded_672',generation_epochs=list(range(9)),
      train_rows='artifacts/compression_lab_v2/expanded_train.jsonl',validation_rows='artifacts/compression_lab_v2/validation.jsonl',
      audit_rows='artifacts/compression_lab_v2/train_audit.jsonl',output=f'artifacts/compression_lab_v2/expanded/{name}',initialization='fresh generator; no pilot warm start')
     path=BASE/'configs'/f'{name}.json';save(path,cfg);rel=str(path.relative_to(ROOT));expanded.append(rel);record['expanded_configs'].append(rel)
    save(BASE/'expanded_configs.json',expanded)
   if record.get('stage')=='expanded_registered' and args.submit:
    record['expanded_jobs']=[submit('scripts/slurm_compression_lab_v2_distributed.sh',p) for p in record['expanded_configs']];record['stage']='expanded_running'
   if record.get('stage')=='expanded_running':
    configs=[load(ROOT/p) for p in record['expanded_configs']]
    if not all((ROOT/c['output']/'complete.json').exists() for c in configs):continue
    gg=[gate(c) for c in configs];record['expanded_gates']=gg
    if not all(g['pass'] for g in gg):record['stage']='expanded_failed_no_large_training';continue
    cases=[]
    for cfg in configs:
     for stage,checkpoint in [('initial','initial.pt'),('epoch8','epoch8.pt')]:
      cases.append(dict(checkpoint=cfg['output']+'/'+checkpoint,output=cfg['output'],rows='artifacts/compression_lab_v2/confirmation.jsonl',stage=stage,cohort='confirmation'))
    for position in ['pre-merger','post-merger']:
     for kind in ['mean','selection','cell_mean','cell_selection']:
      cases.append(dict(config=dict(kind=kind,ratio=ratio,insertion=position),
       output=f'artifacts/compression_lab_v2/confirmation_baselines/{position}_{kind}_r{ratio:g}',
       rows='artifacts/compression_lab_v2/confirmation.jsonl',stage='fixed',cohort='confirmation'))
    suite=BASE/f'confirmation_suite_r{ratio:g}.json';save(suite,cases);record['confirmation_list']=str(suite.relative_to(ROOT));record['stage']='confirmation_registered'
   if record.get('stage')=='confirmation_registered' and args.submit:
    record['confirmation_job']=subprocess.check_output(['sbatch','--parsable','--job-name=v2-confirm','--gres=npu:4','--cpus-per-task=32','--mem=128G',
     'scripts/slurm_compression_lab_v2.sh','suite','4',record['confirmation_list']],cwd=ROOT,text=True).strip().split(';')[0];record['stage']='confirmation_running'
   if record.get('stage')=='confirmation_running':
    cases=load(ROOT/record['confirmation_list'])
    for case in cases:refresh(ROOT/case['output'])
    if not all((ROOT/c['output']/f'confirmation.{c["stage"]}.summary.json').exists() for c in cases):continue
    cc=controls(ratio,'confirmation');checks=[]
    for path in record['expanded_configs']:
     cfg=load(ROOT/path);out=ROOT/cfg['output'];u=paired(read(out/'confirmation.initial.jsonl'),read(out/'confirmation.epoch8.jsonl'))
     measured=load(out/'confirmation.epoch8.summary.json');score=measured['student_macro']
     initial_score=load(out/'confirmation.initial.summary.json')['student_macro']
     checks.append(dict(seed=cfg['seed'],score=score,initial_score=initial_score,teacher_score=measured['teacher_macro'],paired_vs_initial=u,fixed_controls=cc,**{'pass':u['cluster_bootstrap_95ci_points'][0]>0 and score>max(cc)}))
    record['confirmation_checks']=checks
    record['stage']='confirmation_passed_ready_full' if all(c['pass'] for c in checks) else 'confirmation_failed_no_large_training'
  fullpaths=load(BASE/'full_configs.json') or []
  for key,record in state['ratios'].items():
   if record.get('stage')=='confirmation_passed_ready_full':
    parent=load(ROOT/record['expanded_configs'][0]);ratio=float(key)
    rows=read(ROOT/'artifacts/meta_compress_search/full_train.jsonl')
    heldout={r['document_id'] for name in ['validation','confirmation'] for r in read(BASE/f'{name}.jsonl')}
    assert not heldout&{r['document_id'] for r in rows}
    target=BASE/'full_train.jsonl'
    if not target.exists():target.write_bytes((ROOT/'artifacts/meta_compress_search/full_train.jsonl').read_bytes())
    name=f'full_{parent["kind"]}_{parent["insertion"]}_r{ratio:g}'
    cfg={**parent,'name':name,'phase':'v2_full_20k','seed':20261105,'epochs':4,'generation_epochs':list(range(5)),
     'lr':.0003,'batch_size':32,'train_rows':'artifacts/compression_lab_v2/full_train.jsonl',
     'output':f'artifacts/compression_lab_v2/full/{name}',
     'selection':'fresh init; preregistered first seed; fixed epoch4; benchmark never used for selection'}
    if parent['kind']=='sparse_relu':cfg['implementation_version']='sparse-row-renormalized-v1'
    path=BASE/'configs'/f'{name}.json';save(path,cfg);rel=str(path.relative_to(ROOT));record['full_config']=rel
    if rel not in fullpaths:fullpaths.append(rel)
    save(BASE/'full_configs.json',fullpaths);record['stage']='full_registered'
   if record.get('stage')=='full_registered' and args.submit:
    record['full_job']=submit_full('scripts/slurm_compression_lab_v2_distributed.sh',record['full_config']);record['stage']='full_running'
   if record.get('stage')=='full_running':
    cfg=load(ROOT/record['full_config'])
    if (ROOT/cfg['output']/'complete.json').exists():record['full_gate_diagnostic']=gate(cfg);record['stage']='benchmark_registered'
   if record.get('stage')=='benchmark_registered' and args.submit:
    record['benchmark_job']=submit_full('scripts/slurm_compression_lab_v2_benchmark.sh',record['full_config']);record['stage']='benchmark_running'
   if record.get('stage')=='benchmark_running':
    cfg=load(ROOT/record['full_config']);out=ROOT/cfg['output']/'benchmark'
    refresh(out)
    if (out/'complete.json').exists():record['benchmark_summary']=load(out/'summary.json');record['stage']='benchmark_complete'
  # Commit stage transitions before optional cache housekeeping can fail.
  save(BASE/'stages.json',state)
  prefetch=load(BASE/'infrastructure/full_cache_prefetch_submission.json')
  if prefetch and not prefetch.get('cancelled_when_cache_ready'):
   ready=any((ROOT/load(ROOT/r['full_config'])['output']/'epoch1.pt').exists()
             for r in state['ratios'].values() if r.get('full_config'))
   if ready:
    job=prefetch['job']
    if active(job):subprocess.run(['scancel',job],check=True)
    prefetch['cancelled_when_cache_ready']=True;prefetch['reason']='A complete first training epoch has already materialized every original20k teacher cache entry.'
    save(BASE/'infrastructure/full_cache_prefetch_submission.json',prefetch)
  save(BASE/'stages.json',state);print(json.dumps(state,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
