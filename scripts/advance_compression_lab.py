"""Idempotent stage advances. Chooses per budget only after ALL registered pilots finish."""
import argparse,fcntl,json,subprocess,os
from pathlib import Path
from scripts.report_compression_lab import main as report,load,read,ROOT,BASE
from scripts.report_meta_compress_strict import paired

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def submit(script,*args):
 result=subprocess.check_output(['sbatch','--parsable',script,*args],text=True,cwd=ROOT).strip();return result.split(';')[0]
def collect_lists(names):return [p for name in names for p in load(BASE/name)]
def gate(cfg):
 out=ROOT/cfg['output'];a=load(out/'validation.initial.summary.json');b=load(out/f'validation.epoch{cfg["epochs"]}.summary.json');ta=load(out/'train_audit.initial.summary.json');tb=load(out/f'train_audit.epoch{cfg["epochs"]}.summary.json')
 folder='baselines_pre' if cfg.get('insertion')=='pre-merger' else 'baselines'
 controls=[load(BASE/position_folder/f'{kind}_r{cfg["ratio"]:g}'/'validation.fixed.summary.json').get('student_macro') for position_folder in ['baselines','baselines_pre'] for kind in ['mean','selection']]
 if not all([a,b,ta,tb]) or None in controls:return {'pass':False,'reason':'incomplete'}
 reason=[]
 if b['kl']>a['kl']*.95:reason.append('validation KL decrease <5%')
 if tb['kl']>ta['kl']*.95:reason.append('fixed train-audit KL decrease <5%')
 if b['student_macro']<a['student_macro']+2:reason.append('score gain <2points')
 if b['student_macro']<=max(controls):reason.append('not above matched fixed controls')
 return {'pass':not reason,'reasons':reason,'initial':a,'final':b,'fixed_controls':controls}
def main():
 p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
 with (BASE/'advance.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);report();summary=load(BASE/'summary.json');state=load(BASE/'stages.json') or {'ratios':{}}
  expandedpaths=load(BASE/'expanded_configs.json') or []
  for ratio in [.1,.25,.5]:
   key=f'{ratio:g}';rr=[r for r in summary['records'] if r['ratio']==ratio and r['phase'] in ['initial_screen','higher_update_budget','pre_merger_ablation']]
   record=state['ratios'].setdefault(key,{})
   if not record.get('selection_frozen'):
    if not all(r['status']=='完成' for r in rr):continue
    eligible=[r for r in rr if r['screen_gate']]
    record['selection_frozen']=True
    if not eligible:record['stage']='no_pilot_pass';continue
    chosen=max(eligible,key=lambda r:(r['final_score'],r['kl_drop_percent'],r['name']))
    record['selected_pilot']=chosen;record['expanded_configs']=[];record['stage']='expanded_registered'
    position='pre-merger' if chosen['phase']=='pre_merger_ablation' else 'post-merger'
    for seed in [20261105,20261106]:
     name=f'expanded_{chosen["kind"]}_r{ratio:g}_seed{seed}'
     cfg=dict(name=name,kind=chosen['kind'],ratio=ratio,seed=seed,epochs=8,lr=.0003,clip=1.,batch_size=32,insertion=position,phase='expanded_672_images',train_rows='artifacts/compression_lab/expanded_train.jsonl',validation_rows='artifacts/compression_lab/validation.jsonl',output=f'artifacts/compression_lab/expanded/{name}',initialization='fresh generator; no pilot checkpoint warm start',sampler='shuffle + full global batch cyclic padding')
     path=BASE/'configs'/f'{name}.json';save(path,cfg);rel=str(path.relative_to(ROOT));expandedpaths.append(rel);record['expanded_configs'].append(rel)
    save(BASE/'expanded_configs.json',expandedpaths)
   if record.get('stage')=='expanded_registered' and a.submit:
    record['expanded_jobs']=[submit('scripts/slurm_compression_lab_distributed.sh',path) for path in record['expanded_configs']];record['stage']='expanded_running'
   if record.get('stage')=='expanded_running':
    cfgs=[load(ROOT/p) for p in record['expanded_configs']]
    if not all((ROOT/c['output']/'complete.json').exists() for c in cfgs):continue
    gates=[gate(c) for c in cfgs];record['expanded_gates']=gates
    if not all(g['pass'] for g in gates):record['stage']='expanded_failed_no_large_training';continue
    cases=[]
    for cfg in cfgs:
     for stage,checkpoint in [('initial','initial.pt'),('epoch8','epoch8.pt')]:cases.append({'checkpoint':cfg['output']+'/'+checkpoint,'output':cfg['output'],'rows':'artifacts/compression_lab/confirmation.jsonl','stage':stage,'cohort':'confirmation'})
    for insertion in ['pre-merger','post-merger']:
     for kind in ['mean','selection']:
      folder=f'artifacts/compression_lab/confirmation_baselines/{insertion}_r{ratio:g}/{kind}'
      cases.append({'config':{'kind':kind,'ratio':ratio,'insertion':insertion},'output':folder,'rows':'artifacts/compression_lab/confirmation.jsonl','stage':'fixed','cohort':'confirmation'})
    path=BASE/f'confirmation_suite_r{ratio:g}.json';save(path,cases);record['confirmation_list']=str(path.relative_to(ROOT));record['stage']='confirmation_registered'
   if record.get('stage')=='confirmation_registered' and a.submit:
    record['confirmation_job']=submit('scripts/slurm_compression_lab_suite.sh',record['confirmation_list']);record['stage']='confirmation_running'
   if record.get('stage')=='confirmation_running':
    cases=load(ROOT/record['confirmation_list'])
    if not all((ROOT/c['output']/f'confirmation.{c["stage"]}.summary.json').exists() for c in cases):continue
    control=[load(ROOT/c['output']/'confirmation.fixed.summary.json')['student_macro'] for c in cases if 'checkpoint' not in c];checks=[]
    for cfgpath in record['expanded_configs']:
     cfg=load(ROOT/cfgpath);out=ROOT/cfg['output'];initial=read(out/'confirmation.initial.jsonl');final=read(out/'confirmation.epoch8.jsonl');u=paired(initial,final)
     score=load(out/'confirmation.epoch8.summary.json')['student_macro'];checks.append({'seed':cfg['seed'],'score':score,'paired_vs_initial':u,'fixed_controls':control,'pass':u['cluster_bootstrap_95ci_points'][0]>0 and score>max(control)})
    record['confirmation_checks']=checks;record['stage']='confirmation_passed_ready_full' if all(c['pass'] for c in checks) else 'confirmation_failed_no_large_training'
  fullpaths=load(BASE/'full_configs.json') or []
  for key,record in state['ratios'].items():
   if record.get('stage')=='confirmation_passed_ready_full':
    ratio=float(key);parent=load(ROOT/record['expanded_configs'][0])
    rows=read(ROOT/'artifacts/meta_compress_search/full_train.jsonl')
    heldout={r['document_id'] for name in ['validation','confirmation'] for r in read(BASE/f'{name}.jsonl')}
    assert not heldout & {r['document_id'] for r in rows}
    target=BASE/'full_train.jsonl'
    if not target.exists():target.write_bytes((ROOT/'artifacts/meta_compress_search/full_train.jsonl').read_bytes())
    name=f'full_{parent["kind"]}_r{ratio:g}'
    cfg={**parent,'name':name,'phase':'full_20k','seed':20261105,'epochs':4,'lr':.0003,'batch_size':32,'train_rows':'artifacts/compression_lab/full_train.jsonl','output':f'artifacts/compression_lab/full/{name}','selection':'first preregistered seed, final epoch4; no benchmark selection'}
    path=BASE/'configs'/f'{name}.json';save(path,cfg);rel=str(path.relative_to(ROOT));record['full_config']=rel
    if rel not in fullpaths:fullpaths.append(rel)
    save(BASE/'full_configs.json',fullpaths);record['stage']='full_registered'
   if record.get('stage')=='full_registered' and a.submit:
    record['full_job']=submit('scripts/slurm_compression_lab_distributed.sh',record['full_config']);record['stage']='full_running'
   if record.get('stage')=='full_running':
    cfg=load(ROOT/record['full_config'])
    if (ROOT/cfg['output']/'complete.json').exists():record['full_gate_diagnostic']=gate(cfg);record['stage']='benchmark_registered'
   if record.get('stage')=='benchmark_registered' and a.submit:
    record['benchmark_job']=submit('scripts/slurm_compression_lab_benchmark.sh',record['full_config']);record['stage']='benchmark_running'
   if record.get('stage')=='benchmark_running':
    cfg=load(ROOT/record['full_config']);out=ROOT/cfg['output']/'benchmark'
    if (out/'complete.json').exists():record['benchmark_summary']=load(out/'summary.json');record['stage']='benchmark_complete'
  save(BASE/'stages.json',state)
  print(json.dumps(state,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
