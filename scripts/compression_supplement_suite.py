"""Durable, source-frozen controller for feedback tables T1--T6."""
from __future__ import annotations
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.supplement_common import ROOT,BASE,REG,SOURCES,load,save,lines,save_lines,sha,now,verify,benchmark_rows,legacy,teacher_records


def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    if REG.exists():verify();return
    from scripts.column_tree_full_suite import verify as a
    from scripts.column_tree_post_full_suite import verify as b
    a();b()
    template=load(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108/configs/full_quadtree_question_pre-merger_r0.25.json')
    cohorts={k:lines(ROOT/template[k]) for k in ['train_rows','validation_rows','audit_rows']}
    formal=benchmark_rows();assert len(formal)==23565
    from vqa.metrics import normalize
    pages=collections.defaultdict(list)
    for r in formal:pages[r['image']].append(r)
    mapping={};swap=[]
    for image,rr in sorted(pages.items()):
        distinct={}
        for r in sorted(rr,key=lambda x:x['id']):distinct.setdefault(normalize(r['question']),r)
        rr=sorted(distinct.values(),key=lambda r:r['id'])
        if len(rr)<2:continue
        for i,r in enumerate(rr):mapping[r['id']]=rr[(i+1)%len(rr)]['id'];swap.append(r)
    save(BASE/'same_page_mapping.json',mapping);save_lines(BASE/'same_page_rows.jsonl',swap)
    smoke=[]
    for task in sorted({r['task'] for r in formal}):smoke+=sorted([r for r in formal if r['task']==task],key=lambda r:r['id'])[:2]
    save_lines(BASE/'smoke_rows.jsonl',smoke)
    # Development is restricted to the historical full training set, never formal.
    dev=[]
    for task in sorted({r['task'] for r in cohorts['train_rows']}):
        rr=sorted([r for r in cohorts['train_rows'] if r['task']==task],key=lambda r:sha_text('dev20261111:'+r['id']))
        dev+=rr[:32]
    assert len(dev)==224
    save_lines(BASE/'development_train.jsonl',dev)
    save_lines(BASE/'formal_rows.jsonl',formal)
    teacher=teacher_records();ordered=sorted(formal,key=lambda r:(teacher[r['id']]['tokens'],r['id']))
    timing=[]
    for start,end in [(0,len(ordered)//3),(len(ordered)//3,2*len(ordered)//3),(2*len(ordered)//3,len(ordered))]:
        rr=sorted(ordered[start:end],key=lambda r:sha_text('timing20261111:'+r['id']))[:30]
        timing += [dict(r,original_tokens=teacher[r['id']]['tokens']) for r in rr]
    save_lines(BASE/'timing_rows.jsonl',timing)
    records=[]
    def add(name,phase,cfg,deps=(),category='formal'):
        cfg=dict(cfg,name=name,output=f'artifacts/compression_supplement_20261111/runs/{name}')
        path=BASE/'configs'/f'{name}.json';save(path,cfg)
        records.append(dict(name=name,phase=phase,category=category,depends=list(deps),config=str(path.relative_to(ROOT)),config_sha256=sha(path),output=cfg['output']))
        return name,cfg
    base=dict(template);base.pop('kind',None);base.pop('selection',None)
    base.update(phase='feedback_supplement',generation_epochs=[0,1,2,3,4],optimizer='adamw',schedule='cosine',optimizer_policy='fixed',checkpoint_rule='fixed epoch4, never formal-selected')
    add('technical_smoke','smoke',dict(method='smoke',ratio=.25),category='technical')
    dev_names=[]
    settings=[dict(candidate='sgd_author',optimizer='sgd',lr=.001,clip=.01,schedule='constant'),
              dict(candidate='adamw_reference',optimizer='adamw',lr=.0003,clip=1.,schedule='cosine'),
              dict(candidate='adamw_conservative',optimizer='adamw',lr=.00003,clip=1.,schedule='cosine')]
    for method in ['M','T']:
        for setting in settings:
            cfg=dict(base,method=method,ratio=.25,insertion='post-merger' if method=='M' else 'pre-merger',
                     train_rows='artifacts/compression_supplement_20261111/development_train.jsonl',max_steps=100,epochs=1,**setting)
            name,_=add(f'dev_{method}_{setting["candidate"]}','train',cfg,['technical_smoke'],'development');dev_names.append(name)
    train=[]
    for method,ratio,seed in [('M',.25,20261105),('M',.5,20261105),('V',.25,20261105),('J',.25,20261105),
                              ('T',.25,20261106),('T',.25,20261107),('M',.25,20261106),('M',.25,20261107)]:
        name=f'full_{method}_r{ratio:g}_seed{seed}'
        cfg=dict(base,method=method,ratio=ratio,seed=seed,insertion='post-merger' if method=='M' else 'pre-merger',
                 optimizer_policy='development_selected' if method=='M' else 'fixed')
        _,cfg=add(name,'train',cfg,['technical_smoke']+ (dev_names if method=='M' else []),'full_training');train.append((name,cfg))
    # Cheap interventions reuse frozen historical T; full cohort, no seed averaging.
    for method,variants in [('HA',['H','A']),('QSWAP',['QQ','QW','WQ','WW'])]:
        cfg=dict(method=method,ratio=.25,seed=20261105,insertion='pre-merger',variants=variants,
                 checkpoint=legacy('quadtree_question',.25)+'/epoch4.pt',generator_config={'method':'T'})
        if method=='QSWAP':cfg['eval_rows']='artifacts/compression_supplement_20261111/same_page_rows.jsonl'
        add(method+'_r0.25','eval',cfg,['technical_smoke'])
    free_configs=[]
    for method in ['R','Gstar','S']:
        for ratio in [.25,.5]:
            cfg=dict(method=method,ratio=ratio,seed=20261105,insertion='pre-merger',variants=['trained'])
            if method=='Gstar':cfg.update(checkpoint=legacy('quadtree_question',ratio)+'/epoch4.pt',generator_config={'method':'T'})
            _,cfg=add(f'{method}_r{ratio:g}','eval',cfg,['technical_smoke']);free_configs.append(cfg)
    for name,cfg in train:
        evaluation=dict(cfg,checkpoint=cfg['output']+'/epoch4.pt',generator_config=cfg,variants=['trained'])
        # Separate benchmark config shares the run directory, not its phase metadata.
        add('eval_'+name,'eval',dict(evaluation,output=cfg['output']),[name])
        records[-1]['output']=cfg['output']
        path=ROOT/records[-1]['config'];evaluation.update(name='eval_'+name)
        save(path,evaluation);records[-1]['config_sha256']=sha(path)
    m25=next((name,cfg) for name,cfg in train if cfg['method']=='M' and cfg['ratio']==.25 and cfg['seed']==20261105)
    timing_methods=[dict(method='REF',ratio=1.),dict(method='T',ratio=.25,checkpoint=legacy('quadtree_question',.25)+'/epoch4.pt'),
                    dict(method='C',ratio=.25,checkpoint=legacy('col_adaptive',.25)+'/epoch4.pt'),
                    dict(method='M',ratio=.25,checkpoint=m25[1]['output']+'/epoch4.pt')]
    timing_methods += [dict(c) for c in free_configs if c['ratio']==.25]
    save(BASE/'timing_methods.json',timing_methods)
    add('efficiency','efficiency',dict(method='timing',ratio=.25),['technical_smoke',m25[0]])
    dataset=[template[k] for k in ['train_rows','validation_rows','audit_rows']]+[str((BASE/p).relative_to(ROOT)) for p in ['formal_rows.jsonl','smoke_rows.jsonl','same_page_rows.jsonl','same_page_mapping.json','development_train.jsonl','timing_rows.jsonl','timing_methods.json']]
    registration=dict(created_utc=now(),runs=records,source_sha256={p:sha(ROOT/p) for p in SOURCES},data_sha256={p:sha(ROOT/p) for p in dataset},
                      benchmark_rows='artifacts/compression_supplement_20261111/formal_rows.jsonl',
                      full_training_runs=8,development_runs=6,development_updates_per_run=100,
                      protocol=dict(primary='25% T_pre minus M_post, matched seeds20261105/06/07',secondary='50% and single-seed same-position TP_post minus M_post',
                                    frozen_backbone=True,epochs=4,updates=2500,batch=32,world=8,formal_cap=256,teacher_cap=64,
                                    metadata='feedback/experiment_runbook.docx',budget='max(1,min(N,round(r*N))) Python ties-to-even',
                                    intervention_order='For each fixed partition, order by TRUE question representatives. QQ/QW share Q order; WQ/WW share Q order on W partition.',
                                    gridstar='fixed4x4 clipped to legal grid and K; rho region mean softmax allocation, min1 and capped largest remainder; s local topk; not original GridPrune',
                                    s2='official allocation.py pinned a919a8d7f1c139dd054631c34779d1fdbd8dbeed; UND original preprocessing/block order/1D RoPE; grid side min(8 for25%,9 for50%,h,w,isqrtK)',
                                    development_selection='3 identical optimizer trials each for M and T, 224 train QA + fixed56 validation,100updates; choose M by endpoint validation generation macro then KL then registered candidate order; T main remains historical preset (no tuning advantage); selection before any formal M generation',
                                    already_viewed_benchmark=True,max_live_jobs=2,max_attempts=3),
                      cohorts=dict(train_questions=20000,formal_questions=23565,same_page_questions=len(swap),same_page_images=len({r['image'] for r in swap})),
                      author_meta_implementation='vqa/meta_compress.py is paper-based UND adaptation, not claimed official source',
                      sources=dict(S2Prune='https://github.com/yuanyuanjia71-spec/S2Prune/tree/a919a8d7f1c139dd054631c34779d1fdbd8dbeed',GridPrune='https://arxiv.org/abs/2511.10081',MetaCompress='https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.html'))
    save(REG,registration)
    save(BASE/'state.json',dict(created_utc=now(),runs={r['name']:dict(stage='registered',jobs=[]) for r in records}))
    print(json.dumps(dict(registered=len(records),full_training=8,development=6,same_page_questions=len(swap))))


def sha_text(s):return __import__('hashlib').sha256(s.encode()).hexdigest()
def queue():
    text=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%R'],text=True)
    return {p[0]:dict(name=p[1],state=p[2],reason=p[3]) for line in text.splitlines() if len(p:=line.split('|',3))==4}


def marker(r):
    if r['phase']=='smoke':path=BASE/'smoke/complete.json'
    elif r['phase']=='efficiency':path=BASE/'efficiency/complete.json'
    else:path=ROOT/r['output']/('complete.json' if r['phase']=='train' else 'benchmark/complete.json')
    # Shared-FS negative dentries can outlive a successful NPU commit. Refresh
    # directories before deciding a stage failed or submitting a duplicate.
    for parent in [path.parent.parent,path.parent]:
        try:list(parent.iterdir())
        except FileNotFoundError:pass
    return path


def choose_development(reg):
    if (BASE/'development_selection.json').exists():return
    records=[r for r in reg['runs'] if r['category']=='development']
    if not all(marker(r).exists() for r in records):return
    rr=[]
    for r in records:
        cfg=load(ROOT/r['config']);s=load(ROOT/r['output']/'validation.endpoint.summary.json')
        assert s and s['rows']==56
        rr.append(dict(name=r['name'],method=cfg['method'],candidate=cfg['candidate'],validation=s,
                       settings={k:cfg[k] for k in ['optimizer','lr','clip','schedule']}))
    candidates=[r for r in rr if r['method']=='M']
    selected=min(enumerate(candidates),key=lambda item:(-item[1]['validation']['student_macro'],item[1]['validation']['kl'],item[0]))[1]
    save(BASE/'development_selection.json',dict(selected_utc=now(),settings=selected['settings'],selected=selected['name'],all_trials=rr,formal_used=False,
                                               T_preset_retained=True,note='Equal development update budgets; T primary not optimized on its trial outcomes. M selected on validation, shared across budgets/seeds.'))


def advance(reg,state):
    live=queue();active=0
    recovery=load(BASE/'resize_recovery/registration.json')
    if recovery:
        assert recovery['original_registration_sha256']==sha(REG)
        for path,digest in recovery['source_sha256'].items():assert sha(ROOT/path)==digest
    for r in reg['runs']:
        st=state['runs'][r['name']]
        if marker(r).exists():st.update(stage='complete');continue
        if st['jobs'] and st['jobs'][-1]['job'] in live:
            st.update(stage='active',scheduler=live[st['jobs'][-1]['job']]);active+=1;continue
        if st['jobs']:
            job=st['jobs'][-1]
            if not job.get('exit_seen'):job['exit_seen']=now();st['stage']='awaiting_commit';continue
            if ( __import__('datetime').datetime.now(__import__('datetime').timezone.utc)-__import__('datetime').datetime.fromisoformat(job['exit_seen'])).total_seconds()<90:continue
            attempts=6 if r['name'] in recovery.get('runs',[]) else 3
            if len(st['jobs'])>=attempts:st.update(stage='failed',error='Attempt limit reached without committed result; inspect preserved logs and resize amendment if applicable.');continue
        st['stage']='ready'
    choose_development(reg)
    # Interleave a training job with an inference job when both are runnable.
    # Seeds, minibatches, device world and scientific configurations are unchanged.
    while active<2:
        candidates=[]
        for r in reg['runs']:
            st=state['runs'][r['name']]
            if any(state['runs'][d]['stage'] in ['failed','blocked'] for d in r['depends']):
                st.update(stage='blocked',error='Required predecessor failed; inspect logs.');continue
            if st['stage']!='ready' or not all(state['runs'][d]['stage']=='complete' for d in r['depends']):continue
            if r['category']=='full_training' and load(ROOT/r['config'])['method']=='M' and not (BASE/'development_selection.json').exists():continue
            candidates.append(r)
        if not candidates:break
        active_phases=[r['phase'] for r in reg['runs'] if state['runs'][r['name']]['stage']=='active']
        def priority(r):
            if r['category']=='technical':p=0
            elif r['category']=='development':p=1
            elif r['phase']=='train' and 'eval' in active_phases and 'train' not in active_phases:p=2
            elif r['name'] in ['HA_r0.25','QSWAP_r0.25']:p=3
            elif r['phase']=='eval':p=4
            elif r['phase']=='train':p=5
            else:p=6
            return p,reg['runs'].index(r)
        r=min(candidates,key=priority);st=state['runs'][r['name']]
        cmd=['sbatch','--parsable','--job-name=cs-'+r['name'][:70],'scripts/slurm_compression_supplement_stage.sh',r['phase'],r['config']]
        if r['name'] in recovery.get('runs',[]):
            cmd=['sbatch','--parsable','--job-name=cs-resize-repair-'+r['name'],'scripts/slurm_supplement_resize_recovery.sh',r['config']]
            st['recovery_amendment']='artifacts/compression_supplement_20261111/resize_recovery/registration.json'
        job=subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0]
        st['jobs'].append(dict(job=job,submitted_utc=now(),command=cmd));st['stage']='active';active+=1;save(BASE/'state.json',state)
    state['updated_utc']=now();save(BASE/'state.json',state)


def report(require=False):
    cmd=[str(ROOT/'.venv/bin/python'),'-m','scripts.report_compression_supplement']
    if require:cmd.append('--require-complete')
    subprocess.run(cmd,cwd=ROOT,check=True)


def control():
    with (BASE/'controller.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        save(BASE/'controller_runtime.json',dict(job=os.environ.get('SLURM_JOB_ID'),started_utc=now()))
        reg=verify()
        while True:
            verify();state=load(BASE/'state.json');advance(reg,state);report()
            stages={k:v['stage'] for k,v in state['runs'].items()}
            print(json.dumps(dict(updated_utc=now(),stages=stages)),flush=True)
            if all(s in ['complete','failed','blocked'] for s in stages.values()):
                if all(s=='complete' for s in stages.values()):report(True);save(BASE/'complete.json',dict(completed_utc=now(),runs=len(stages)))
                else:save(BASE/'attention_required.json',dict(stages=stages,updated_utc=now()))
                return
            time.sleep(45)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','resume','control','status','verify']);a=p.parse_args()
    if a.action=='prepare':prepare();report()
    elif a.action=='resume':
        prepare()
        if any(v['name']=='compression-supplement' for v in queue().values()):print('Existing controller retained.');return
        job=subprocess.check_output(['sbatch','--parsable','scripts/slurm_compression_supplement_controller.sh'],cwd=ROOT,text=True).strip().split(';')[0]
        save(BASE/'controller_submission.json',dict(job=job,submitted_utc=now()));print('Controller submitted:',job)
    elif a.action=='control':control()
    elif a.action=='verify':verify();print('Supplement source/data/config hashes verified.')
    else:report();print(json.dumps(load(BASE/'state.json'),ensure_ascii=False));print(json.dumps(queue()))


if __name__=='__main__':main()
