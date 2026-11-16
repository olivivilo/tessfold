"""Persistent bounded-concurrency, adaptive full-paper-loss experiment controller.

Every change is a new immutable config/run. Confirmation never picks a runner-up.
"""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import math
import os
import hashlib
import subprocess
import time
from scripts.report_meta_compress_strict import ROOT,BASE,read,collect,save

STATE=BASE/'controller.json'
def stamp():return datetime.datetime.now().astimezone().isoformat()
def shell(args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def accepted(result):
    key='requested_stability' if (BASE/'primary_endpoint_addendum.json').exists() else 'stability'
    return result[key]['pass']

def update(state,phase,**extra):
    state.update(phase=phase,updated_at=stamp(),**extra);save(STATE,state)
    lines=['# 严格 MetaCompress 实时进度','',f'更新：{state["updated_at"]}',f'阶段：**{phase}**','',
        '结构与 full loss 不变，主干冻结；本文件的“提交/运行”不代表达成稳定收益。',
        '[实验结果](RESULTS.md) · [逐题作答](answers.csv) · [协议](protocol.json)','',
        '| 名称 | 作业 | 状态 |','|---|---|---|']
    for name,r in state.get('jobs',{}).items():lines.append(f'| {name} | {r["job"]} | {r.get("state","submitted")} |')
    if state.get('error'):lines+=['',f'错误：{state["error"]}']
    (BASE/'RUNNING_STATUS.md').write_text('\n'.join(lines)+'\n')

def report():
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.report_meta_compress_strict'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

def submit(state,name,config,extra=None,script_override=None):
    cfg=json.loads((ROOT/config).read_text())
    script={'pca-whitened':'scripts/slurm_meta_compress_strict_whitened.sh',
            'pca-pool-null':'scripts/slurm_meta_compress_strict_pool_null.sh',
            'fourier-dense-position':'scripts/slurm_meta_compress_strict_dense_position.sh',
            'multiscale-position':'scripts/slurm_meta_compress_strict_multiscale.sh'}.get(
                cfg.get('initialization'),'scripts/slurm_meta_compress_strict.sh')
    if cfg['insertion'] in ['post-norm','post-gelu']:
        script='scripts/slurm_meta_compress_strict_inner.sh'
    if cfg.get('training_devices')==8:
        script='scripts/slurm_meta_compress_strict_full.sh'
    if script_override:script=script_override
    job=shell(['sbatch','--parsable',script,'--config',config,*(extra or [])]).split(';')[0]
    state['jobs'][name]={'job':job,'config':config,'extra':extra or [],'script':script,'state':'submitted','at':stamp()}
    update(state,state['phase'])

def batch(state,phase,work,limit=8,stop_when_repeated_stable=None,npu_budget=24):
    """work = name/config/marker/extra. Restart preempted/failed jobs twice."""
    def devices(w):
        if w.get('script')=='scripts/slurm_meta_compress_strict_benchmark.sh':return 8
        return json.loads((ROOT/w['config']).read_text()).get('training_devices',2)
    update(state,phase)
    while True:
        unfinished=[];active=0;active_devices=0
        for w in work:
            name=w['name'];record=state['jobs'].get(name)
            run=(ROOT/w['marker']).parent
            if (run/'early_stop.json').exists():
                if record:record['state']='early_stopped'
                continue
            if (ROOT/w['marker']).exists():
                if record:record['state']='complete'
                continue
            if record and record.get('state')=='failed':continue
            unfinished.append(w)
            if not record:continue
            if name.startswith('screen') and (run/'manifest.json').exists():
                result,_=collect(run)
                per_epoch=math.ceil(len(read(ROOT/result['config']['train_rows']))/result['config']['batch_size'])
                for epoch in [2,3]:
                    v=result['stages'].get(f'validation/epoch{epoch}',{})
                    if (v.get('complete') and v['score']<.25*v['teacher']
                            and v.get('kl_drop_fraction',0)<.01
                            and result['updates']<=per_epoch*(result['config']['epochs']-1)):
                        save(run/'early_stop.json',{'at':stamp(),'rule':'futility_addendum.json',
                            'validation':v,'completed_validation_epoch':epoch,
                            'observed_updates':result['updates'],'planned_epochs':result['config']['epochs'],
                            'reason':'After >=2 epochs score below 25% teacher and KL drop <1%; not a completed planned run.'})
                        subprocess.run(['scancel',record['job']],cwd=ROOT,check=True)
                        record['state']='early_stopped';break
                if record['state']=='early_stopped':continue
            status=shell(['sacct','-j',record['job'],'--noheader','--format=State','--parsable2']).splitlines()
            status=status[0].split('|')[0].split('+')[0].strip() if status else 'PENDING'
            record['state']=status
            if status in ['COMPLETED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','CANCELLED','PREEMPTED','BOOT_FAIL']:
                if (ROOT/w['marker']).exists():continue
                # Allow one filesystem visibility cycle after Slurm completion.
                record['terminal_seen']=record.get('terminal_seen',0)+1
                if record['terminal_seen']<3:active+=1;active_devices+=devices(w);continue
                retries=state.setdefault('retries',{}).get(name,0)
                if retries<2:
                    state.setdefault('history',[]).append({'name':name,**record})
                    state['retries'][name]=retries+1
                    submit(state,name,w['config'],w.get('extra'),w.get('script'));active+=1;active_devices+=devices(w)
                else:
                    record['state']='failed';state.setdefault('failed',[]).append(name)
            else:active+=1;active_devices+=devices(w)
        if stop_when_repeated_stable:
            for config in stop_when_repeated_stable:
                cfg=json.loads((ROOT/config).read_text());run=ROOT/cfg['output']
                repeated=BASE/'runs'/(cfg['name']+'_seed2')
                if not (run/'complete.json').exists() or not (repeated/'complete.json').exists():continue
                first_result,_=collect(run);second_result,_=collect(repeated)
                if accepted(first_result) and accepted(second_result):
                    record={'at':stamp(),'family':cfg['name'],
                        'reason':'Both seeds completed expanded training and passed current total-loss/VQA endpoint protocol; original stricter KL-tail diagnostic remains reported. Advance to unchanged independent confirmation.',
                        'unsubmitted':[w['name'] for w in unfinished if w['name'] not in state['jobs']],
                        'running_jobs_retained':[w['name'] for w in unfinished if w['name'] in state['jobs']]}
                    save(BASE/'early_repeated_stable_selection.json',record)
                    report();update(state,'early_repeated_stable_family',early_family=cfg['name']);return
        for w in unfinished:
            if active>=limit:break
            if w['name'] not in state['jobs']:
                # Reserve room for a higher-priority eight-rank expansion;
                # do not refill the gap with two-rank initializer screens.
                requested=devices(w)
                if active_devices+requested>npu_budget:break
                submit(state,w['name'],w['config'],w.get('extra'),w.get('script'));active+=1;active_devices+=requested
        report();update(state,phase)
        if all((ROOT/w['marker']).exists() or ((ROOT/w['marker']).parent/'early_stop.json').exists()
               or state['jobs'].get(w['name'],{}).get('state')=='failed' for w in work):return
        time.sleep(30)

def training_work(config):
    cfg=json.loads((ROOT/config).read_text())
    return {'name':cfg['name'],'config':config,'marker':cfg['output']+'/complete.json'}

def ranked(configs):
    candidates=[]
    for path in configs:
        cfg=json.loads((ROOT/path).read_text());run=ROOT/cfg['output']
        if not (run/'complete.json').exists():continue
        result,_=collect(run)
        v=result['stages'].get(f'validation/epoch{cfg["epochs"]}',{})
        if v.get('complete'):candidates.append((result,path,v))
    # Avoid spending expanded training on a 5->8pt floor gain while a useful
    # initialization already retains most evidence. Still validation only.
    return sorted(candidates,key=lambda x:(accepted(x[0]),
        x[2]['score']>=.5*x[2]['teacher'],x[2].get('gain_points',-1e9),x[2]['score']),reverse=True)

def register(cfg):
    path=BASE/'configs'/f'{cfg["name"]}.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    else:save(path,cfg)
    return str(path.relative_to(ROOT))

def refine(state,previous,roundno):
    selection=ranked(previous)[:2]
    settings=[dict(optimizer='adamw',lr=3e-5,clip=1.),
              dict(optimizer='adamw',lr=1e-5,clip=10.),
              dict(optimizer='sgd',lr=1e-3,momentum=.9,clip=10.),
              dict(optimizer='adamw',lr=3e-4,clip=1.)]
    configs=[]
    for j,(r,_,v) in enumerate(selection):
        for k,setting in enumerate(settings):
            cfg=dict(r['config']);name=f'round{roundno}_parent{j}_opt{k}'
            cfg.update(setting);cfg.update(name=name,output=f'artifacts/meta_compress_strict/runs/{name}',
                train_rows='artifacts/meta_compress_strict/train.jsonl',epochs=8,
                parent_validation_candidate=r['name'])
            if cfg['initialization'] in ['fourier-dense-position','multiscale-position']:
                # Tiny Q/K weights compensate a large positional projection;
                # scale optimizer search to the parent's registered LR.
                cfg['lr']=r['config']['lr']*[.3,1.,10.,3.][k]
            if roundno==3:
                # If 50% cannot retain evidence, test less aggressive reduction;
                # this is a new budget, not a training improvement over 50%.
                cfg['ratio']=.75
            configs.append(register(cfg))
    decision={'at':stamp(),'round':roundno,'parents':[r['name'] for r,_,_ in selection],
              'settings':settings,'configs':configs,'selection_source':'validation only; no confirmation inspected'}
    path=BASE/f'round{roundno}_registration.json'
    if not path.exists():save(path,decision)
    else:assert json.loads(path.read_text())['configs']==configs
    return configs

def confirm(state,winner):
    r,path,_=winner;cfg=r['config'];name=cfg['name'];epoch=cfg['epochs']
    work=[]
    for stage,checkpoint in [('initial','initial.pt'),('selected',f'epoch{epoch}.pt')]:
        work.append({'name':f'confirm_{name}_{stage}','config':path,
            'marker':cfg['output']+f'/confirm.{stage}.complete.json',
            'extra':['--evaluate','artifacts/meta_compress_strict/confirm.jsonl','--checkpoint',cfg['output']+'/'+checkpoint,
                     '--cohort','confirm','--stage',stage]})
    batch(state,'confirm_frozen_winner',work,limit=2)
    result,_=collect(ROOT/cfg['output']);s=result['stages'].get('confirm/selected',{})
    ok=bool(s.get('complete') and s.get('total_drop_fraction',0)>=.03 and s.get('kl_drop_fraction',0)>=.05
            and s.get('gain_points',0)>=2 and s.get('score',0)>=.7*s.get('teacher',float('inf'))
            and s.get('paired',{}).get('cluster_bootstrap_95ci_points',[-1])[0]>0)
    record={'candidate':name,'stage':f'epoch{epoch}','confirmation':s,'pass':ok,
            'at':stamp(),'note':'No other candidate is selected from confirmation outcomes'}
    save(BASE/f'confirmation.{name}.json',record)
    return ok

def full_config(winner):
    cfg=dict(winner[0]['config']);forbidden={r['document_id'] for f in ['validation','confirm'] for r in read(BASE/f'{f}.jsonl')}
    from collections import defaultdict
    import hashlib
    grouped=defaultdict(list)
    for r in read(ROOT/'data/full/train.jsonl'):
        if r['document_id'] not in forbidden:grouped[r['task']].append(r)
    # Keep the pilot/audit questions in the full training set. Otherwise a
    # random 20k draw could turn the named train_audit into an unseen subset.
    required=read(BASE/'train.jsonl')
    selected=[]
    for task,rr in sorted(grouped.items()):
        keep=[r for r in required if r['task']==task];ids={r['id'] for r in keep}
        extra=sorted((r for r in rr if r['id'] not in ids),
                     key=lambda r:hashlib.sha256(('strict_full:'+r['id']).encode()).hexdigest())
        selected+=keep+extra[:max(0,2857-len(keep))]
    path=BASE/'full_train.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    # Expanding image coverage must not silently multiply the confirmed
    # schedule's integrated LR by another order of magnitude. Full eight
    # epochs still visit all examples; scale LR, not the original objective.
    from vqa.meta_compress_tuning import learning_rate_factor
    parent_steps=math.ceil(len(read(ROOT/cfg['train_rows']))/cfg['batch_size'])*cfg['epochs']
    full_steps=math.ceil(len(selected)/cfg['batch_size'])*8
    parent_sum=sum(learning_rate_factor(s,parent_steps,cfg) for s in range(parent_steps))
    full_sum=sum(learning_rate_factor(s,full_steps,cfg) for s in range(full_steps))
    parent_lr=cfg['lr']
    cfg.update(lr=parent_lr*parent_sum/full_sum,
        full_learning_rate_policy='Preserve schedule-integrated LR of validation/confirmation parent; all data for8epochs',
        parent_learning_rate=parent_lr,parent_optimizer_steps=parent_steps,
        parent_integrated_learning_rate=parent_lr*parent_sum)
    cfg.update(name='full_verified_family',output='artifacts/meta_compress_strict/runs/full_verified_family',
               train_rows='artifacts/meta_compress_strict/full_train.jsonl',epochs=8,training_devices=8)
    return register(cfg)

def main():
    p=argparse.ArgumentParser();p.add_argument('--adopt-config');p.add_argument('--adopt-job');a=p.parse_args()
    lock=(BASE/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'started_at':stamp(),'jobs':{},'phase':'registered'}
    source=Path(__file__);digest=hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot=BASE/'controller_sources'/f'{digest}.py';snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    state.setdefault('controller_versions',[]).append({'at':stamp(),'job':os.environ.get('SLURM_JOB_ID'),
        'sha256':digest,'snapshot':str(snapshot.relative_to(ROOT))})
    if a.adopt_config:
        cfg=json.loads((ROOT/a.adopt_config).read_text())
        state['jobs'].setdefault(cfg['name'],{'job':a.adopt_job,'config':a.adopt_config,'state':'adopted'})
    try:
        configs=json.loads((BASE/'screen_configs.json').read_text())
        earlypath=BASE/'early_expansion.json'
        early=json.loads(earlypath.read_text())['configs'] if earlypath.exists() else []
        repeatsfile=BASE/'early_repeats.json'
        repeats=json.loads(repeatsfile.read_text())['configs'] if repeatsfile.exists() else []
        # Keep both seeds of a family adjacent. Otherwise new first seeds wait
        # behind every family's second seed, delaying any repeated conclusion.
        repeat_by_family={Path(c).stem.removesuffix('_seed2'):c for c in repeats}
        paired_early=[]
        for c in early:
            paired_early.append(c)
            if Path(c).stem in repeat_by_family:paired_early.append(repeat_by_family[Path(c).stem])
        paired_early+= [c for c in repeats if c not in paired_early]
        batch(state,'screen_and_early_multiimage_expansion',[training_work(c) for c in paired_early+configs],
              stop_when_repeated_stable=early if repeats else None)
        configs+=early
        allconfigs=list(configs)
        # 4-epoch screen is deliberately insufficient for a final acceptance.
        # Repeat on the larger cohort before checking stability/confirmation.
        for roundno in [1,2,3]:
            if roundno==1:
                eligible=early
            else:
                registration=BASE/f'round{roundno}_registration.json'
                configs=json.loads(registration.read_text())['configs'] if registration.exists() else refine(state,configs,roundno)
                if not configs:break
                batch(state,f'round{roundno}_expanded_multiquestion',[training_work(c) for c in configs])
                allconfigs+=configs;eligible=configs
            passing=[w for w in ranked(eligible) if accepted(w[0])]
            if roundno==1 and repeats:
                repeated_names={Path(p).stem for p in repeats}
                passing=[w for w in passing if (w[0]['name']+'_seed2') in repeated_names
                    and (BASE/'runs'/(w[0]['name']+'_seed2')/'complete.json').exists()
                    and accepted(collect(BASE/'runs'/(w[0]['name']+'_seed2'))[0])]
            if passing:
                winner=passing[0]
                repeat=dict(winner[0]['config']);repeat.update(name=repeat['name']+'_seed2',seed=20261102)
                repeat['output']='artifacts/meta_compress_strict/runs/'+repeat['name']
                repeatpath=register(repeat)
                batch(state,'repeat_selected_family_seed2',[training_work(repeatpath)])
                repeats=ranked([repeatpath])
                if not repeats or not accepted(repeats[0][0]):
                    update(state,'seed_repeat_failed');continue
                # Family and epochs fixed before observing confirmation results.
                ok1=confirm(state,winner);ok2=confirm(state,repeats[0])
                if not (ok1 and ok2):
                    update(state,'confirmation_failed_no_test_reselection');return
                full=full_config(winner)
                batch(state,'full_retraining_verified_family',[training_work(full)],limit=1)
                fresult=ranked([full])
                if not fresult:
                    update(state,'full_retraining_failed',selected_family=winner[0]['name']);return
                fullcfg=fresult[0][0]['config'];checkpoint=fullcfg['output']+f'/epoch{fullcfg["epochs"]}.pt'
                benchmark={'name':'final_full_benchmark','config':full,
                    'marker':fullcfg['output']+'/benchmark/complete.json',
                    'extra':['--checkpoint',checkpoint],'script':'scripts/slurm_meta_compress_strict_benchmark.sh'}
                batch(state,'final_23565_question_benchmark',[benchmark],limit=1)
                if not (ROOT/benchmark['marker']).exists():
                    update(state,'final_benchmark_failed',selected_family=winner[0]['name']);return
                update(state,'completed' if accepted(fresult[0][0]) else 'full_retraining_not_stable',
                       selected_family=winner[0]['name'],full_config=full,full_checkpoint=checkpoint,
                       final_report='docs/METACOMPRESS_STRICT_FINAL_RESULTS.md')
                return
        update(state,'no_stable_candidate',error='Registered initialization/optimizer/placement/budget rounds finished without repeated, confirmed stability. Negative results retained; further diagnosis required.')
    except Exception as e:
        update(state,'controller_error',error=repr(e));raise

if __name__=='__main__':main()
