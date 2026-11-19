"""Persisted Slurm controller: search -> confirmation -> fresh 20k retrain -> benchmark.

No training is launched without the registered validation gates. All failed
candidates and protocol changes stay on disk. Controller can safely restart.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time
from scripts.report_meta_compress_search import BASE,ROOT,collect,read,summarize

STATE=BASE/'controller.json'


def stamp():return datetime.datetime.now().astimezone().isoformat()


def save(path,data):
    tmp=path.with_suffix(path.suffix+'.controller.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def shell(args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()


def report(cohort='search'):
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.report_meta_compress_search','--cohort',cohort],cwd=ROOT,check=True)


def update(state,phase,**extra):
    if phase not in {'error','no_verified_candidate'}:state.pop('error',None)
    state.update(phase=phase,updated_at=stamp(),**extra);save(STATE,state)
    lines=['# Hyperparameter Search and Retraining: Live Status','',f"Updated:{state['updated_at']}",'',f'Current stage: **{phase}**','',
           'Written by the Slurm controller; submitted does not mean training/evaluation completed.','',
           '[Stagewise search results](RESULTS.search.md) · [Confirmation results](RESULTS.confirm.md)','',
           '## Job Submissions','', '| key | job ID |','|---|---|']
    for key,value in state.get('jobs',{}).items():lines.append(f'| {key} | {value} |')
    if state.get('selection'):lines+=['','## Selection Records','', '```json',json.dumps(state['selection'],ensure_ascii=False,indent=2),'```']
    if state.get('error'):lines+=['',f"Error:{state['error']}"]
    (BASE/'RUNNING_STATUS.md').write_text('\n'.join(lines)+'\n')


def submit(state,key,script,args):
    for early in [BASE/'early_round2_jobs.json',BASE/'early_round3_jobs.json']:
        if key not in state['jobs'] and early.exists():
            for record in json.loads(early.read_text()).get('jobs',[]):
                if record['name']==key:
                    state['jobs'][key]=record.get('active_job',record['job']);update(state,state['phase'])
    if key not in state['jobs']:
        flags=[]
        if key=='full_retrain':
            cfg=json.loads((ROOT/args[0]).read_text())
            if cfg['ratio']>.5:flags=['--nodes=6']
        elif script=='scripts/slurm_meta_compress_search.sh':
            cfg=json.loads((ROOT/args[0]).read_text())
            if cfg['ratio']>.5:flags=['--mem=400G']
        job=shell(['sbatch','--parsable',*flags,script,*args]).split(';')[0]
        state['jobs'][key]=job;update(state,state['phase'])
    return state['jobs'][key]


def wait(state,phase,work):
    update(state,phase)
    def finished(marker):
        return marker.exists() or (marker.parent.parent==BASE/'runs' and (marker.parent/'early_stop.json').exists())
    while not all(finished(marker) for job,marker in work):
        for index,(job,marker) in enumerate(work):
            if finished(marker):continue
            rulepath=BASE/'pilot_culling_addendum.json'
            if rulepath.exists() and marker.parent.name in json.loads(rulepath.read_text())['scope']:
                result=collect(marker.parent);initial=result['stages'].get('initial/trained');mean=result.get('matched_mean')
                for epoch in range(2,result['config']['epochs']+1):
                    stage=result['stages'].get(f'epoch{epoch}/trained')
                    if not (initial and mean and stage and mean['complete'] and initial['complete'] and stage['complete']):continue
                    if (stage['score']<mean['score']-10 and stage['score']<.25*stage['teacher'] and
                            stage['prediction']>=initial['prediction']):
                        record={'at':stamp(),'rule':str(rulepath.relative_to(ROOT)),'job':job,'last_eligible_complete_epoch':epoch,
                            'epoch_result':stage,'matched_mean':mean,'observed_updates':result.get('updates_completed'),
                            'planned_updates':112,'training_complete':False,'reason':'Pilot futility rule; not a completed 4-epoch run'}
                        save(marker.parent/'early_stop.json',record)
                        subprocess.run(['scancel',job],check=True)
                        state.setdefault('early_stops',[]).append({'name':marker.parent.name,'job':job,'epoch':epoch,'at':stamp()})
                        update(state,phase);break
                if finished(marker):continue
            statuses=shell(['sacct','-j',job,'--noheader','--format=State','--parsable2']).splitlines()
            if statuses:
                status=statuses[0].split('|')[0].split()[0].split('+')[0]
                if status=='COMPLETED':
                    # GPFS can expose Slurm completion before a different
                    # node sees a just-renamed completion marker.
                    misses=state.setdefault('marker_visibility_retries',{})
                    misses[str(job)]=misses.get(str(job),0)+1
                    if misses[str(job)]<=3:continue
                if status in {'TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL'}:
                    name=marker.parent.name;cfgpath=BASE/f'configs/{name}.json'
                    retries=state.setdefault('resume_counts',{})
                    if cfgpath.exists() and (marker.parent/'latest.pt').exists() and retries.get(name,0)<2:
                        cfg=json.loads(cfgpath.read_text())
                        script=('scripts/slurm_meta_compress_search_full.sh' if cfg.get('cohort')=='full_original_20k'
                                else 'scripts/slurm_meta_compress_search.sh')
                        flags=[]
                        if cfg['ratio']>.5:
                            flags=(['--nodes=6'] if cfg.get('cohort')=='full_original_20k' else ['--mem=400G'])
                        new=shell(['sbatch','--parsable',*flags,script,str(cfgpath.relative_to(ROOT))]).split(';')[0]
                        retries[name]=retries.get(name,0)+1
                        state.setdefault('resumed_jobs',[]).append({'name':name,'from':job,'to':new,'cause':status,'at':stamp()})
                        for key,value in list(state['jobs'].items()):
                            if str(value)==str(job):state['jobs'][key]=new
                        work[index]=(new,marker);update(state,phase);continue
                if status in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','COMPLETED'}:
                    raise RuntimeError(f'Job {job} terminal state {status} without {marker.relative_to(ROOT)}')
                reason=shell(['squeue','-j',job,'--noheader','--format=%r'])
                if 'DependencyNeverSatisfied' in reason:
                    raise RuntimeError(f'Job {job} has a failed prerequisite; inspect its cache/training log')
        report();update(state,phase);time.sleep(60)


def round2():
    """Contingency registered in source before round1 results; explicit changes."""
    path=BASE/'round2_protocol.json'
    template=json.loads((BASE/'configs/r1_adamw_cos25.json').read_text())
    settings=[
        dict(name='r2_dense_kl_sgdm25',optimizer='sgd',lr=.003,momentum=.9,clip=10.,entropy_weight=0.,collapse_weight=0.),
        dict(name='r2_dense_balanced25',entropy_weight=.01,collapse_weight=.001),
        dict(name='r2_dense_balanced50',ratio=.5,entropy_weight=.01,collapse_weight=.001),
        dict(name='r2_local_full25',local_bias=20.),
        dict(name='r2_local_full50',local_bias=20.,ratio=.5),
        dict(name='r2_local_sgdm50',local_bias=20.,ratio=.5,optimizer='sgd',lr=.003,momentum=.9),
        dict(name='r2_local_balanced25',local_bias=20.,entropy_weight=.01,collapse_weight=.01),
        dict(name='r2_local_balanced50',local_bias=20.,ratio=.5,entropy_weight=.01,collapse_weight=.01),
    ]
    if not path.exists():
        save(path,{'registered_at':stamp(),'trigger':'Round1 has no independently confirmed candidate',
            'changes':'Dense candidates test KL/regularizer gradient conflict. Local candidates add fixed -20 logit bias outside balanced contiguous groups; this IS a generator structure constraint beyond paper, not an optimizer-only change.',
            'settings':settings,'same_cohorts':True,'same_gates':True})
    for setting in settings:
        cfg={**template,**setting,'output':f"artifacts/meta_compress_search/runs/{setting['name']}"}
        destination=BASE/f"configs/{setting['name']}.json"
        if destination.exists():assert json.loads(destination.read_text())==cfg
        else:save(destination,cfg)
    return [s['name'] for s in settings]


def passing(names):
    options=[]
    for name in names:
        run=BASE/'runs'/name
        if not (run/'complete.json').exists():continue
        r=collect_ready(run)
        if not r.get('best_passing_stage'):continue
        stage=r['best_passing_stage'];epoch=int(stage.split('/')[0][5:]);s=r['stages'][stage]
        options.append({'name':name,'epoch':epoch,'search_score':s['score'],'search_kl':s['prediction'],
            'search_gain':s['paired_vs_initial']['gain_points'],'ratio':r['config']['ratio']})
    return sorted(options,key=lambda o:(-o['search_score'],o['search_kl'],o['ratio']))


def collect_ready(run,cohort='search'):
    for attempt in range(7):
        r=collect(run,cohort)
        if cohort=='search':required=['initial/trained']+[f'epoch{i}/trained' for i in range(1,r['config']['epochs']+1)]
        else:required=[f'{cohort}/initial',f'{cohort}/selected']
        if all(r['stages'].get(k,{}).get('complete') for k in required) and (r.get('matched_mean') or {}).get('complete'):
            return r
        if attempt<6:time.sleep(30)
    raise RuntimeError(f'Completion marker exists but expected evaluation rows are not fully visible: {run.name}/{cohort}')


def round3():
    """Broader image coverage if the first two rounds do not generalize."""
    from collections import Counter
    path=BASE/'pilot_expanded.jsonl';metadata=BASE/'round3_protocol.json'
    if not metadata.exists():
        original=read(BASE/'full_train.jsonl');rows=read(BASE/'pilot.jsonl')
        used={r['document_id'] for r in rows};counts=Counter(r['task'] for r in rows)
        forbidden={r['document_id'] for n in ['search','confirm'] for r in read(BASE/f'{n}.jsonl')}
        forbidden|={r['document_id'] for f in ['val','test'] for r in read(ROOT/f'data/full/{f}.jsonl')}
        forbidden|={r['document_id'] for r in read(ROOT/'artifacts/meta_compress_diagnosis/samples.jsonl')}
        for row in sorted(original,key=lambda r:hashlib.sha256(('expanded20261028:'+r['id']).encode()).hexdigest()):
            if row['task'] not in counts or counts[row['task']]>=96 or row['document_id'] in used or row['document_id'] in forbidden:continue
            rows.append(row);used.add(row['document_id']);counts[row['task']]+=1
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        save(metadata,{'registered_at':stamp(),'trigger':'First two rounds have no confirmed candidate',
            'train_images':len(rows),'task_counts':dict(counts),'max_per_task':96,
            'rows_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'change':'Larger image-disjoint pilot from the SAME original 20k; all teacher targets already cached. Fixed local prior retained; KL-only / weak regularization / attention temperature / 75-percent budget. No official benchmark used.'})
    template=json.loads((BASE/'configs/r1_adamw_cos25.json').read_text())
    settings=[
        dict(name='r3_local_kl_adamw50',ratio=.5,lr=1e-4,entropy_weight=0.,collapse_weight=0.),
        dict(name='r3_local_balanced_constant50',ratio=.5,entropy_weight=.01,collapse_weight=.01,scheduler='constant'),
        dict(name='r3_local_sharp50',ratio=.5,entropy_weight=.01,collapse_weight=.01,attention_scale=4.),
        dict(name='r3_local_kl75',ratio=.75,entropy_weight=0.,collapse_weight=0.),
    ]
    for setting in settings:
        cfg={**template,'local_bias':20.,**setting,'train_rows':str(path.relative_to(ROOT)),
             'cohort':'expanded_pilot','output':f"artifacts/meta_compress_search/runs/{setting['name']}"}
        destination=BASE/f"configs/{setting['name']}.json"
        if destination.exists():assert json.loads(destination.read_text())==cfg
        else:save(destination,cfg)
    return [s['name'] for s in settings]


def advance(state,names):
    candidates=passing(names)
    # At most the search winner from this round sees confirmation. Do not
    # repeatedly choose runner-ups using the same confirmation outcomes.
    if not candidates:return None
    selected=candidates[0];name=selected['name'];cfg=f'artifacts/meta_compress_search/configs/{name}.json'
    ckpt=f'artifacts/meta_compress_search/runs/{name}/epoch{selected["epoch"]}.pt'
    state['selection']={'search_winner':selected,'confirmation_seen_before':len(state.get('confirmation_attempts',[]))}
    update(state,'confirm_search_winner')
    ratio=selected['ratio'];mean=BASE/f'baselines/ratio{ratio:g}/confirm/complete.json'
    job=submit(state,f'mean_confirm_{ratio:g}','scripts/slurm_meta_compress_baseline.sh',[str(ratio),'confirm'])
    job2=submit(state,f'confirm_{name}','scripts/slurm_meta_compress_confirm.sh',[cfg,ckpt])
    wait(state,'confirm_search_winner',[(job,mean),(job2,BASE/f'runs/{name}/confirm/complete.json')])
    report('confirm');result=collect_ready(BASE/'runs'/name,'confirm');stage=result['stages']['confirm/selected']
    record={'candidate':selected,'confirmation':stage,'matched_mean':result['matched_mean']}
    target=BASE/f'confirmation_decision.{name}.json'
    if not target.exists():save(target,record)
    attempts=state.setdefault('confirmation_attempts',[])
    if name not in attempts:attempts.append(name)
    if not stage.get('practical_pass'):
        update(state,'confirmation_rejected',last_rejected=record);return None
    state['selection']={'candidate':selected,'confirmation':stage,'matched_mean':result['matched_mean']}
    update(state,'confirmed_candidate')
    return selected


def main():
    state=json.loads(STATE.read_text()) if STATE.exists() else {'started_at':stamp(),'jobs':{}}
    if state.get('phase')=='completed':return
    content=Path(__file__).read_bytes();digest=hashlib.sha256(content).hexdigest()
    snapshot=BASE/f'controller_source_snapshots/{os.environ.get("SLURM_JOB_ID","manual")}.{digest}.py'
    snapshot.parent.mkdir(exist_ok=True);snapshot.write_bytes(content)
    state.setdefault('controller_versions',[]).append({'job':os.environ.get('SLURM_JOB_ID'),
        'sha256':digest,'snapshot':str(snapshot.relative_to(ROOT)),'at':stamp()})
    try:
        names=json.loads((BASE/'protocol.json').read_text())['round1_configs']
        existing=json.loads((BASE/'round1_jobs.json').read_text());means=json.loads((BASE/'baseline_jobs.json').read_text())
        work=[]
        for r in existing:
            actual=state['jobs'].setdefault(r['name'],r['job']);work.append((actual,BASE/f"runs/{r['name']}/complete.json"))
        for r in means:
            actual=state['jobs'].setdefault(f"mean_search_{r['ratio']:g}",r['job']);work.append((actual,BASE/f"baselines/ratio{r['ratio']:g}/search/complete.json"))
        wait(state,'round1_training_and_search',work);report();selected=advance(state,names)
        if selected is None:
            names=round2();work=[];update(state,'round2_training_and_search')
            for name in names:
                job=submit(state,name,'scripts/slurm_meta_compress_search.sh',[f'artifacts/meta_compress_search/configs/{name}.json'])
                work.append((job,BASE/f'runs/{name}/complete.json'))
            wait(state,'round2_training_and_search',work);report();selected=advance(state,names)
        if selected is None:
            names=round3();work=[];update(state,'round3_expanded_image_search')
            job=submit(state,'mean_search_0.75','scripts/slurm_meta_compress_baseline.sh',['0.75','search'])
            work.append((job,BASE/'baselines/ratio0.75/search/complete.json'))
            for name in names:
                job=submit(state,name,'scripts/slurm_meta_compress_search.sh',[f'artifacts/meta_compress_search/configs/{name}.json'])
                work.append((job,BASE/f'runs/{name}/complete.json'))
            wait(state,'round3_expanded_image_search',work);report();selected=advance(state,names)
        if selected is None:
            update(state,'no_verified_candidate',error='All three registered rounds failed the frozen multi-image/practical gates. Full retraining deliberately not launched. Further evidence-based search is required.')
            return
        source=json.loads((BASE/f"configs/{selected['name']}.json").read_text())
        name=f"full_{selected['name']}"
        cfg={**source,'name':name,'epochs':2,'batch_size':36,'cohort':'full_original_20k',
             'train_rows':'artifacts/meta_compress_search/full_train.jsonl','output':f'artifacts/meta_compress_search/runs/{name}',
             'initialization':'fresh original initial.pt; no pilot generator or optimizer state reused'}
        path=BASE/f'configs/{name}.json'
        if path.exists():assert json.loads(path.read_text())==cfg
        else:save(path,cfg)
        config_rel=str(path.relative_to(ROOT));update(state,'full_retrain_selected_recipe',full_config=config_rel)
        job=submit(state,'full_retrain','scripts/slurm_meta_compress_search_full.sh',[config_rel])
        wait(state,'full_retrain_selected_recipe',[(job,BASE/f'runs/{name}/complete.json')])
        checkpoint=f'artifacts/meta_compress_search/runs/{name}/epoch2.pt'
        update(state,'final_fixed_benchmark',benchmark_checkpoint=checkpoint)
        job=submit(state,'final_benchmark','scripts/slurm_meta_compress_search_benchmark.sh',[config_rel,checkpoint])
        wait(state,'final_fixed_benchmark',[(job,BASE/f'runs/{name}/benchmark/complete.rank{r}.json') for r in range(32)])
        report();subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.plot_meta_compress_search'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.final_report_meta_compress_search'],cwd=ROOT,check=True)
        update(state,'completed')
    except Exception as e:
        update(state,'error',error=str(e));raise


if __name__=='__main__':main()
