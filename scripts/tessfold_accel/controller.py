"""Durable CPU orchestration; fixed pilot gates, full training, locked8B eval.

Entry: .venv/bin/python -m scripts.tessfold_accel.controller status|resume|control
Never retries an unexplained failed stage automatically. Keeps negative pilot
results and does not use any formal scores to select a checkpoint.
"""
import argparse,fcntl,os,subprocess,time
from pathlib import Path
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def queue():
    r=subprocess.run(['squeue','-h','-u','jchen524','-o','%i|%j|%T|%R'],capture_output=True,text=True,check=True)
    return {p[0]:dict(name=p[1],state=p[2],reason=p[3]) for line in r.stdout.splitlines() if len(p:=line.split('|',3))==4}


def submit(args):
    return subprocess.check_output(['sbatch','--parsable',*args],cwd=ROOT,text=True).strip().split(';')[0]


def advance(state):
    q=queue()
    raw_jobs={'system_exact':'165866','formal_equivalence':'165869','prefix_formal':'165873',
        'bidirectional_formal':'165924','bidirectional_timing':'165925','hard_formal':'165902','interleaved':'165877','multi_deployment':'165881','batched_serving':'165880',
        'qwen_full_pass1':'165861','qwen_full_tree25':'165871','qwen_full_stable50':'165884','qwen_memory_smoke':'165914','qwen_pilot_stable':'165879'}
    raw_jobs.update(load(BASE/'job_overrides.json'))
    import datetime
    for label,job in raw_jobs.items():
        stage=state.setdefault('registered_stages',{}).setdefault(label,dict(job=job))
        if stage.get('job')!=job:
            stage.setdefault('previous_jobs',[]).append(stage['job']);stage.update(job=job,status='resubmitted');stage.pop('last_absent',None)
        if (BASE/label/'complete.json').exists():stage['status']='complete'
        elif job in q:stage['status']=q[job]['state'];stage.pop('last_absent',None)
        else:
            stage.setdefault('last_absent',now())
            if datetime.datetime.now(datetime.timezone.utc).timestamp()-datetime.datetime.fromisoformat(stage['last_absent']).timestamp()>180:stage['status']='attention_required'
    for label in ['bidirectional_formal','qwen_formal']:
        pools=load(BASE/label/'parallel_jobs.json')
        if not pools:continue
        pool_state=state.setdefault('parallel_pools',{}).setdefault(label,{})
        pool_state.update(jobs=pools['jobs'],live={j:q[j] for j in pools['jobs'] if j in q},complete=(BASE/label/'complete.json').exists())
        if not (BASE/label/'complete.json').exists():
            for job in pools['jobs']:
                marker=state['parallel_pools'][label].setdefault('absent_since',{})
                if job in q:marker.pop(job,None)
                else:
                    marker.setdefault(job,now())
                    if datetime.datetime.now(datetime.timezone.utc).timestamp()-datetime.datetime.fromisoformat(marker[job]).timestamp()>180:
                        if label in state:state[label]['status']='attention_required'
                        state.setdefault('registered_stages',{}).setdefault(label,{})['status']='attention_required'
    # Stable50 is the fourth configuration; do not launch it unless its
    # independently recorded multi-image pilot meets the same paired gate.
    pilot=BASE/'qwen_pilot_stable/Column_r0.5/complete.json'
    if pilot.exists() and 'stable_full' not in state:
        r=load(pilot)
        if r['pilot_gate_passed']:
            args=['--gres=npu:4','--cpus-per-task=32','--mem=192G','--export=ALL,TESSFOLD_NPROC=4',
                'scripts/tessfold_accel/qwen_train_stable.sh','--label','qwen_full_stable50','--limit','20000','--epochs','4',
                '--configs','Column:.5','--lr','1e-3','--batch','16','--gate-root','artifacts/tessfold_acceleration_20261115/qwen_pilot_stable']
            state['stable_full']=dict(status='submitted',job=submit(args),time=now(),pilot_sha256=sha(pilot),command=args)
        else:state['stable_full']=dict(status='not_launched_pilot_failed',time=now(),pilot_sha256=sha(pilot))
        save(BASE/'controller_state.json',state)
    definitions=[('Column',.25,'qwen_full_pass1'),('Tree',.5,'qwen_full_pass1'),('Tree',.25,'qwen_full_tree25')]
    if state.get('stable_full',{}).get('status')=='submitted':definitions.append(('Column',.5,'qwen_full_stable50'))
    ready=('stable_full' in state and all((BASE/label/f'{kind}_r{ratio:g}/complete.json').exists() for kind,ratio,label in definitions))
    if ready and not (BASE/'qwen_selected_runs.json').exists():
        runs=[]
        for kind,ratio,label in definitions:
            out=BASE/label/f'{kind}_r{ratio:g}';r=load(out/'complete.json');manifest=load(out/'manifest.json')
            # Same preregistered dev-only eligible criterion. If none of the
            # trained epochs qualifies, preserve it as a negative full run;
            # formal evaluates the deterministic initial checkpoint instead.
            stage=r['best']['stage'] if r['best'] else 'initial'
            checkpoint=out/(stage+'.pt')
            runs.append(dict(id=f'{kind}_r{ratio:g}',kind=kind,ratio=ratio,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
                stage=stage,allocation=manifest.get('allocation'),pilot_gate_sha256=manifest.get('gate_sha256'),
                full_run=str(out.relative_to(ROOT)),dev_selected=True,full_train_improved_gate=bool(r['best']),
                selection='Choose highest dev macro among epochs reducing KL>=5percent and improving generation>=.25pp; absent means initial. No formal feedback.'))
        save(BASE/'qwen_selected_runs.json',dict(time=now(),runs=runs,formal_used_for_selection=False))
    if (BASE/'qwen_selected_runs.json').exists():
        for stage,label,extra in [('qwen_formal','qwen_formal',[]),('qwen_timing','qwen_timing',['--timing']),('qwen_memory','qwen_memory',[])]:
            if stage=='qwen_memory' and not (BASE/'qwen_memory_smoke/complete.json').exists():continue
            if stage not in state:
                args=['scripts/tessfold_accel/qwen_memory.sh' if stage=='qwen_memory' else 'scripts/tessfold_accel/qwen_eval.sh','--runs','artifacts/tessfold_acceleration_20261115/qwen_selected_runs.json',*(['--label',label] if stage!='qwen_memory' else []),*extra]
                if stage=='qwen_timing':args=['--gres=npu:4','--cpus-per-task=32','--mem=192G','--export=ALL,TESSFOLD_NPROC=4',*args]
                state[stage]=dict(status='submitted',job=submit(args),time=now(),command=args);save(BASE/'controller_state.json',state)
            if (BASE/label/'complete.json').exists():state[stage]['status']='complete'
            elif state[stage]['job'] in q:state[stage].pop('last_absent',None)
            elif state[stage]['job'] not in q:
                state[stage].setdefault('last_absent',now())
                if __import__('datetime').datetime.now(__import__('datetime').timezone.utc).timestamp()-__import__('datetime').datetime.fromisoformat(state[stage]['last_absent']).timestamp()>180:
                    state[stage]['status']='attention_required'
    state['updated']=now();state['live_queue']=q;save(BASE/'controller_state.json',state)
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.progress'],cwd=ROOT,check=True)
    expected=['system_exact','formal_equivalence','prefix_formal','interleaved','multi_deployment','batched_serving','qwen_formal','qwen_timing','qwen_memory','hard_formal','bidirectional_formal','bidirectional_timing','bidir_interleaved']
    finished=all((BASE/p/'complete.json').exists() for p in expected)
    signature=[p for p in expected if (BASE/p/'complete.json').exists()]
    if signature!=state.get('last_reported_completed_stages'):
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.finish_report'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.target_gap'],cwd=ROOT,check=True)
        state['last_reported_completed_stages']=signature;save(BASE/'controller_state.json',state)
    if finished:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.finish_report','--require-complete'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.paper_tables'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.final_audit'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.paper_extensions'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.tessfold_accel.target_gap','--publish-complete'],cwd=ROOT,check=True)
        save(BASE/'all_stages_complete.json',dict(time=now(),native_und_preserved=True,all_models_frozen_except_new8b_routers=True))
    return finished or any(state.get(s,{}).get('status')=='attention_required' for s in ['qwen_formal','qwen_timing','qwen_memory']) or any(r.get('status')=='attention_required' for r in state.get('registered_stages',{}).values())


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','resume','control']);a=p.parse_args()
    if a.action=='status':
        print(load(BASE/'controller_state.json'));print(queue());return
    if a.action=='resume':
        if any(r['name']=='tessfold-accel-control' for r in queue().values()):print('Existing controller retained');return
        job=submit(['scripts/tessfold_accel/controller.sh']);save(BASE/'controller_submission.json',dict(job=job,time=now()));print(job);return
    with (BASE/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        while True:
            state=load(BASE/'controller_state.json')
            if advance(state):break
            time.sleep(45)

if __name__=='__main__':main()
