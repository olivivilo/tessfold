"""Durable pilot -> full -> locked test/timing/memory -> independent audit."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from scripts.tessfold_budget1075.common import (
    ROOT, BASE, REGISTRY, MODELS, model_id, candidates, pilot_label, full_label,
    completed, refresh, register, verify, load, save, sha, now,
)


def queue():
    result = subprocess.run(['squeue', '-h', '-u', 'jchen524', '-o', '%i|%j|%T|%R'],
                            capture_output=True, text=True, check=True)
    return {p[0]: dict(name=p[1], state=p[2], reason=p[3])
            for line in result.stdout.splitlines() if len(p := line.split('|', 3)) == 4}


def submit(args):
    return subprocess.check_output(['sbatch', '--parsable', *args], cwd=ROOT, text=True).strip().split(';')[0]


def npu_command(name, npu, action, args):
    return [f'--job-name=tessfold-1075-{name}', f'--gres=npu:{npu}',
            f'--cpus-per-task={npu * 8}', f'--mem={max(64, npu * 48)}G',
            f'--export=ALL,TESSFOLD_NPROC={npu}', 'scripts/tessfold_budget1075/npu.sh', action, *args]


def ensure(state, label, command, q, marker=None):
    stage = state['jobs'].get(label)
    if stage is None:
        stage = dict(job=submit(command), command=command, time=now(), status='submitted')
        state['jobs'][label] = stage
        save(BASE / 'controller_state.json', state)
    if load(refresh(marker or BASE / label / 'complete.json')):
        stage['status'] = 'complete'
        stage.pop('absent_since', None)
    elif stage['job'] in q:
        stage.update(status=q[stage['job']]['state'], reason=q[stage['job']]['reason'])
        stage.pop('absent_since', None)
    else:
        import datetime
        stage.setdefault('absent_since', now())
        elapsed = datetime.datetime.now(datetime.timezone.utc).timestamp() - datetime.datetime.fromisoformat(stage['absent_since']).timestamp()
        if elapsed > 180:
            stage['status'] = 'attention_required'
    return stage


def training_command(kind, ratio, attempt, label, full=False):
    args = ['--label', 'budget1075/' + label, '--limit', '20000' if full else '512',
            '--epochs', '4' if full else '3', '--configs', f'{kind}:{ratio:g}',
            '--lr', str(attempt['lr']), '--batch', str(attempt['batch'])]
    if attempt['stable']:
        args = ['--stable-implementation', *args]
    if full:
        gate = BASE / pilot_label(kind, ratio, attempt)
        args += ['--gate-root', str(gate.relative_to(ROOT))]
    return npu_command(('full-' if full else 'pilot-') + model_id(kind, ratio), 4, 'train', args)


def advance(state):
    q = queue()
    state.setdefault('jobs', {})
    # A completed pilot is not visited again after its choices are locked.
    # Refresh every recorded stage so its displayed status cannot remain RUNNING.
    for label, stage in state['jobs'].items():
        marker = BASE / ('full_training_audit.json' if label == 'full_training_audit' else label + '/complete.json')
        if load(refresh(marker)):
            stage['status'] = 'complete'
            stage.pop('absent_since', None)
        elif stage['job'] in q:
            stage.update(status=q[stage['job']]['state'], reason=q[stage['job']]['reason'])
    state.setdefault('pilot_choices', {})
    for kind, ratio in MODELS:
        identity = model_id(kind, ratio)
        chosen = state['pilot_choices'].get(identity)
        if chosen is None:
            for attempt in candidates(kind):
                label = pilot_label(kind, ratio, attempt)
                run = completed(label + '/' + identity)
                if not run:
                    ensure(state, label, training_command(kind, ratio, attempt, label), q)
                    break
                if run['pilot_gate_passed']:
                    chosen = dict(attempt, pilot=label, pilot_sha256=sha(BASE / label / identity / 'complete.json'))
                    state['pilot_choices'][identity] = chosen
                    break
            else:
                state.setdefault('scientific_attention', {})[identity] = 'All fixed pilot candidates failed; no gate bypass or fabricated full run.'
        if chosen:
            label = full_label(kind, ratio)
            ensure(state, label, training_command(kind, ratio, chosen, label, full=True), q)
    full_ready = all(completed(full_label(k, r) + '/' + model_id(k, r)) for k, r in MODELS)
    if full_ready and not REGISTRY.exists():
        runs = []
        for kind, ratio in MODELS:
            identity = model_id(kind, ratio)
            out = BASE / full_label(kind, ratio) / identity
            result, manifest = load(out / 'complete.json'), load(out / 'manifest.json')
            stage = result['best']['stage'] if result['best'] else 'initial'
            checkpoint = out / (stage + '.pt')
            runs.append(dict(id=identity, kind=kind, ratio=ratio, checkpoint=str(checkpoint.relative_to(ROOT)),
                             checkpoint_sha256=sha(checkpoint), stage=stage, allocation=manifest.get('allocation'),
                             pilot_gate_sha256=manifest['gate_sha256'], full_run=str(out.relative_to(ROOT)),
                             dev_selected=True, full_train_improved_gate=bool(result['best']),
                             selection='Same registered dev gate and macro/KL tie break as 25%/50%; initial if no eligible epoch.'))
        save(REGISTRY, dict(time=now(), runs=runs, formal_used_for_selection=False,
                            registration_sha256=sha(BASE / 'registration.json')))
    if REGISTRY.exists():
        run_arg = ['--runs', str(REGISTRY.relative_to(ROOT))]
        ensure(state, 'qwen_formal', npu_command('formal', 8, 'eval', [*run_arg, '--label', 'qwen_formal']), q)
        ensure(state, 'qwen_timing', npu_command('timing', 2, 'eval', [*run_arg, '--label', 'qwen_timing', '--timing']), q)
        ensure(state, 'qwen_memory_smoke', npu_command('mem-smoke', 2, 'memory', [*run_arg, '--smoke']), q)
        ensure(state, 'full_training_audit', npu_command('audit', 1, 'audit', []), q,
               marker=BASE / 'full_training_audit.json')
        if completed('qwen_memory_smoke'):
            ensure(state, 'qwen_memory', npu_command('memory', 2, 'memory', run_arg), q)
    state.update(updated=now(), live_queue=q)
    save(BASE / 'controller_state.json', state)
    from scripts.tessfold_budget1075.report import write_report
    write_report()
    ready = all(completed(x) for x in ['qwen_formal', 'qwen_timing', 'qwen_memory']) and load(BASE / 'full_training_audit.json')
    if ready:
        from scripts.tessfold_budget1075.report import audit
        audit()
        write_report()
        return True
    return bool(state.get('scientific_attention')) or any(s['status'] == 'attention_required' for s in state['jobs'].values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['register', 'status', 'resume', 'control', 'report', 'audit'])
    args = p.parse_args()
    if args.action == 'register':
        register()
        state = load(BASE / 'controller_state.json')
        state.setdefault('jobs', {}).setdefault('pilot_default',
            dict(job='166098', time=now(), status='adopted_running',
                 command=['scripts/tessfold_accel/qwen_train_safe.sh', '--label', 'budget1075/pilot_default',
                          '--limit', '512', '--epochs', '3', '--configs', 'Column:.1,Tree:.1,Column:.75,Tree:.75',
                          '--lr', '3e-4', '--batch', '32']))
        save(BASE / 'controller_state.json', state)
        from scripts.tessfold_budget1075.report import write_report
        write_report()
        print(json.dumps(dict(registration=str(BASE / 'registration.json'))))
        return
    if args.action == 'status':
        print(json.dumps(dict(state=load(BASE / 'controller_state.json'), queue=queue(),
                              all_complete=bool(load(BASE / 'all_stages_complete.json'))), ensure_ascii=False))
        return
    verify()
    if args.action in ['report', 'audit']:
        from scripts.tessfold_budget1075.report import write_report, audit
        if args.action == 'audit':
            audit()
        write_report()
        return
    if args.action == 'resume':
        if load(BASE / 'all_stages_complete.json'):
            print('All stages already audited and complete')
            return
        if any(r['name'] == 'tessfold-1075-control' for r in queue().values()):
            print('Existing controller retained')
            return
        job = submit(['scripts/tessfold_budget1075/controller.sh'])
        save(BASE / 'controller_submission.json', dict(job=job, time=now()))
        print(job)
        return
    with (BASE / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while True:
            verify()
            if advance(load(BASE / 'controller_state.json')):
                break
            time.sleep(45)


if __name__ == '__main__':
    main()
