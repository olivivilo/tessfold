"""Register and schedule the fixed column/tree comparison; never select on scores.

CPU entry: .venv/bin/python -m scripts.column_tree_full_suite prepare|resume|status|control
"""
from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/compression_lab_v2'
SUITE = BASE / 'column_tree_full_20261108'
REGISTRATION = SUITE / 'registration.json'
STATE = SUITE / 'state.json'
CPU = ROOT / '.venv/bin/python'
RATIOS = [.1, .25, .5, .75]
KINDS = ['col_adaptive', 'quadtree_question']
SCIENTIFIC_SOURCES = [
    'vqa/compression_generators_v2.py', 'vqa/compression_generators.py',
    'vqa/compression_generators_v2_repair.py', 'vqa/official_und.py',
    'vqa/decoder_dependent_loss.py', 'vqa/benchmark_metrics.py',
    'scripts/train_compression_lab_v2.py', 'scripts/train_compression_lab_v2_distributed.py',
    'scripts/train_compression_lab_v2_repaired.py', 'scripts/train_compression_lab.py',
    'scripts/train_meta_compress.py', 'scripts/train_meta_compress_strict.py',
    'scripts/eval_token_proxy.py', 'scripts/benchmark_compression_lab_v2.py',
    'scripts/compression_lab_v2_io.py', 'scripts/compression_lab_v2_io_retry.py',
    'scripts/benchmark_compression_lab_v2_io.py', 'scripts/run_column_tree_full_stage.py',
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def benchmark_rows():
    return [r for r in rows(ROOT / 'data/full/val.jsonl') if r['task'] in ['DocVQA', 'InfoVQA']] + rows(ROOT / 'data/full/test.jsonl')


def prepare():
    SUITE.mkdir(parents=True, exist_ok=True)
    if REGISTRATION.exists():
        verify()
        print('Existing immutable registration retained.')
        return
    template_path = BASE / 'configs/full_quadtree_question_pre-merger_r0.5.json'
    template = load(template_path)
    legacy = load(BASE / 'full/full_quadtree_question_pre-merger_r0.5/manifest.json')
    assert template == legacy['config']
    for p, expected in legacy['source_sha256'].items():
        assert sha(ROOT / p) == expected, f'Legacy scientific source changed: {p}'
    paths = {k: template[k] for k in ['train_rows', 'validation_rows', 'audit_rows']}
    paths.update(formal_val_source='data/full/val.jsonl', formal_test_source='data/full/test.jsonl')
    cohort = {k: rows(ROOT / p) for k, p in paths.items()}
    cohort['formal_benchmark'] = benchmark_rows()
    assert len(cohort['train_rows']) == 20000 and len(cohort['formal_benchmark']) == 23565
    assert len(cohort['validation_rows']) == 56 and len(cohort['audit_rows']) == 14
    ids = {k: {r['document_id'] for r in rr} for k, rr in cohort.items()}
    for a, b in [('train_rows', 'validation_rows'), ('train_rows', 'formal_benchmark'),
                 ('validation_rows', 'formal_benchmark'), ('audit_rows', 'formal_benchmark')]:
        assert not ids[a] & ids[b], f'Document leakage: {a}/{b}'
    records = []
    config_dir = SUITE / 'configs'
    config_dir.mkdir(exist_ok=True)
    for kind in KINDS:
        for ratio in RATIOS:
            name = f'full_{kind}_pre-merger_r{ratio:g}'
            reuse = kind == 'quadtree_question' and ratio in [.1, .5]
            path = BASE / 'configs' / f'{name}.json' if reuse else config_dir / f'{name}.json'
            if reuse:
                cfg = load(path)
                assert (ROOT / cfg['output'] / 'benchmark/complete.json').exists()
            else:
                cfg = dict(template, name=name, kind=kind, ratio=ratio,
                           output=f'artifacts/compression_lab_v2/full/{name}',
                           selection='User-authorized fixed paired comparison 20261108; no screening gate, no test-based seed/epoch/architecture selection',
                           initialization='fresh generator seed20261105; no pilot/expanded warm start')
                assert not path.exists() and not (ROOT / cfg['output']).exists()
                save(path, cfg)
            assert all(cfg[k] == template[k] for k in ['train_rows', 'validation_rows', 'audit_rows', 'seed', 'epochs', 'lr', 'clip', 'batch_size', 'insertion'])
            records.append(dict(name=name, kind=kind, ratio=ratio, reuse=reuse,
                                config=str(path.relative_to(ROOT)), config_sha256=sha(path), output=cfg['output']))
    # Preserve the exact filtered benchmark, including source order, for external auditing.
    snapshot = SUITE / 'formal_benchmark.jsonl'
    snapshot.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cohort['formal_benchmark']))
    data = {k: dict(path=p, sha256=sha(ROOT / p), questions=len(cohort[k]),
                   images=len(ids[k]), by_task=dict(collections.Counter(r['task'] for r in cohort[k]))) for k, p in paths.items()}
    data['formal_benchmark'] = dict(path=str(snapshot.relative_to(ROOT)), sha256=sha(snapshot),
                                  questions=23565, images=len(ids['formal_benchmark']),
                                  by_task=dict(collections.Counter(r['task'] for r in cohort['formal_benchmark'])))
    registration = dict(
        registered_utc=now(), suite='column_tree_full_20261108',
        user_request='Full Column training at 10%/25%/50%/75%, new Tree75%, and combined existing Tree10%/25%/50% reporting with identical data splits',
        clarification='Legacy25% was global_topk, not quadtree_question; also train missing tree25% to make complete2x4 comparison. Preserve legacyglobal_topk as separate reference.',
        runs=records, new_runs=sum(not r['reuse'] for r in records), reused_runs=sum(r['reuse'] for r in records),
        dataset=data, train_validation_benchmark_document_disjoint=True,
        historical_audit_note=dict(images_shared_with_full_train=len(ids['audit_rows'] & ids['train_rows']),
                                   total_audit_images=len(ids['audit_rows']),
                                   text='Keep original14QA/7image historical train-audit set exactly; it is not wholly a subset of the20k split. Report full-train epoch KL separately.'),
        scientific_source_sha256={p: sha(ROOT / p) for p in SCIENTIFIC_SOURCES},
        reference_config=str(template_path.relative_to(ROOT)), reference_config_sha256=sha(template_path),
        protocol=dict(insertion='pre-merger', backbone='models/MonkeyOCRv2-B-Und',
                      frozen=['encoder', 'native merger', 'Qwen3-1.7B'], loss='KL-only teacher||student, common teacher greedy prefixes; T=1',
                      epochs=4, global_updates=2500, updates_per_epoch=625, batch_size=32,
                      seed=20261105, optimizer='AdamW', lr=3e-4, weight_decay=0., clip=1.,
                      warmup_updates=250, schedule='cosine to10%peak', initialization='fresh',
                      checkpoint_every_updates=25, endpoint='fixedepoch4; no validation/test checkpoint reselection',
                      train_teacher_generation_cap=64, validation_generation_cap=64, benchmark_generation_cap=256,
                      benchmark_variants=['initial', 'trained', 'mean', 'selection', 'cell_mean', 'cell_selection'],
                      train_world=8, benchmark_world=8, max_live_stage_jobs=2, max_stage_attempts=3,
                      benchmark_note='Same locked23565QA as prior experiments, including DocVQA/InfoVQA official validation labels plus other tasks test; already viewed historical benchmark, not a new blind holdout.'),
        legacy_global_topk_reference='artifacts/compression_lab_v2/configs/full_global_topk_pre-merger_r0.25.json',
        created_by='scripts.column_tree_full_suite')
    save(REGISTRATION, registration)
    save(STATE, dict(created_utc=now(), runs={r['name']: dict(stage='complete' if r['reuse'] else 'registered', jobs=[], reused=r['reuse']) for r in records}))
    print(json.dumps(dict(new_runs=registration['new_runs'], reused_runs=registration['reused_runs'], registration=str(REGISTRATION)), ensure_ascii=False))


def verify(config=None):
    reg = load(REGISTRATION)
    assert reg and len(reg['runs']) == 8
    for p, expected in reg['scientific_source_sha256'].items():
        assert sha(ROOT / p) == expected, f'Registered scientific source changed: {p}'
    for meta in reg['dataset'].values():
        assert sha(ROOT / meta['path']) == meta['sha256'], f'Dataset changed: {meta["path"]}'
    for record in reg['runs']:
        assert sha(ROOT / record['config']) == record['config_sha256']
    if config:
        assert str(Path(config)) in {r['config'] for r in reg['runs'] if not r['reuse']}
    return reg


def queue():
    text = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%j|%T|%R'], text=True)
    return {s[0]: dict(name=s[1], state=s[2], reason=s[3]) for line in text.splitlines() if len(s := line.split('|', 3)) == 4}


def report():
    subprocess.run([str(CPU), '-m', 'scripts.report_column_tree_full'], cwd=ROOT, check=True)


def ensure_controller():
    prepare()
    live = queue()
    if any(j['name'] == 'column-tree-full' for j in live.values()):
        print('Existing suite controller retained.')
        return
    job = subprocess.check_output(['sbatch', '--parsable', 'scripts/slurm_column_tree_full_controller.sh'], cwd=ROOT, text=True).strip().split(';')[0]
    save(SUITE / 'controller_submission.json', dict(job=job, submitted_utc=now()))
    print(json.dumps(dict(controller_job=job, status='submitted')))


def advance(reg, state):
    live = queue()
    active = 0
    for record in reg['runs']:
        st = state['runs'][record['name']]
        if record['reuse']:
            continue
        out = ROOT / record['output']
        last = st['jobs'][-1] if st['jobs'] else None
        if last and last['job'] in live:
            st.update(stage=last['phase'] + '_active', scheduler=live[last['job']])
            active += 1
            continue
        if (out / 'benchmark/complete.json').exists():
            st.update(stage='complete', scheduler=None)
            continue
        phase = 'benchmark' if (out / 'complete.json').exists() else 'train'
        if last and last['phase'] == phase:
            if not last.get('observed_exit_utc'):
                last['observed_exit_utc'] = now()
            if (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(last['observed_exit_utc'])).total_seconds() < 90:
                st['stage'] = phase + '_awaiting_committed_artifacts'
                continue
            attempts = sum(j['phase'] == phase for j in st['jobs'])
            if attempts >= reg['protocol']['max_stage_attempts']:
                st.update(stage=phase + '_failed', error='Stage exited without completion after3attempts; see per-job logs; no scientific protocol relaxed.')
                continue
        st['stage'] = phase + '_ready'
    save(STATE, state)
    # Finish an already trained run before starting a new one; registration order fixed.
    waiting = sorted([r for r in reg['runs'] if state['runs'][r['name']]['stage'] in ['train_ready', 'benchmark_ready']],
                     key=lambda r: state['runs'][r['name']]['stage'] != 'benchmark_ready')
    for record in waiting:
        if active >= reg['protocol']['max_live_stage_jobs']:
            break
        st = state['runs'][record['name']]
        phase = st['stage'].removesuffix('_ready')
        jobname = ('ct-tr-' if phase == 'train' else 'ct-ev-') + record['name'].replace('full_', '').replace('quadtree_question', 'tree').replace('col_adaptive', 'col').replace('_pre-merger_r', '-r')
        cmd = ['sbatch', '--parsable', '--job-name=' + jobname, 'scripts/slurm_column_tree_full_stage.sh', phase, record['config']]
        job = subprocess.check_output(cmd, cwd=ROOT, text=True).strip().split(';')[0]
        st['jobs'].append(dict(job=job, phase=phase, submitted_utc=now(), command=cmd))
        st.update(stage=phase + '_active', scheduler=dict(state='SUBMITTED'))
        save(STATE, state)
        active += 1
        print(json.dumps(dict(run=record['name'], phase=phase, job=job)), flush=True)
    state['updated_utc'] = now()
    save(STATE, state)


def control():
    SUITE.mkdir(parents=True, exist_ok=True)
    with (SUITE / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing controller retained.')
            return
        save(SUITE / 'controller_runtime.json', dict(job=os.environ.get('SLURM_JOB_ID'), pid=os.getpid(), started_utc=now()))
        reg = verify()
        while True:
            verify()
            state = load(STATE)
            advance(reg, state)
            report()
            stages = {k: v['stage'] for k, v in state['runs'].items()}
            print(json.dumps(dict(updated_utc=now(), stages=stages)), flush=True)
            if all(s == 'complete' or s.endswith('_failed') for s in stages.values()):
                if all(s == 'complete' for s in stages.values()):
                    subprocess.run([str(CPU), '-m', 'scripts.report_column_tree_full', '--require-complete'], cwd=ROOT, check=True)
                    save(SUITE / 'complete.json', dict(completed_utc=now(), runs=8, newly_trained=6, reused=2))
                else:
                    save(SUITE / 'attention_required.json', dict(updated_utc=now(), stages=stages))
                return
            time.sleep(45)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'resume', 'status', 'report', 'control', 'verify'])
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(); report()
    elif a.action == 'resume':
        ensure_controller()
    elif a.action in ['status', 'report']:
        report()
        print(json.dumps(load(STATE), ensure_ascii=False, indent=2))
        if a.action == 'status':
            print(json.dumps(queue(), ensure_ascii=False, indent=2))
    elif a.action == 'verify':
        verify(); print('Registered sources, all split hashes and all8configs verified.')
    else:
        control()


if __name__ == '__main__':
    main()
