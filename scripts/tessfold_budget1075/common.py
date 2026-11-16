from pathlib import Path
from scripts.tessfold_repair.common import ROOT, load, lines, save, sha, now

PARENT = ROOT / 'artifacts/tessfold_acceleration_20261115'
BASE = PARENT / 'budget1075'
REGISTRY = BASE / 'qwen_selected_runs.json'
MODELS = [('Column', .1), ('Tree', .1), ('Column', .75), ('Tree', .75)]
REPORT = ROOT / 'docs/TESSFOLD_8B_10_75_RESULTS.md'


def model_id(kind, ratio):
    return f'{kind}_r{ratio:g}'


def refresh(path):
    """Invalidate shared-filesystem stale negative directory entries."""
    path = Path(path)
    directory = path.parent
    # Refresh ancestors first: a newly created run directory itself can be
    # hidden by a cached negative dentry on a different Slurm node.
    for ancestor in reversed([directory, *list(directory.parents)[:2]]):
        try:
            list(ancestor.iterdir())
        except FileNotFoundError:
            pass
    return path


def completed(label):
    return load(refresh(BASE / label / 'complete.json'))


def candidates(kind):
    result = [dict(name='default', lr=3e-4, batch=32, stable=False),
              dict(name='low', lr=1e-4, batch=32, stable=False),
              dict(name='high', lr=1e-3, batch=32, stable=False)]
    if kind == 'Column':
        result.append(dict(name='stable', lr=1e-3, batch=16, stable=True))
    return result


def pilot_label(kind, ratio, attempt):
    return 'pilot_default' if attempt['name'] == 'default' else f'pilot_{attempt["name"]}_{model_id(kind, ratio)}'


def full_label(kind, ratio):
    return f'full_{model_id(kind, ratio)}'


def register():
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / 'registration.json'
    if path.exists():
        return load(path)
    sources = [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'scripts/tessfold_budget1075').glob('*'))
               if p.suffix in ['.py', '.sh']]
    sources += ['scripts/tessfold_accel/qwen_training.py', 'scripts/tessfold_accel/qwen_training_stable.py',
                'scripts/tessfold_accel/run_qwen_safe.py', 'scripts/tessfold_accel/qwen_formal.py',
                'scripts/tessfold_accel/qwen_eval_safe.py', 'scripts/tessfold_accel/qwen_memory.py',
                'scripts/tessfold_accel/training_audit_npu.py', 'scripts/tessfold_accel/qwen_timing_stats.py',
                'scripts/tessfold_accel/qwen_control_stats.py', 'vqa/tessfold_qwen3vl.py',
                'vqa/tessfold_qwen3vl_stable.py', 'vqa/tessfold_qwen3vl_timing.py',
                'vqa/tessfold_native_partition.py', 'vqa/tessfold_partition_native.cpp', 'vqa/benchmark_metrics.py']
    config = load(ROOT / 'artifacts/compression_lab_v2/configs/full_quadtree_question_pre-merger_r0.5.json')
    data = [config['train_rows'], config['validation_rows'],
            'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl',
            'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl',
            str((PARENT / 'qwen_initial_control_rows.jsonl').relative_to(ROOT))]
    # Carry forward already verified identity gates without changing their sources.
    for label in ['qwen_identity56', 'qwen_timing_identity']:
        original = PARENT / label / 'complete.json'
        save(BASE / label / 'complete.json', load(original))
    train, dev, formal = (lines(ROOT / p) for p in data[:3])
    sets = [{r['document_id'] for r in rows} for rows in [train, dev, formal]]
    assert [len(rows) for rows in [train, dev, formal]] == [20000, 56, 23565]
    assert all(not sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    result = dict(time=now(), models=[dict(id=model_id(k, r), kind=k, ratio=r) for k, r in MODELS],
                  source_sha256={s: sha(ROOT / s) for s in sources}, data_sha256={s: sha(ROOT / s) for s in data},
                  inherited_final_audit_sha256=sha(PARENT / 'final_audit.json'),
                  base_model='Qwen3-VL-8B-Instruct', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                  seed=20261115, train_questions=20000, train_images=len(sets[0]), epochs=4,
                  dev_questions=56, formal_questions=23565, formal_images=len(sets[2]),
                  pilot_questions=512, pilot_epochs=3, candidates={k: candidates(k) for k in ['Column', 'Tree']},
                  candidate_selection='First passing pilot in fixed default/low/high/(Column stable) order; development data only.',
                  full_selection='Same eligible KL>=5% reduction and generation>=0.25pp gain gate as previous suite; choose best dev macro, tie lower KL; otherwise initial.',
                  loss='Full-vocabulary common teacher greedy-prefix KL only, no entropy or gold training.',
                  optimizer='AdamW wd0, clip5, 10% warmup/cosine to 0.1LR; fixed initial seed and 4 epochs.',
                  frozen='All base parameters, native vision, mergers, DeepStack and language model. Train P only.',
                  insertion='After native mergers; identical P on final features and all 3 DeepStack branches; representative MRoPE.',
                  controls='Same pre-fixed 512 document clusters / 2404 QA; no test-driven selection.',
                  measurement='90 QA x 5 models x 2 modes x 3 repeats = 2700 timing; independent 1350 memory observations.',
                  formal_expected_records=127441, single_seed=True,
                  original_pending_pilot_cancelled='166097 requested16NPU; no work executed; replaced by8NPU job166098.',
                  adopted_default_pilot_job='166098')
    save(path, result)
    return result


def verify():
    registration = load(BASE / 'registration.json')
    assert registration
    for field in ['source_sha256', 'data_sha256']:
        for path, digest in registration[field].items():
            assert sha(ROOT / path) == digest, f'Changed registered file: {path}'
    return registration
