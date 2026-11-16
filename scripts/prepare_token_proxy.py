"""Pre-register deterministic sample, rates, random trials, and analysis choices."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_SPLITS = {'DocVQA': 'val', 'InfoVQA': 'val', 'ChartQA': 'test', 'DT-VQA': 'test',
               'DeepForm': 'test', 'KLC': 'test', 'WTQ': 'test', 'OCRBench': 'test'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='random_v1')
    p.add_argument('--per-task', type=int, default=24)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--cap-to-available', action='store_true',
                   help='Use all available distinct images when a task has fewer than --per-task')
    a = p.parse_args()
    out = ROOT/'artifacts/token_proxy'/a.label
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ('val', 'test'):
        with (ROOT/f'data/full/{split}.jsonl').open() as handle:
            rows.extend(r for r in map(json.loads, handle) if TASK_SPLITS.get(r['task']) == split)
    rows.sort(key=lambda r: hashlib.sha256(('token-proxy-v1-independent-'+r['id']).encode()).hexdigest())
    counts, seen, selected = collections.Counter(), set(), []
    for row in rows:
        if counts[row['task']] < a.per_task and row['image'] not in seen:
            selected.append(row)
            counts[row['task']] += 1
            seen.add(row['image'])
    available = {t: len({r['image'] for r in rows if r['task'] == t}) for t in TASK_SPLITS}
    targets = {t: min(a.per_task, available[t]) if a.cap_to_available else a.per_task for t in TASK_SPLITS}
    assert all(counts[t] == targets[t] for t in TASK_SPLITS), (dict(counts), targets)
    rates = [.125, .5] if a.smoke else [.125, .25, .375, .5, .625, .875]
    seeds = [0] if a.smoke else [0, 1]
    trials = [{'family': f, 'ratio': r, 'replicate': s, 'kind': 'random', 'logit_bound': 2.}
              for r in rates for f in ('local', 'global') for s in seeds]
    if not a.smoke:
        trials.extend({'family': 'local', 'ratio': r, 'replicate': 0, 'kind': 'mean_pool_control', 'logit_bound': 0.}
                      for r in (.25, .5))
    config = {'label': a.label, 'per_task': a.per_task, 'sample_seed': 'token-proxy-v1-independent',
              'tasks': TASK_SPLITS, 'trial_configs': trials, 'smoke': a.smoke,
              'model': 'models/MonkeyOCRv2-B-Und', 'adapter': None, 'max_new_tokens': 256,
              'max_pixels': 1003520, 'do_sample': False, 'dtype': 'bfloat16',
              'matrix_uniform_mix': .3, 'matrix_global_floor': 1e-6,
              'loss_version': 'lossplus section12 V1: .5rec+rel+2op+.1reg',
              'loss_constants': {'ridge': 1e-4, 'charbonnier_delta': .001, 'tau_op': 1., 'tau_rel': .1,
                                 'position_alpha': .25, 'fourier_frequencies': 4, 'attention_dim': 128,
                                 'anchors': 256, 'candidates': 512, 'random_queries': 64, 'self_queries': 64,
                                 'projection_seed': 20261016},
              'analysis': {'unit': 'individual question x random compression; no replicate score averaging',
                           'primary': 'Spearman(loss, teacher_score-student_score)',
                           'strong_threshold': .7, 'bootstrap_unit': 'image', 'bootstrap_repeats': 1000,
                           'strata': ['task', 'ratio', 'family', 'question'],
                           'conditioning': ['teacher_score>0', 'teacher_score==1'],
                           'ablations': ['A_rec', 'B_rec_rel', 'C_rec_op', 'D_rec_rel_op', 'total']},
              'counts': dict(counts), 'questions': len(selected),
              'random_trials': len(selected)*sum(t['kind'] == 'random' for t in trials),
              'control_trials': len(selected)*sum(t['kind'] != 'random' for t in trials)}
    if a.cap_to_available:
        config.update({'cap_to_available': True, 'available_unique_images': available,
                       'target_unique_images': targets})
    data = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected)
    config['sample_sha256'] = hashlib.sha256(data.encode()).hexdigest()
    config['lossplus_sha256'] = hashlib.sha256((ROOT/'lossplus.md').read_bytes()).hexdigest()
    for name, content in [('samples.jsonl', data), ('protocol.json', json.dumps(config, ensure_ascii=False, indent=2)+'\n')]:
        path = out/name
        if path.exists():
            assert path.read_text() == content, f'Existing preregistration differs: {path}'
        else:
            path.write_text(content)
    print(json.dumps({'output': str(out), 'questions': len(selected), 'random_trials': config['random_trials'],
                      'control_trials': config['control_trials']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
