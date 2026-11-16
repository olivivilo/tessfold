"""Verify record pairing, scores, hashes, matrix bounds, and sampled loss replay."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from vqa.benchmark_metrics import native_score
from vqa.token_proxy import BalancedMatrix, VisualProxy, stable_seed

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='random_v1')
    p.add_argument('--replay', type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(4)
    out = ROOT/'artifacts/token_proxy'/a.label
    protocol = json.loads((out/'protocol.json').read_text())
    rows, teachers = [], {}
    for path in out.glob('trials.rank*.jsonl'):
        rows.extend(map(json.loads, path.read_text().splitlines()))
    for path in out.glob('teachers.rank*.jsonl'):
        for r in map(json.loads, path.read_text().splitlines()):
            assert r['id'] not in teachers
            teachers[r['id']] = r
    assert len(teachers) == protocol['questions']
    assert len(rows) == protocol['random_trials']+protocol['control_trials']
    assert len({r['id'] for r in rows}) == len(rows)
    samples = {r['id']: r for r in map(json.loads, (out/'samples.jsonl').read_text().splitlines())}
    for r in teachers.values():
        assert abs(native_score(r['prediction'], samples[r['id']])-r['score']) < 1e-12
    pairings, max_error = collections.Counter(), 0.
    for r in rows:
        t = teachers[r['sample_id']]
        assert r['teacher_prediction'] == t['prediction'] and r['teacher_score'] == t['score']
        assert r['feature_sha256'] == t['feature_sha256']
        assert r['answers'] == t['answers'] and r['question'] == t['question']
        assert abs(native_score(r['student_prediction'], samples[r['sample_id']])-r['student_score']) < 1e-12
        assert abs(r['score_drop']-(r['teacher_score']-r['student_score'])) < 1e-12
        matrix_path = ROOT/r['matrix_file']
        assert hashlib.sha256(matrix_path.read_bytes()).hexdigest() == r['matrix_file_sha256']
        mat = np.load(matrix_path)
        group, weight = mat['group'], mat['weight']
        m, n = int(mat['m']), len(group)
        assert int(mat['seed']) == r['matrix_seed']
        assert m == max(1, int(n*r['ratio'])) and n == t['n']
        sums = np.bincount(group, weights=weight, minlength=m)
        error = float(np.max(np.abs(sums-1)))
        max_error = max(error, max_error)
        assert error < 1e-5 and np.min(weight) > 0
        size = np.bincount(group, minlength=m)
        assert size.min() > 0 and size.max()-size.min() <= 1
        assert r['matrix_diagnostics']['column_mass_min_relative'] >= .149
        if r['kind'] == 'random':
            assert r['matrix_diagnostics']['non_singleton_max_entry'] <= .851
        pairings[r['sample_id']] += 1
    assert set(pairings.values()) == {len(protocol['trial_configs'])}
    replayed, max_loss_error = [], 0.
    for r in sorted(rows, key=lambda r: stable_seed('audit', r['id']))[:a.replay]:
        f = torch.load(out/'features'/f"{r['document_id']}.pt", weights_only=True)
        mat = np.load(ROOT/r['matrix_file'])
        matrix = BalancedMatrix(torch.from_numpy(mat['group']), torch.from_numpy(mat['weight']), int(mat['m']), float(mat['floor']))
        proxy = VisualProxy(f['x'].float(), f['u'], f['edge_weights'], stable_seed('probes-v1', r['document_id']))
        _, loss = proxy(matrix)
        error = max(abs(float(loss[k])-r['loss'][k]) for k in loss)
        assert error < 2e-4, (r['id'], error)
        max_loss_error = max(max_loss_error, error)
        replayed.append({'trial_id': r['id'], 'max_loss_error': error})
    result = {'passed': True, 'questions': len(teachers), 'trials': len(rows), 'scores_recomputed': True,
              'matrix_hashes_verified': len(rows), 'max_support_row_sum_error': max_error,
              'loss_replay': replayed, 'max_loss_replay_error': max_loss_error}
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
