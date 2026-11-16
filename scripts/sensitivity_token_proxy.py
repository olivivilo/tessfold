"""Exploratory operator-temperature sensitivity; reuses tokens/P, never a decoder.

The primary pre-registered tau=1 result is unchanged. The grid is motivated by
observed near-uniform probe entropy, not selected to maximize downstream scores.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vqa.token_proxy import BalancedMatrix, VisualProxy, stable_seed

ROOT = Path(__file__).resolve().parents[1]
TAUS = [1., .1, .03]


def work(item):
    label, doc, rows = item
    torch.set_num_threads(1)
    out = ROOT/'artifacts/token_proxy'/label
    f = torch.load(out/'features'/f'{doc}.pt', weights_only=True)
    proxy = VisualProxy(f['x'].float(), f['u'], f['edge_weights'], stable_seed('probes-v1', doc))
    outputs, entropy = [], {}
    teacher_outputs = {}
    for tau in TAUS:
        a = (proxy.q @ proxy.kx.T/(math.sqrt(proxy.da)*tau)).softmax(-1)
        teacher_outputs[tau] = a @ proxy.vx
        entropy[str(tau)] = float(-(a*a.clamp_min(1e-30).log()).sum(-1).mean()/math.log(len(f['x'])))
    for row in rows:
        mat = np.load(ROOT/row['matrix_file'])
        matrix = BalancedMatrix(torch.from_numpy(mat['group']), torch.from_numpy(mat['weight']), int(mat['m']), float(mat['floor']))
        c = matrix.compress(proxy.x)
        zc = F.layer_norm(c, (c.shape[-1],))
        k, v = zc @ proxy.wk, zc @ proxy.wv
        values = {}
        for tau in TAUS:
            yt = teacher_outputs[tau]
            ys = (proxy.q @ k.T/(math.sqrt(proxy.da)*tau)).softmax(-1) @ v
            cos = (1-(yt*ys).sum(-1)/(yt.norm(dim=-1)*ys.norm(dim=-1)+1e-8)).mean()
            l2 = ((yt-ys).square().sum(-1)/(yt.square().sum(-1)+1e-8)).mean()
            op = float(cos+.5*l2)
            total = row['loss']['total']+2*(op-row['loss']['op'])
            values[str(tau)] = {'op': op, 'total': total}
            if tau == 1.:
                assert abs(op-row['loss']['op']) < 2e-4
        outputs.append({'id': row['id'], 'values': values})
    return outputs, entropy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='random_v1')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    out = ROOT/'artifacts/token_proxy'/a.label
    rows = []
    for path in out.glob('trials.rank*.jsonl'):
        rows.extend(r for r in map(json.loads, path.read_text().splitlines()) if r['kind'] == 'random')
    protocol = json.loads((out/'protocol.json').read_text())
    assert len(rows) == protocol['random_trials']
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r['document_id'], []).append(r)
    outputs, entropies = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, (result, entropy) in enumerate(pool.map(work, [(a.label, doc, rr) for doc, rr in by_doc.items()])):
            outputs.extend(result)
            entropies.append(entropy)
            if i % 24 == 0:
                print(json.dumps({'documents': i+1, 'total': len(by_doc)}), flush=True)
    (out/'temperature_sensitivity.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
    values = {r['id']: r['values'] for r in outputs}
    from scripts.report_token_proxy import describe, within
    summary = {'exploratory': True, 'primary_tau_unchanged': 1., 'temperature_grid': TAUS,
               'motivation': 'Unit-normalized queries plus sqrt(da) scaling produced entropy >.9995; sensitivity only, no model selection.',
               'per_tau': {}}
    for tau in TAUS:
        modified = [{**r, 'loss': {**r['loss'], **values[r['id']][str(tau)]}} for r in rows]
        summary['per_tau'][str(tau)] = {'all_random': describe(modified),
                                      'teacher_positive': describe([r for r in modified if r['teacher_score'] > 0]),
                                      'within_question': within(modified, ['sample_id']),
                                      'within_question_ratio_family': within(modified, ['sample_id', 'ratio', 'family']),
                                      'entropy_range': [min(e[str(tau)] for e in entropies), max(e[str(tau)] for e in entropies)]}
    (out/'temperature_sensitivity.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    # spawn avoids inherited BLAS worker state on hosts that default to fork.
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
