"""Closed-form geometric Q initializer, using training image grid shapes only.

This is an initializer diagnostic, not VQA training or downstream evidence.
It assigns the existing Q weight once. The forward and full paper loss stay
unchanged. No questions, answers, decoder forwards, or validation images enter
the fit. Original K/position parameters are left at the dense-position init.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from scripts.report_meta_compress_strict import ROOT, BASE, read, save
from vqa.meta_compress_dense_position_initialization import initialize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--ratio', type=float, default=.75)
    p.add_argument('--ridge', type=float, default=.001)
    p.add_argument('--fit-grids', type=int, default=24)
    p.add_argument('--rows-per-grid', type=int, default=256)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    cfg = {'seed': a.seed, 'insertion': 'post-merger', 'initialization': 'fourier-dense-position',
           'position_strength': 512., 'projection_gain': .03125}
    shapes = set()
    for doc in sorted({r['document_id'] for r in read(BASE/'pilot.jsonl')}):
        item = torch.load(BASE/'npu_cache/features'/f'{doc}.pt', map_location='cpu', weights_only=True, mmap=True)
        assert item['source'] == 'MonkeyOCRv2-B-Und-native-bf16-v1'
        shapes.add(tuple(int(x) for x in item['grid'].reshape(-1).tolist()))
    shapes = sorted(shapes, key=lambda s: hashlib.sha256(f'geometry:{s}'.encode()).hexdigest())
    count = min(a.fit_grids, len(shapes)*3//4)
    fit, audit = shapes[:count], shapes[count:count+8]
    assert fit and audit
    module = initialize(cfg)
    us, ys = [], []
    with torch.no_grad():
        for shape in fit:
            grid = torch.tensor([shape]); n = int(grid.prod()); m = module.output_length(n, a.ratio)
            pos = module._position(n, grid, torch.device('cpu'), torch.float32)
            u = module.pool(pos, m)
            target = ((torch.arange(m)+.5)*n/m).long().clamp_max(n-1)
            sample = torch.linspace(0, m-1, min(a.rows_per_grid, m)).round().long()
            us.append(u[sample].double()); ys.append(module.key(pos[target[sample]]).double())
        u = torch.cat(us); y = torch.cat(ys)
        cov = u.T@u/len(u); cross = u.T@y/len(u)
        ridge = a.ridge*float(cov.diagonal().mean())
        weight = torch.linalg.solve(cov+ridge*torch.eye(cov.shape[0], dtype=cov.dtype), cross).T.float()
        diagnostics = {}
        for kind in ['original', 'geometric-ridge']:
            original = module.query.weight.detach().clone()
            if kind == 'geometric-ridge':
                module.query.weight.copy_(weight)
            stats = []
            for shape in audit:
                grid = torch.tensor([shape]); n = int(grid.prod()); m = module.output_length(n, a.ratio)
                _, prob = module(torch.zeros(n, module.feature_dim), grid, a.ratio)
                idx = prob.argmax(-1); target = ((torch.arange(m)+.5)*n/m).long().clamp_max(n-1)
                stats.append({'grid': shape, 'tokens': n, 'rows': m,
                              'argmax_unique_fraction': len(idx.unique())/m,
                              'uniform_target_accuracy': float((idx == target).float().mean()),
                              'serialized_displacement_fraction': float((idx-target).abs().float().mean()/n),
                              'entropy': float(-(prob*prob.clamp_min(1e-12).log()).sum(-1).mean()),
                              'collapse': float(prob.sum(0).max()),
                              'row_sum_error': float((prob.sum(-1)-1).abs().max())})
            diagnostics[kind] = stats
            module.query.weight.copy_(original)
    name = f'seed{a.seed}_keep{a.ratio:g}_ridge{a.ridge:g}'
    out = BASE/'geometric_calibration'/name
    out.mkdir(parents=True, exist_ok=True)
    path = out/'query.pt'
    assert not path.exists(), 'Immutable initialization; choose a new configuration rather than overwrite.'
    torch.save({'query_weight': weight, 'config': cfg, 'fit_shapes': fit, 'audit_shapes': audit,
                'rows': len(u), 'ridge_scale': a.ridge, 'actual_ridge': ridge,
                'source': 'Training-only grid geometry; one-time closed-form Q parameter initialization'}, path)
    result = {'config': cfg, 'ratio': a.ratio, 'fit_shapes': fit, 'audit_shapes': audit,
              'fit_samples': len(u), 'ridge_fraction': a.ridge, 'actual_ridge': ridge,
              'query_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'diagnostics': diagnostics,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'pilot_sha256': hashlib.sha256((BASE/'pilot.jsonl').read_bytes()).hexdigest(),
              'limitation': 'Geometry audit only, source features zero; neither an actual VQA result nor proof of training improvement.'}
    save(out/'RESULTS.json', result)
    for kind, rr in diagnostics.items():
        print(kind, {k: sum(r[k] for r in rr)/len(rr) for k in ['argmax_unique_fraction',
              'uniform_target_accuracy', 'serialized_displacement_fraction', 'entropy', 'collapse']}, flush=True)


if __name__ == '__main__':
    main()
