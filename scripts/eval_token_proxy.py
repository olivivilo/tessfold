"""Independent teacher/student generations for every random convex compression."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from vqa.official_und import ROOT, OfficialUND
from vqa.benchmark_metrics import native_score
from vqa.jsonl import recover_predictions
from vqa.token_proxy import BalancedMatrix, VisualProxy, image_edge_weights, patch_geometry, random_matrix, stable_seed


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def cached_vision(tower, projected, raw):
    """Replace only the already-executed encoder result, restoring it after each call."""
    original = tower.forward
    calls = [0]

    def forward(*args, **kwargs):
        calls[0] += 1
        assert calls[0] == 1, 'Vision must only be consumed once, at independent prefill'
        return projected, raw

    tower.forward = forward
    try:
        yield
        assert calls[0] == 1
    finally:
        tower.forward = original


def shortened_inputs(inputs, m, image_token):
    ids = inputs['input_ids']
    positions = (ids[0] == image_token).nonzero().flatten()
    assert len(positions) > 0 and int(positions[-1]-positions[0]+1) == len(positions)
    lo, hi = int(positions[0]), int(positions[-1])+1
    newids = torch.cat((ids[:, :lo], ids.new_full((1, m), image_token), ids[:, hi:]), -1)
    result = {**inputs, 'input_ids': newids, 'attention_mask': torch.ones_like(newids)}
    assert int((newids == image_token).sum()) == m
    # No cached positions, past_key_values, answer text, or history is ever passed.
    assert not any(k in result for k in ('past_key_values', 'position_ids', 'cache_position'))
    return result


@torch.no_grad()
def generate(engine, inputs, max_new_tokens, projected=None, raw=None):
    start = time.monotonic()
    if projected is None:
        outputs = engine.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens)
    else:
        with cached_vision(engine.model.vision_tower, projected, raw):
            outputs = engine.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens)
    generated = outputs[0, inputs['input_ids'].shape[1]:]
    tokenids = generated.cpu().tolist()
    prediction = engine.processor.decode(generated, skip_special_tokens=True)
    return prediction, {'input_tokens': inputs['input_ids'].shape[1], 'new_tokens': len(tokenids),
                        'hit_token_limit': len(tokenids) == max_new_tokens,
                        'seconds': time.monotonic()-start, 'generated_ids': tokenids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='random_v1')
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    out = ROOT/'artifacts/token_proxy'/args.label
    protocol = json.loads((out/'protocol.json').read_text())
    rows = [json.loads(s) for s in (out/'samples.jsonl').read_text().splitlines()]
    assert hashlib.sha256((out/'samples.jsonl').read_bytes()).hexdigest() == protocol['sample_sha256']
    code_files = ['vqa/token_proxy.py', 'scripts/eval_token_proxy.py', 'vqa/official_und.py', 'vqa/benchmark_metrics.py']
    manifest = {'protocol_sha256': hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),
                'code_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in code_files},
                'rank': rank, 'world': world, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    mp = out/f'run.rank{rank}.json'
    if mp.exists():
        old = json.loads(mp.read_text())
        assert all(old[k] == manifest[k] for k in ('protocol_sha256', 'code_sha256', 'rank', 'world')), 'Resume settings changed'
    else:
        mp.write_text(json.dumps(manifest, indent=2))
    path, teacher_path = out/f'trials.rank{rank}.jsonl', out/f'teachers.rank{rank}.jsonl'
    done, teachers = recover_predictions(path), recover_predictions(teacher_path)
    (out/'matrices').mkdir(exist_ok=True)
    (out/'features').mkdir(exist_ok=True)
    engine = OfficialUND()
    engine.model.requires_grad_(False)
    tower = engine.model.vision_tower
    old_baseline = {}
    wanted = {r['id'] for r in rows}
    for p in (ROOT/'artifacts/official_und').glob('baseline_full.rank*.jsonl'):
        with p.open() as h:
            for line in h:
                row = json.loads(line)
                if row['id'] in wanted:
                    old_baseline[row['id']] = row['prediction']
    start = time.monotonic()
    with path.open('a') as handle, teacher_path.open('a') as thandle, torch.no_grad():
        for row_index, row in enumerate(rows[rank::world]):
            def trialid(t):
                return f"{row['id']}__{t['kind']}__{t['family']}__r{t['ratio']}__s{t['replicate']}"
            if row['id'] in teachers and all(trialid(t) in done for t in protocol['trial_configs']):
                continue
            inputs = engine.inputs(row)
            projected, raw = tower(inputs['pixel_values'], inputs['image_grid_thw'])
            assert raw.shape == (int(inputs['image_grid_thw'].prod()), 768)
            raw_cpu = raw.cpu()
            x = raw_cpu.float()
            grid = inputs['image_grid_thw'][0].cpu().tolist()
            u, edges = image_edge_weights(inputs['pixel_values'].cpu(), grid,
                                         engine.processor.image_processor.image_mean,
                                         engine.processor.image_processor.image_std)
            _, yy, xx = patch_geometry(grid)
            official_positions = tower.get_pos_ids_by_grid(inputs['image_grid_thw'].cpu())[0]
            assert torch.equal(official_positions, torch.stack((yy, xx), -1))
            feature_sha = digest(raw_cpu)
            feature_file = out/'features'/f"{row['document_id']}.pt"
            if not feature_file.exists():
                torch.save({'x': raw_cpu, 'u': u, 'edge_weights': edges, 'grid': grid, 'sha256': feature_sha}, feature_file)
            proxy = VisualProxy(x, u, edges, stable_seed('probes-v1', row['document_id']))
            if row['id'] in teachers:
                teacher = teachers[row['id']]
                assert teacher['feature_sha256'] == feature_sha
            else:
                prediction, runtime = generate(engine, inputs, protocol['max_new_tokens'], projected, raw)
                teacher = {**row, 'prediction': prediction, 'score': native_score(prediction, row), 'runtime': runtime,
                           'feature_sha256': feature_sha, 'grid': grid, 'n': len(x),
                           'prior_baseline_prediction': old_baseline.get(row['id']),
                           'prior_baseline_match': old_baseline.get(row['id']) == prediction,
                           'teacher_attention_entropy_fraction': proxy.teacher_attention_entropy_fraction,
                           'teacher_attention_max': proxy.teacher_attention_max}
                if row_index == 0:
                    # Native image path and cached-token path must give identical prefill logits AND generation.
                    native_prediction, native_runtime = generate(engine, inputs, protocol['max_new_tokens'])
                    assert native_runtime['generated_ids'] == runtime['generated_ids'], 'Native/cached generation mismatch'
                    native_logits = engine.model(**inputs, use_cache=False, logits_to_keep=1).logits.cpu()
                    with cached_vision(tower, projected, raw):
                        cached_logits = engine.model(**inputs, use_cache=False, logits_to_keep=1).logits.cpu()
                    torch.testing.assert_close(native_logits, cached_logits, atol=0, rtol=0)
                    identity = BalancedMatrix(torch.arange(len(x)), torch.ones(len(x)), len(x), floor=0.)
                    identity_c, identity_loss = proxy(identity)
                    identity_projected = tower.merger(identity_c.to(device=engine.device, dtype=raw.dtype))
                    assert torch.equal(identity_projected, projected), 'Identity changed projected features'
                    identity_pred, identity_runtime = generate(engine, inputs, protocol['max_new_tokens'], identity_projected, raw)
                    assert identity_runtime['generated_ids'] == runtime['generated_ids']
                    teacher['interface_checks'] = {'native_cached_logits_max_abs': float((native_logits-cached_logits).abs().max()),
                                                   'native_cached_generated_ids_equal': True,
                                                   'identity_projection_equal': True, 'identity_generation_equal': True,
                                                   'identity_loss': {k: float(v) for k,v in identity_loss.items()}}
                thandle.write(json.dumps(teacher, ensure_ascii=False)+'\n')
                thandle.flush()
                teachers[row['id']] = teacher
            for trial in protocol['trial_configs']:
                tid = trialid(trial)
                if tid in done:
                    continue
                trial_start = time.monotonic()
                seed = stable_seed('random-matrix-v1', row['document_id'], trial['family'], trial['ratio'], trial['replicate'], trial['kind'])
                matrix = random_matrix(u, trial['ratio'], trial['family'], seed, logit_bound=trial['logit_bound'])
                diagnostic = matrix.diagnostics()
                assert diagnostic['row_sum_max_error'] < 1e-5
                assert diagnostic['column_mass_min_relative'] >= .149
                assert diagnostic['column_mass_max_relative'] <= 8.001
                c, losses = proxy(matrix)
                losses = {k: float(v) for k,v in losses.items()}
                assert all(np.isfinite(v) for v in losses.values())
                matrix_file = out/'matrices'/f'{hashlib.sha256(tid.encode()).hexdigest()}.npz'
                np.savez_compressed(matrix_file, group=matrix.group.numpy(), weight=matrix.weight.numpy(),
                                    m=matrix.m, floor=matrix.floor, seed=seed)
                compressed = c.to(device=engine.device, dtype=raw.dtype)
                projection = tower.merger(compressed)
                student_inputs = shortened_inputs(inputs, matrix.m, engine.model.config.image_token_id)
                prediction, runtime = generate(engine, student_inputs, protocol['max_new_tokens'], projection, compressed)
                score = native_score(prediction, row)
                result = {'id': tid, 'sample_id': row['id'], 'task': row['task'], 'document_id': row['document_id'],
                          'question': row['question'], 'image': row['image'], 'answers': row['answers'],
                          **trial, 'matrix_seed': seed, 'matrix_file': str(matrix_file.relative_to(ROOT)),
                          'matrix_file_sha256': hashlib.sha256(matrix_file.read_bytes()).hexdigest(),
                          'feature_sha256': feature_sha, 'compressed_feature_sha256': digest(compressed),
                          'matrix_diagnostics': diagnostic, 'loss': losses,
                          'teacher_prediction': teacher['prediction'], 'teacher_score': teacher['score'],
                          'student_prediction': prediction, 'student_score': score,
                          'score_drop': teacher['score']-score, 'runtime': runtime,
                          'bf16_quantization_relative_l2': float((compressed.cpu().float()-c).norm()/(c.norm()+1e-8)),
                          'trial_seconds': time.monotonic()-trial_start}
                handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                handle.flush()
                done[tid] = result
            print(json.dumps({'rank': rank, 'questions_done': row_index+1, 'questions_total': len(rows[rank::world]),
                              'trials_done': len(done), 'elapsed_seconds': time.monotonic()-start,
                              'task': row['task'], 'teacher': teacher['prediction'], 'gold': row['answers']}, ensure_ascii=False), flush=True)
    (out/f'rank{rank}.complete.json').write_text(json.dumps({'questions': len(rows[rank::world]), 'trials': len(done)}))


if __name__ == '__main__':
    main()
