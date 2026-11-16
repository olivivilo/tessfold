"""Recompute complete-corpus scores and verify saved checkpoints/public inference."""
import argparse
import collections
import hashlib
import json
import math
import os
import subprocess
import sys

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from safetensors import safe_open
from vqa.model import resolve
from vqa.benchmark_metrics import native_score, due_field_f1
from vqa.full_train import atomic_json


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def read_weight(directory, key):
    index = directory / 'model.safetensors.index.json'
    shard = json.loads(index.read_text())['weight_map'][key] if index.exists() else 'model.safetensors'
    with safe_open(directory / shard, framework='pt', device='cpu') as handle:
        return handle.get_tensor(key).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/full.json')
    parser.add_argument('--run-infer', action='store_true', help='Requires the configured accelerator on a compute node')
    args = parser.parse_args()
    cfg = json.loads(resolve(args.config).read_text())
    out = resolve(cfg['output_dir'])
    fingerprint = json.loads((out / 'corpus_fingerprint.json').read_text())
    for split, digest in fingerprint.items():
        assert hashlib.sha256((resolve(cfg['data_dir']) / f'{split}.jsonl').read_bytes()).hexdigest() == digest, split
    encoder = resolve(cfg['vision_model']) / 'model.safetensors'
    digest = hashlib.sha256()
    with encoder.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == 'a9f059fc5a7afb6e9d2ad64b5db02dfc8e846d4b713460f4b3c0b2468a4d52e1', 'Encoder weights changed'
    train_size = len(read_jsonl(resolve(cfg['data_dir']) / 'train.jsonl'))
    stage_info = {}
    projectors = {}
    for stage in (1, 2):
        directory = out / f'stage{stage}'
        info = json.loads((directory / 'complete.json').read_text())
        expected = math.ceil(train_size / cfg['effective_batch_size']) * cfg[f'stage{stage}_epochs']
        assert info['steps'] == expected and info['train_samples'] == train_size, info
        projectors[stage] = torch.load(directory / 'projector.pt', map_location='cpu', weights_only=True)
        assert all(value.device.type == 'cpu' and torch.isfinite(value).all() for value in projectors[stage].values())
        stage_info[stage] = info
    assert any(not torch.equal(projectors[1][key], projectors[2][key]) for key in projectors[1]), 'Stage 2 MLP did not change'
    key = 'model.layers.0.self_attn.q_proj.weight'
    delta = (read_weight(out / 'stage2/language_model', key) - read_weight(resolve(cfg['language_model']), key)).abs()
    assert float(delta.max()) > 0, 'Stage 2 Qwen did not change'
    metrics_recomputed = {}
    infer_row = None
    for split in cfg['eval_splits']:
        source = read_jsonl(resolve(cfg['data_dir']) / f'{split}.jsonl')
        source_by_id = {row['id']: row for row in source}
        predicted = {}
        for path in sorted(out.glob(f'final_{split}.rank*.jsonl')):
            for row in read_jsonl(path):
                assert row['id'] in source_by_id, f'Unexpected prediction: {row["id"]}'
                if row['id'] in predicted:
                    assert predicted[row['id']] == row, f'Conflicting duplicate: {row["id"]}'
                predicted[row['id']] = row
        assert set(predicted) == set(source_by_id), f'Incomplete {split} evaluation'
        groups = collections.defaultdict(list)
        for row in predicted.values():
            original = source_by_id[row['id']]
            assert all(row[key] == value for key, value in original.items()), row['id']
            score = native_score(row['prediction'], original)
            assert math.isclose(score, row['score'], abs_tol=1e-12), row['id']
            groups[row['task']].append(row)
        reported = json.loads((out / f'final_{split}_metrics.json').read_text())
        assert reported['samples'] == len(source) and not reported['limit_per_task'] and not reported['blank_image']
        assert set(groups) == set(reported['benchmarks'])
        computed = {}
        for task, rows in groups.items():
            score = due_field_f1(rows)['score_percent'] if task in ('DeepForm', 'KLC') else 100 * sum(r['score'] for r in rows) / len(rows)
            recorded = reported['benchmarks'][task]
            assert recorded['samples'] == len(rows) and math.isclose(score, recorded['score_percent'], abs_tol=1e-8), task
            computed[task] = {'samples': len(rows), 'score_percent': score}
        metrics_recomputed[split] = computed
        if split == 'test':
            infer_row = predicted[source[0]['id']]
    result = {'passed': True, 'encoder_original_sha256': digest.hexdigest(), 'corpus_fingerprints_unchanged': True,
              'stages': stage_info, 'projector_updated_in_stage2': True,
              'llm_weight_check': {'tensor': key, 'mean_absolute_update': float(delta.mean()), 'max_absolute_update': float(delta.max())},
              'full_metrics_recomputed': metrics_recomputed}
    if args.run_infer:
        assert infer_row is not None, 'Public inference check requires a test split'
        command = [sys.executable, '-m', 'vqa.full_infer', '--config', args.config,
                   '--image', infer_row['image'], '--question', infer_row['question']]
        completed = subprocess.run(command, cwd=resolve('.'), text=True, capture_output=True, timeout=900)
        (out / 'infer_verification_stdout.txt').write_text(completed.stdout)
        (out / 'infer_verification_stderr.txt').write_text(completed.stderr)
        completed.check_returncode()
        output = json.loads(completed.stdout.strip().splitlines()[-1])
        assert output['answer'] == infer_row['prediction'], (output['answer'], infer_row['prediction'])
        result['public_infer_entry'] = {'passed': True, 'sample_id': infer_row['id'], 'prediction': output['answer'],
                                        'image_features': 'Recomputed from the original image; saved stage-2 MLP and Qwen reloaded'}
    atomic_json(out / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
