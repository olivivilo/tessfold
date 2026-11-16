"""Audit independent VQA requests using the actual trained model (no weight changes)."""
import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from vqa.model import VLM, resolve
from vqa.full_train import amp, set_runtime, atomic_json

TARGET = 'Dr. Robert E. Shank'
OUT = resolve('artifacts/full/context_audit')


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_model(device):
    cfg = json.loads(resolve('configs/full.json').read_text())
    cfg.update(device=device, threads=8, language_model='artifacts/full/stage2/language_model',
               mixed_precision='bfloat16' if device == 'npu' else None)
    set_runtime(cfg)
    model = VLM(cfg)
    model.projector.load_state_dict(torch.load(resolve('artifacts/full/stage2/projector.pt'), map_location='cpu', weights_only=True))
    model.eval()
    return cfg, model


@torch.inference_mode()
def probe(model, cfg, row, condition, *, blank=False, use_cache=True, reference_id=None):
    captured = {}
    with amp(cfg):
        expected = model.embeds(row, blank_image=blank)
    expected_hash = tensor_hash(expected)
    expected_length = expected.shape[1]
    del expected

    def before_forward(module, args, kwargs):
        if captured:
            return
        cache = kwargs.get('past_key_values')
        length = int(cache.get_seq_length()) if cache is not None else 0
        embeddings = kwargs.get('inputs_embeds')
        captured.update(cache_type=type(cache).__name__, initial_kv_length=length,
                        prefill_length=embeddings.shape[1] if embeddings is not None else None,
                        prefill_sha256=tensor_hash(embeddings) if embeddings is not None else None,
                        cache_position_start=int(kwargs['cache_position'][0]) if kwargs.get('cache_position') is not None else None)

    hook = model.lm.register_forward_pre_hook(before_forward, with_kwargs=True)
    started = time.monotonic()
    try:
        with amp(cfg):
            if use_cache:
                prediction = model.predict(row, blank_image=blank)
            else:
                embeds = model.embeds(row, blank_image=blank)
                ids = model.lm.generate(inputs_embeds=embeds,
                                        attention_mask=torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long),
                                        max_new_tokens=cfg['max_new_tokens'], do_sample=False, use_cache=False,
                                        pad_token_id=model.tokenizer.pad_token_id, eos_token_id=model.tokenizer.eos_token_id)
                prediction = model.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    finally:
        hook.remove()
    assert captured['initial_kv_length'] == 0, captured
    assert captured['prefill_length'] == expected_length and captured['prefill_sha256'] == expected_hash, captured
    left, right = model.prompt(row['question'])
    result = {'id': reference_id or row['id'], 'condition': condition, 'question': row['question'],
              'document_id_used': row['document_id'], 'prediction': prediction, 'historical_prediction': row['prediction'],
              'gold_answers': row['answers'], 'blank_image': blank, 'use_cache': use_cache,
              'prompt_text_without_image_embeddings': left + '[CURRENT IMAGE EMBEDDINGS]' + right,
              'expected_prefill_length': expected_length, 'expected_prefill_sha256': expected_hash,
              'first_forward': captured, 'seconds': time.monotonic() - started}
    with (OUT / 'runtime_predictions.jsonl').open('a') as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ('id', 'condition', 'prediction', 'seconds')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['npu', 'cpu'], default='npu')
    parser.add_argument('--fresh-id')
    parser.add_argument('--full-group', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    all_rows = [json.loads(line) for line in resolve('artifacts/full/docvqa_comparison.jsonl').read_text().splitlines()]
    lookup = {r['id']: r for r in all_rows}
    cfg, model = load_model(args.device)
    if args.full_group:
        targets = sorted([r for r in all_rows if r['prediction'] == TARGET], key=lambda r: int(r['id'].rsplit('_', 1)[1]))
        baseline = {r['id']: probe(model, cfg, r, 'full_group_forward_order') for r in targets}
        reverse = {}
        for row in reversed(targets):
            model.features.clear()
            reverse[row['id']] = probe(model, cfg, row, 'full_group_reverse_order_cache_cleared')
        summary = {'samples': len(targets), 'distinct_documents': len({r['document_id'] for r in targets}),
                   'first_question_started_in_fresh_process': True,
                   'forward_matches_historical': sum(v['prediction'] == TARGET for v in baseline.values()),
                   'reverse_and_cache_clear_matches_forward': sum(v['prediction'] == baseline[k]['prediction'] for k, v in reverse.items()),
                   'prefill_hash_invariant_to_order_and_feature_cache': sum(v['expected_prefill_sha256'] == baseline[k]['expected_prefill_sha256'] for k,v in reverse.items()),
                   'all_request_initial_kv_lengths_zero': True,
                   'first_forward_matches_current_image_and_current_question': True,
                   'per_sample': {k: {'forward': v['prediction'], 'reverse_cache_cleared': reverse[k]['prediction']} for k,v in baseline.items()}}
        atomic_json(OUT / 'full_group_summary.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k != 'per_sample'},ensure_ascii=False), flush=True)
        return
    if args.fresh_id:
        result = probe(model, cfg, lookup[args.fresh_id], 'fresh_process')
        atomic_json(OUT / f'fresh_{args.fresh_id}.json', result)
        return
    ids = [10, 77, 86, 87, 92, 157, 1, 2, 3, 4]
    rows = [lookup[f'DocVQA_validation_{i}'] for i in ids]
    atomic_json(OUT / 'runtime_manifest.json', {'device': cfg['device'], 'precision': cfg['mixed_precision'] or 'float32',
                'sample_ids': [r['id'] for r in rows], 'checkpoint': cfg['language_model'],
                'scope': 'Six early target-name predictions across all four original ranks, plus four controls; diagnostic, not benchmark'})
    baseline = {r['id']: probe(model, cfg, r, 'baseline_forward_order') for r in rows}
    reversed_results = {r['id']: probe(model, cfg, r, 'reversed_order') for r in reversed(rows)}
    cleared = {}
    for row in rows:
        model.features.clear()
        cleared[row['id']] = probe(model, cfg, row, 'feature_cache_cleared')
    poisoned = probe(model, cfg, {**rows[0], 'answers': ['CONTEXT_AUDIT_GOLD_MUST_NEVER_ENTER_INPUT']}, 'gold_metadata_poisoned')
    no_cache = {r['id']: probe(model, cfg, r, 'kv_cache_disabled', use_cache=False) for r in rows[:2]}
    blank = {r['id']: probe(model, cfg, r, 'zero_image_features', blank=True) for r in rows}
    wrong = {}
    for i, row in enumerate(rows):
        donor = next(r for r in rows[i+1:] + rows[:i] if r['document_id'] != row['document_id'])
        wrong[row['id']] = probe(model, cfg, {**row, 'document_id': donor['document_id'], 'image': donor['image']},
                                 'wrong_image_same_question', reference_id=row['id'])
    features = []
    for doc in sorted({r['document_id'] for r in rows}):
        sample = next(r for r in rows if r['document_id'] == doc)
        loaded = model.feature(sample)
        disk = torch.load(resolve(cfg['output_dir']) / 'features' / f'{doc}.pt', map_location='cpu', weights_only=True)['features'].float()
        exact = torch.equal(loaded.cpu(), disk)
        assert exact, 'Feature cache changed or incorrect document mapping'
        features.append({'document_id': doc, 'feature_sha256': tensor_hash(disk), 'shape': list(disk.shape), 'memory_equals_disk': exact,
                         'image_sha256_matches_document_id': hashlib.sha256(resolve(sample['image']).read_bytes()).hexdigest() == doc})
    assert len({r['feature_sha256'] for r in features}) == len(features), 'Different documents have identical features'
    first = rows[0]['id']
    summary = {'scope': 'Small causal diagnostic; no training or benchmark score changes', 'samples': len(rows),
               'baseline_matches_historical': sum(v['prediction'] == lookup[k]['prediction'] for k, v in baseline.items()),
               'order_invariant': sum(v['prediction'] == baseline[k]['prediction'] for k, v in reversed_results.items()),
               'feature_cache_clear_invariant': sum(v['prediction'] == baseline[k]['prediction'] for k, v in cleared.items()),
               'gold_metadata_unused': poisoned['expected_prefill_sha256'] == baseline[first]['expected_prefill_sha256'] and poisoned['prediction'] == baseline[first]['prediction'],
               'kv_cache_disabled_matches': sum(v['prediction'] == baseline[k]['prediction'] for k, v in no_cache.items()),
               'kv_cache_disabled_samples': len(no_cache),
               'all_tested_request_initial_kv_lengths_zero': True,
               'all_tested_prefills_match_current_image_and_current_question_only': True,
               'zero_image_same_answer': sum(v['prediction'] == baseline[k]['prediction'] for k, v in blank.items()),
               'wrong_image_same_answer': sum(v['prediction'] == baseline[k]['prediction'] for k, v in wrong.items()),
               'shank_baseline_samples': sum(v['prediction'] == TARGET for v in baseline.values()),
               'shank_zero_image_samples': sum(v['prediction'] == TARGET for v in blank.values()),
               'shank_wrong_image_samples': sum(v['prediction'] == TARGET for v in wrong.values()),
               'features': features, 'per_sample': {k: {'baseline': v['prediction'], 'historical': lookup[k]['prediction'],
                           'reversed': reversed_results[k]['prediction'], 'cache_cleared': cleared[k]['prediction'],
                           'zero_image': blank[k]['prediction'], 'wrong_image': wrong[k]['prediction']} for k, v in baseline.items()}}
    del model
    gc.collect()
    if args.device == 'npu':
        torch.npu.empty_cache()
    fresh = []
    for row in rows[:2]:
        subprocess.run([sys.executable, '-m', 'scripts.audit_vqa_context', '--device', args.device, '--fresh-id', row['id']], check=True)
        fresh.append(json.loads((OUT / f'fresh_{row["id"]}.json').read_text()))
    summary['fresh_process_samples'] = len(fresh)
    summary['fresh_process_matches'] = sum(r['prediction'] == baseline[r['id']]['prediction'] for r in fresh)
    atomic_json(OUT / 'runtime_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
