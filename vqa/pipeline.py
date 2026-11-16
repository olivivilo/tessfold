"""ENTRY: python -m vqa.pipeline --config configs/alignment.json run"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / 'artifacts/hf_cache'))

import torch
from PIL import Image

from .metrics import anls, exact_match, relaxed_accuracy
from .model import ROOT, VLM, load_vision, resolve

def read_data(config, split):
    path = resolve(config['data_dir']) / f'{split}.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def output(config):
    path = resolve(config['output_dir'])
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_data(config):
    splits = {s: read_data(config, s) for s in ('train', 'val', 'test')}
    ids = {s: set(x['document_id'] for x in records) for s, records in splits.items()}
    for a, b in (('train','val'), ('train','test'), ('val','test')):
        if ids[a] & ids[b]:
            raise ValueError(f'Document leakage: {a}, {b}')
    for records in splits.values():
        for sample in records:
            if not sample['answers'] or not sample['question']:
                raise ValueError('Empty question or answer')
            actual = hashlib.sha256(resolve(sample['image']).read_bytes()).hexdigest()
            if actual != sample['document_id']:
                raise ValueError('Image does not match manifest SHA256')
    return splits

def cache_features(config):
    splits = validate_data(config)
    path = output(config) / 'features'
    path.mkdir(exist_ok=True)
    signature = {'model': config['vision_model'], 'max_pixels': config['max_pixels'],
                 'min_pixels': config['min_pixels'], 'dtype': config['dtype'], 'format': 1}
    records = {s['document_id']: s for samples in splits.values() for s in samples}
    pending = []
    for doc, sample in records.items():
        cache = path / f'{doc}.pt'
        if not cache.exists() or torch.load(cache, map_location='cpu', weights_only=True)['signature'] != signature:
            pending.append(sample)
    if not pending:
        print('All visual features cached', flush=True)
        return
    vision, processor = load_vision(config)
    for i, sample in enumerate(pending):
        image = Image.open(resolve(sample['image'])).convert('RGB')
        processed = processor(images=[image], return_tensors='pt')
        pixels = processed['pixel_values'].to(config['device'], dtype=getattr(torch, config['dtype']))
        grid = processed['image_grid_thw'].to(config['device'])
        with torch.no_grad():
            features = vision(pixels, grid, bf16=config['dtype']=='bfloat16')
        assert features.ndim == 2 and features.shape[1] == vision.config.embed_dim
        torch.save({'features': features.cpu(), 'grid': grid.cpu(), 'signature': signature},
                   path / f"{sample['document_id']}.pt")
        print(f'cache {i+1}/{len(pending)}: {tuple(features.shape)}', flush=True)

@torch.no_grad()
def evaluate(model, config, split, label, blank_image=False):
    model.eval()
    results, nll, tokens = [], 0.0, 0
    start = time.monotonic()
    for sample in read_data(config, split):
        loss, n = model.loss(sample)
        prediction = model.predict(sample, blank_image=blank_image)
        result = {k: sample[k] for k in ('id','document_id','image','question','answers','task')}
        result.update(prediction=prediction, exact_match=exact_match(prediction, sample['answers']),
                      anls=anls(prediction, sample['answers']), relaxed_accuracy=relaxed_accuracy(prediction, sample['answers']),
                      answer_nll=float(loss), answer_tokens=n)
        results.append(result)
        nll += float(loss)*n
        tokens += n
        print(f"{label} {len(results)}: {prediction!r} / {sample['answers']}", flush=True)
    def aggregate(rows):
        return {k: 100*sum(r[k] for r in rows)/len(rows) for k in ('exact_match','anls','relaxed_accuracy')}
    metrics = {'split': split, 'label': label, 'samples': len(results), 'documents': len(set(r['document_id'] for r in results)),
               **aggregate(results), 'answer_nll': nll/tokens, 'answer_perplexity': math.exp(min(nll/tokens, 80)),
               'by_task': {task: {'samples': len(rows), **aggregate(rows)} for task in sorted(set(r['task'] for r in results))
                           for rows in [[r for r in results if r['task']==task]]},
               'seconds': time.monotonic()-start, 'blank_image': blank_image,
               'scope': 'Derived MonkeyDocv2 QA subset; not official DocVQA/ChartQA or paper Table 8'}
    path = output(config)
    (path / f'{label}_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results))
    (path / f'{label}_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics

def train(model, config, stage):
    steps = config[f'stage{stage}_steps']
    if not steps:
        return
    joint = stage == 2
    model.lm.requires_grad_(joint)
    model.projector.requires_grad_(True)
    model.projector.train()
    model.lm.train(joint)
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all(p.requires_grad for p in model.projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config[f'stage{stage}_lr'], weight_decay=config['weight_decay'])
    samples = read_data(config, 'train')
    rng = random.Random(config['seed']+stage)
    rng.shuffle(samples)
    accumulation = config['gradient_accumulation']
    start = time.monotonic()
    log = output(config) / 'train.jsonl'
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(accumulation):
            index = (step*accumulation+micro) % len(samples)
            loss, _ = model.loss(samples[index])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            (loss/accumulation).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'])
        optimizer.step()
        row = {'stage': stage, 'step': step+1, 'loss': sum(losses)/len(losses),
               'gradient_norm': float(norm), 'lr': optimizer.param_groups[0]['lr'],
               'elapsed_seconds': time.monotonic()-start, 'trainable_parameters': sum(p.numel() for p in parameters)}
        with log.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    checkpoint = output(config) / f'stage{stage}'
    checkpoint.mkdir(exist_ok=True)
    torch.save(model.projector.state_dict(), checkpoint / 'projector.pt')
    if joint:
        model.lm.save_pretrained(checkpoint / 'language_model', safe_serialization=True)
        model.tokenizer.save_pretrained(checkpoint / 'language_model')
    (checkpoint / 'config.json').write_text(json.dumps(config, indent=2))
    model.lm.requires_grad_(False)
    model.eval()
    del optimizer

def load_checkpoint(model, config, stage):
    checkpoint = output(config) / f'stage{stage}'
    if stage == 2:
        from transformers import AutoModelForCausalLM
        # Free original LM before loading the jointly trained LM to bound memory usage.
        del model.lm
        model.lm = AutoModelForCausalLM.from_pretrained(checkpoint / 'language_model', local_files_only=True,
                                                      dtype=model.dtype, attn_implementation='sdpa').to(config['device'])
        model.lm.requires_grad_(False)
    model.projector.load_state_dict(torch.load(checkpoint / 'projector.pt', map_location=config['device'], weights_only=True))
    model.eval()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/alignment.json')
    parser.add_argument('command', choices=['cache','run','eval','infer'])
    parser.add_argument('--stage', type=int, choices=[1,2], default=2)
    parser.add_argument('--split', choices=['val','test'], default='test')
    parser.add_argument('--image')
    parser.add_argument('--question')
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text())
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    path = output(config)
    (path / 'config.json').write_text(json.dumps(config, indent=2))
    if args.command in ('cache','run'):
        cache_features(config)
    if args.command == 'cache':
        return
    model = VLM(config)
    if args.command == 'run':
        # Every run starts from pretrained components; logs belong to this run only.
        (path / 'train.jsonl').write_text('')
        environment = {'torch': torch.__version__, 'device': config['device'], 'dtype': config['dtype'],
                       'threads': config['threads'], 'vision_frozen': True,
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'hostname': os.uname().nodename,
                       'vision_model': config['vision_model'], 'language_model': config['language_model'],
                       'projector_parameters': sum(p.numel() for p in model.projector.parameters()),
                       'language_parameters': sum(p.numel() for p in model.lm.parameters())}
        (path / 'environment.json').write_text(json.dumps(environment, indent=2))
        # Baseline/final use identical held-out data and decoding; no test-driven tuning.
        summaries = {'baseline_test': evaluate(model, config, 'test', 'baseline_test')}
        train(model, config, 1)
        summaries['stage1_val'] = evaluate(model, config, 'val', 'stage1_val')
        train(model, config, 2)
        summaries['final_test'] = evaluate(model, config, 'test', 'final_test')
        # Verify checkpoint reload reproduces greedy output on a held-out example.
        sample = read_data(config, 'test')[0]
        before = model.predict(sample)
        load_checkpoint(model, config, 2 if config['stage2_steps'] else 1)
        after = model.predict(sample)
        if before != after:
            raise RuntimeError('Checkpoint roundtrip changed greedy prediction')
        summaries['checkpoint_roundtrip'] = {'passed': True, 'sample_id': sample['id'], 'prediction': after}
        (path / 'results.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    elif args.command == 'eval':
        load_checkpoint(model, config, args.stage)
        evaluate(model, config, args.split, f'reeval_stage{args.stage}_{args.split}')
    else:
        if not args.image or not args.question:
            parser.error('infer requires --image and --question')
        load_checkpoint(model, config, args.stage)
        image_path = resolve(args.image)
        vision, processor = load_vision(config)
        media = processor(images=[Image.open(image_path).convert('RGB')], return_tensors='pt')
        with torch.no_grad():
            features = vision(media['pixel_values'].to(config['device'], dtype=model.dtype),
                              media['image_grid_thw'].to(config['device']), bf16=config['dtype']=='bfloat16')
        sample = {'document_id': 'inference', 'question': args.question}
        model.features['inference'] = features
        print(model.predict(sample))

if __name__ == '__main__':
    main()
