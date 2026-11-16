"""Warm immutable frozen-UND teacher caches; never updates a model or selects one."""
import hashlib
import json
import os
import random
import torch
import torch.distributed as dist
from scripts import compression_lab_v2_io as io
from scripts.train_meta_compress import init_runtime, atomic_json, sha
from scripts.train_meta_compress_strict import Examples
from scripts.train_compression_lab_v2 import ROOT, BASE, read
from vqa.official_und import OfficialUND


def main():
    rank, world, _, device = init_runtime('npu', 4)
    io.common.atomic_save = io.guarded_save
    engine = OfficialUND(device=str(device))
    engine.model.eval().requires_grad_(False)
    data = Examples(engine, 'npu', capacity=2)
    rows = read(BASE/'full_train.jsonl')
    order = list(range(len(rows)))
    random.Random(20261105).shuffle(order)
    order.reverse()  # Work ahead of the forward-order first training epoch.
    output = BASE/'infrastructure/full_cache_prefetch'
    output.mkdir(exist_ok=True)
    atomic_json(output/f'manifest.rank{rank}.json', {
        'job': os.environ.get('SLURM_JOB_ID'), 'world': world,
        'train_sha256': sha(BASE/'full_train.jsonl'),
        'source_sha256': {p: sha(ROOT/p) for p in [
            'scripts/prefetch_compression_lab_v2.py', 'scripts/compression_lab_v2_io.py',
            'scripts/train_meta_compress.py', 'scripts/train_meta_compress_strict.py',
            'vqa/official_und.py']},
        'frozen_teacher_only': True, 'model_updates': 0})
    processed = skipped = 0
    for i in order[rank::world]:
        row = rows[i]
        key = hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'
        paths = [data.directory/'features'/(row['document_id']+'.pt'),
                 data.directory/'teachers'/key, data.directory/'logits'/key]
        if all(p.exists() for p in paths):
            skipped += 1
        else:
            data.get(row)
            processed += 1
        if (processed+skipped) % 100 == 0:
            print(json.dumps({'rank': rank, 'processed': processed, 'skipped': skipped}), flush=True)
    atomic_json(output/f'complete.rank{rank}.json', {'processed': processed, 'skipped': skipped,
                                                   'model_updates': 0})
    dist.barrier()
    if rank == 0:
        atomic_json(output/'complete.json', {'world': world, 'questions': len(rows), 'model_updates': 0})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
