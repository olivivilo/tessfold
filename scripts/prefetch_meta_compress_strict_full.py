"""Populate the unchanged full-training teacher cache, without training or scoring.

Sharded helpers may run alongside the trainer. Existing cache writes are atomic;
all values use the trainer's own Examples implementation and NPU BF16 path.
"""
import argparse
import hashlib
import json
import os
import random
import time

import torch.distributed as dist

from scripts.train_meta_compress import init_runtime, barrier, atomic_json, sha
from scripts.train_meta_compress_strict import Examples, BASE, ROOT, read
from vqa.official_und import OfficialUND


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args()
    assert 0<=args.shard<args.shards
    rank,world,_,device=init_runtime('npu',8)
    path=BASE/'full_train.jsonl'
    rows=read(path)
    # Fill the far end of epoch one's order first while training consumes its
    # beginning. This is cache scheduling only, never a change in train order.
    random.Random(20261101).shuffle(rows)
    rows=list(reversed(rows))[args.shard::args.shards][rank::world]
    out=BASE/'full_cache_prefetch'/f'shard{args.shard}'
    out.mkdir(parents=True,exist_ok=True)
    signature={'rows_sha256':sha(path),'shard':args.shard,'shards':args.shards,
        'rank':rank,'world':world,'max_new_tokens':64,'device':'npu',
        'training':False,'job':os.environ.get('SLURM_JOB_ID'),
        'code_sha256':{p:sha(ROOT/p) for p in [
            'scripts/prefetch_meta_compress_strict_full.py',
            'scripts/train_meta_compress_strict.py','scripts/train_meta_compress.py',
            'vqa/official_und.py']}}
    atomic_json(out/f'manifest.rank{rank}.json',signature)
    engine=OfficialUND(device=str(device))
    engine.model.eval().requires_grad_(False)
    examples=Examples(engine,'npu')
    cached=0;created=0;started=time.monotonic()
    for index,row in enumerate(rows):
        key=hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'
        files=[examples.directory/'features'/(row['document_id']+'.pt'),
            examples.directory/'teachers'/key,examples.directory/'logits'/key]
        if all(p.exists() for p in files):cached+=1
        else:
            examples.get(row);created+=1
        if (index+1)%50==0 or index==len(rows)-1:
            progress={'rank':rank,'done':index+1,'total':len(rows),
                'cached':cached,'created':created,'elapsed':time.monotonic()-started}
            atomic_json(out/f'progress.rank{rank}.json',progress)
            print(json.dumps(progress),flush=True)
    atomic_json(out/f'complete.rank{rank}.json',progress)
    barrier(world)
    if rank==0:atomic_json(out/'complete.json',{'shard':args.shard,'world':world,
        'rows_sha256':sha(path),'training':False})
    barrier(world)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
