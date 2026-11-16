"""Prepare only frozen teacher targets for the already fixed final benchmark.

No candidate model, gold-answer scoring, model selection, or training runs here.
The final evaluator consumes the same separate NPU/cap256 cache unchanged.
"""
import argparse
import hashlib
import json
import os
import time

import torch.distributed as dist

from scripts.train_meta_compress import init_runtime, barrier, atomic_json, sha, prepare_example, teacher_logits_for
from scripts.train_meta_compress_strict import BASE, ROOT, read
from vqa.official_und import OfficialUND


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args()
    assert 0<=args.shard<args.shards
    rank,world,_,device=init_runtime('npu',8)
    rows=[r for r in read(ROOT/'data/full/val.jsonl') if r['task'] in ['DocVQA','InfoVQA']]+read(ROOT/'data/full/test.jsonl')
    assert len(rows)==23565
    rows=rows[args.shard::args.shards][rank::world]
    out=BASE/'benchmark_cache_prefetch'/f'shard{args.shard}'
    out.mkdir(parents=True,exist_ok=True)
    signature={'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/full/val.jsonl',ROOT/'data/full/test.jsonl']},
        'shard':args.shard,'shards':args.shards,'rank':rank,'world':world,
        'max_new_tokens':256,'device':'npu','candidate_evaluation':False,
        'job':os.environ.get('SLURM_JOB_ID'),'code_sha256':sha(__file__)}
    atomic_json(out/f'manifest.rank{rank}.json',signature)
    engine=OfficialUND(device=str(device))
    engine.model.eval().requires_grad_(False)
    cache=BASE/'benchmark_npu_cache'
    for name in ['features','teachers','logits']:(cache/name).mkdir(parents=True,exist_ok=True)
    cached=0;created=0;started=time.monotonic()
    for index,row in enumerate(rows):
        key=hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'
        files=[cache/'features'/(row['document_id']+'.pt'),cache/'teachers'/key,cache/'logits'/key]
        if all(p.exists() for p in files):cached+=1
        else:
            prompt,raw,grid,x,response=prepare_example(engine,row,cache,256)
            teacher_logits_for(engine,row,prompt,x,response,cache)
            created+=1
        if (index+1)%50==0 or index==len(rows)-1:
            progress={'rank':rank,'done':index+1,'total':len(rows),
                'cached':cached,'created':created,'elapsed':time.monotonic()-started}
            atomic_json(out/f'progress.rank{rank}.json',progress)
            print(json.dumps(progress),flush=True)
    atomic_json(out/f'complete.rank{rank}.json',progress)
    barrier(world)
    if rank==0:atomic_json(out/'complete.json',{'shard':args.shard,'world':world,'candidate_evaluation':False})
    barrier(world)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
