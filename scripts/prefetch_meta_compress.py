"""Precompute frozen UND teacher targets; never changes generator or losses.

Uses exactly the training helpers and same 16 CPU threads. Independent samples
are claimed atomically to reduce teacher-generation stragglers in synchronous
SGD. All targets remain full FP32; gold is unused.
"""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import hashlib
import json
import random
import time
import torch
from vqa.official_und import ROOT,OfficialUND
from scripts.train_meta_compress import choose_rows,read_rows,prepare_example,teacher_logits_for,sha,atomic_json

def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(16)
    root=ROOT/'artifacts/meta_compress';cache=ROOT/'artifacts/meta_compress_cache'
    claims=cache/'prefetch_claims';claims.mkdir(exist_ok=True)
    rows=choose_rows(read_rows('train'),20000);order=list(range(len(rows)));random.Random(42).shuffle(order)
    # Begin ahead of live SGD; the first batches are already being processed.
    order=order[1152:]+order[:1152]
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    manifest={'rank':rank,'world':world,'device':'cpu','threads':16,'purpose':'frozen teacher-only precompute',
              'training_helper_sha256':sha(ROOT/'scripts/train_meta_compress.py'),
              'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    atomic_json(cache/f'prefetch.rank{rank}.json',manifest)
    count=0;start=time.monotonic()
    for j in range(len(order)):
        ix=order[(j+rank)%len(order)];row=rows[ix];key=hashlib.sha256(row['id'].encode()).hexdigest()
        if (cache/'logits'/f'{key}.pt').exists():continue
        lock=claims/key
        try:lock.mkdir()
        except FileExistsError:continue
        try:
            if (cache/'logits'/f'{key}.pt').exists():continue
            prompt,raw,grid,projected,response=prepare_example(engine,row,cache,64)
            logits=teacher_logits_for(engine,row,prompt,projected,response,cache)
            del prompt,raw,grid,projected,response,logits
            count+=1
            if count%10==0:print(json.dumps({'rank':rank,'cached':count,'seconds':time.monotonic()-start}),flush=True)
        finally:lock.rmdir()
    atomic_json(cache/f'prefetch.rank{rank}.complete.json',{'cached':count,'seconds':time.monotonic()-start})

if __name__=='__main__':main()
