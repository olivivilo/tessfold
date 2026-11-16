"""Prepare teacher targets for new internal validation images using Slurm workers."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import hashlib
import json
import time
import torch
from scripts.train_meta_compress_search import CACHE,read
from scripts.train_meta_compress import prepare_example,teacher_logits_for,atomic_json,ROOT
from vqa.official_und import OfficialUND


def main():
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));torch.set_num_threads(8)
    rows=read(ROOT/'artifacts/meta_compress_search/cache_rows.jsonl')
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    start=time.monotonic();count=0
    for row in rows[rank::world]:
        key=hashlib.sha256(row['id'].encode()).hexdigest()
        if (CACHE/'logits'/f'{key}.pt').exists():continue
        prompt,raw,grid,x,response=prepare_example(engine,row,CACHE,64)
        tl=teacher_logits_for(engine,row,prompt,x,response,CACHE);count+=1
        print(json.dumps({'rank':rank,'new_targets':count,'id':row['id'],'seconds':time.monotonic()-start}),flush=True)
    atomic_json(ROOT/f'artifacts/meta_compress_search/cache.rank{rank}.complete.json',{'new_targets':count})


if __name__=='__main__':main()
