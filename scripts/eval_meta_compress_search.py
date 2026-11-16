"""One-shot image-disjoint confirmation for a search-selected checkpoint."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
import torch.distributed as dist
from scripts.train_meta_compress_search import read,load_module,Examples,evaluate,ROOT
from scripts.train_meta_compress import init_runtime,atomic_json,sha,barrier
from vqa.official_und import OfficialUND


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--cohort',default='confirm');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();cfg=json.loads((ROOT/a.config).read_text())
    out=ROOT/cfg['output']/a.cohort;out.mkdir(parents=True,exist_ok=True)
    rank,world,local,device=init_runtime(a.device,a.threads)
    signature={'config':cfg,'checkpoint_sha256':sha(ROOT/a.checkpoint),'checkpoint':a.checkpoint,
        'cohort':a.cohort,'rows_sha256':sha(ROOT/f'artifacts/meta_compress_search/{a.cohort}.jsonl'),
        'world':world,'device':a.device,'threads':a.threads,
        'code_sha256':{f:sha(ROOT/f) for f in ['scripts/eval_meta_compress_search.py','scripts/train_meta_compress_search.py','vqa/meta_compress_tuning.py']}}
    if rank==0:
        manifest=out/'manifest.json'
        if manifest.exists():assert json.loads(manifest.read_text())==signature,'Changed confirmation protocol'
        atomic_json(manifest,signature)
    barrier(world)
    if not (out/'complete.json').exists():
        engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
        examples=Examples(engine,capacity=24);rows=read(ROOT/f'artifacts/meta_compress_search/{a.cohort}.jsonl')
        for variant,path in [('initial',None),('selected',ROOT/a.checkpoint)]:
            module=load_module(cfg,path).to(device)
            evaluate(engine,module,rows,examples,cfg,out,a.cohort,rank,world,variant=variant)
        if rank==0:atomic_json(out/'complete.json',{'rows':len(rows),'variants':['initial','selected'],'checkpoint_sha256':signature['checkpoint_sha256']})
    barrier(world)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
