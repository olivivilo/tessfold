"""Evaluate deterministic contiguous mean pooling at the matched token budget."""
import argparse
import json
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
import torch.distributed as dist
from scripts.train_meta_compress_search import read,Examples,evaluate,ROOT
from scripts.train_meta_compress import init_runtime,atomic_json,sha,barrier
from vqa.official_und import OfficialUND
from vqa.meta_compress import MetaCompress


class BalancedMean(torch.nn.Module):
    def forward(self,x,grid,ratio):
        n=len(x);m=MetaCompress.output_length(n,ratio)
        groups=torch.arange(n,device=x.device)*m//n
        p=torch.nn.functional.one_hot(groups,m).T.float();p=p/p.sum(-1,keepdim=True)
        return p@x,p


def main():
    p=argparse.ArgumentParser();p.add_argument('--ratio',type=float,required=True);p.add_argument('--cohort',default='search')
    p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    rank,world,local,device=init_runtime('cpu',a.threads)
    out=ROOT/f'artifacts/meta_compress_search/baselines/ratio{a.ratio:g}/{a.cohort}';out.mkdir(parents=True,exist_ok=True)
    cfg={'ratio':a.ratio,'entropy_weight':1.,'collapse_weight':1.}
    signature={'cohort':a.cohort,'ratio':a.ratio,'world':world,'row_sha256':sha(ROOT/f'artifacts/meta_compress_search/{a.cohort}.jsonl'),
        'code_sha256':sha(ROOT/'scripts/baseline_meta_compress_search.py')}
    if rank==0:
        manifest=out/'manifest.json'
        if manifest.exists():assert json.loads(manifest.read_text())==signature
        atomic_json(manifest,signature)
    barrier(world)
    if not (out/'complete.json').exists():
        engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
        rows=read(ROOT/f'artifacts/meta_compress_search/{a.cohort}.jsonl')
        evaluate(engine,BalancedMean(),rows,Examples(engine,24),cfg,out,a.cohort,rank,world,variant='mean')
        if rank==0:atomic_json(out/'complete.json',{'rows':len(rows),'ratio':a.ratio})
    barrier(world)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
