"""Same-budget fixed controls, not learned MetaCompress candidates."""
import argparse
import torch
import torch.distributed as dist
from torch import nn
from scripts.train_meta_compress_strict import Examples,evaluate,BASE,ROOT
from scripts.train_meta_compress import init_runtime,atomic_json
from vqa.official_und import OfficialUND
from scripts.report_meta_compress_strict import read

class Fixed(nn.Module):
    def __init__(self,kind):super().__init__();self.kind=kind
    def forward(self,x,grid,keep_ratio):
        n=len(x);m=max(1,round(n*keep_ratio))
        p=x.new_zeros(m,n);idx=torch.arange(n,device=x.device)
        if self.kind=='mean':
            p[idx*m//n,idx]=1.;p=p/p.sum(-1,keepdim=True)
        else:
            selected=torch.arange(m,device=x.device)*n//m
            p[torch.arange(m,device=x.device),selected]=1.
        return p@x,p

def main():
    a=argparse.ArgumentParser();a.add_argument('--threads',type=int,default=8)
    a.add_argument('--ratio',type=float,default=.5);args=a.parse_args()
    rank,world,local,device=init_runtime('npu',args.threads)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    examples=Examples(engine,'npu');rows=read(BASE/'validation.jsonl')
    for insertion in ['pre-merger','post-merger']:
        for kind in ['mean','uniform-selection']:
            cfg={'insertion':insertion,'ratio':args.ratio}
            suffix='' if args.ratio==.5 else f'_ratio{args.ratio:g}'
            out=BASE/'baselines'/f'{insertion}_{kind}{suffix}';out.mkdir(parents=True,exist_ok=True)
            evaluate(engine,Fixed(kind),rows,examples,cfg,out,'fixed','validation',rank,world)
            if rank==0:atomic_json(out/'complete.json',{'rows':len(rows),'ratio':args.ratio,'insertion':insertion,'kind':kind})
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
