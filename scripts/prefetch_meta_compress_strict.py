"""Warm the device-consistent frozen teacher cache for expanded training."""
import argparse
import torch.distributed as dist
from vqa.official_und import OfficialUND
from scripts.train_meta_compress import init_runtime,barrier,atomic_json,sha
from scripts.train_meta_compress_strict import Examples,BASE,ROOT,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',default='artifacts/meta_compress_strict/train.jsonl');a=p.parse_args()
    rank,world,_,device=init_runtime('npu',8)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    examples=Examples(engine,'npu');rows=read(ROOT/a.rows)
    for i,row in enumerate(rows[rank::world]):
        examples.get(row)
        if i%50==0:print({'rank':rank,'done':i+1,'rank_total':len(rows[rank::world])},flush=True)
    barrier(world)
    if rank==0:atomic_json(BASE/'prefetch.complete.json',{'rows':len(rows),'rows_sha256':sha(ROOT/a.rows),
        'device':'npu','cache':'npu_cache','not_training':True})
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
