"""All56QA native8B manual+cached path against stock HF generation tokens."""
import json,os
import torch
import torch.distributed as dist
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.common import ROOT,lines,load,save,sha
from scripts.tessfold_accel.qwen_training import Data
from vqa.tessfold_qwen3vl import NativeQwen

@torch.no_grad()
def main():
    rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115/qwen_identity56';base.mkdir(exist_ok=True)
    engine=NativeQwen(device);data=Data(engine)
    rows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')[rank::world]
    old={r['id']:r for p in (ROOT/'artifacts/tessfold_qwen3vl_short_extension_20261115').glob('predictions.rank*.jsonl') for r in lines(p)}
    results=[]
    for row in rows:
        d=engine.from_features(row,data.features(row));ids=engine.greedy(d)
        r=dict(id=row['id'],task=row['task'],equal=ids==old[row['id']]['generated_ids'],current_ids=ids,original_ids=old[row['id']]['generated_ids'])
        results.append(r);assert r['equal'],row['id']
    save(base/f'rows.rank{rank}.json',results);dist.barrier()
    if rank==0:save(base/'complete.json',dict(cases=56,all_stock_HF_tokens_equal=True))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
