"""Verify instrumented natural output and strict HF-style fixed64 suppression."""
import json,os,time
import torch
import torch.distributed as dist
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.common import ROOT,lines,load,save,sha
from scripts.tessfold_accel.qwen_training import Data
from vqa.tessfold_qwen3vl_timing import TimingQwen

@torch.no_grad()
def main():
    rank,world,_,device=init_runtime('npu',4);metadata=dist.new_group(backend='gloo')
    base=ROOT/'artifacts/tessfold_acceleration_20261115/qwen_timing_identity';base.mkdir(exist_ok=True)
    engine=TimingQwen(device);data=Data(engine);allrows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')
    old={r['id']:r for p in (ROOT/'artifacts/tessfold_qwen3vl_short_extension_20261115').glob('predictions.rank*.jsonl') for r in lines(p)}
    first={}
    for row in allrows:first.setdefault(row['task'],row['id'])
    results=[]
    for row in allrows[rank::world]:
        d=engine.from_features(row,data.features(row));start=time.perf_counter();ids=engine.greedy(d,cap=64)
        assert ids==old[row['id']]['generated_ids']
        assert engine.first_token_timestamp>=start and engine.visual_setup_ms>=0
        result=dict(id=row['id'],natural_equal=True)
        if row['id'] in first.values():
            stock_inputs=engine.batch(row)
            stock=engine.model.generate(**stock_inputs,do_sample=False,use_cache=True,min_new_tokens=64,max_new_tokens=64)[0,stock_inputs['input_ids'].shape[1]:].cpu().tolist()
            ids=engine.greedy(d,cap=64,fixed=True)
            assert len(ids)==64 and ids==stock,'Fixed64 instrumentation differs from official HF'
            result['fixed64_equal']=True
        results.append(result)
    save(base/f'rows.rank{rank}.json',results);dist.barrier(group=metadata)
    if rank==0:
        rr=[r for f in base.glob('rows.rank*.json') for r in load(f,[])]
        assert len(rr)==56 and sum(r.get('fixed64_equal',False) for r in rr)==7
        save(base/'complete.json',dict(cases=56,natural_equal=56,fixed64_equal=7,timing_source_sha256=sha(ROOT/'vqa/tessfold_qwen3vl_timing.py')))
    dist.barrier(group=metadata);dist.destroy_process_group()

if __name__=='__main__':main()
