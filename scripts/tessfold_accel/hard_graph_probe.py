"""Measure hard representative graph including CPU partition/transfer."""
import json,os
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.tessfold_accel.probe import milliseconds
from vqa.official_und import OfficialUND
from vqa.tessfold_fast_runtime import dimensions
from vqa.tessfold_hard_graph import HardGraph

@torch.no_grad()
def main():
    reg=verify();rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115/hard_graph1';base.mkdir(exist_ok=True)
    save(base/f'manifest.rank{rank}.json',dict(job=os.environ.get('SLURM_JOB_ID'),sources={s:sha(ROOT/s) for s in ['vqa/tessfold_hard_graph.py',str(__import__('pathlib').Path(__file__).relative_to(ROOT))]}))
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    meta=reg['models'][2+rank];module=load_module(engine,meta['kind'],meta['ratio']).requires_grad_(False)
    records=[]
    rows=lines(ROOT/'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl')[:12]
    for row in rows:
        prompt,raw,grid,x,response=prepare_example(engine,row,ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache',256)
        qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device)
        q=engine.model.get_input_embeddings()(qids) if module.kind=='quadtree_question' else None
        _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,meta['ratio'],raw=raw,question=q if q is not None else x[:1])
        reps=p.argmax(-1);reference=raw.index_select(0,reps)
        plan=HardGraph(module,raw,q,dimensions(grid),meta['ratio'])
        c,d=plan(raw,q);torch.npu.synchronize()
        r=dict(id=row['id'],kind=meta['kind'],n=len(raw),m=len(c),pure_compressor_ms=milliseconds(lambda:plan(raw,q),20),setup_ms=plan.setup_ms,
            persistent_allocated_bytes=plan.persistent_allocated_bytes,hard_feature_equal=torch.equal(c,reference),representatives_equal=torch.equal(reps,d['representatives']),
            note='Hard vs original hard readout, NOT equivalence to original soft model')
        records.append(r);print(json.dumps(r),flush=True)
        del plan,c,d,p,reference;torch.npu.synchronize();torch.npu.empty_cache()
    save(base/f'rows.rank{rank}.json',records);dist.barrier()
    if rank==0:save(base/'complete.json',dict(cases=len(rows)*world))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
