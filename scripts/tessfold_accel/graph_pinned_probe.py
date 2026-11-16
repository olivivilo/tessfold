"""Whole-compressor graph correctness, setup/memory, steady-state costs."""
import json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.tessfold_accel.probe import milliseconds
from vqa.official_und import OfficialUND
from vqa.tessfold_fast_runtime import GraphCompressor,compress,dimensions

@torch.no_grad()
def main():
    reg=verify();rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115/graph_full2';base.mkdir(exist_ok=True)
    save(base/f'manifest.rank{rank}.json',dict(job=os.environ.get('SLURM_JOB_ID'),source_sha256=sha(__file__),runtime_sha256=sha(ROOT/'vqa/tessfold_fast_runtime.py')))
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    meta=reg['models'][2+rank];module=load_module(engine,meta['kind'],meta['ratio']).requires_grad_(False)
    rows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')[:4];records=[]
    for row in rows:
        prompt,raw,grid,x,response=prepare_example(engine,row,ROOT/'artifacts/meta_compress_strict/npu_cache',64)
        qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device)
        q=engine.model.get_input_embeddings()(qids) if module.kind=='quadtree_question' else None
        oldq=q if q is not None else x[:1]
        _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,meta['ratio'],raw=raw,question=oldq)
        ref=(p@raw.float()).to(raw.dtype)
        plan=GraphCompressor(module,raw,q,dimensions(grid),meta['ratio'])
        c,d=plan(raw,q);torch.npu.synchronize();assert torch.equal(c,ref) and torch.equal(d['weights'],p)
        elapsed=milliseconds(lambda:plan(raw,q),20)
        eager=milliseconds(lambda:compress(module,raw,grid,q,meta['ratio'],shape=dimensions(grid)),5)
        r=dict(id=row['id'],kind=meta['kind'],n=len(raw),m=len(c),pure_compressor_ms=elapsed,eager_fast_ms=eager,
            setup_ms=plan.setup_ms,persistent_allocated_bytes=plan.persistent_allocated_bytes,feature_exact=True,p_exact=True,host_pinned=plan.host_pinned)
        records.append(r);print(json.dumps(r),flush=True)
        del plan,c,d,p,ref;torch.npu.synchronize();torch.npu.empty_cache()
    save(base/f'rows.rank{rank}.json',records);dist.barrier()
    if rank==0:save(base/'complete.json',dict(cases=4*world,all_exact=True))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
