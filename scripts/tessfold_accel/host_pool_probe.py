"""Check true O(N) NPU aggregation, instability, downstream and pure wall time."""
import json,os
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.tessfold_accel.probe import milliseconds
from scripts.eval_token_proxy import generate,shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_fast_runtime import dimensions
from vqa.tessfold_host_pool import HostPool as SparseGraphCompressor
from vqa.benchmark_metrics import native_score

@torch.no_grad()
def main():
    reg=verify();rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115/host_pool1';base.mkdir(exist_ok=True)
    save(base/f'manifest.rank{rank}.json',dict(job=os.environ.get('SLURM_JOB_ID'),source_sha256=sha(__file__),runtime_sha256=sha(ROOT/'vqa/tessfold_fast_runtime.py')))
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    meta=reg['models'][2+rank];module=load_module(engine,meta['kind'],meta['ratio']).requires_grad_(False)
    records=[]
    for row in lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')[:8]:
        prompt,raw,grid,x,response=prepare_example(engine,row,ROOT/'artifacts/meta_compress_strict/npu_cache',64)
        qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device)
        q=engine.model.get_input_embeddings()(qids) if module.kind=='quadtree_question' else None
        _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,meta['ratio'],raw=raw,question=q if q is not None else x[:1])
        ref=(p@raw.float()).to(raw.dtype)
        plan=SparseGraphCompressor(module,raw,q,dimensions(grid),meta['ratio'])
        c,d=plan(raw,q);torch.npu.synchronize();first=c.clone();assert torch.isfinite(first).all(),'Sparse output nonfinite before repetitions'
        repeated=[]
        for _ in range(3):
            cc,_=plan(raw,q);torch.npu.synchronize();repeated.append(torch.equal(first,cc))
        si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
        pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,engine.model.vision_tower.merger(c),raw)
        ms=milliseconds(lambda:plan(raw,q),20);assert torch.isfinite(c).all(),'Sparse output nonfinite after timing'
        r=dict(id=row['id'],kind=meta['kind'],n=len(raw),m=len(c),pure_compressor_ms=ms,setup_ms=plan.setup_ms,
            persistent_allocated_bytes=plan.persistent_allocated_bytes,feature_exact=torch.equal(c,ref),max_abs_error=float((c.float()-ref.float()).abs().max()),
            repeat_features_equal=repeated,prediction=pred,score=native_score(pred,row),generated_ids=rt['generated_ids'])
        records.append(r);print(json.dumps(r),flush=True)
        del plan,c,cc,d,first,p,ref;torch.npu.synchronize();torch.npu.empty_cache()
    save(base/f'rows.rank{rank}.json',records);dist.barrier()
    if rank==0:save(base/'complete.json',dict(cases=8*world))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
