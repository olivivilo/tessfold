"""Real frozen UND pilot, precise compressor-only costs and numerical gates."""
import argparse,json,os,time,shutil
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,verify,load_module
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.eval_token_proxy import generate,shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import compress as original_compress
from vqa.tessfold_fast_runtime import compress,dimensions,ScoreGraph
from vqa.benchmark_metrics import native_score


def milliseconds(fn,repeats=5):
    for _ in range(2):fn()
    torch.npu.synchronize();start=time.perf_counter()
    for _ in range(repeats):fn()
    torch.npu.synchronize();return (time.perf_counter()-start)*1e3/repeats


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='pilot1');p.add_argument('--limit',type=int,default=8);p.add_argument('--graph-only',action='store_true');a=p.parse_args()
    reg=verify();rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115'/a.label;base.mkdir(parents=True,exist_ok=True)
    files=['vqa/tessfold_partition_native.cpp','vqa/tessfold_native_partition.py','vqa/tessfold_fast_runtime.py',str(Path(__file__).relative_to(ROOT))]
    save(base/f'manifest.rank{rank}.json',dict(job=os.environ.get('SLURM_JOB_ID'),world=world,rank=rank,sources={f:sha(ROOT/f) for f in files}))
    if rank==0:
        for f in files:
            dst=base/'sources'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    meta=reg['models'][rank if world==8 else 2];module=load_module(engine,meta['kind'],meta['ratio']).requires_grad_(False)
    rows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')[:a.limit]
    cache=ROOT/'artifacts/meta_compress_strict/npu_cache';path=base/f'rows.rank{rank}.jsonl'
    assert not path.exists(),'Use a new versioned label for each probe'
    old={r['id']:r for r in lines(ROOT/meta['output']/'validation.epoch4.jsonl')}
    with path.open('w') as f:
        for row in rows:
            prompt,raw,grid,x,response=prepare_example(engine,row,cache,64)
            shape=dimensions(grid)
            qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device)
            q=engine.model.get_input_embeddings()(qids)
            _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,meta['ratio'],raw=raw,question=q)
            reference=(p@raw.float()).to(raw.dtype)
            result=dict(id=row['id'],task=row['task'],kind=meta['kind'],ratio=meta['ratio'],n=len(raw),variants={})
            backends=['dense_exact','bucket_fp32','bucket_fold','hard_linear'] if not a.graph_only else ['dense_exact','bucket_fp32']
            graph=None
            if a.graph_only:
                before=int(torch.npu.memory_allocated());graph=ScoreGraph(module,raw,q if module.kind=='quadtree_question' else None,*shape)
                result['graph_setup_ms']=graph.setup_ms;result['graph_persistent_allocated_bytes']=int(torch.npu.memory_allocated())-before
            for backend in backends:
                c,d=compress(module,raw,grid,q,meta['ratio'],backend,shape,graph,True)
                equal=torch.equal(c,reference)
                if backend=='dense_exact':assert equal and torch.equal(p,d['weights']),'Compiled strict path changed features'
                projected=engine.model.vision_tower.merger(c)
                si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
                pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,projected,raw)
                exact_ids=rt['generated_ids']==old[row['id']]['student_ids']
                if backend=='dense_exact':assert exact_ids,'Strict features match but generation differs'
                result['variants'][backend]=dict(feature_equal=equal,representatives_equal=torch.equal(d['representatives'],p.argmax(-1)),
                    max_abs_error=float((c.float()-reference.float()).abs().max()),generation_equal=exact_ids,prediction=pred,
                    score=native_score(pred,row),original_score=native_score(old[row['id']]['student_prediction'] if 'student_prediction' in old[row['id']] else old[row['id']].get('student',''),row),
                    student_ids=rt['generated_ids'],timings_ms=d['timings_ms'],compressor_wall_ms=milliseconds(lambda:compress(module,raw,grid,q,meta['ratio'],backend,shape,graph)),
                    padded_support=d.get('padded_support'))
                del c,d,projected
            result['original_compressor_wall_ms']=milliseconds(lambda:original_compress(module,raw,grid,q,meta['ratio']))
            f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();print(json.dumps(dict(rank=rank,id=row['id'],strict_ms=result['variants']['dense_exact']['compressor_wall_ms'])),flush=True)
            del p,reference
            if graph is not None:del graph;torch.npu.synchronize();torch.npu.empty_cache()
    dist.barrier()
    if rank==0:
        rr=[r for path in base.glob('rows.rank*.jsonl') for r in lines(path)]
        save(base/'complete.json',dict(cases=len(rr),strict_all_exact=True,world=world,graph=a.graph_only))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
