"""Paired full-prompt/semantic-prefix VQA; pilot or all23,565formalQA.

REF and Column receive exactly the same text/feature/cache reuse permissions.
Tree is excluded from cross-question visual-KV reuse because P depends on Q.
"""
import argparse,collections,json,os,time,shutil
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime
from scripts.eval_token_proxy import shortened_inputs
from scripts.tessfold_repair.benchmark import clean_generate
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import raw_vision_only
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.tessfold_prefix_cache import build,answer,text_inputs
from vqa.benchmark_metrics import native_score


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='prefix_pilot1');p.add_argument('--method',choices=['REF','Column'],default='Column');p.add_argument('--ratio',type=float,default=.25);p.add_argument('--full',action='store_true');a=p.parse_args()
    verify();rank,world,_,device=init_runtime('npu',4);base=ROOT/'artifacts/tessfold_acceleration_20261115'/a.label/a.method;base.mkdir(parents=True,exist_ok=True)
    source=ROOT/('artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl' if a.full else 'artifacts/compression_supplement_20261111/same_page_rows.jsonl')
    pages=collections.defaultdict(list)
    for row in lines(source):pages[row['document_id']].append(row)
    if not a.full:
        selected=load(ROOT/'artifacts/tessfold_system_repair_20261115/multi_pages.json',[])
        wanted={p['document_id'] for p in selected}
        for id_ in ['KLC_test_1569','DeepForm_test_1410']:
            wanted.update(k for k,v in pages.items() if any(r['id']==id_ for r in v))
        pages={k:sorted(v,key=lambda r:r['id'])[:10] for k,v in pages.items() if k in wanted}
    bins=[[] for _ in range(world)];loads=[0]*world
    for doc,rows in sorted(pages.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        i=min(range(world),key=lambda i:loads[i]);bins[i].append((doc,rows));loads[i]+=len(rows)
    files=['vqa/tessfold_prefix_cache.py','vqa/tessfold_fast_runtime.py','vqa/tessfold_partition_native.cpp',str(Path(__file__).relative_to(ROOT))]
    manifest=dict(job=os.environ.get('SLURM_JOB_ID'),method=a.method,ratio=a.ratio,full=a.full,source_sha256={f:sha(ROOT/f) for f in files},data_sha256=sha(source),world=world,rank=rank,questions=loads[rank],pages=len(bins[rank]))
    if (base/f'manifest.rank{rank}.json').exists():
        old=load(base/f'manifest.rank{rank}.json');assert all(old[k]==manifest[k] for k in ['method','ratio','full','source_sha256','data_sha256','world','rank','questions','pages'])
    save(base/f'manifest.rank{rank}.json',manifest)
    if rank==0:
        for f in files:
            dest=base/'sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    module=None if a.method=='REF' else load_module(engine,'col_adaptive',a.ratio).requires_grad_(False)
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={r['id'] for r in previous}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    with path.open('a') as f:
        for doc,rows in bins[rank]:
            if all(r['id'] in done for r in rows):continue
            torch.npu.synchronize();start=time.perf_counter();inp=engine.inputs(rows[0]);shape=dimensions(inp['image_grid_thw'])
            with raw_vision_only(engine.model.vision_tower):_,raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
            if module is None:projected=engine.model.vision_tower.merger(raw)
            else:
                c,d=compress(module,raw,inp['image_grid_thw'],None,a.ratio,shape=shape);projected=engine.model.vision_tower.merger(c);del c,d
            ids=text_inputs(engine,rows[0],len(projected));expected=inp['input_ids'] if module is None else shortened_inputs(inp,len(projected),engine.model.config.image_token_id)['input_ids']
            assert torch.equal(ids,expected),'Cached text path changed native prompt'
            del raw,inp
            cache,prefix=build(engine.model,ids,projected);torch.npu.synchronize();ingest=(time.perf_counter()-start)*1e3
            for row in rows:
                if row['id'] in done:continue
                ids=text_inputs(engine,row,len(projected))
                torch.npu.synchronize();ts=time.perf_counter();reference=clean_generate(engine,ids,projected,cap=256);torch.npu.synchronize();full_ms=(time.perf_counter()-ts)*1e3
                ts=time.perf_counter();cached=answer(engine.model,ids,cache,prefix,cap=256);torch.npu.synchronize();cached_ms=(time.perf_counter()-ts)*1e3
                rp=engine.processor.decode(reference,skip_special_tokens=True);cp=engine.processor.decode(cached,skip_special_tokens=True)
                result=dict(row,method=a.method,ratio=a.ratio,reference_prediction=rp,cached_prediction=cp,reference_ids=reference,cached_ids=cached,
                    tokens_equal=reference==cached,reference_score=native_score(rp,row),cached_score=native_score(cp,row),full_decoder_ms=full_ms,cached_decoder_ms=cached_ms,
                    ingest_ms=ingest,questions_on_page=len(rows),visual_tokens=len(projected),prefix_tokens=prefix.shape[1],cache_immutable=True,
                    cache_policy='complete visual block including vision_end; shallow layer-object fork, read-only shared prefix tensors; DynamicLayer concatenates new storage')
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(row['id'])
            del cache,prefix,projected;torch.npu.empty_cache()
            print(json.dumps(dict(rank=rank,page=doc,questions_done=len(done))),flush=True)
    dist.barrier()
    if rank==0:
        rr=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)]
        assert len(rr)==sum(loads) and len({r['id'] for r in rr})==len(rr)
        save(base/'complete.json',dict(questions=len(rr),pages=len(pages),token_mismatches=sum(not r['tokens_equal'] for r in rr),
            task_score_sum_delta=sum(r['cached_score']-r['reference_score'] for r in rr),all_cache_immutable=True,full=a.full))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
