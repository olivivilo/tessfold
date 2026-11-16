"""Real24-page deployment: remove repeated image preprocessing fairly.

4scenarios: cold, features+original processor, features+exact text-only template,
semantic prefix KV (explicit numerical approximation). All three methods get the
same rights; ingestion and per-request costs retained, no repeated fake queries.
"""
import collections,json,os,random,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.benchmark import clean_generate,get_question
from scripts.eval_token_proxy import shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import raw_vision_only
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.tessfold_prefix_cache import build,answer,text_inputs
from vqa.benchmark_metrics import native_score

@torch.no_grad()
def projected(engine,module,row,raw,grid,shape):
    if module is None:return engine.model.vision_tower.merger(raw)
    q=None if module.kind=='col_adaptive' else get_question(engine,row)
    c,_=compress(module,raw,grid,q,.25,shape=shape)
    return engine.model.vision_tower.merger(c)


@torch.no_grad()
def run(engine,module,page,method,scenario):
    rows=page['questions'];torch.npu.synchronize();start=time.perf_counter();ingest=0.;cache=prefix=raw=visual=grid=None;shape=None
    if scenario!='cold':
        inp=engine.inputs(rows[0]);grid=inp['image_grid_thw'];shape=dimensions(grid)
        with raw_vision_only(engine.model.vision_tower):_,raw=engine.model.vision_tower(inp['pixel_values'],grid)
        n=len(raw)
        if method!='Tree':
            visual=projected(engine,module,rows[0],raw,grid,shape);m=len(visual)
            original=shortened_inputs(inp,m,engine.model.config.image_token_id)['input_ids']
            assert torch.equal(text_inputs(engine,rows[0],m),original),'Text-only prompt differs from official processor'
            if scenario=='prefix_kv':cache,prefix=build(engine.model,original,visual);del visual;visual=None
            del raw;raw=None
        del inp
        torch.npu.synchronize();ingest=(time.perf_counter()-start)*1e3
    result=[]
    for row in rows:
        torch.npu.synchronize();ts=time.perf_counter()
        if scenario=='cold':
            inp=engine.inputs(row);grid=inp['image_grid_thw'];shape=dimensions(grid)
            with raw_vision_only(engine.model.vision_tower):_,raw=engine.model.vision_tower(inp['pixel_values'],grid)
            visual=projected(engine,module,row,raw,grid,shape);m=len(visual)
            ids=shortened_inputs(inp,m,engine.model.config.image_token_id)['input_ids'];del inp,raw;raw=None
            output=clean_generate(engine,ids,visual,cap=256)
            del visual;visual=None
        else:
            if method=='Tree':visual=projected(engine,module,row,raw,grid,shape);m=len(visual)
            if scenario=='feature_processor':
                inp=engine.inputs(row);ids=shortened_inputs(inp,m,engine.model.config.image_token_id)['input_ids'];del inp
            else:ids=text_inputs(engine,row,m)
            if scenario=='prefix_kv':
                if method=='Tree':local,local_prefix=build(engine.model,ids,visual);output=answer(engine.model,ids,local,local_prefix);del local,local_prefix
                else:output=answer(engine.model,ids,cache,prefix)
            else:output=clean_generate(engine,ids,visual,cap=256)
            if method=='Tree':del visual;visual=None
        torch.npu.synchronize();elapsed=(time.perf_counter()-ts)*1e3
        prediction=engine.processor.decode(output,skip_special_tokens=True)
        result.append(dict(id=row['id'],task=row['task'],milliseconds=elapsed,generated_ids=output,prediction=prediction,score=native_score(prediction,row)))
    return dict(page=page['document_id'],task=page['task'],method=method,scenario=scenario,ingest_ms=ingest,queries=result,
        amortization={str(k):(ingest+sum(r['milliseconds'] for r in result[:k]))/k for k in [1,2,5,10]},
        prefix_policy='Complete visual block, immutable DynamicLayer fork, numerical variant; feature_text_only is independent of this variant')


@torch.no_grad()
def main():
    verify();rank,world,_,device=init_runtime('npu',4);meta_group=dist.new_group(backend='gloo');base=ROOT/'artifacts/tessfold_acceleration_20261115/multi_deployment';base.mkdir(exist_ok=True)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    modules={'REF':None,'Column':load_module(engine,'col_adaptive',.25).requires_grad_(False),'Tree':load_module(engine,'quadtree_question',.25).requires_grad_(False)}
    pages=load(ROOT/'artifacts/tessfold_system_repair_20261115/multi_pages.json',[])[rank::world]
    sources=['scripts/tessfold_accel/multi_deployment.py','vqa/tessfold_prefix_cache.py','vqa/tessfold_fast_runtime.py','vqa/tessfold_partition_native.cpp']
    manifest=dict(world=world,rank=rank,sources={s:sha(ROOT/s) for s in sources},pages=len(pages),scenarios=['cold','feature_processor','feature_text_only','prefix_kv'],repeats=3,expected_page_runs=864)
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={(r['page'],r['method'],r['scenario'],r['repeat']) for r in previous}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    jobs=[(page,m,s,k) for page in pages for m in ['REF','Column','Tree'] for s in manifest['scenarios'] for k in range(3)]
    random.Random(20261115+rank).shuffle(jobs)
    with path.open('a') as f:
        for page,method,scenario,repeat in jobs:
            key=(page['document_id'],method,scenario,repeat)
            if key in done:continue
            r=run(engine,modules[method],page,method,scenario);r['repeat']=repeat
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(key)
            if len(done)%8==0:print(json.dumps(dict(rank=rank,runs=len(done))),flush=True)
    dist.barrier(group=meta_group)
    if rank==0:
        rr=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)];assert len(rr)==864
        reference={(r['page'],r['method'],r['repeat']):r for r in rr if r['scenario']=='cold'}
        mismatches=collections.Counter();score_delta=collections.Counter()
        for r in rr:
            original=reference[(r['page'],r['method'],r['repeat'])]
            for a,b in zip(r['queries'],original['queries']):
                if a['generated_ids']!=b['generated_ids']:mismatches[r['scenario']]+=1
                score_delta[r['scenario']]+=a['score']-b['score']
        assert mismatches['feature_processor']==mismatches['feature_text_only']==0,'Pure feature/text caching changed tokens'
        save(base/'complete.json',dict(page_runs=864,questions=8640,feature_paths_exact=True,prefix_token_mismatches=mismatches['prefix_kv'],prefix_task_score_sum_delta=score_delta['prefix_kv'],
            note='Prefix cache evaluated as separate numerical variant; no unconditional bit-exact claim.'))
    dist.barrier(group=meta_group);dist.destroy_process_group()

if __name__=='__main__':main()
