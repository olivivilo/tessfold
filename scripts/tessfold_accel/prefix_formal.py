"""Evaluate approximate semantic-KV caching on all23,565QA, both REF/Column.

Reference is the independent original full generation, cap256, with saved token
IDs; do not rerun full decoder merely to generate a favorable timing baseline.
No throughput inference; this stage assesses TASK impact and causal isolation.
"""
import collections,hashlib,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime
from scripts.eval_token_proxy import shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.tessfold_prefix_cache import build,answer
from vqa.benchmark_metrics import native_score

@torch.no_grad()
def main():
    verify();rank,world,_,device=init_runtime('npu',4)
    meta_group=dist.new_group(backend='gloo')
    base=ROOT/'artifacts/tessfold_acceleration_20261115/prefix_formal';base.mkdir(exist_ok=True)
    source=ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl';rows=lines(source);assert len(rows)==23565
    pages=collections.defaultdict(list)
    for r in rows:pages[r['document_id']].append(r)
    assignments=[[] for _ in range(world)];counts=[0]*world
    for page,rr in sorted(pages.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        i=min(range(world),key=lambda i:counts[i]);assignments[i].append((page,rr));counts[i]+=len(rr)
    manifest=dict(world=world,rank=rank,questions=23565,source_sha256={s:sha(ROOT/s) for s in ['scripts/tessfold_accel/prefix_formal.py','vqa/tessfold_prefix_cache.py','vqa/tessfold_fast_runtime.py','vqa/tessfold_native_partition.py','vqa/tessfold_partition_native.cpp']},data_sha256=sha(source),cap=256,methods=['REF','Column'],ratio=.25)
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    wanted={r['id'] for _,rr in assignments[rank] for r in rr};old={}
    directory=ROOT/'artifacts/compression_lab_v2/full/full_col_adaptive_pre-merger_r0.25/benchmark'
    for f in directory.glob('predictions.rank*.jsonl'):
        with f.open() as h:
            for line in h:
                r=json.loads(line)
                if r['variant']=='trained' and r['id'] in wanted:old[r['id']]=dict(ids=r['student_ids'],prediction=r['student_prediction'])
    assert len(old)==len(wanted)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    module=load_module(engine,'col_adaptive',.25).requires_grad_(False)
    cache=ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache'
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={(r['id'],r['method']) for r in previous}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    with path.open('a') as f:
      for page,rr in assignments[rank]:
        if all((r['id'],m) in done for r in rr for m in ['REF','Column']):continue
        features=torch.load(cache/'features'/f'{page}.pt',map_location='cpu',weights_only=True)
        raw,grid=features['features'].to(device),features['grid'].to(device);shape=dimensions(grid)
        projected={'REF':engine.model.vision_tower.merger(raw)}
        c,d=compress(module,raw,grid,None,.25,shape=shape);projected['Column']=engine.model.vision_tower.merger(c);del c,d
        first=torch.load(cache/'teachers'/f'{hashlib.sha256(rr[0]["id"].encode()).hexdigest()}.pt',map_location='cpu',weights_only=True)
        prompt=first['input_ids'].to(device)
        ids={method:prompt if method=='REF' else shortened_inputs({'input_ids':prompt},len(x),engine.model.config.image_token_id)['input_ids'] for method,x in projected.items()}
        prefixes={method:build(engine.model,ids[method],x) for method,x in projected.items()}
        del raw,features
        for row in rr:
            t=torch.load(cache/'teachers'/f'{hashlib.sha256(row["id"].encode()).hexdigest()}.pt',map_location='cpu',weights_only=True)
            assert t['max_new_tokens']==256;prompt=t['input_ids'].to(device)
            for method,x in projected.items():
                if (row['id'],method) in done:continue
                ids=prompt if method=='REF' else shortened_inputs({'input_ids':prompt},len(x),engine.model.config.image_token_id)['input_ids']
                cached=answer(engine.model,ids,*prefixes[method],cap=256)
                prediction=engine.processor.decode(cached,skip_special_tokens=True)
                reference_ids=t['generated_ids'] if method=='REF' else old[row['id']]['ids']
                reference_prediction=t['prediction'] if method=='REF' else old[row['id']]['prediction']
                result=dict(row,method=method,reference_prediction=reference_prediction,cached_prediction=prediction,
                    reference_ids=reference_ids,cached_ids=cached,tokens_equal=reference_ids==cached,
                    reference_score=native_score(reference_prediction,row),cached_score=native_score(prediction,row),
                    cache_immutable=True,visual_tokens=len(x),prefix_tokens=prefixes[method][1].shape[1],reference_generation_reused=True,
                    metric='Native QA score;DeepForm/KLC per-field EM here;corpus F1 additionally aggregated in final report')
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add((row['id'],method))
        del projected,prefixes;torch.npu.empty_cache()
        print(json.dumps(dict(rank=rank,page=page,records=len(done))),flush=True)
    dist.barrier(group=meta_group)
    if rank==0:
        rows=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)]
        assert len(rows)==47130 and len({(r['id'],r['method']) for r in rows})==47130
        save(base/'complete.json',dict(questions=23565,records=47130,methods={m:dict(token_mismatches=sum(not r['tokens_equal'] for r in rows if r['method']==m),task_score_sum_delta=sum(r['cached_score']-r['reference_score'] for r in rows if r['method']==m)) for m in ['REF','Column']},not_strict_equivalence=True))
    dist.barrier(group=meta_group);dist.destroy_process_group()

if __name__=='__main__':main()
