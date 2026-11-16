"""Complete23,565QA x4 frozen configurations vs saved original tensor/generation.

No model selection, no retraining. A failure stays a failure, never loosens the
comparison to equality of task scores or decoded strings.
"""
import collections,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_accel.system_common import ROOT,BASE as SYSTEM,verify,load,lines,save,sha
from scripts.tessfold_repair.common import load_module,REG
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.tessfold_repair.benchmark import clean_generate,get_question
from scripts.eval_token_proxy import shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_fast_runtime import compress,dimensions

@torch.no_grad()
def main():
    verify();rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115/formal_equivalence';base.mkdir(exist_ok=True)
    reg=load(REG);metas=[r for r in reg['models'] if r['ratio'] in [.25,.5]]
    rows=lines(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl');assert len(rows)==23565
    manifest=dict(world=world,rank=rank,sources={s:sha(ROOT/s) for s in ['scripts/tessfold_accel/formal_equivalence.py','vqa/tessfold_fast_runtime.py','vqa/tessfold_native_partition.py','vqa/tessfold_partition_native.cpp']},
        checkpoint_sha256={m['id']:m['checkpoint_sha256'] for m in metas},data_sha256=sha(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl'),
        cap=256,questions=23565,configurations=4,expected_records=94260)
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path)
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest);path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous));done={(r['id'],r['model']) for r in previous}
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    modules={m['id']:load_module(engine,m['kind'],m['ratio']).requires_grad_(False) for m in metas}
    wanted={r['id'] for r in rows[rank::world]};historical={}
    for m in metas:
        lookup={}
        for p in (ROOT/m['output']/'benchmark').glob('predictions.rank*.jsonl'):
            with p.open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['variant']=='trained' and r['id'] in wanted:lookup[r['id']]=dict(ids=r['student_ids'],score=r['student_score'])
        assert len(lookup)==len(wanted);historical[m['id']]=lookup
    started=time.perf_counter();cache=ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache'
    with path.open('a') as f:
        for i,row in enumerate(rows[rank::world]):
            if all((row['id'],m['id']) in done for m in metas):continue
            prompt,raw,grid,x,response=prepare_example(engine,row,cache,256)
            q=get_question(engine,row);shape=dimensions(grid)
            for m in metas:
                if (row['id'],m['id']) in done:continue
                module=modules[m['id']]
                _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,m['ratio'],raw=raw,question=q)
                reference=(p@raw.float()).to(raw.dtype)
                c,d=compress(module,raw,grid,q,m['ratio'],shape=shape)
                assert torch.equal(p,d['weights']) and torch.equal(reference,c),f'Formal feature mismatch {row["id"]} {m["id"]}'
                projected=engine.model.vision_tower.merger(c)
                si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)['input_ids']
                ids=clean_generate(engine,si,projected,cap=256)
                assert ids==historical[m['id']][row['id']]['ids'],f'Formal generation mismatch {row["id"]} {m["id"]}'
                record=dict(id=row['id'],document_id=row['document_id'],task=row['task'],model=m['id'],ratio=m['ratio'],feature_exact=True,p_exact=True,generation_exact=True,score=historical[m['id']][row['id']]['score'],n=len(raw),m=len(c))
                f.write(json.dumps(record)+'\n');f.flush();done.add((row['id'],m['id']));del p,reference,c,d,projected
            if i%100==0:print(json.dumps(dict(rank=rank,questions=i+1,records=len(done),seconds=time.perf_counter()-started)),flush=True)
    dist.barrier()
    if rank==0:
        rr=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)];assert len(rr)==94260 and len({(r['id'],r['model']) for r in rr})==94260
        tasks={}
        for m in metas:
            mm=[r for r in rr if r['model']==m['id']];tt={t:100*sum(r['score'] for r in mm if r['task']==t)/sum(r['task']==t for r in mm) for t in sorted({r['task'] for r in mm})}
            tasks[m['id']]=dict(tasks=tt,macro=sum(tt.values())/len(tt))
        save(base/'complete.json',dict(questions=23565,records=94260,all_feature_exact=True,all_generation_exact=True,results=tasks,
            metric='Same historical benchmark: DeepForm/KLC fieldEM, task-equal macro. Not the native8B DUE-F1 small-development metric.'))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
