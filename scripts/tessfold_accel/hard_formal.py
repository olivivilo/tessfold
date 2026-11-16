"""Full23,565QA evaluation of no-dense-P top-token readout against legacyH.

Do not assume soft equivalence; even historicalH equivalence is measured, not
silently inherited. Fixed checkpoint, no benchmark-based selection.
"""
import json,os
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.tessfold_repair.benchmark import clean_generate,get_question
from scripts.eval_token_proxy import shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.benchmark_metrics import native_score

@torch.no_grad()
def main():
    verify();rank,world,_,device=init_runtime('npu',4);metadata=dist.new_group(backend='gloo')
    base=ROOT/'artifacts/tessfold_acceleration_20261115/hard_formal';base.mkdir(exist_ok=True)
    rows=lines(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl');assert len(rows)==23565
    wanted={r['id'] for r in rows[rank::world]};old={}
    for f in (ROOT/'artifacts/compression_supplement_20261111/runs/HA_r0.25/benchmark').glob('predictions.rank*.jsonl'):
        with f.open() as h:
            for line in h:
                r=json.loads(line)
                if r['variant']=='H' and r['id'] in wanted:old[r['id']]=r
    assert len(old)==len(wanted)
    manifest=dict(rank=rank,world=world,sources={s:sha(ROOT/s) for s in ['scripts/tessfold_accel/hard_formal.py','vqa/tessfold_fast_runtime.py','vqa/tessfold_partition_native.cpp']},ratio=.25,kind='quadtree_question',cap=256,mode='hard_linear',weights_frozen=True)
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);module=load_module(engine,'quadtree_question',.25).requires_grad_(False)
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={r['id'] for r in previous};path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    with path.open('a') as f:
      for i,row in enumerate(rows[rank::world]):
        if row['id'] in done:continue
        prompt,raw,grid,x,response=prepare_example(engine,row,ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache',256)
        c,d=compress(module,raw,grid,get_question(engine,row),.25,backend='hard_linear',shape=dimensions(grid))
        projected=engine.model.vision_tower.merger(c);ids=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)['input_ids']
        output=clean_generate(engine,ids,projected,cap=256);prediction=engine.processor.decode(output,skip_special_tokens=True)
        result=dict(row,prediction=prediction,score=native_score(prediction,row),generated_ids=output,original_prediction=old[row['id']]['student_prediction'],original_score=old[row['id']]['student_score'],
            legacy_h_token_equal=output==old[row['id']]['student_ids'],n=len(raw),m=len(c),readout='argmax original leaf score, no dense P, independent hard variant')
        f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(row['id'])
        if i%100==0:print(json.dumps(dict(rank=rank,records=len(done))),flush=True)
    dist.barrier(group=metadata)
    if rank==0:
        rr=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)];assert len(rr)==len({r['id'] for r in rr})==23565
        by={t:100*sum(r['score'] for r in rr if r['task']==t)/sum(r['task']==t for r in rr) for t in sorted({r['task'] for r in rr})}
        save(base/'complete.json',dict(questions=23565,legacy_h_token_mismatches=sum(not r['legacy_h_token_equal'] for r in rr),tasks=by,macro=sum(by.values())/len(by),
            score_sum_delta=sum(r['score']-r['original_score'] for r in rr),not_soft_equivalent=True,benchmark_used_for_selection=False))
    dist.barrier(group=metadata);dist.destroy_process_group()

if __name__=='__main__':main()
