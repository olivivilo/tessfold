"""Gated full UND numerical-variant evaluation of explicit bidirectional PFA.

No training or test selection. Never replaces/merges strict eager scores.
"""
import argparse,collections,hashlib,json
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now,verify,load_module
from scripts.train_meta_compress import init_runtime
from scripts.eval_token_proxy import shortened_inputs
from scripts.tessfold_repair.benchmark import clean_generate,get_question
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import raw_vision_only
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.tessfold_bidirectional_vision import fused_vision
from vqa.benchmark_metrics import native_score,due_field_f1
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def gate():
    verify();pilot=load(BASE/'bidirectional_vision_pilot/complete.json')
    assert pilot['cases']==56 and pilot['expansion_gate_passed']
    for p in (BASE/'bidirectional_vision_pilot').glob('manifest.rank*.json'):
        for path,digest in load(p)['source_sha256'].items():assert sha(ROOT/path)==digest
    return pilot


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timing',action='store_true');a=parser.parse_args()
    pilot=gate();rank,world,_,device=init_runtime('npu',4);group=dist.new_group(backend='gloo')
    base=BASE/('bidirectional_timing' if a.timing else 'bidirectional_formal');base.mkdir(exist_ok=True)
    data_path=ROOT/('artifacts/tessfold_system_repair_20261115/timing_rows.jsonl' if a.timing else 'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl')
    rows=lines(data_path);assert len(rows)==(90 if a.timing else 23565)
    reg=verify();metas=[m for m in reg['models'] if m['ratio'] in [.25,.5]];assert len(metas)==4
    manifest=dict(rank=rank,world=world,timing=a.timing,pilot_sha256=sha(BASE/'bidirectional_vision_pilot/complete.json'),data_sha256=sha(data_path),
        sources={s:sha(ROOT/s) for s in ['scripts/tessfold_accel/bidirectional_formal.py','vqa/tessfold_bidirectional_vision.py','vqa/tessfold_fast_runtime.py','scripts/tessfold_accel/system_benchmark.py']},
        note='FrozenUNDauthorPFA with explicitnoncausalwindow; separate numericalvariant, no reuseofeagerformalaccuracy. No testselection.')
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    modules={m['id']:load_module(engine,m['kind'],m['ratio']).requires_grad_(False) for m in metas}
    if a.timing:
        from scripts.tessfold_accel.system_benchmark import single_request
        (base/'traces').mkdir(exist_ok=True);configs=[('REF',1.)]+[(name,r) for r in [.25,.5] for name in ['C_FAST','T_FAST']]
        path=base/f'rows.rank{rank}.jsonl';previous=lines(path);key=lambda r:(r['id'],r['method'],r['ratio'],r['mode'],r['repeat'],r['instrumentation'])
        done={key(r) for r in previous};path.write_text(''.join(json.dumps(r)+'\n' for r in previous))
        with fused_vision(engine.model.vision_tower),path.open('a') as f:
          for row in rows[rank::world]:
            for method,ratio in configs:
                meta=None if method=='REF' else next(m for m in metas if m['kind']==('col_adaptive' if method=='C_FAST' else 'quadtree_question') and m['ratio']==ratio)
                module=modules[meta['id']] if meta else None
                single_request(engine,module,row,method,ratio,cap=4)
                jobs=[(fixed,k,False) for fixed in [False,True] for k in range(3)]+[(False,0,True)]
                for fixed,repeat,memory in jobs:
                    k=(row['id'],method,ratio,'fixed64' if fixed else 'natural_eos',repeat,'memory1ms' if memory else 'synchronized_wall_and_NPU_events')
                    if k in done:continue
                    trace=base/'traces'/(hashlib.sha256(str(k).encode()).hexdigest()+'.npz')
                    r=single_request(engine,module,row,method,ratio,fixed,memory,trace);r.update(repeat=repeat,numerical_variant='bidirectional_PFA')
                    if memory:r.update(trace=str(trace.relative_to(ROOT)),trace_sha256=sha(trace))
                    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(k)
            print(json.dumps(dict(rank=rank,records=len(done))),flush=True)
    else:
        # Assign whole pages to one rank: one true encoder execution per image,
        # independent fresh decoder generation for every question/model.
        pages=collections.defaultdict(list)
        for row in rows:pages[row['document_id']].append(row)
        assignments=[[] for _ in range(world)];counts=[0]*world
        for page,rr in sorted(pages.items(),key=lambda kv:(-len(kv[1]),kv[0])):
            i=min(range(world),key=lambda k:counts[k]);assignments[i].append((page,rr));counts[i]+=len(rr)
        path=base/f'rows.rank{rank}.jsonl';previous=lines(path);key=lambda r:(r['id'],r['model']);done={key(r) for r in previous}
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
        teacher_cache=ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache/teachers'
        methods=[dict(id='REF',ratio=1.)]+metas
        with fused_vision(engine.model.vision_tower),path.open('a') as f:
          for page,rr in assignments[rank]:
            if all((r['id'],m['id']) in done for r in rr for m in methods):continue
            inp=engine.inputs(rr[0]);grid=inp['image_grid_thw'];shape=dimensions(grid)
            with raw_vision_only(engine.model.vision_tower):_,raw=engine.model.vision_tower(inp['pixel_values'],grid)
            del inp
            for row in rr:
                teacher=torch.load(teacher_cache/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True)
                assert teacher['max_new_tokens']==256
                prompt=teacher['input_ids'].to(device);q=get_question(engine,row)
                for meta in methods:
                    if (row['id'],meta['id']) in done:continue
                    if meta['id']=='REF':c=raw
                    else:c,info=compress(modules[meta['id']],raw,grid,q,meta['ratio'],shape=shape);del info
                    projected=engine.model.vision_tower.merger(c)
                    ids=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)['input_ids']
                    output=clean_generate(engine,ids,projected,cap=256);prediction=engine.processor.decode(output,skip_special_tokens=True)
                    r=dict(row,model=meta['id'],ratio=meta['ratio'],prediction=prediction,score=native_score(prediction,row),generated_ids=output,
                        hit_token_limit=len(output)==256,n=len(raw),m=len(c),numerical_variant='bidirectional_PFA',weights_frozen=True,benchmark_used_for_selection=False)
                    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add((row['id'],meta['id']));del c,projected
                del q,prompt
            del raw,grid
            print(json.dumps(dict(rank=rank,page=page,records=len(done))),flush=True)
    dist.barrier(group=group)
    if rank==0:
        records=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)]
        expected=3150 if a.timing else 117825;assert len(records)==len({key(r) for r in records})==expected
        result=dict(time=now(),records=expected,timing=a.timing,not_bit_exact=True,original_frozen_weights=True,benchmark_used_for_selection=False)
        if not a.timing:
            result['shared_fieldEM_results']={};result['native_metric_results']={}
            for meta in methods:
                ss=[r for r in records if r['model']==meta['id']];assert len(ss)==23565
                by={t:100*sum(r['score'] for r in ss if r['task']==t)/sum(r['task']==t for r in ss) for t in sorted({r['task'] for r in ss})}
                result['shared_fieldEM_results'][meta['id']]=dict(macro=sum(by.values())/len(by),by_task=by)
                native=dict(by)
                for t in ['KLC','DeepForm']:native[t]=due_field_f1([r for r in ss if r['task']==t])['score_percent']
                result['native_metric_results'][meta['id']]=dict(macro=sum(native.values())/len(native),by_task=native)
        save(base/'complete.json',result)
    dist.barrier(group=group);dist.destroy_process_group()

if __name__=='__main__':main()
