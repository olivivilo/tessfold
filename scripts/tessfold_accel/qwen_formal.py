"""Independent native8B teacher and trainedP formal VQA and deployment timing.

The run list is frozen before formal execution from completed dev-selected
checkpoints; no test-based epoch, architecture, seed, or prompt selection.
"""
import argparse,collections,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
from scripts.tessfold_accel.qwen_training import Data,summary
from vqa.tessfold_qwen3vl import NativeQwen,NativeSpatialRouter
from vqa.tessfold_qwen3vl_stable import StableColumn
from vqa.tessfold_qwen3vl_timing import TimingQwen
from vqa.benchmark_metrics import native_score
from scripts.tessfold_accel.qwen_control_stats import control_rows,paired_results
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--label',default='qwen_formal');p.add_argument('--timing',action='store_true');a=p.parse_args()
    rank,world,_,device=init_runtime('npu',4);meta_group=dist.new_group(backend='gloo');reg=load(ROOT/a.runs);assert reg['runs']
    base=BASE/a.label;base.mkdir(exist_ok=True)
    rows=lines(ROOT/('artifacts/tessfold_system_repair_20261115/timing_rows.jsonl' if a.timing else 'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl'))
    assert len(rows)==(90 if a.timing else 23565)
    files=['vqa/tessfold_qwen3vl.py','vqa/tessfold_qwen3vl_stable.py','vqa/tessfold_qwen3vl_timing.py','scripts/tessfold_accel/qwen_training.py','scripts/tessfold_accel/qwen_control_stats.py','scripts/tessfold_accel/qwen_eval_safe.py','scripts/tessfold_accel/run_qwen_safe.py','scripts/tessfold_accel/qwen_eval.sh',str(Path(__file__).relative_to(ROOT))]
    manifest=dict(rank=rank,world=world,run_registry_sha256=sha(ROOT/a.runs),source_sha256={s:sha(ROOT/s) for s in files},
        cap=256,short_answer=True,weights_frozen=True,timing=a.timing,
        interpretation='Native8B visual backbone, original mergers+3DeepStack branches; same P on4branches, representative MRoPE. No UND LLM transplant.')
    controls=[];control_ids=set()
    if not a.timing:
        subset,registration=control_rows();control_ids={r['id'] for r in subset}
        assert control_ids <= {r['id'] for r in rows}
        for meta in reg['runs']:
            initial=ROOT/meta['full_run']/'initial.pt';assert initial.exists()
            controls.append(dict(meta,id=meta['id']+'_initial',checkpoint=str(initial.relative_to(ROOT)),checkpoint_sha256=sha(initial),
                stage='initial',role='initial_control',parent_model=meta['id']))
        manifest.update(initial_control_subset_sha256=registration['data_sha256'],initial_control_questions=len(control_ids),initial_controls=controls)
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    assert load(BASE/'qwen_identity56/complete.json').get('all_stock_HF_tokens_equal')
    if a.timing:
        timing_gate=load(BASE/'qwen_timing_identity/complete.json')
        assert timing_gate.get('natural_equal')==56 and timing_gate.get('fixed64_equal')==7
        assert timing_gate['timing_source_sha256']==sha(ROOT/'vqa/tessfold_qwen3vl_timing.py')
    engine=(TimingQwen if a.timing else NativeQwen)(device);data=Data(engine,64);modules={}
    for meta in reg['runs']+controls:
        assert sha(ROOT/meta['checkpoint'])==meta['checkpoint_sha256']
        state=torch.load(ROOT/meta['checkpoint'],map_location='cpu',weights_only=False)
        router=(StableColumn if meta.get('allocation') else NativeSpatialRouter)(meta['kind']).to(device).eval().requires_grad_(False)
        router.load_state_dict(state['router']);modules[meta['id']]=router
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    key=lambda r:(r['id'],r['model'],r.get('mode','natural_eos'),r.get('repeat',0))
    done={key(r) for r in previous};methods=[dict(id='REF',kind='REF',ratio=1.)]+reg['runs']
    with path.open('a') as f:
      for index,row in enumerate(rows[rank::world]):
        if not a.timing:
            d=engine.from_features(row,data.features(row))
            for meta in methods+(controls if row['id'] in control_ids else []):
                k=(row['id'],meta['id'],'natural_eos',0)
                if k in done:continue
                if meta['id']=='REF':p=reps=None
                else:p,reps=modules[meta['id']](d['x'],d['shape'],d['question'],meta['ratio'])
                ids=engine.greedy(d,p,reps,cap=256);prediction=engine.decode(ids)
                record=dict(row,model=meta['id'],ratio=meta['ratio'],prediction=prediction,score=native_score(prediction,row),generated_ids=ids,
                    hit_token_limit=len(ids)==256,visual_tokens=len(d['x']) if p is None else len(p),original_visual_tokens=len(d['x']),checkpoint=meta.get('checkpoint'),role=meta.get('role','primary'),parent_model=meta.get('parent_model'),benchmark_used_for_selection=False)
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(k)
                if p is not None:del p,reps
        else:
            # Equal native processor/full visual costs for all methods. Warmup
            # uses this fixed request and is outside recorded repetitions.
            for meta in methods:
              warm=engine.prepare(row);engine.greedy(warm,cap=4);del warm
              for mode in ['natural_eos','fixed64']:
               for repeat in range(3):
                k=(row['id'],meta['id'],mode,repeat)
                if k in done:continue
                torch.npu.synchronize();torch.npu.reset_peak_memory_stats();start=time.perf_counter()
                d=engine.prepare(row);torch.npu.synchronize();front=(time.perf_counter()-start)*1e3
                ts=time.perf_counter()
                if meta['id']=='REF':p=reps=None
                else:p,reps=modules[meta['id']](d['x'],d['shape'],d['question'],meta['ratio'])
                torch.npu.synchronize();comp=(time.perf_counter()-ts)*1e3
                calls=[0];first=[None]
                def pre(mod,args,kwargs):
                    if calls[0]==0:torch.npu.synchronize();first[0]=time.perf_counter()
                    calls[0]+=1
                def post(mod,args,kwargs,out):
                    if calls[0]==1:torch.npu.synchronize();first.append(time.perf_counter())
                h=engine.model.model.language_model.register_forward_pre_hook(pre,with_kwargs=True);hh=engine.model.model.language_model.register_forward_hook(post,with_kwargs=True)
                try:ids=engine.greedy(d,p,reps,cap=64 if mode=='fixed64' else 256,fixed=mode=='fixed64')
                finally:h.remove();hh.remove()
                torch.npu.synchronize();finish=time.perf_counter()
                record=dict(id=row['id'],document_id=row['document_id'],task=row['task'],model=meta['id'],ratio=meta['ratio'],mode=mode,repeat=repeat,
                    end_to_end_ms=(finish-start)*1e3,front_processor_vision_mergers_ms=front,p_generator_ms=comp,visual_setup_ms=engine.visual_setup_ms,
                    compressor_ms=comp+engine.visual_setup_ms,ttft_ms=(engine.first_token_timestamp-start)*1e3,
                    post_front_ttft_ms=(engine.first_token_timestamp-start)*1e3-front,prefill_ms=(first[1]-first[0])*1e3,
                    prefill_done_ms=(first[1]-start)*1e3,peak_allocated_bytes=int(torch.npu.max_memory_allocated()),
                    original_visual_tokens=len(d['x']),visual_tokens=len(d['x']) if p is None else len(p),generated_ids=ids,output_tokens=len(ids),prediction=engine.decode(ids),
                    note='ttft_ms includes first LM head+argmax. compressor_ms includes P generation+4branch aggregation+position selection+prompt embedding/scatter; not router-only. prefill_done_ms remains separately recorded. No interrequest cache.')
                if mode=='fixed64':assert len(ids)==64
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(k)
                del d,p,reps
        if index%50==0:print(json.dumps(dict(rank=rank,questions=index+1,records=len(done),mode='timing' if a.timing else 'formal')),flush=True)
    dist.barrier(group=meta_group)
    if rank==0:
        rr=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)]
        expected=len(rows)*len(methods)*(6 if a.timing else 1)+len(control_ids)*len(controls)
        assert len(rr)==expected and len({key(r) for r in rr})==expected
        expected_keys={(r['id'],m['id'],mode,repeat) for r in rows for m in methods
            for mode in (['natural_eos','fixed64'] if a.timing else ['natural_eos']) for repeat in (range(3) if a.timing else [0])}
        expected_keys|={(i,m['id'],'natural_eos',0) for i in control_ids for m in controls}
        assert {key(r) for r in rr}==expected_keys,'Formal Cartesian coverage mismatch'
        for r in rr:
            assert r['visual_tokens']==(r['original_visual_tokens'] if r['model']=='REF' else max(1,round(r['original_visual_tokens']*r['ratio'])))
            if not a.timing:assert abs(r['score']-native_score(r['prediction'],r))<1e-10
        for f in base.glob('manifest.rank*.json'):
            stored=load(f)
            for source,digest in stored['source_sha256'].items():assert sha(ROOT/source)==digest

        if a.timing:result=dict(time=now(),records=expected,timing=True,questions=90,models=len(methods))
        else:
            results={m['id']:summary([r for r in rr if r['model']==m['id']]) for m in methods}
            shared={}
            for m in methods:
                rrr=[r for r in rr if r['model']==m['id']];by={t:100*sum(r['score'] for r in rrr if r['task']==t)/sum(r['task']==t for r in rrr) for t in sorted({r['task'] for r in rrr})};shared[m['id']]=dict(macro=sum(by.values())/len(by),by_task=by)
            control_result=paired_results(rr,reg['runs'])
            result=dict(initial_control_results=control_result,primary_records=len(rows)*len(methods),initial_control_records=len(control_ids)*len(controls),
                time=now(),records=expected,questions=23565,models=len(methods),native_metric_results=results,shared_fieldEM_results=shared,
                selection='Models fixed by dev before test; benchmark never used for selection',cap=256,short_answer=True)
        save(base/'complete.json',result)
    dist.barrier(group=meta_group);dist.destroy_process_group()

if __name__=='__main__':main()
