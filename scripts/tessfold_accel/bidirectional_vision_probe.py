"""Explicit bidirectional fused-attention pilot: all56devQA, frozen weights, new outputs."""
import collections,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.tessfold_repair.benchmark import clean_generate,get_question
from scripts.eval_token_proxy import shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import raw_vision_only
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.tessfold_bidirectional_vision import fused_vision
from vqa.benchmark_metrics import native_score

@torch.no_grad()
def encoder(engine,inputs,fused):
    from contextlib import nullcontext
    context=fused_vision(engine.model.vision_tower) if fused else nullcontext()
    with context,raw_vision_only(engine.model.vision_tower):
        for _ in range(2):engine.model.vision_tower(inputs['pixel_values'],inputs['image_grid_thw'])
        torch.npu.synchronize();torch.npu.reset_peak_memory_stats();start=time.perf_counter()
        for _ in range(3):_,raw=engine.model.vision_tower(inputs['pixel_values'],inputs['image_grid_thw'])
        torch.npu.synchronize();elapsed=(time.perf_counter()-start)*1e3/3
        return raw,elapsed,int(torch.npu.max_memory_allocated())

@torch.no_grad()
def main():
    reg=verify();rank,world,_,device=init_runtime('npu',4);metadata=dist.new_group(backend='gloo')
    base=ROOT/'artifacts/tessfold_acceleration_20261115/bidirectional_vision_pilot';base.mkdir(exist_ok=True)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    metas=[m for m in reg['models'] if m['ratio'] in [.25,.5]]
    modules={m['id']:load_module(engine,m['kind'],m['ratio']).requires_grad_(False) for m in metas}
    old={m['id']:{r['id']:r for r in lines(ROOT/m['output']/'validation.epoch4.jsonl')} for m in metas}
    save(base/f'manifest.rank{rank}.json',dict(world=world,rank=rank,source_sha256={s:sha(ROOT/s) for s in ['vqa/tessfold_bidirectional_vision.py','scripts/tessfold_accel/bidirectional_vision_probe.py']},
        note='Same frozen encoder weights; Ascend fused full-bidirectional attention with explicit pre/next INTMAX, not a strict replacement.',
        expansion_gate='Encoder median wall speedup>=1.5 and REF QA mean drop<=2pp on fixed56dev. If passed, full evaluation remains separate numerical candidate.'))
    path=base/f'rows.rank{rank}.jsonl';assert not path.exists()
    with path.open('w') as f:
      for row in lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')[rank::world]:
        inp=engine.inputs(row);shape=dimensions(inp['image_grid_thw'])
        raw,old_ms,old_mem=encoder(engine,inp,False);reference_raw=raw.cpu();del raw
        raw,new_ms,new_mem=encoder(engine,inp,True)
        old_on_device=reference_raw.to(device);difference=float((old_on_device.float()-raw.float()).abs().max());cosine=float(torch.nn.functional.cosine_similarity(old_on_device.flatten().float(),raw.flatten().float(),dim=0));del old_on_device,reference_raw
        projected=engine.model.vision_tower.merger(raw);ids=inp['input_ids'];new_ids=clean_generate(engine,ids,projected,cap=64)
        prediction=engine.processor.decode(new_ids,skip_special_tokens=True)
        teacher=torch.load(ROOT/'artifacts/meta_compress_strict/npu_cache/teachers'/(__import__('hashlib').sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True)
        methods={'REF':dict(prediction=prediction,score=native_score(prediction,row),original_prediction=teacher['prediction'],original_score=native_score(teacher['prediction'],row),generated_ids=new_ids,token_equal=new_ids==teacher['generated_ids'])}
        q=get_question(engine,row)
        for m in metas:
            c,d=compress(modules[m['id']],raw,inp['image_grid_thw'],q,m['ratio'],shape=shape);projected=engine.model.vision_tower.merger(c)
            ids=shortened_inputs(inp,len(c),engine.model.config.image_token_id)['input_ids'];new_ids=clean_generate(engine,ids,projected,cap=64)
            prediction=engine.processor.decode(new_ids,skip_special_tokens=True);original=old[m['id']][row['id']]
            methods[m['id']]=dict(prediction=prediction,score=native_score(prediction,row),original_prediction=original['student_prediction'],original_score=original['student_score'],generated_ids=new_ids,token_equal=new_ids==original['student_ids'])
        record=dict(id=row['id'],task=row['task'],n=len(raw),eager_encoder_ms=old_ms,fused_encoder_ms=new_ms,eager_peak_bytes=old_mem,fused_peak_bytes=new_mem,
            max_abs_feature_difference=difference,feature_cosine=cosine,methods=methods,all_weights_frozen=True)
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();print(json.dumps(dict(rank=rank,id=row['id'],encoder_speedup=old_ms/new_ms)),flush=True)
        del raw,projected,c,d,inp,q
    dist.barrier(group=metadata)
    if rank==0:
        import numpy as np
        rr=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)];assert len(rr)==56
        ratio=float(np.median([r['eager_encoder_ms']/r['fused_encoder_ms'] for r in rr]));delta=100*np.mean([r['methods']['REF']['score']-r['methods']['REF']['original_score'] for r in rr])
        save(base/'complete.json',dict(cases=56,median_encoder_speedup=ratio,ref_mean_qa_delta_pp=float(delta),expansion_gate_passed=bool(ratio>=1.5 and delta>=-2.),
            token_changes={m:sum(not r['methods'][m]['token_equal'] for r in rr) for m in rr[0]['methods']},not_bit_exact=True))
    dist.barrier(group=metadata);dist.destroy_process_group()

if __name__=='__main__':main()
