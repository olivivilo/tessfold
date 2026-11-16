"""Actual batch1/2/4/8 throughput, NOT inferred4x from logical visual KV.

Three fixed length strata, eight distinct requests each, nested batch cohorts.
Batching may change floating-point outputs: keep perQA scores and IDs, do not
inherit batch1 accuracy or label this a production concurrency capacity test.
"""
import json,os,random,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,lines,load,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime
from scripts.eval_token_proxy import shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import raw_vision_only
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.tessfold_generation_compat import preserve_prefill_ids
from vqa.benchmark_metrics import native_score

@torch.no_grad()
def request(engine,module,rows,method,fixed):
    torch.npu.synchronize();torch.npu.reset_peak_memory_stats();start=time.perf_counter()
    inputs=[engine.inputs(row) for row in rows]
    pixels=torch.cat([x['pixel_values'] for x in inputs],dim=0);grid=torch.cat([x['image_grid_thw'] for x in inputs],dim=0)
    with raw_vision_only(engine.model.vision_tower):_,raw=engine.model.vision_tower(pixels,grid)
    offset=0;visual=[];ids=[];original=[];actual=[]
    for row,inp in zip(rows,inputs):
        n=int(inp['image_grid_thw'].prod());h=raw[offset:offset+n];offset+=n;original.append(n)
        if method=='REF':p=engine.model.vision_tower.merger(h);si=inp['input_ids']
        else:
            c,d=compress(module,h,inp['image_grid_thw'],None,.25,shape=dimensions(inp['image_grid_thw']))
            p=engine.model.vision_tower.merger(c);si=shortened_inputs(inp,len(c),engine.model.config.image_token_id)['input_ids'];del c,d
        visual.append(p);ids.append(si[0]);actual.append(len(p))
    assert offset==len(raw)
    longest=max(len(x) for x in ids);pad=engine.processor.tokenizer.pad_token_id
    batch=ids[0].new_full((len(rows),longest),pad);attention=torch.zeros_like(batch)
    for i,x in enumerate(ids):batch[i,-len(x):]=x;attention[i,-len(x):]=1
    embeds=engine.model.get_input_embeddings()(batch);mask=batch==engine.model.config.image_token_id
    projected=torch.cat(visual,dim=0);embeds=embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds),projected.to(embeds.dtype))
    del inputs,pixels,grid,raw,visual,projected
    with preserve_prefill_ids(engine.model):
        outputs=engine.model.generate(input_ids=batch,inputs_embeds=embeds,attention_mask=attention,
            use_cache=True,do_sample=False,max_new_tokens=64 if fixed else 256,min_new_tokens=64 if fixed else 0)
    torch.npu.synchronize();elapsed=(time.perf_counter()-start)*1e3;peak=int(torch.npu.max_memory_allocated())
    records=[]
    for i,row in enumerate(rows):
        tokens=outputs[i,longest:].cpu().tolist();eos=engine.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        if not fixed:
            for j,x in enumerate(tokens):
                if x in eos:tokens=tokens[:j+1];break
        prediction=engine.processor.decode(tokens,skip_special_tokens=True)
        records.append(dict(row,prediction=prediction,generated_ids=tokens,score=native_score(prediction,row),original_tokens=original[i],visual_tokens=actual[i]))
    return dict(method=method,batch_size=len(rows),mode='fixed64' if fixed else 'natural_eos',batch_ms=elapsed,
        ms_per_query=elapsed/len(rows),queries_per_second=len(rows)*1000/elapsed,peak_allocated_bytes=peak,queries=records)


@torch.no_grad()
def main():
    verify();rank,world,_,device=init_runtime('npu',4);base=ROOT/'artifacts/tessfold_acceleration_20261115/batched_serving';base.mkdir(exist_ok=True)
    rows=sorted(lines(ROOT/'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl'),key=lambda r:(r['original_tokens'],r['id']))
    cohorts={name:sorted(rows[i*30:(i+1)*30],key=lambda r:r['id'])[:8] for i,name in enumerate(['short','medium','long'])}
    assert all(len({r['id'] for r in rr})==8 for rr in cohorts.values())
    manifest=dict(rank=rank,world=world,source_sha256=sha(__file__),cohorts={k:[r['id'] for r in v] for k,v in cohorts.items()},
        stages='processor+onebatchedfullvision+compression+nativeLLMgreedy',models_frozen=True,batch_sizes=[1,2,4,8],repeats=3,modes=['natural_eos','fixed64'],expected_requests=144,
        distinction='Real batching, finite cohorts, not async serving/max concurrent capacity. Exact generation gate applies only tobatch1; all batch sizes keep task scores.')
    save(base/f'manifest.rank{rank}.json',manifest)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);module=load_module(engine,'col_adaptive',.25).requires_grad_(False)
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={(r['stratum'],r['batch_size'],r['method'],r['mode'],r['repeat']) for r in previous}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    jobs=[(s,b,fixed,k) for s in ['short','medium','long'] for b in [1,2,4,8] for fixed in [False,True] for k in range(3)][rank::world]
    rng=random.Random(20261115+rank);rng.shuffle(jobs)
    with path.open('a') as f:
        for stratum,batch,fixed,repeat in jobs:
            for method in rng.sample(['REF','Column'],2):
                key=(stratum,batch,method,'fixed64' if fixed else 'natural_eos',repeat)
                if key in done:continue
                if repeat==0:request(engine,module,cohorts[stratum][:batch],method,fixed)
                r=request(engine,module,cohorts[stratum][:batch],method,fixed);r.update(stratum=stratum,repeat=repeat)
                f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(key)
                print(json.dumps(dict(rank=rank,records=len(done),method=method,batch=batch,stratum=stratum)),flush=True)
    dist.barrier()
    if rank==0:
        rr=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)];assert len(rr)==144
        save(base/'complete.json',dict(requests=144,batch_sizes=[1,2,4,8],strata=3,repeats=3,models=2))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
