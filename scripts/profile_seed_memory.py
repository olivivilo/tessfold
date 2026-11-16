"""Stage-resolved allocator profiling of all24 frozen-backbone configurations.

Uses native Hugging Face generate unchanged. Hooks split first decoder forward
(prefill) from subsequent generation; separate native merger from visual trunk.
Every synchronization/measurement is confined to this versioned diagnostic.
"""
from __future__ import annotations
import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from scripts.seed_memory_common import *
from scripts.train_meta_compress import init_runtime
from scripts.eval_token_proxy import cached_vision,shortened_inputs
from scripts.train_compression_lab_v2_repaired import build
from vqa.official_und import OfficialUND
from vqa.stage_memory import StageMemory,cache_nbytes


def load_model(engine,meta):
 if meta['kind']=='REF':return None
 cfg=load(ROOT/meta['config']);module=build(dict(kind=meta['kind'])).to(engine.device).eval()
 module.load_state_dict(torch.load(ROOT/meta['checkpoint'],map_location='cpu',weights_only=False)['generator']);return module


@torch.no_grad()
def reference_generation(engine,module,row,meta,cap=16):
 """Uninstrumented route for validating identical tokens, no saved features."""
 inp=engine.inputs(row);x,raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw']);grid=inp['image_grid_thw']
 if module is None:sp=x;si=inp
 else:
  qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(engine.device);q=engine.model.get_input_embeddings()(qids)
  _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,meta['ratio'],raw=raw,question=q)
  sp=engine.model.vision_tower.merger((p@raw.float()).to(raw.dtype));si=shortened_inputs(inp,len(sp),engine.model.config.image_token_id)
 with cached_vision(engine.model.vision_tower,sp,raw):
  outputs=engine.model.generate(**si,do_sample=False,use_cache=True,max_new_tokens=cap)
 return outputs[0,si['input_ids'].shape[1]:].cpu().tolist()


@torch.no_grad()
def profile_request(engine,module,row,meta,meter,trace_path,fixed=False,cap=None):
 meter.reset();api=torch.npu;api.synchronize(engine.device)
 resident=dict(allocated_bytes=int(api.memory_allocated(engine.device)),reserved_bytes=int(api.memory_reserved(engine.device)))
 meter.begin('processor');inp=engine.inputs(row);meter.end()
 tower=engine.model.vision_tower;merger=tower.merger;merger_calls=[0]
 def merger_pre(mod,args):
  merger_calls[0]+=1;assert merger_calls[0]==1
  meter.end();meter.begin('native_merger')
 def merger_post(mod,args,out):meter.end()
 h1=merger.register_forward_pre_hook(merger_pre);h2=merger.register_forward_hook(merger_post)
 try:
  meter.begin('encoder');x,raw=tower(inp['pixel_values'],inp['image_grid_thw'])
  assert meter.active is None and merger_calls[0]==1
 finally:h1.remove();h2.remove()
 n=len(raw);grid=inp['image_grid_thw'];m=max(1,round(n*meta['ratio']))
 meter.begin('compression');p=None
 if module is None:sp=x;si=inp
 else:
  qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(engine.device);q=engine.model.get_input_embeddings()(qids)
  _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,meta['ratio'],raw=raw,question=q)
  sp=merger((p@raw.float()).to(raw.dtype));si=shortened_inputs(inp,len(sp),engine.model.config.image_token_id)
  assert len(sp)==m and p.shape==(m,n)
  assert float(p.min())>=0 and float((p.sum(-1)-1).abs().max())<1e-4
 meter.end()
 calls=[0];state={}
 def pre(mod,args,kwargs):
  if calls[0]==0:
   assert meter.active is None;meter.begin('qwen_prefill')
   ids=kwargs.get('input_ids',args[0] if args else None)
   state['prefill_sequence_tokens']=int(ids.shape[-1]);state['prefill_visual_tokens']=int((ids==engine.model.config.image_token_id).sum())
  else:assert meter.active=='generation'
 def post(mod,args,kwargs,output):
  calls[0]+=1
  if calls[0]==1:
   state['prefill_kv_storage_bytes']=cache_nbytes(output.past_key_values)
   state['prefill_logits_shape']=list(output.logits.shape)
   meter.end();meter.begin('generation')
  state['last_forward_kv_storage_bytes']=cache_nbytes(output.past_key_values)
 before=engine.model.register_forward_pre_hook(pre,with_kwargs=True)
 after=engine.model.register_forward_hook(post,with_kwargs=True)
 try:
  with cached_vision(tower,sp,raw):
   outputs=engine.model.generate(**si,do_sample=False,use_cache=True,max_new_tokens=cap if cap is not None else (64 if fixed else 256),min_new_tokens=64 if fixed else 0)
  assert calls[0]>=1 and meter.active=='generation';meter.end()
 finally:
  before.remove();after.remove()
  if meter.active is not None:meter.end()
 ids=outputs[0,si['input_ids'].shape[1]:].cpu().tolist()
 assert state['prefill_visual_tokens']==m
 if fixed:assert len(ids)==64
 assert set(meter.phases)=={'processor','encoder','native_merger','compression','qwen_prefill','generation'}
 phases=meter.export(trace_path)
 for name,v in phases.items():
  assert 0<=v['time_average_allocated_bytes']<=v['peak_allocated_bytes']+1
  assert v['peak_allocated_bytes']>=max(v['start_allocated_bytes'],v['end_allocated_bytes'])
 return dict(id=row['id'],document_id=row['document_id'],task=row['task'],model=meta['id'],kind=meta['kind'],ratio=meta['ratio'],seed=meta['seed'],
  mode='fixed64' if fixed else 'natural_eos',tokens=n,compressed_tokens=m,generated_ids=ids,generated_tokens=len(ids),
  prediction=engine.processor.decode(ids,skip_special_tokens=True),resident=resident,phases=phases,
  whole_request_peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in phases.values()),
  forward_calls=calls[0],decoder_state=state,trace=str(Path(trace_path).relative_to(ROOT)),
  trace_sha256=sha(trace_path),cache_scope='no inter-request visual features or KV; weights resident',
  lifecycle='matches previous full-original-feature pipeline; keeps input/raw/x/P through generation, no cleanup optimization',
  average_definition='actual1ms target sampling, time-weighted trapezoid; includes CPU gaps, synchronized phase boundaries; not a kernel-time or total-device average')


def main():
 reg=verify();rank,world,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
 out=BASE/'memory';out.mkdir(exist_ok=True);(out/'traces').mkdir(exist_ok=True)
 meter=StageMemory(device,reg['protocol']['sampling_interval_seconds'])
 models=reg['models']+[dict(id='REF',kind='REF',ratio=1.,seed=None)]
 rows=lines(BASE/'timing_rows.jsonl')[rank::world];assert rows
 path=out/f'memory.rank{rank}.jsonl';rr=lines(path);save_lines(path,rr)
 done={(r['model'],r['id'],r['mode'],r['repeat']) for r in rr}
 signature=dict(registration_sha256=sha(REG),world=world,rank=rank,device=torch.npu.get_device_name(),
  source_sha256=reg['source_sha256'],checkpoints={m['id']:sha(ROOT/m['checkpoint']) for m in reg['models']},
  warmup_requests_per_model=10,memory_sampling_seconds=meter.interval)
 manifest=out/f'manifest.rank{rank}.json'
 if manifest.exists():assert load(manifest)==signature
 else:save(manifest,signature)
 try:
  with path.open('a') as f:
   # Per-rank model order is fixed but varied; exactly one generator resident.
   order=list(models);random.Random(20261113+rank).shuffle(order)
   for meta in order:
    expected={(meta['id'],r['id'],mode,k) for r in rows for mode in ['natural_eos','fixed64'] for k in range(3)}
    if expected<=done:continue
    module=load_model(engine,meta)
    for ix in range(10):reference_generation(engine,module,rows[ix%len(rows)],meta,cap=16)
    torch.npu.synchronize();torch.npu.empty_cache()
    jobs=[(r,fixed,repeat) for repeat in range(3) for r in rows for fixed in [False,True]]
    random.Random(20261113+rank).shuffle(jobs)
    for row,fixed,repeat in jobs:
     mode='fixed64' if fixed else 'natural_eos';key=(meta['id'],row['id'],mode,repeat)
     if key in done:continue
     name=hashlib.sha256(json.dumps(key).encode()).hexdigest()+'.npz'
     result=profile_request(engine,module,row,meta,meter,out/'traces'/name,fixed)
     result['repeat']=repeat
     f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(key)
    del module;torch.npu.empty_cache();print(json.dumps(dict(rank=rank,profiled_model=meta['id'],records=len(done))),flush=True)
 finally:meter.close()
 dist.barrier()
 if rank==0:
  rr=[r for p in out.glob('memory.rank*.jsonl') for r in lines(p)]
  expected={(m['id'],r['id'],mode,k) for m in models for r in lines(BASE/'timing_rows.jsonl') for mode in ['natural_eos','fixed64'] for k in range(3)}
  assert len(rr)==len(expected)==13500 and {(r['model'],r['id'],r['mode'],r['repeat']) for r in rr}==expected
  save(out/'complete.json',dict(records=len(rr),models=25,questions=90,repeats=3,modes=['natural_eos','fixed64']))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
