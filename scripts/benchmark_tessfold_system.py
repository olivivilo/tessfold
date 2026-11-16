"""Zero-training measured system study; never infer speed from token ratios."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,time
import numpy as np
import torch
import torch.distributed as dist
from scripts.tessfold_system_common import *
from scripts.train_meta_compress import init_runtime
from scripts.eval_token_proxy import cached_vision,shortened_inputs
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import compress,raw_vision_only,build_prefix,answer_prefix,kv_visual_bytes
from vqa.stage_memory import StageMemory,cache_nbytes

class Timer:
 def __init__(self,device,memory=False):
  self.device=device;self.memory=StageMemory(device) if memory else None;self.records={};self.active=None;self.events={}
 def begin(self,name):
  assert self.active is None;torch.npu.synchronize(self.device)
  if self.memory:self.memory.begin(name)
  else:torch.npu.reset_peak_memory_stats(self.device)
  start=torch.npu.Event(enable_timing=True);start.record();self.events[name]=[start,None];self.active=(name,time.perf_counter())
 def end(self):
  name,ts=self.active;end=torch.npu.Event(enable_timing=True);end.record();torch.npu.synchronize(self.device)
  wall=(time.perf_counter()-ts)*1000
  mem=self.memory.end() if self.memory else dict(peak_allocated_bytes=int(torch.npu.max_memory_allocated(self.device)),peak_reserved_bytes=int(torch.npu.max_memory_reserved(self.device)))
  self.events[name][1]=end;self.records[name]=dict(wall_ms=wall,device_event_ms=float(self.events[name][0].elapsed_time(end)),**mem);self.active=None
 def close(self):
  if self.active:self.end()
  if self.memory:self.memory.close()


def get_question(engine,row):
 ids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(engine.device)
 return engine.model.get_input_embeddings()(ids)


@torch.no_grad()
def projected_features(engine,module,row,inp,method,ratio,raw=None):
 """Return only aligned features and minimal metadata, no lingering P/H refs."""
 if raw is None:
  with raw_vision_only(engine.model.vision_tower):pair=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
  raw=pair[1];del pair
 n=len(raw)
 if method=='REF':projected=engine.model.vision_tower.merger(raw)
 else:
  q=None if module.kind=='col_adaptive' else get_question(engine,row)
  c,info=compress(module,raw,inp['image_grid_thw'],q,ratio,hard=method=='T_HARD')
  projected=engine.model.vision_tower.merger(c);del c,info
 return projected,n


@torch.no_grad()
def clean_generate(engine,ids,projected,fixed=False,cap=256,streamer=None):
 embeds=engine.model.get_input_embeddings()(ids)
 mask=ids==engine.model.config.image_token_id
 embeds=embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds),projected.to(embeds.dtype))
 outputs=engine.model.generate(input_ids=ids,inputs_embeds=embeds,attention_mask=torch.ones_like(ids),
  do_sample=False,use_cache=True,max_new_tokens=64 if fixed else cap,min_new_tokens=64 if fixed else 0,streamer=streamer)
 return outputs[0,ids.shape[1]:].cpu().tolist()


@torch.no_grad()
def single_request(engine,module,row,method,ratio,fixed=False,memory=False,trace_path=None,cap=256):
 from scripts.benchmark_compression_supplement import FirstToken
 timer=Timer(engine.device,memory);torch.npu.synchronize();start=time.perf_counter();first=[None];state={}
 def first_event():torch.npu.synchronize();first[0]=time.perf_counter()
 try:
  timer.begin('processor')
  if method=='R':
   from scripts.repair_supplement_resize import install
   original=install();from PIL import Image
   with Image.open(ROOT/row['image']) as im:
    # N from locked original timing cohort, avoids preprocessing the large image twice.
    n=row['original_tokens'];inp,details=original.resized_inputs(engine,row,n,ratio,im.convert('RGB'))
  else:inp=engine.inputs(row);details={}
  timer.end();tower=engine.model.vision_tower
  timer.begin('encoder')
  if method in ['T_DENSE','C_DENSE','S']:
   x,raw=tower(inp['pixel_values'],inp['image_grid_thw'])
  else:
   with raw_vision_only(tower):pair=tower(inp['pixel_values'],inp['image_grid_thw'])
   raw=pair[1];del pair;x=None
  timer.end();n=len(raw)
  timer.begin('compression_and_merger')
  if method in ['REF','R','S']:projected=x if x is not None else tower.merger(raw);m=len(projected);ids=inp['input_ids']
  elif method in ['T_DENSE','C_DENSE']:
   q=get_question(engine,row);_,p=module(torch.zeros_like(x,dtype=torch.float32),inp['image_grid_thw'],ratio,raw=raw,question=q)
   projected=tower.merger((p@raw.float()).to(raw.dtype));m=len(projected);ids=shortened_inputs(inp,m,engine.model.config.image_token_id)['input_ids']
  else:
   q=None if method=='C_STRICT' else get_question(engine,row)
   c,info=compress(module,raw,inp['image_grid_thw'],q,ratio,hard=method=='T_HARD');projected=tower.merger(c)
   m=len(projected);ids=shortened_inputs(inp,m,engine.model.config.image_token_id)['input_ids'];del c,info
  timer.end()
  if method not in ['T_DENSE','C_DENSE','S']:
   # Clean deployment variant only; no reference held by a dummy tower callback.
   del raw,x,inp
   if 'q' in locals():del q
  calls=[0]
  def pre(mod,args,kwargs):
   if calls[0]==0:timer.begin('qwen_prefill')
  def post(mod,args,kwargs,out):
   calls[0]+=1
   if calls[0]==1:
    state['prefill_kv_bytes']=cache_nbytes(out.past_key_values);state['visual_kv_bytes']=kv_visual_bytes(out.past_key_values,m)
    timer.end();timer.begin('generation')
   state['last_kv_bytes']=cache_nbytes(out.past_key_values)
  h1=engine.model.register_forward_pre_hook(pre,with_kwargs=True);h2=engine.model.register_forward_hook(post,with_kwargs=True)
  streamer=FirstToken(first_event)
  try:
   if method=='S':
    # Original S2Prune hooks layer internals; cannot safely combine an outer
    # per-forward timer with manual decoder-layer execution. Use its stage total.
    h1.remove();h2.remove();timer.begin('decoder_total')
    from vqa.supplement_s2prune import s2_generate
    from PIL import Image
    with Image.open(ROOT/row['image']) as im:pred,rt=s2_generate(engine,ids,projected,inp['image_grid_thw'],im.convert('RGB'),ratio,64 if fixed else cap,fixed,on_first=first_event)
    generated=rt['generated_ids'];m=rt['suffix_visual_tokens'];details.update(rt);timer.end()
   elif method in ['T_DENSE','C_DENSE']:
    si=shortened_inputs(inp,m,engine.model.config.image_token_id)
    with cached_vision(tower,projected,raw):outputs=engine.model.generate(**si,do_sample=False,use_cache=True,max_new_tokens=64 if fixed else cap,min_new_tokens=64 if fixed else 0,streamer=streamer)
    generated=outputs[0,ids.shape[1]:].cpu().tolist();timer.end()
   else:generated=clean_generate(engine,ids,projected,fixed,cap,streamer);timer.end()
  finally:h1.remove();h2.remove()
  torch.npu.synchronize();finish=time.perf_counter();assert first[0] is not None
  if fixed:assert len(generated)==64
  if memory:timer.memory.export(trace_path)
  return dict(id=row['id'],task=row['task'],document_id=row['document_id'],method=method,ratio=ratio,n_original=row['original_tokens'],encoder_tokens=n,visual_tokens=m,
   mode='fixed64' if fixed else 'natural_eos',instrumentation='memory1ms' if memory else 'synchronized_wall_and_NPU_events',
   end_to_end_ms=(finish-start)*1000,ttft_ms=(first[0]-start)*1000,
   post_encoder_ttft_ms=(first[0]-start)*1000-timer.records['processor']['wall_ms']-timer.records['encoder']['wall_ms'],
   stages=timer.records,decoder_state=state,generated_ids=generated,student_prediction=engine.processor.decode(generated,skip_special_tokens=True),
   output_tokens=len(generated),peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in timer.records.values()),detail={k:v for k,v in details.items() if k!='generated_ids'})
 finally:timer.close()


def single_main(engine,rank,world):
 reg=load(REG);out=BASE/'single';out.mkdir(exist_ok=True);(out/'traces').mkdir(exist_ok=True)
 rows=lines(BASE/'timing_rows.jsonl')[rank::world]
 cfgs=[('REF',1.)]+[(m,r) for r in [.25,.5] for m in ['T_DENSE','T_STRICT','C_DENSE','C_STRICT','T_HARD','R','S']]
 path=out/f'results.rank{rank}.jsonl';prev=lines(path);done={(r['id'],r['method'],r['ratio'],r['mode'],r['repeat'],r['instrumentation']) for r in prev}
 with path.open('a') as f:
  for method,ratio in cfgs:
   kind='quadtree_question' if method.startswith('T') else 'col_adaptive'
   module=None if method in ['REF','R','S'] else load_module(engine,kind,ratio)
   for row in rows[:2]:single_request(engine,module,row,method,ratio,cap=16)
   jobs=[(row,fixed,k,False) for k in range(3) for row in rows for fixed in [False,True]]
   # Memory sampling is a separate request pass, excluded from latency summaries.
   jobs += [(row,False,0,True) for row in rows]
   random.Random(20261114+rank).shuffle(jobs)
   for row,fixed,k,memory in jobs:
    key=(row['id'],method,ratio,'fixed64' if fixed else 'natural_eos',k,'memory1ms' if memory else 'synchronized_wall_and_NPU_events')
    if key in done:continue
    trace=out/'traces'/(hashlib.sha256(str(key).encode()).hexdigest()+'.npz')
    result=single_request(engine,module,row,method,ratio,fixed,memory,trace);result['repeat']=k
    if memory:result.update(trace=str(trace.relative_to(ROOT)),trace_sha256=sha(trace))
    f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(key)
   del module;torch.npu.empty_cache();print(json.dumps(dict(rank=rank,finished_method=method,ratio=ratio,records=len(done))),flush=True)
 dist.barrier()
 if rank==0:
  rr=[r for p in out.glob('results.rank*.jsonl') for r in lines(p)];assert len(rr)==90*15*7
  save(out/'complete.json',dict(records=len(rr),latency_records=90*15*6,memory_records=90*15,methods=15,questions=90))
 dist.barrier()


@torch.no_grad()
def multi_main(engine,rank,world):
 reg=load(REG);out=BASE/'multi';out.mkdir(exist_ok=True)
 pages=load(BASE/'multi_pages.json',[])[rank::world];prefix_pass=load(BASE/'validation/complete.json')['prefix_reuse_token_mismatches']==0
 path=out/f'results.rank{rank}.jsonl';prev=lines(path);done={(r['page'],r['method'],r['scenario'],r['repeat']) for r in prev}
 with path.open('a') as f:
  for method in ['REF','Column','Tree']:
   module=None if method=='REF' else load_module(engine,'col_adaptive' if method=='Column' else 'quadtree_question',.25)
   scenarios=['cold','feature_cache']+(['prefix_cache'] if prefix_pass else [])
   for page in pages:
    questions=page['questions'];assert len(questions)==10
    for scenario in scenarios:
     for repeat in range(3):
      key=(page['document_id'],method,scenario,repeat)
      if key in done:continue
      torch.npu.synchronize();start=time.perf_counter();raw=None;vis=None;cache=None;prefix=None;ingest=0.;qrows=[]
      if scenario!='cold':
       inp=engine.inputs(questions[0])
       with raw_vision_only(engine.model.vision_tower):pair=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
       raw=pair[1];del pair
       if method!='Tree':
        vis,n=projected_features(engine,module,questions[0],inp,'REF' if method=='REF' else 'C_STRICT',1. if method=='REF' else .25,raw)
        if scenario=='prefix_cache':
         ids=shortened_inputs(inp,len(vis),engine.model.config.image_token_id)['input_ids'];cache,prefix=build_prefix(engine.model,ids,vis)
       torch.npu.synchronize();ingest=(time.perf_counter()-start)*1000
      for row in questions:
       torch.npu.synchronize();qs=time.perf_counter()
       if scenario=='cold':
        req=single_request(engine,module,row,'REF' if method=='REF' else 'C_STRICT' if method=='Column' else 'T_STRICT',1. if method=='REF' else .25)
        idsout=req['generated_ids']
       else:
        inp=engine.inputs(row)
        if method=='Tree':
         projected,n=projected_features(engine,module,row,inp,'T_STRICT',.25,raw)
         ids=shortened_inputs(inp,len(projected),engine.model.config.image_token_id)['input_ids']
         if scenario=='prefix_cache':
          local_cache,local_prefix=build_prefix(engine.model,ids,projected);idsout=answer_prefix(engine.model,ids,local_cache,local_prefix)
          del local_cache,local_prefix
         else:idsout=clean_generate(engine,ids,projected)
         del projected
        else:
         ids=shortened_inputs(inp,len(vis),engine.model.config.image_token_id)['input_ids']
         if scenario=='prefix_cache':idsout=answer_prefix(engine.model,ids,cache,prefix)
         else:idsout=clean_generate(engine,ids,vis)
       torch.npu.synchronize();qrows.append(dict(id=row['id'],milliseconds=(time.perf_counter()-qs)*1000,generated_ids=idsout,prediction=engine.processor.decode(idsout,skip_special_tokens=True)))
      result=dict(page=page['document_id'],method=method,scenario=scenario,repeat=repeat,ingest_ms=ingest,queries=qrows,
       cache_policy='clone immutable visual prefix for each independent question; cloning included; full model receives same feature/KV reuse privileges',
       amortization={str(k):(ingest+sum(x['milliseconds'] for x in qrows[:k]))/k for k in [1,2,5,10]})
      f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(key)
      del raw,vis,cache,prefix
    print(json.dumps(dict(rank=rank,multi_method=method,page=page['document_id'],records=len(done))),flush=True)
   del module;torch.npu.empty_cache()
 dist.barrier()
 if rank==0:
  rr=[r for p in out.glob('results.rank*.jsonl') for r in lines(p)];expected=24*3*len(scenarios)*3;assert len(rr)==expected
  save(out/'complete.json',dict(page_runs=len(rr),pages=24,questions_per_page=10,scenarios=scenarios,repeats=3,cache_enabled=prefix_pass))
 dist.barrier()


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['single','multi'],required=True);a=p.parse_args();verify()
 rank,world,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
 if a.mode=='single':single_main(engine,rank,world)
 else:multi_main(engine,rank,world)
 dist.destroy_process_group()
if __name__=='__main__':main()
