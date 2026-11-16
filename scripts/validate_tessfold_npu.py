"""Independent dense reference, actual checkpoints, and prefix caching gates."""
import os,time,math
import numpy as np
import torch
import torch.distributed as dist
from scripts.tessfold_system_common import *
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.eval_token_proxy import generate,shortened_inputs,cached_vision
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import compress,score_energy,budget,partition_integral_candidate,build_prefix,answer_prefix
from vqa.compression_generators_v2 import adaptive_rectangles


def main():
 reg=verify();rank,world,_,device=init_runtime('npu',4);assert world==8
 engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
 meta=reg['models'][rank];module=load_module(engine,meta['kind'],meta['ratio']);out=BASE/'validation';out.mkdir(exist_ok=True)
 path=out/f'rows.rank{rank}.jsonl';previous=lines(path);done={r['id'] for r in previous}
 old={r['id']:r for r in lines(ROOT/meta['output']/'validation.epoch4.jsonl')}
 rows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl');cache=ROOT/'artifacts/meta_compress_strict/npu_cache'
 entries=list(previous)
 with torch.no_grad(),path.open('a') as f:
  for row in rows:
   if row['id'] in done:continue
   prompt,raw,grid,x,response=prepare_example(engine,row,cache,64)
   ids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device);q=engine.model.get_input_embeddings()(ids)
   _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,meta['ratio'],raw=raw,question=q)
   ref=(p@raw.float()).to(raw.dtype);fast,detail=compress(module,raw,grid,q,meta['ratio'])
   strict=torch.equal(ref,fast) and torch.equal(p,detail['weights']);assert strict
   refproj=engine.model.vision_tower.merger(ref);proj=engine.model.vision_tower.merger(fast);assert torch.equal(refproj,proj)
   si=shortened_inputs({'input_ids':prompt},len(proj),engine.model.config.image_token_id)
   pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,proj,raw)
   assert rt['generated_ids']==old[row['id']]['student_ids'],'Strict path must preserve saved generation tokens'
   from scripts.benchmark_tessfold_system import single_request
   timingrow=dict(row,original_tokens=len(raw))
   clean=single_request(engine,module,timingrow,'C_STRICT' if meta['kind']=='col_adaptive' else 'T_STRICT',meta['ratio'],cap=64)
   assert clean['generated_ids']==rt['generated_ids'],'Clean lifetime/inputs_embeds path changed tokens'
   # Measure candidate precision but never substitute it silently for strict.
   variants={}
   for backend in ['segmented_fp32','segmented_bf16']:
    try:
     c,d=compress(module,raw,grid,q,meta['ratio'],backend)
     variants[backend]=dict(equal=torch.equal(c,ref),max_abs_error=float((c.float()-ref.float()).abs().max()),representatives_equal=torch.equal(d['representatives'],detail['representatives']))
    except (RuntimeError,NotImplementedError) as e:variants[backend]=dict(equal=False,error=str(e)[:500])
   h,w,block,inverse=__import__('vqa.compression_generators',fromlist=['coordinates']).coordinates(grid,device)
   energy=detail['energy'][inverse].cpu().numpy().reshape(h,w);mode='col' if meta['kind']=='col_adaptive' else 'tree';m=len(proj)
   g0=adaptive_rectangles(torch.from_numpy(energy),h,w,m,mode);g1=partition_integral_candidate(energy,m,mode)
   item=dict(id=row['id'],model=meta['id'],kind=meta['kind'],ratio=meta['ratio'],strict_feature_bit_exact=strict,
    projection_bit_exact=True,generation_tokens_equal=True,student_prediction=pred,student_ids=rt['generated_ids'],
    candidates=variants,integral_partition_equal=bool(np.array_equal(g0,g1)),n=len(raw),m=m)
   f.write(__import__('json').dumps(item,ensure_ascii=False)+'\n');f.flush();entries.append(item)
   del p,ref,fast,detail,refproj,proj
 # Prefix cache gate: compare native full prompt and independent same-prefix KV.
 swap=lines(ROOT/'artifacts/compression_supplement_20261111/same_page_rows.jsonl');pages={}
 for r in swap:pages.setdefault(r['document_id'],[]).append(r)
 page=sorted(pages)[rank];questions=sorted(pages[page],key=lambda r:r['id'])[:2]
 prefix_tests=[]
 for method in ['REF','Column','Tree']:
  mod=None if method=='REF' else load_module(engine,'col_adaptive' if method=='Column' else 'quadtree_question',.25)
  shared=None;prefix=None;initial_visual=None
  for row in questions:
   inp=engine.inputs(row);x,raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
   if mod is None:sp=x;si=inp
   else:
    qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device);q=engine.model.get_input_embeddings()(qids)
    c,_=compress(mod,raw,inp['image_grid_thw'],q,.25);sp=engine.model.vision_tower.merger(c);si=shortened_inputs(inp,len(sp),engine.model.config.image_token_id)
   with cached_vision(engine.model.vision_tower,sp,raw):original=engine.model.generate(**si,do_sample=False,use_cache=True,max_new_tokens=32)[0,si['input_ids'].shape[1]:].cpu().tolist()
   if shared is None or method=='Tree':shared,prefix=build_prefix(engine.model,si['input_ids'],sp)
   length_before=int(shared.get_seq_length());answer=answer_prefix(engine.model,si['input_ids'],shared,prefix,cap=32)
   assert int(shared.get_seq_length())==length_before,'Base cache must not include another question'
   prefix_tests.append(dict(method=method,id=row['id'],equal=answer==original,original_ids=original,cached_ids=answer,cache_unchanged=True,prefix_tokens=length_before))
  del mod,shared
 save(out/f'summary.rank{rank}.json',dict(model=meta['id'],rows=len(entries),strict_all_exact=all(r['strict_feature_bit_exact'] and r['generation_tokens_equal'] for r in entries),prefix_tests=prefix_tests))
 dist.barrier()
 if rank==0:
  summaries=[load(out/f'summary.rank{k}.json') for k in range(world)];assert all(r['rows']==56 and r['strict_all_exact'] for r in summaries)
  allrows=[r for k in range(world) for r in lines(out/f'rows.rank{k}.jsonl')]
  save(out/'complete.json',dict(cases=448,strict_passed=True,sparse_promoted=False,
    reason='Sparse reductions are candidates only; historical exact path is retained regardless of tolerance.',
    integral_partition_mismatches=sum(not r['integral_partition_equal'] for r in allrows),
    candidate_mismatches={b:sum(not r['candidates'][b]['equal'] for r in allrows) for b in ['segmented_fp32','segmented_bf16']},
    prefix_reuse_token_mismatches=sum(not x['equal'] for s in summaries for x in s['prefix_tests']),summaries=summaries))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
