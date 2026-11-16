"""Full23565QA evaluation of fixedepoch4; trained variant only, no selection."""
import argparse,hashlib,json,os,csv
import torch
import torch.distributed as dist
from scripts.seed_memory_common import *
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.eval_token_proxy import generate,shortened_inputs
from scripts.train_compression_lab_v2_repaired import build
from vqa.official_und import OfficialUND
from vqa.benchmark_metrics import native_score

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();reg=verify();meta=next(m for m in reg['models'] if m['id']==a.model)
 rank,world,_,device=init_runtime('npu',4)
 cfg=load(ROOT/meta['config']);out=ROOT/meta['output']/'benchmark';out.mkdir(parents=True,exist_ok=True)
 rows=lines(BASE/'formal_rows.jsonl');assert len(rows)==23565
 sig=dict(model=meta['id'],config=cfg,checkpoint_sha256=sha(ROOT/meta['checkpoint']),world=world,cap=256,benchmark_used_for_selection=False,source_sha256=reg['source_sha256'])
 path=out/f'manifest.rank{rank}.json'
 if path.exists():assert load(path)==sig
 else:save(path,sig)
 engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
 module=build(cfg).to(device).eval();module.load_state_dict(torch.load(ROOT/meta['checkpoint'],map_location='cpu',weights_only=False)['generator'])
 cache=ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache'
 for sub in ['features','teachers','logits']:(cache/sub).mkdir(parents=True,exist_ok=True)
 path=out/f'predictions.rank{rank}.jsonl';rr=lines(path);save_lines(path,rr);done={r['id'] for r in rr}
 with torch.no_grad(),path.open('a') as f:
  for ix,row in enumerate(rows[rank::world]):
   if row['id'] in done:continue
   prompt,raw,grid,x,response=prepare_example(engine,row,cache,256)
   tp=torch.load(cache/'teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True)['prediction']
   qids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device);q=engine.model.get_input_embeddings()(qids)
   _,matrix=module(torch.zeros_like(x,dtype=torch.float32),grid,cfg['ratio'],raw=raw,question=q)
   sp=engine.model.vision_tower.merger((matrix@raw.float()).to(raw.dtype));si=shortened_inputs({'input_ids':prompt},len(sp),engine.model.config.image_token_id)
   pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},256,sp,raw)
   item=dict(id=row['id'],document_id=row['document_id'],task=row['task'],question=row['question'],answers=row['answers'],variant='trained',
    seed=cfg['seed'],ratio=cfg['ratio'],tokens=len(raw),compressed_tokens=len(sp),student_prediction=pred,student_ids=rt['generated_ids'],student_score=float(native_score(pred,row)),
    teacher_prediction=tp,teacher_score=float(native_score(tp,row)),hit_token_limit=rt['hit_token_limit'],row_sum_error=float((matrix.sum(-1)-1).abs().max()),min_weight=float(matrix.min()))
   assert item['row_sum_error']<1e-4 and item['min_weight']>=0
   f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
   if ix%100==0:print(json.dumps(dict(model=meta['id'],rank=rank,questions=ix+1,total=len(rows[rank::world]))),flush=True)
 dist.barrier()
 if rank==0:
  rr=[r for p in out.glob('predictions.rank*.jsonl') for r in lines(p)];assert len(rr)==23565 and {r['id'] for r in rr}=={r['id'] for r in rows}
  result=summarize(rr);result.update(config=cfg,cap=256,teacher_macro=summarize([dict(r,student_score=r['teacher_score']) for r in rr])['macro'],benchmark_used_for_selection=False)
  save(out/'summary.json',result)
  with (out/'answers.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
  save(out/'complete.json',dict(questions=23565,records=23565,checkpoint_sha256=sig['checkpoint_sha256'],job=os.environ.get('SLURM_JOB_ID')))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
