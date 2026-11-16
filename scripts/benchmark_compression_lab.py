"""Locked full benchmark; neither its scores nor checkpoints feed back into selection."""
import argparse,csv,hashlib,json,os
from collections import defaultdict
import torch
import torch.distributed as dist
from scripts.train_meta_compress import init_runtime,prepare_example,atomic_json,sha
from scripts.train_compression_lab import BASE,read,macro
from scripts.eval_token_proxy import generate,shortened_inputs
from vqa.official_und import OfficialUND,ROOT
from vqa.compression_generators import ConvexGenerator,FixedGenerator
from vqa.benchmark_metrics import native_score
from scripts.report_meta_compress_strict import paired

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--rows');a=p.parse_args();cfg=json.loads((ROOT/a.config).read_text());run=ROOT/cfg['output'];out=run/('benchmark_smoke' if a.smoke else 'benchmark');out.mkdir(exist_ok=True)
 rank,world,local,device=init_runtime('npu',4);rows=[r for r in read(ROOT/'data/full/val.jsonl') if r['task'] in ['DocVQA','InfoVQA']]+read(ROOT/'data/full/test.jsonl')
 if a.rows:rows=read(ROOT/a.rows)
 if a.smoke:
  seen=set();rr=[]
  for r in rows:
   if r['task'] not in seen:rr.append(r);seen.add(r['task'])
  rows=rr
 else:assert not a.rows and len(rows)==23565
 assert len({r['id'] for r in rows})==len(rows)
 assert not {r['document_id'] for r in rows}&{r['document_id'] for r in read(ROOT/cfg['train_rows'])}
 signature={'config':cfg,'checkpoint_sha256':sha(run/f'epoch{cfg["epochs"]}.pt'),'initial_sha256':sha(run/'initial.pt'),'world':world,'cap':256,'smoke':a.smoke,'rows_override':a.rows,'backbone_frozen':True,'benchmark_used_for_selection':False,'source_sha256':{s:sha(ROOT/s) for s in ['scripts/benchmark_compression_lab.py','vqa/compression_generators.py','vqa/official_und.py','scripts/eval_token_proxy.py','vqa/benchmark_metrics.py']},'teacher_cache':'same-device official frozen UND native BF16 cache from strict benchmark, cap256; identity audit matches native generation'}
 if (out/f'manifest.rank{rank}.json').exists():assert json.loads((out/f'manifest.rank{rank}.json').read_text())==signature
 atomic_json(out/f'manifest.rank{rank}.json',signature)
 engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);modules={}
 for kind,checkpoint in [('initial','initial.pt'),('trained',f'epoch{cfg["epochs"]}.pt')]:
  state=torch.load(run/checkpoint,map_location='cpu',weights_only=False);module=ConvexGenerator(cfg['kind']).to(device).eval();module.load_state_dict(state['generator']);modules[kind]=module
 modules.update(mean=FixedGenerator('mean'),selection=FixedGenerator('selection'))
 cache=ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache'
 for key in ['features','teachers','logits']:(cache/key).mkdir(parents=True,exist_ok=True)
 path=out/f'predictions.rank{rank}.jsonl';items=read(path);done={(r['id'],r['variant']) for r in items}
 with torch.no_grad(),path.open('a') as f:
  for index,row in enumerate(rows[rank::world]):
   if all((row['id'],v) in done for v in modules):continue
   prompt,raw,grid,x,response=prepare_example(engine,row,cache,256)
   record=torch.load(cache/'teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True);tp=record['prediction']
   ids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(device);q=engine.model.get_input_embeddings()(ids)
   for variant,module in modules.items():
    if (row['id'],variant) in done:continue
    if cfg.get('insertion')=='pre-merger':
     _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,cfg['ratio'],raw=raw,question=q);sp=engine.model.vision_tower.merger((p@raw.float()).to(raw.dtype))
    else:c,p=module(x.float(),grid,cfg['ratio'],raw=raw,question=q);sp=c.to(raw.dtype)
    si=shortened_inputs({'input_ids':prompt},len(sp),engine.model.config.image_token_id);pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},256,sp,raw)
    item={'id':row['id'],'document_id':row['document_id'],'task':row['task'],'question':row['question'],'answers':row['answers'],'variant':variant,'teacher_prediction':tp,'student_prediction':pred,'teacher_score':float(native_score(tp,row)),'student_score':float(native_score(pred,row)),'tokens':len(raw),'compressed_tokens':len(sp),'ratio':cfg['ratio'],'hit_token_limit':rt['hit_token_limit'],'teacher_hit_token_limit':len(response)==256,'student_ids':rt['generated_ids'],'row_sum_error':float((p.sum(-1)-1).abs().max())}
    f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
   if index%100==0:print(json.dumps({'rank':rank,'questions':index+1,'total':len(rows[rank::world])}),flush=True)
 dist.barrier()
 if rank==0:
  allrows=[r for part in out.glob('predictions.rank*.jsonl') for r in read(part)];assert len(allrows)==len(rows)*len(modules) and len({(r['id'],r['variant']) for r in allrows})==len(allrows)
  groups=defaultdict(list)
  for r in allrows:groups[(r['variant'],r['task'])].append(r)
  tasks=sorted({r['task'] for r in rows});stats={t:{v:100*sum(r['student_score'] for r in groups[(v,t)])/len(groups[(v,t)]) for v in modules} for t in tasks}
  for t in tasks:stats[t]['teacher']=100*sum(r['teacher_score'] for r in groups[('trained',t)])/len(groups[('trained',t)])
  result={'questions':len(rows),'tasks':stats,'macro':{v:sum(stats[t][v] for t in tasks)/len(tasks) for v in [*modules,'teacher']},'paired_trained_vs_initial':paired([r for r in allrows if r['variant']=='initial'],[r for r in allrows if r['variant']=='trained']),'config':cfg,'benchmark_used_for_selection':False,'cap':256,'metric_limitation':'DeepForm/KLC field exact, not document-level F1'}
  atomic_json(out/'summary.json',result)
  with (out/'answers.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
  atomic_json(out/'complete.json',{'questions':len(rows),'records':len(allrows),'job':os.environ.get('SLURM_JOB_ID')})
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
