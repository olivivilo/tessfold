"""Second-round frozen-UND training/evaluation, independent NPU workers.

Entry: python -m scripts.train_compression_lab_v2 --mode smoke|screen|baseline|cache
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,time,traceback
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
from vqa.compression_generators_v2 import SpatialGenerator,SpatialFixed,NAMES
from vqa.compression_generators import FixedGenerator
from vqa.official_und import OfficialUND,ROOT
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score
from scripts.eval_token_proxy import generate,shortened_inputs
from scripts.train_meta_compress import init_runtime,atomic_json,atomic_save,sha
from scripts.train_compression_lab import Data,macro
BASE=ROOT/'artifacts/compression_lab_v2'
SOURCES=['vqa/compression_generators_v2.py','scripts/train_compression_lab_v2.py',
 'scripts/prepare_compression_lab_v2.py','vqa/compression_generators.py',
 'scripts/train_compression_lab.py','scripts/train_meta_compress_strict.py','scripts/train_meta_compress.py',
 'vqa/official_und.py','vqa/decoder_dependent_loss.py','scripts/eval_token_proxy.py','vqa/benchmark_metrics.py']

def read(path):
 if not path.exists():return []
 rows=path.read_text().splitlines();out=[]
 for i,line in enumerate(rows):
  try:out.append(json.loads(line))
  except json.JSONDecodeError:
   if i!=len(rows)-1:raise
 return out

def build(cfg):
 kind=cfg['kind']
 if kind in ['mean','selection']:return FixedGenerator(kind)
 if kind in ['cell_mean','cell_selection']:return SpatialFixed(kind)
 return SpatialGenerator(kind)

def student(engine,module,row,data,cfg):
 prompt,raw,grid,x,response,tl,tp,q=data.get_all(row)
 if cfg['insertion']=='pre-merger':
  dummy=raw.new_zeros((len(raw),2048),dtype=torch.float32)
  _,p=module(dummy,grid,cfg['ratio'],raw=raw,question=q)
  c=p@raw.float();sp=engine.model.vision_tower.merger(c.to(raw.dtype))
 else:
  c,p=module(x.float(),grid,cfg['ratio'],raw=raw,question=q);sp=c.to(raw.dtype)
 ids=shortened_inputs({'input_ids':prompt},len(p),engine.model.config.image_token_id)['input_ids']
 sl=common_prefix_logits(engine.model,ids,sp[None],response)
 logt=tl.detach().float().log_softmax(-1);logs=sl.float().log_softmax(-1)
 kl=(logt.exp()*(logt-logs)).sum(-1).mean()
 return {'kl':kl,'total':kl},(p,sp,ids,raw,grid,tp,response)

def evaluate(engine,module,rows,data,cfg,out,stage,cohort='validation',do_generate=True):
 path=out/f'{cohort}.{stage}.jsonl';items=read(path);done={r['id'] for r in items}
 if path.exists():path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
 module.eval()
 with torch.no_grad(),path.open('a') as f:
  for row in rows:
   if row['id'] in done:continue
   losses,(p,sp,ids,raw,grid,tp,response)=student(engine,module,row,data,cfg)
   pred='';rt={}
   if do_generate:pred,rt=generate(engine,{'input_ids':ids,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,sp,raw)
   rowmax=p.argmax(-1)
   item={'id':row['id'],'document_id':row['document_id'],'task':row['task'],'question':row['question'],'answers':row['answers'],
    'stage':stage,'cohort':cohort,'student_prediction':pred if do_generate else None,'teacher_prediction':tp,
    'student_score':float(native_score(pred,row)) if do_generate else None,'teacher_score':float(native_score(tp,row)),
    **{k:float(v) for k,v in losses.items()},'tokens':p.shape[1],'compressed_tokens':p.shape[0],
    'row_sum_error':float((p.sum(-1)-1).abs().max()),'min_weight':float(p.min()),
    'entropy':float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean()),'row_max_mean':float(p.max(-1).values.mean()),
    'distinct_row_argmax':int(rowmax.unique().numel()),'hit_token_limit':rt.get('hit_token_limit'),
    'teacher_hit_token_limit':len(response)==64,'student_ids':rt.get('generated_ids')}
   assert item['min_weight']>=-1e-6 and item['row_sum_error']<1e-4
   f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush();items.append(item)
 assert len(items)==len(rows) and len({x['id'] for x in items})==len(rows)
 summary={'rows':len(items),'kl':sum(x['kl'] for x in items)/len(items),'teacher_macro':100*macro(items,'teacher_score')}
 if do_generate:
  summary['student_macro']=100*macro(items,'student_score')
  summary['by_task']={t:100*sum(x['student_score'] for x in items if x['task']==t)/sum(x['task']==t for x in items) for t in sorted({x['task'] for x in items})}
 atomic_json(out/f'{cohort}.{stage}.summary.json',summary);module.train();return summary

def train(engine,data,cfg):
 out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return
 torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);module=build(cfg).to(engine.device)
 assert sum(p.numel() for p in engine.model.parameters() if p.requires_grad)==0
 rows=read(ROOT/cfg['train_rows']);valid=read(ROOT/cfg['validation_rows']);audit=read(ROOT/cfg['audit_rows'])
 assert not {r['document_id'] for r in rows}&{r['document_id'] for r in valid}
 manifest={'config':cfg,'source_sha256':{f:sha(ROOT/f) for f in SOURCES},
 'data_sha256':{k:sha(ROOT/cfg[k]) for k in ['train_rows','validation_rows','audit_rows']},
 'trainable_backbone':0,'trainable_generator':sum(p.numel() for p in module.parameters()),'job':os.environ.get('SLURM_JOB_ID')}
 if (out/'manifest.json').exists():
  old=json.loads((out/'manifest.json').read_text());assert old['source_sha256']==manifest['source_sha256'] and old['config']==cfg,'Source/config changed on resume'
 else:
  atomic_json(out/'manifest.json',manifest)
  for f in SOURCES:
   dest=out/'source_snapshots'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
 optimizer=torch.optim.AdamW(module.parameters(),lr=cfg['lr'],weight_decay=0.)
 per_epoch=math.ceil(len(rows)/cfg['batch_size']);total=per_epoch*cfg['epochs'];first=0
 if (out/'latest.pt').exists():
  saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);module.load_state_dict(saved['generator']);optimizer.load_state_dict(saved['optimizer']);first=saved['step']
 else:atomic_save(out/'initial.pt',{'generator':module.state_dict(),'config':cfg,'step':0})
 logs=read(out/'training.jsonl');(out/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs if r['step']<=first))
 # Complete any evaluation interrupted after the corresponding checkpoint commit.
 if first==0 or first%per_epoch==0:
  epoch=first//per_epoch;stage='initial' if first==0 else f'epoch{epoch}'
  evaluate(engine,module,valid,data,cfg,out,stage,do_generate=epoch in cfg['generation_epochs'])
  evaluate(engine,module,audit,data,cfg,out,stage,'train_audit',False)
 started=time.monotonic()
 for step in range(first,total):
  epoch=step//per_epoch;order=list(range(len(rows)));random.Random(cfg['seed']+epoch).shuffle(order)
  indices=order[(step%per_epoch)*cfg['batch_size']:(step%per_epoch+1)*cfg['batch_size']]
  warm=max(1,round(total*.1));factor=(step+1)/warm if step<warm else .1+.9*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2
  lr=cfg['lr']*factor
  for group in optimizer.param_groups:group['lr']=lr
  optimizer.zero_grad(set_to_none=True);values=[]
  for ix in indices:
   losses,details=student(engine,module,rows[ix],data,cfg);assert torch.isfinite(losses['total'])
   (losses['total']/len(indices)).backward();values.append(float(losses['kl'].detach()));del losses,details
  norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),cfg['clip']));assert math.isfinite(norm)
  before=[p.detach().clone() for p in module.parameters()];optimizer.step()
  change=math.sqrt(sum(float((p.detach()-v).square().sum()) for p,v in zip(module.parameters(),before)));assert change>0
  assert all(p.grad is None for p in engine.model.parameters()),'Backbone parameter received gradient'
  record={'step':step+1,'epoch':epoch+1,'lr':lr,'grad_norm':norm,'clip_multiplier':min(1.,cfg['clip']/(norm+1e-6)),
   'update_l2':change,'seconds':time.monotonic()-started,'kl':sum(values)/len(values),'total':sum(values)/len(values)}
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps({'run':cfg['name'],**record}),flush=True)
  if (step+1)%per_epoch==0:
   state={'generator':{k:v.detach().cpu() for k,v in module.state_dict().items()},'optimizer':optimizer.state_dict(),'config':cfg,'step':step+1}
   atomic_save(out/f'epoch{epoch+1}.pt',state);atomic_save(out/'latest.pt',state)
   evaluate(engine,module,valid,data,cfg,out,f'epoch{epoch+1}',do_generate=epoch+1 in cfg['generation_epochs'])
   evaluate(engine,module,audit,data,cfg,out,f'epoch{epoch+1}','train_audit',False)
 atomic_json(out/'complete.json',{'steps':total,'epochs':cfg['epochs'],'trainable_backbone':0,'job':os.environ.get('SLURM_JOB_ID')})
 (out/'failure.txt').unlink(missing_ok=True)

def smoke(engine,data):
 out=BASE/'smoke';out.mkdir(exist_ok=True);rows=read(BASE/'train.jsonl');results=[]
 for position in ['pre-merger','post-merger']:
  for kind in NAMES:
   torch.manual_seed(20261105);module=SpatialGenerator(kind).to(engine.device)
   # Use a second task on alternate insertion positions; each is a real UND gradient.
   row=rows[0 if position=='pre-merger' else -1];cfg=dict(kind=kind,ratio=.1,insertion=position)
   optimizer=torch.optim.AdamW(module.parameters(),lr=.001,weight_decay=0.)
   losses,details=student(engine,module,row,data,cfg);p=details[0]
   losses['total'].backward();norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.))
   before=[v.detach().clone() for v in module.parameters()];optimizer.step()
   change=math.sqrt(sum(float((v.detach()-old).square().sum()) for v,old in zip(module.parameters(),before)))
   assert math.isfinite(norm) and norm>0 and change>0
   assert not any(v.requires_grad or v.grad is not None for v in engine.model.parameters())
   item=dict(kind=kind,insertion=position,kl=float(losses['kl']),gradient_norm=norm,update_l2=change,
    row_sum_error=float((p.sum(-1)-1).abs().max()),min_p=float(p.min()),trainable_backbone=0)
   assert item['row_sum_error']<1e-4 and item['min_p']>=-1e-6
   results.append(item);atomic_json(out/'results.json',results);print(json.dumps(item),flush=True)
   del module,optimizer,losses,details,p
 atomic_json(out/'complete.json',{'cases':len(results),'architectures':len(NAMES),'source_sha256':{f:sha(ROOT/f) for f in SOURCES},'job':os.environ.get('SLURM_JOB_ID')})

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','screen','baseline','cache','suite'],default='screen')
 p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
 p.add_argument('--config-list',default='artifacts/compression_lab_v2/screen_configs.json');a=p.parse_args()
 os.environ.pop('RANK',None);os.environ.pop('WORLD_SIZE',None)
 _,_,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);data=Data(engine)
 if a.mode=='smoke':smoke(engine,data);return
 if a.mode=='cache':
  rows={r['id']:r for name in ['train','validation'] for r in read(BASE/f'{name}.jsonl')}
  for row in list(rows.values())[a.worker::a.workers]:data.get_all(row);print('CACHED',row['id'],flush=True)
  return
 if a.mode=='baseline':
  cases=[dict(kind=k,ratio=r,insertion=s) for r in [.1,.25,.5] for s in ['pre-merger','post-merger'] for k in ['mean','selection','cell_mean','cell_selection']]
  for cfg in cases[a.worker::a.workers]:
   out=BASE/'baselines'/f'{cfg["insertion"]}_{cfg["kind"]}_r{cfg["ratio"]:g}';out.mkdir(parents=True,exist_ok=True)
   evaluate(engine,build(cfg).to(device),read(BASE/'validation.jsonl'),data,cfg,out,'fixed')
  return
 if a.mode=='suite':
  cases=json.loads((ROOT/a.config_list).read_text())
  for case in cases[a.worker::a.workers]:
   if 'checkpoint' in case:
    saved=torch.load(ROOT/case['checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config'];module=build(cfg).to(device);module.load_state_dict(saved['generator'])
   else:cfg=case['config'];module=build(cfg).to(device)
   out=ROOT/case['output'];out.mkdir(parents=True,exist_ok=True)
   evaluate(engine,module,read(ROOT/case['rows']),data,cfg,out,case['stage'],case.get('cohort','confirmation'))
  return
 for path in json.loads((ROOT/a.config_list).read_text())[a.worker::a.workers]:
  cfg=json.loads((ROOT/path).read_text())
  try:train(engine,data,cfg)
  except Exception:
   out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
if __name__=='__main__':main()
