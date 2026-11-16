"""Independent single-NPU workers; load frozen UND once, train all assigned candidates."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,time,traceback
from pathlib import Path
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
from vqa.compression_generators import ConvexGenerator,FixedGenerator,NAMES
from vqa.official_und import OfficialUND,ROOT
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score
from scripts.eval_token_proxy import generate,shortened_inputs
from scripts.train_meta_compress import init_runtime,atomic_json,atomic_save,sha
from scripts.train_meta_compress_strict import Examples
BASE=ROOT/'artifacts/compression_lab'
OVERLAP={'anchor_resampler','balanced_transport','soft_kmeans','local_global_mixture','multihead_transport','monotone_density'}
SOURCES=['vqa/compression_generators.py','scripts/train_compression_lab.py','scripts/prepare_compression_lab.py','vqa/official_und.py','vqa/decoder_dependent_loss.py','scripts/eval_token_proxy.py','vqa/benchmark_metrics.py']
def read(path):
 if not path.exists():return []
 return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def macro(items,key):
 tasks=sorted({x['task'] for x in items});return sum(sum(x[key] for x in items if x['task']==t)/sum(x['task']==t for x in items) for t in tasks)/len(tasks)
class Data(Examples):
 def __init__(self,engine):
  super().__init__(engine,'npu',capacity=4);self.qcache={}
 def get_all(self,row):
  item=self.get(row)
  if row['id'] not in self.qcache:
   ids=self.engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(self.engine.device)
   assert len(ids)>0
   with torch.no_grad():q=self.engine.model.get_input_embeddings()(ids)
   self.qcache[row['id']]=q.detach()
  return (*item,self.qcache[row['id']])
def student(engine,module,row,data,cfg):
 prompt,raw,grid,x,response,tl,tp,q=data.get_all(row)
 c,p=module(x.float(),grid,cfg['ratio'],raw=raw,question=q)
 sp=c.to(raw.dtype);ids=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)['input_ids']
 sl=common_prefix_logits(engine.model,ids,sp[None],response)
 logt=tl.detach().float().log_softmax(-1);logs=sl.float().log_softmax(-1)
 kl=(logt.exp()*(logt-logs)).sum(-1).mean()
 col=p.sum(0)/(p.shape[0]/p.shape[1]);load=(col-1).square().mean()
 penalty=.01*load if cfg['kind'] in OVERLAP else load*0
 total=kl+penalty
 return {'kl':kl,'load_variance':load,'total':total},(p,sp,ids,raw,grid,tp,response)
def evaluate(engine,module,rows,data,cfg,out,stage,cohort='validation',do_generate=True):
 path=out/f'{cohort}.{stage}.jsonl';items=read(path);done={x['id'] for x in items}
 module.eval()
 with torch.no_grad(),path.open('a') as f:
  for row in rows:
   if row['id'] in done:continue
   loss,(p,sp,ids,raw,grid,tp,response)=student(engine,module,row,data,cfg)
   pred='';rt={}
   if do_generate:pred,rt=generate(engine,{'input_ids':ids,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,sp,raw)
   item={'id':row['id'],'document_id':row['document_id'],'task':row['task'],'question':row['question'],'answers':row['answers'],'stage':stage,'cohort':cohort,
    'student_prediction':pred,'teacher_prediction':tp,'student_score':float(native_score(pred,row)) if do_generate else None,'teacher_score':float(native_score(tp,row)),
    **{k:float(v) for k,v in loss.items()},'tokens':p.shape[1],'compressed_tokens':p.shape[0],'row_sum_error':float((p.sum(-1)-1).abs().max()),'min_weight':float(p.min()),
    'entropy':float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean()),'row_max_mean':float(p.max(-1).values.mean()),'hit_token_limit':rt.get('hit_token_limit'),'teacher_hit_token_limit':len(response)==64,'student_ids':rt.get('generated_ids')}
   assert item['min_weight']>=-1e-6 and item['row_sum_error']<1e-4
   f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush();items.append(item)
 summary={'rows':len(items),'kl':sum(x['kl'] for x in items)/len(items),'total':sum(x['total'] for x in items)/len(items),'teacher_macro':100*macro(items,'teacher_score')}
 if do_generate:summary['student_macro']=100*macro(items,'student_score')
 atomic_json(out/f'{cohort}.{stage}.summary.json',summary);module.train();return summary

def train(engine,data,cfg):
 out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return
 torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
 module=ConvexGenerator(cfg['kind']).to(engine.device);module.train()
 assert sum(p.numel() for p in engine.model.parameters() if p.requires_grad)==0
 trainrows=read(ROOT/cfg['train_rows']);valid=read(ROOT/cfg['validation_rows']);audit=read(BASE/'train_audit.jsonl')
 assert not {r['document_id'] for r in trainrows}&{r['document_id'] for r in valid}
 manifest={'config':cfg,'source_sha256':{f:sha(ROOT/f) for f in SOURCES},'data_sha256':{k:sha(ROOT/cfg[k]) for k in ['train_rows','validation_rows']},'trainable_backbone':0,'trainable_generator':sum(p.numel() for p in module.parameters()),'job':os.environ.get('SLURM_JOB_ID')}
 if (out/'manifest.json').exists():
  old=json.loads((out/'manifest.json').read_text());assert old['source_sha256']==manifest['source_sha256'] and old['config']==cfg,'Source/config changed on resume'
 else:
  atomic_json(out/'manifest.json',manifest)
  for f in SOURCES:
   dest=out/'source_snapshots'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
 optimizer=torch.optim.AdamW(module.parameters(),lr=cfg['lr'],weight_decay=0.)
 per_epoch=math.ceil(len(trainrows)/cfg['batch_size']);total=per_epoch*cfg['epochs'];first=0
 if (out/'latest.pt').exists():
  saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);module.load_state_dict(saved['generator']);optimizer.load_state_dict(saved['optimizer']);first=saved['step']
 else:atomic_save(out/'initial.pt',{'generator':module.state_dict(),'config':cfg,'step':0})
 # Repair any interrupted write and discard updates newer than committed checkpoint.
 training=read(out/'training.jsonl');(out/'training.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in training if x['step']<=first))
 evaluate(engine,module,valid,data,cfg,out,'initial') if first==0 else None
 evaluate(engine,module,audit,data,cfg,out,'initial','train_audit',False) if first==0 else None
 started=time.monotonic()
 for step in range(first,total):
  epoch=step//per_epoch;order=list(range(len(trainrows)));random.Random(cfg['seed']+epoch).shuffle(order)
  indices=order[(step%per_epoch)*cfg['batch_size']:(step%per_epoch+1)*cfg['batch_size']]
  warm=max(1,round(total*.1))
  factor=(step+1)/warm if step<warm else .1+.9*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2
  lr=cfg['lr']*factor
  for group in optimizer.param_groups:group['lr']=lr
  optimizer.zero_grad(set_to_none=True);values=[]
  for ix in indices:
   losses,_=student(engine,module,trainrows[ix],data,cfg);assert torch.isfinite(losses['total'])
   (losses['total']/len(indices)).backward();values.append({k:float(v.detach()) for k,v in losses.items()})
  norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),cfg['clip']));assert math.isfinite(norm)
  old=[p.detach().clone() for p in module.parameters()];optimizer.step()
  update=math.sqrt(sum(float((p.detach()-v).square().sum()) for p,v in zip(module.parameters(),old)))
  assert update>0,'Generator failed to update'
  assert all(p.grad is None for p in engine.model.parameters()),'Backbone received parameter gradients'
  record={'step':step+1,'epoch':epoch+1,'lr':lr,'grad_norm':norm,'clip_multiplier':min(1.,cfg['clip']/(norm+1e-6)),'update_l2':update,'seconds':time.monotonic()-started,**{k:sum(x[k] for x in values)/len(values) for k in values[0]}}
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps({'run':cfg['name'],**record}),flush=True)
  if (step+1)%per_epoch==0:
   state={'generator':{k:v.detach().cpu() for k,v in module.state_dict().items()},'optimizer':optimizer.state_dict(),'config':cfg,'step':step+1}
   atomic_save(out/f'epoch{epoch+1}.pt',state);atomic_save(out/'latest.pt',state)
   evaluate(engine,module,valid,data,cfg,out,f'epoch{epoch+1}')
   evaluate(engine,module,audit,data,cfg,out,f'epoch{epoch+1}','train_audit',False)
 atomic_json(out/'complete.json',{'steps':total,'epochs':cfg['epochs'],'trainable_backbone':0,'job':os.environ.get('SLURM_JOB_ID')})

def smoke(engine,data):
 out=BASE/'smoke';out.mkdir(exist_ok=True);row=read(BASE/'train.jsonl')[0];results=[]
 for kind in NAMES:
  torch.manual_seed(20261105);module=ConvexGenerator(kind).to(engine.device)
  cfg={'kind':kind,'ratio':.1};optimizer=torch.optim.AdamW(module.parameters(),lr=1e-4)
  losses,details=student(engine,module,row,data,cfg);p=details[0]
  losses['total'].backward();norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.));before=[v.detach().clone() for v in module.parameters()];optimizer.step()
  change=math.sqrt(sum(float((v.detach()-old).square().sum()) for v,old in zip(module.parameters(),before)))
  frozen=sum(v.numel() for v in engine.model.parameters() if v.requires_grad)
  assert math.isfinite(norm) and norm>0 and change>0 and frozen==0 and all(v.grad is None for v in engine.model.parameters())
  item={'kind':kind,'kl':float(losses['kl']),'gradient_norm':norm,'update_l2':change,'row_sum_error':float((p.sum(-1)-1).abs().max()),'min_p':float(p.min()),'backbone_trainable':frozen}
  assert item['row_sum_error']<1e-4 and item['min_p']>=-1e-6
  results.append(item);atomic_json(out/'results.json',results);print(json.dumps(item),flush=True)
  del module,optimizer,losses,details,p
 atomic_json(out/'complete.json',{'schemes':len(results),'real_UND_teacher_KL_gradient':True,'source_sha256':{f:sha(ROOT/f) for f in SOURCES},'job':os.environ.get('SLURM_JOB_ID')})

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','screen','cache','evaluate','baseline'],default='screen');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--config-list',default='artifacts/compression_lab/screen_configs.json');p.add_argument('--checkpoint');p.add_argument('--rows',default='artifacts/compression_lab/confirmation.jsonl');p.add_argument('--cohort',default='confirmation');a=p.parse_args()
 # No DDP: each process owns a separate scheme and a single NPU.
 os.environ.pop('RANK',None);os.environ.pop('WORLD_SIZE',None)
 _,_,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);data=Data(engine)
 if a.mode=='smoke':smoke(engine,data);return
 if a.mode=='cache':
  allrows={r['id']:r for name in ['train','validation','confirmation'] for r in read(BASE/f'{name}.jsonl')}
  rows=sorted(allrows.values(),key=lambda r:(r['document_id'],r['id']));docs=sorted({r['document_id'] for r in rows});mine=set(docs[a.worker::a.workers])
  for row in rows:
   if row['document_id'] in mine:data.get_all(row);print('CACHED',row['id'],flush=True)
  return
 if a.mode=='baseline':
  for ratio in [.1,.25,.5]:
   for kind in ['mean','selection']:
    cfg={'kind':kind,'ratio':ratio};out=BASE/'baselines'/f'{kind}_r{ratio:g}';out.mkdir(parents=True,exist_ok=True)
    evaluate(engine,FixedGenerator(kind),read(BASE/'validation.jsonl'),data,cfg,out,'fixed')
  return
 if a.mode=='evaluate':
  checkpoint=ROOT/a.checkpoint;saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];module=ConvexGenerator(cfg['kind']).to(device);module.load_state_dict(saved['generator'])
  evaluate(engine,module,read(ROOT/a.rows),data,cfg,ROOT/cfg['output'],checkpoint.stem,a.cohort);return
 configs=json.loads((ROOT/a.config_list).read_text())
 for path in configs[a.worker::a.workers]:
  cfg=json.loads((ROOT/path).read_text())
  try:train(engine,data,cfg)
  except Exception:
   out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
   raise # infrastructure failure is NOT a negative scientific result; fix before continuing
if __name__=='__main__':main()
