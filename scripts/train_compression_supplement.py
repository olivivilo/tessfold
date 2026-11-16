"""Second-round four-NPU multi-image expansion/full training; only generator wrapped in DDP."""
import argparse,contextlib,json,math,os,random,time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts import compression_supplement_math as lab
from scripts.train_meta_compress import init_runtime,atomic_json,atomic_save,sha
from vqa.compression_generators_v2 import SpatialGenerator
from vqa.supplement_generators import build
REPAIR_SOURCES=[]
from vqa.official_und import OfficialUND,ROOT
BASE=lab.BASE

def evaluation(engine,module,rows,data,cfg,out,stage,cohort,rank,world):
 localout=out/f'eval_rank{rank}';localout.mkdir(exist_ok=True)
 lab.evaluate(engine,module,rows[rank::world],data,cfg,localout,stage,cohort,cohort=='validation')
 dist.barrier()
 if rank==0:
  items=[]
  for r in range(world):items+=lab.read(out/f'eval_rank{r}'/f'{cohort}.{stage}.jsonl')
  assert len(items)==len(rows) and len({r['id'] for r in items})==len(rows)
  path=out/f'{cohort}.{stage}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
  summary={'rows':len(items),'kl':sum(r['kl'] for r in items)/len(items),'teacher_macro':100*lab.macro(items,'teacher_score')}
  if cohort=='validation':summary['student_macro']=100*lab.macro(items,'student_score')
  atomic_json(out/f'{cohort}.{stage}.summary.json',summary)
 dist.barrier();module.train()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads((ROOT/a.config).read_text());requested_cfg=dict(cfg)
 if cfg.get('optimizer_policy')=='development_selected':
  selection=json.loads((BASE/'development_selection.json').read_text());cfg.update(selection['settings']);cfg['development_selection_sha256']=sha(BASE/'development_selection.json')
 out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
 if int(os.environ.get('RANK',0))==0:atomic_json(out/'effective_config.json',cfg)
 rank,world,local,device=init_runtime('npu',4);assert world>1 and cfg['batch_size']%world==0
 if (out/'complete.json').exists():dist.destroy_process_group();return
 sources=list(lab.SOURCES)
 torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
 engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);data=lab.Data(engine)
 # OfficialUND does not reset the RNG. Explicit seed immediately before module init.
 torch.manual_seed(cfg['seed']);module=build(cfg).to(device)
 rows=lab.read(ROOT/cfg['train_rows']);valid=lab.read(ROOT/cfg['validation_rows']);audit=lab.read(ROOT/cfg['audit_rows'])
 assert not {r['document_id'] for r in rows}&{r['document_id'] for r in valid}
 manifest={'config':cfg,'requested_config':requested_cfg,'world':world,'source_sha256':{f:sha(ROOT/f) for f in sources},'data_sha256':{k:sha(ROOT/cfg[k]) for k in ['train_rows','validation_rows','audit_rows']},'trainable_backbone':0,'trainable_generator':sum(p.numel() for p in module.parameters()),'job':os.environ.get('SLURM_JOB_ID')}
 if rank==0:
  if (out/'manifest.json').exists():
   old=json.loads((out/'manifest.json').read_text());assert old['source_sha256']==manifest['source_sha256'] and old['config']==cfg and old['world']==world
  else:
   atomic_json(out/'manifest.json',manifest)
   for f in sources:
    dest=out/'source_snapshots'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
 optimizer=(torch.optim.SGD(module.parameters(),lr=cfg['lr']) if cfg.get('optimizer')=='sgd' else torch.optim.AdamW(module.parameters(),lr=cfg['lr'],weight_decay=0.))
 first=0
 if (out/'latest.pt').exists():
  saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);module.load_state_dict(saved['generator']);optimizer.load_state_dict(saved['optimizer']);first=saved['step']
 elif rank==0:atomic_save(out/'initial.pt',{'generator':{k:v.detach().cpu() for k,v in module.state_dict().items()},'config':cfg,'step':0})
 if rank==0:
  previous=lab.read(out/'training.jsonl');(out/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in previous if r['step']<=first))
 dist.barrier()
 per_epoch=math.ceil(len(rows)/cfg['batch_size'])
 if first>0 and first%per_epoch==0 and not cfg.get('max_steps'):
  for cohort,rr in [('validation',valid),('train_audit',audit)]:evaluation(engine,module,rr,data,cfg,out,f'epoch{first//per_epoch}',cohort,rank,world)
 if first==0:
  for cohort,rr in [('validation',valid),('train_audit',audit)]:evaluation(engine,module,rr,data,cfg,out,'initial',cohort,rank,world)
 wrapped=DDP(module,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
 per_epoch=math.ceil(len(rows)/cfg['batch_size']);total=cfg.get('max_steps',per_epoch*cfg['epochs']);accum=cfg['batch_size']//world;started=time.monotonic()
 for step in range(first,total):
  epoch=step//per_epoch;order=list(range(len(rows)));random.Random(cfg['seed']+epoch).shuffle(order)
  batch=[order[i%len(order)] for i in range((step%per_epoch)*cfg['batch_size'],(step%per_epoch+1)*cfg['batch_size'])]
  warm=max(1,round(total*.1));factor=(step+1)/warm if step<warm else .1+.9*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2;lr=cfg['lr']*factor
  if cfg.get('schedule')=='constant':lr=cfg['lr']
  for group in optimizer.param_groups:group['lr']=lr
  optimizer.zero_grad(set_to_none=True);values=torch.zeros(4,device=device)
  for micro in range(accum):
   row=rows[batch[micro*world+rank]];ctx=wrapped.no_sync() if micro<accum-1 else contextlib.nullcontext()
   with ctx:
    losses,_=lab.student(engine,wrapped,row,data,cfg);assert torch.isfinite(losses['total'])
    (losses['total']/accum).backward()
   values+=torch.tensor([float(losses[k].detach()) for k in ['kl','total','entropy','collapse']],device=device)
   del losses
  norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),cfg['clip']));assert math.isfinite(norm)
  before=[p.detach().clone() for p in module.parameters()] if rank==0 else []
  optimizer.step();assert all(p.grad is None for p in engine.model.parameters())
  dist.all_reduce(values);values/=cfg['batch_size']
  if rank==0:
   change=math.sqrt(sum(float((p.detach()-old).square().sum()) for p,old in zip(module.parameters(),before)));assert change>0
   record={'step':step+1,'epoch':epoch+1,'lr':lr,'grad_norm':norm,'clip_multiplier':min(1.,cfg['clip']/(norm+1e-6)),'update_l2':change,'seconds':time.monotonic()-started,**dict(zip(['kl','total','entropy','collapse'],values.tolist()))}
   with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print(json.dumps({'run':cfg['name'],**record}),flush=True)
  # Commit recovery state every 25 global updates, without changing optimizer,
  # scheduler, data order or fixed-epoch evaluation/checkpoint selection.
  if (step+1)%25==0 or (step+1)%per_epoch==0:
   if rank==0:
    state={'generator':{k:v.detach().cpu() for k,v in module.state_dict().items()},'optimizer':optimizer.state_dict(),'config':cfg,'step':step+1}
    atomic_save(out/'latest.pt',state)
    if (step+1)%per_epoch==0:atomic_save(out/f'epoch{epoch+1}.pt',state)
   dist.barrier()
  if ((step+1)%per_epoch==0 and not cfg.get('max_steps')) or step+1==total:
   if rank==0 and (step+1)%per_epoch:
    atomic_save(out/'endpoint.pt',{'generator':{k:v.detach().cpu() for k,v in module.state_dict().items()},'config':cfg,'step':step+1})
   for cohort,rr in [('validation',valid),('train_audit',audit)]:evaluation(engine,module,rr,data,cfg,out,f'epoch{epoch+1}' if (step+1)%per_epoch==0 else 'endpoint',cohort,rank,world)
 dist.barrier()
 if rank==0:atomic_json(out/'complete.json',{'steps':total,'epochs':cfg['epochs'],'world':world,'trainable_backbone':0,'job':os.environ.get('SLURM_JOB_ID'),'padded_samples_per_epoch':per_epoch*cfg['batch_size']-len(rows)})
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
