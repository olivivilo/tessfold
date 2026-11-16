"""DDP multi-image pilot and full retraining with isolated, resumable configs.

Main entrypoint: scripts/slurm_meta_compress_search.sh. Never overwrites old run.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vqa.official_und import ROOT,OfficialUND
from vqa.meta_compress import meta_losses
from vqa.meta_compress_tuning import TunableMetaCompress,make_optimizer,learning_rate_factor
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score
from scripts.train_meta_compress import init_runtime,barrier,prepare_example,teacher_logits_for,atomic_json,atomic_save,sha
from scripts.eval_token_proxy import shortened_inputs,generate

CACHE=ROOT/'artifacts/meta_compress_cache'
SOURCE_FILES=['scripts/train_meta_compress_search.py','vqa/meta_compress_tuning.py','vqa/meta_compress.py',
    'vqa/official_und.py','vqa/decoder_dependent_loss.py','scripts/train_meta_compress.py',
    'scripts/eval_token_proxy.py','vqa/benchmark_metrics.py']


def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def load_module(cfg,path=None):
    module=TunableMetaCompress(cfg)
    state=torch.load(path or ROOT/'artifacts/meta_compress/initial.pt',map_location='cpu',weights_only=False)
    module.load_state_dict(state['generator'])
    return module


class Examples:
    def __init__(self,engine,capacity=16):self.engine=engine;self.capacity=capacity;self.cache=OrderedDict()
    def get(self,row):
        if row['id'] in self.cache:
            item=self.cache.pop(row['id']);self.cache[row['id']]=item;return item
        prompt,raw,grid,x,response=prepare_example(self.engine,row,CACHE,64)
        teacher=teacher_logits_for(self.engine,row,prompt,x,response,CACHE)
        item=(prompt,raw,grid,x,response,teacher);self.cache[row['id']]=item
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return item


def student(engine,module,row,examples,cfg):
    prompt,raw,grid,x,response,teacher=examples.get(row)
    compressed,p=module(x.float(),grid,cfg['ratio'])
    si=shortened_inputs({'input_ids':prompt},len(compressed),engine.model.config.image_token_id)
    sp=compressed.to(raw.dtype)
    sl=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
    losses=meta_losses(p,teacher,sl)
    objective=losses['prediction']+cfg['entropy_weight']*losses['entropy']+cfg['collapse_weight']*losses['collapse']
    return objective,losses,(compressed,p,si,sp,raw,grid,response)


def evaluate(engine,module,rows,examples,cfg,out,stage,rank,world,max_new_tokens=64,variant='trained'):
    path=out/f'eval.{stage}.{variant}.rank{rank}.jsonl'
    done={r['id'] for r in read(path)} if path.exists() else set()
    module.eval()
    with torch.no_grad(),path.open('a') as h:
        for row in rows[rank::world]:
            if row['id'] in done:continue
            tick=time.monotonic()
            objective,losses,(c,p,si,sp,raw,grid,response)=student(engine,module,row,examples,cfg)
            prediction,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},max_new_tokens,sp,raw)
            teacher_record=torch.load(CACHE/'teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True)
            teacher_prediction=teacher_record['prediction']
            record={'id':row['id'],'document_id':row['document_id'],'task':row['task'],
                'question':row['question'],'answers':row['answers'],'teacher_prediction':teacher_prediction,
                'student_prediction':prediction,'teacher_score':float(native_score(teacher_prediction,row)),
                'student_score':float(native_score(prediction,row)),
                **{k:float(v) for k,v in losses.items()},'objective':float(objective),
                'tokens':p.shape[1],'compressed_tokens':len(p),'ratio':cfg['ratio'],
                'entropy_fraction':float(losses['entropy'])/math.log(p.shape[1]),
                'column_mass_fraction':float(losses['collapse'])/len(p),
                'row_max_mean':float(p.max(-1).values.mean()),'row_sum_error':float((p.sum(-1)-1).abs().max()),
                'hit_token_limit':rt['hit_token_limit'],'seconds':time.monotonic()-tick,
                'teacher_generated_ids':response,'student_generated_ids':rt['generated_ids'],'stage':stage,'variant':variant}
            h.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');h.flush()
    barrier(world)
    module.train()


def run(a):
    cfg=json.loads((ROOT/a.config).read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    rank,world,local,device=init_runtime(a.device,a.threads)
    if (out/'complete.json').exists():
        if rank==0:print('Complete; nothing to resume',flush=True)
        if world>1:dist.destroy_process_group()
        return
    assert cfg['batch_size']%world==0
    rows=read(ROOT/cfg['train_rows']);valid=read(ROOT/cfg['search_rows'])
    assert not ({r['document_id'] for r in rows}&{r['document_id'] for r in valid})
    signature={'config':cfg,'config_sha256':sha(ROOT/a.config),'train_sha256':sha(ROOT/cfg['train_rows']),
        'validation_sha256':sha(ROOT/cfg['search_rows']), 'world':world,'device':a.device,'threads':a.threads,
        'code_sha256':{f:sha(ROOT/f) for f in SOURCE_FILES}}
    manifest=out/'manifest.json'
    if rank==0:
        if manifest.exists():assert json.loads(manifest.read_text())==signature,'Changed restart protocol'
        atomic_json(manifest,signature)
        for f in SOURCE_FILES:
            dest=out/'source_snapshots'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    barrier(world)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    module=load_module(cfg).to(device);optimizer=make_optimizer(module,cfg)
    checkpoint=out/'latest.pt';first=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        module.load_state_dict(state['generator']);optimizer.load_state_dict(state['optimizer']);first=state['step']
    # A crash can occur between appending a log and committing its checkpoint.
    # Remove only those uncommitted updates before replaying them.
    for logfile in [out/f'examples.rank{rank}.jsonl']+([out/'training.jsonl'] if rank==0 else []):
        if logfile.exists():
            kept=[r for r in read(logfile) if r['step']<=first]
            logfile.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
    examples=Examples(engine,capacity=24)
    if rank==0 and first==0:atomic_save(out/'initial.pt',{'generator':module.state_dict(),'config':cfg})
    if first==0:evaluate(engine,module,valid,examples,cfg,out,'initial',rank,world)
    barrier(world)
    wrapped=DDP(module,device_ids=[local] if a.device=='npu' else None,broadcast_buffers=False) if world>1 else module
    accum=cfg['batch_size']//world;per_epoch=math.ceil(len(rows)/cfg['batch_size']);total=per_epoch*cfg['epochs']
    if first and first%per_epoch==0:
        evaluate(engine,module,valid,examples,cfg,out,f'epoch{first//per_epoch}',rank,world)
    start=time.monotonic()
    for step in range(first,total):
        epoch=step//per_epoch;order=list(range(len(rows)));random.Random(42+epoch).shuffle(order)
        batch=[order[i%len(order)] for i in range((step%per_epoch)*cfg['batch_size'],(step%per_epoch+1)*cfg['batch_size'])]
        lr=cfg['lr']*learning_rate_factor(step,total,cfg)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);values=torch.zeros(8,device=device,dtype=torch.float64)
        import contextlib
        for micro in range(accum):
            row=rows[batch[micro*world+rank]];tick=time.monotonic()
            context=wrapped.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            with context:
                obj,losses,(c,p,si,sp,raw,grid,response)=student(engine,wrapped,row,examples,cfg)
                assert torch.isfinite(obj),row['id']
                (obj/accum).backward()
            entry={'step':step+1,'epoch':epoch+1,'id':row['id'],'task':row['task'],
                **{k:float(v.detach()) for k,v in losses.items()},'objective':float(obj.detach()),
                'tokens':p.shape[1],'compressed_tokens':len(p),
                'column_mass_fraction':float(losses['collapse'].detach())/len(p),
                'seconds':time.monotonic()-tick}
            with (out/f'examples.rank{rank}.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            values+=torch.tensor([entry[k] for k in ['prediction','entropy','collapse','objective','tokens','compressed_tokens','column_mass_fraction','seconds']],device=device,dtype=torch.float64)
            del obj,losses,c,p,sp
        threshold=cfg['clip'] if cfg['clip'] is not None else float('inf')
        norm=torch.nn.utils.clip_grad_norm_(module.parameters(),threshold)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        before={name:param.detach().clone() for name,param in module.named_parameters()} if rank==0 else None
        optimizer.step()
        if world>1:dist.all_reduce(values)
        values/=cfg['batch_size']
        if rank==0:
            updates={name:float((param.detach()-before[name]).norm()/(before[name].norm()+1e-12))
                     for name,param in module.named_parameters()}
            update_l2=math.sqrt(sum(float((param.detach()-before[name]).square().sum()) for name,param in module.named_parameters()))
            entry={'step':step+1,'epoch':epoch+1,'total_steps':total,'lr':lr,'grad_norm':float(norm),
                'clip_multiplier':min(1.,threshold/(float(norm)+1e-6)),
                'update_l2':update_l2,'relative_parameter_updates':updates,
                **dict(zip(['prediction','entropy','collapse','objective','tokens','compressed_tokens','column_mass_fraction','example_seconds'],values.tolist())),
                'elapsed_seconds':time.monotonic()-start}
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            print(json.dumps({'name':cfg['name'],**entry}),flush=True)
            if (step+1)%10==0 or (step+1)%per_epoch==0:
                state={'generator':{k:v.detach().cpu() for k,v in module.state_dict().items()},'optimizer':optimizer.state_dict(),
                       'step':step+1,'epoch':epoch+1,'config':cfg}
                atomic_save(checkpoint,state)
                if (step+1)%per_epoch==0:atomic_save(out/f'epoch{epoch+1}.pt',state)
        if (step+1)%per_epoch==0:
            barrier(world)
            evaluate(engine,module,valid,examples,cfg,out,f'epoch{epoch+1}',rank,world)
    barrier(world)
    if rank==0:atomic_json(out/'complete.json',{'steps':total,'epochs':cfg['epochs'],'rows':len(rows),
        'world':world,'base_trainable_parameters':sum(v.numel() for v in engine.model.parameters() if v.requires_grad),
        'seconds':time.monotonic()-start,'job':os.environ.get('SLURM_JOB_ID')})
    barrier(world)
    if world>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',choices=['cpu','npu'],default='cpu')
    p.add_argument('--threads',type=int,default=8);a=p.parse_args();run(a)


if __name__=='__main__':main()
