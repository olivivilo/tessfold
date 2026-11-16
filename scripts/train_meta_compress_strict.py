"""Resumable full-paper-loss training; frozen UND; pre/post merger placement."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from collections import OrderedDict
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vqa.official_und import ROOT, OfficialUND
from vqa.meta_compress import MetaCompress, meta_losses
from vqa.meta_compress_initialization import initialize
from vqa.meta_compress_tuning import make_optimizer, learning_rate_factor
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score
from scripts.train_meta_compress import init_runtime, barrier, prepare_example, teacher_logits_for, atomic_json, atomic_save, sha
from scripts.eval_token_proxy import shortened_inputs, generate

BASE=ROOT/'artifacts/meta_compress_strict'
SOURCES=['scripts/train_meta_compress_strict.py','vqa/meta_compress_initialization.py',
         'vqa/meta_compress.py','vqa/meta_compress_tuning.py','vqa/official_und.py',
         'vqa/decoder_dependent_loss.py','scripts/train_meta_compress.py',
         'scripts/eval_token_proxy.py','vqa/benchmark_metrics.py']

def read(path):
    if not path.exists():return []
    # Atomic newline commits; a killed writer may leave only its last line incomplete.
    result=[];lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i<len(lines)-1:raise
    return result

class Examples:
    def __init__(self,engine,device,capacity=4):
        self.engine=engine;self.cache=OrderedDict();self.capacity=capacity
        self.directory=BASE/f'{device}_cache'
        for name in ['features','teachers','logits']:(self.directory/name).mkdir(parents=True,exist_ok=True)
    def get(self,row):
        if row['id'] in self.cache:
            item=self.cache.pop(row['id']);self.cache[row['id']]=item;return item
        prompt,raw,grid,x,response=prepare_example(self.engine,row,self.directory,64)
        teacher=teacher_logits_for(self.engine,row,prompt,x,response,self.directory)
        record=torch.load(self.directory/'teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),
                          map_location='cpu',weights_only=True)
        item=(prompt,raw,grid,x,response,teacher,record['prediction']);self.cache[row['id']]=item
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return item

def student(engine,module,row,examples,cfg):
    prompt,raw,grid,x,response,teacher,tp=examples.get(row)
    source=x if cfg['insertion']=='post-merger' else raw
    c,p=module(source.float(),grid,cfg['ratio'])
    sp=c.to(raw.dtype)
    if cfg['insertion']=='pre-merger':sp=engine.model.vision_tower.merger(sp)
    si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
    sl=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
    losses=meta_losses(p,teacher,sl)
    return losses,(c,p,sp,si,raw,grid,response,tp)

def evaluate(engine,module,rows,examples,cfg,out,stage,cohort,rank,world):
    path=out/f'eval.{cohort}.{stage}.rank{rank}.jsonl'
    committed=read(path)
    # Drop an interrupted last line before appending resumed records.
    if path.exists():path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in committed))
    done={r['id'] for r in committed};module.eval()
    with torch.no_grad(),path.open('a') as f:
        for row in rows[rank::world]:
            if row['id'] in done:continue
            losses,(c,p,sp,si,raw,grid,response,tp)=student(engine,module,row,examples,cfg)
            pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,sp,raw)
            item={'id':row['id'],'document_id':row['document_id'],'task':row['task'],
                  'question':row['question'],'answers':row['answers'],'stage':stage,'cohort':cohort,
                  'student_prediction':pred,'teacher_prediction':tp,
                  'student_score':float(native_score(pred,row)),'teacher_score':float(native_score(tp,row)),
                  **{k:float(v) for k,v in losses.items()},'tokens':p.shape[1],'compressed_tokens':p.shape[0],
                  'row_sum_error':float((p.sum(-1)-1).abs().max()),
                  'row_max_mean':float(p.max(-1).values.mean()),'column_mass_fraction':float(losses['collapse'])/len(p),
                  'hit_token_limit':rt['hit_token_limit'],'teacher_hit_token_limit':len(response)==64,
                  'student_generated_ids':rt['generated_ids'],'teacher_generated_ids':response}
            f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    barrier(world);module.train()

def run(a):
    cfg=json.loads((ROOT/a.config).read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    rank,world,local,device=init_runtime(a.device,a.threads)
    if (out/'complete.json').exists() and not a.evaluate:
        if rank==0:print('Already complete',flush=True)
        if world>1:dist.destroy_process_group()
        return
    rows=read(ROOT/cfg['train_rows']);valid=read(ROOT/cfg['validation_rows']);audit=read(ROOT/cfg['audit_rows'])
    assert not ({r['document_id'] for r in rows}&{r['document_id'] for r in valid})
    assert cfg['batch_size']%world==0
    assert not any(k in cfg for k in ['local_bias','attention_scale','entropy_weight','collapse_weight'])
    signature={'config':cfg,'world':world,'device':a.device,'threads':a.threads,
        'data_sha256':{k:sha(ROOT/cfg[k]) for k in ['train_rows','validation_rows','audit_rows']},
        'code_sha256':{s:sha(ROOT/s) for s in SOURCES}}
    if rank==0 and not a.evaluate:
        if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==signature,'Changed restart protocol'
        atomic_json(out/'manifest.json',signature)
        for s in SOURCES:
            dest=out/'source_snapshots'/s;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/s).read_bytes())
    barrier(world)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    # Native model already emits only answer-position logits, avoiding an
    # unnecessary full-context vocabulary projection; no loss change.
    module=initialize(cfg).to(device);assert type(module) is MetaCompress
    examples=Examples(engine,a.device)
    if a.evaluate:
        state=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False)
        module.load_state_dict(state['generator'])
        evaluate(engine,module,read(ROOT/a.evaluate),examples,cfg,out,a.stage,a.cohort,rank,world)
        if rank==0:atomic_json(out/f'{a.cohort}.{a.stage}.complete.json',{'rows':len(read(ROOT/a.evaluate))})
        if world>1:dist.destroy_process_group()
        return
    optimizer=make_optimizer(module,cfg);checkpoint=out/'latest.pt';first=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        module.load_state_dict(saved['generator']);optimizer.load_state_dict(saved['optimizer']);first=saved['step']
    for path in [out/f'examples.rank{rank}.jsonl']+([out/'training.jsonl'] if rank==0 else []):
        if path.exists():path.write_text(''.join(json.dumps(r)+'\n' for r in read(path) if r['step']<=first))
    per_epoch=math.ceil(len(rows)/cfg['batch_size']);total=per_epoch*cfg['epochs']
    if rank==0 and first==0:atomic_save(out/'initial.pt',{'generator':module.state_dict(),'step':0,'config':cfg})
    if first==0:
        for cohort,data in [('validation',valid),('train_audit',audit)]:evaluate(engine,module,data,examples,cfg,out,'initial',cohort,rank,world)
    elif first%per_epoch==0:
        for cohort,data in [('validation',valid),('train_audit',audit)]:evaluate(engine,module,data,examples,cfg,out,f'epoch{first//per_epoch}',cohort,rank,world)
    wrapped=DDP(module,device_ids=[local] if a.device=='npu' else None,broadcast_buffers=False) if world>1 else module
    accum=cfg['batch_size']//world;started=time.monotonic()
    for step in range(first,total):
        epoch=step//per_epoch;order=list(range(len(rows)));random.Random(cfg['seed']+epoch).shuffle(order)
        batch=[order[i%len(order)] for i in range((step%per_epoch)*cfg['batch_size'],(step%per_epoch+1)*cfg['batch_size'])]
        lr=cfg['lr']*learning_rate_factor(step,total,cfg)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);values=torch.zeros(4,device=device)
        for micro in range(accum):
            row=rows[batch[micro*world+rank]]
            ctx=wrapped.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            with ctx:
                losses,details=student(engine,wrapped,row,examples,cfg)
                assert torch.isfinite(losses['total']),row['id']
                (losses['total']/accum).backward()
            item={'step':step+1,'epoch':epoch+1,'id':row['id'],'task':row['task'],
                  **{k:float(v.detach()) for k,v in losses.items()}}
            with (out/f'examples.rank{rank}.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            values+=torch.tensor([item[k] for k in ['prediction','entropy','collapse','total']],device=device)
            del losses,details
        threshold=cfg['clip'] if cfg['clip'] is not None else float('inf')
        norm=torch.nn.utils.clip_grad_norm_(module.parameters(),threshold)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        before=[p.detach().clone() for p in module.parameters()] if rank==0 else []
        optimizer.step()
        if world>1:dist.all_reduce(values)
        values/=cfg['batch_size']
        if rank==0:
            change=math.sqrt(sum(float((p.detach()-old).square().sum()) for p,old in zip(module.parameters(),before)))
            item={'step':step+1,'epoch':epoch+1,'total_steps':total,'lr':lr,'grad_norm':float(norm),
                  'update_l2':change,'clip_multiplier':min(1.,threshold/(float(norm)+1e-6)),
                  **dict(zip(['prediction','entropy','collapse','total'],values.tolist())),
                  'elapsed_seconds':time.monotonic()-started}
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            print(json.dumps({'name':cfg['name'],**item}),flush=True)
            if (step+1)%10==0 or (step+1)%per_epoch==0:
                state={'generator':{k:v.detach().cpu() for k,v in module.state_dict().items()},
                       'optimizer':optimizer.state_dict(),'step':step+1,'epoch':epoch+1,'config':cfg}
                atomic_save(checkpoint,state)
                if (step+1)%per_epoch==0:atomic_save(out/f'epoch{epoch+1}.pt',state)
        if (step+1)%per_epoch==0:
            barrier(world)
            for cohort,data in [('validation',valid),('train_audit',audit)]:
                evaluate(engine,module,data,examples,cfg,out,f'epoch{epoch+1}',cohort,rank,world)
    barrier(world)
    if rank==0:atomic_json(out/'complete.json',{'steps':total,'epochs':cfg['epochs'],'rows':len(rows),
        'world':world,'backbone_trainable_parameters':sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
        'job':os.environ.get('SLURM_JOB_ID')})
    barrier(world)
    if world>1:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--device',choices=['cpu','npu'],default='npu');p.add_argument('--threads',type=int,default=8)
    p.add_argument('--evaluate');p.add_argument('--checkpoint');p.add_argument('--stage',default='selected');p.add_argument('--cohort',default='confirm')
    run(p.parse_args())

if __name__=='__main__':main()
