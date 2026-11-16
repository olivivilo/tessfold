"""Resumable complete-corpus two-stage training: torchrun -m vqa.full_train."""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / 'artifacts/hf_cache'))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from .model import VLM, load_vision, resolve
from .metrics import anls, exact_match, relaxed_accuracy, normalize
from .benchmark_metrics import native_score, due_field_f1, dtvqa_scores
from .jsonl import recover_predictions

STOP = False
def request_stop(*_):
    global STOP
    STOP = True

def read_split(cfg, split):
    return [json.loads(line) for line in (resolve(cfg['data_dir']) / f'{split}.jsonl').read_text().splitlines() if line]

def amp(cfg):
    if cfg.get('mixed_precision'):
        return torch.autocast(device_type=cfg['device'].split(':')[0], dtype=getattr(torch,cfg['mixed_precision']))
    return contextlib.nullcontext()

def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    temp.replace(path)

def cpu_tree(value):
    """Portable checkpoints: never pickle NPU-specific tensor rebuild functions."""
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def barrier(world):
    if world > 1:
        dist.barrier()

def set_runtime(cfg):
    rank, world, local = (int(os.environ.get(k,d)) for k,d in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)])
    device = cfg['device'].split(':')[0]
    if device=='npu':
        import torch_npu
        torch.npu.set_device(local)
        torch.npu.matmul.allow_hf32=False
        torch.npu.conv.allow_hf32=False
        cfg['device'] = f'npu:{local}'
    elif device=='cuda':
        torch.cuda.set_device(local)
        cfg['device'] = f'cuda:{local}'
    if world > 1:
        dist.init_process_group('hccl' if device=='npu' else ('nccl' if device=='cuda' else 'gloo'), timeout=__import__('datetime').timedelta(hours=2))
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    return rank,world,local

def validate(cfg):
    splits = {s:read_split(cfg,s) for s in ['train','val','test']}
    ids = {s:{r['document_id'] for r in rows} for s,rows in splits.items()}
    if ids['train'] & (ids['val'] | ids['test']):
        raise ValueError('Train and heldout image leakage')
    if not splits['train']:
        raise ValueError('Empty training corpus')
    return splits

def cache(cfg,rank,world):
    out = resolve(cfg['output_dir']) / 'features'
    out.mkdir(parents=True,exist_ok=True)
    if rank==0:
        atomic_json(out.parent/'status.json',{'state':'caching','world_size':world})
    splits = validate(cfg)
    docs = {r['document_id']:r for rows in splits.values() for r in rows}
    signature = {k:cfg.get(k) for k in ['vision_model','max_pixels','min_pixels','vision_dtype','vision_attention','feature_storage_dtype']}
    signature['weights_sha256'] = 'a9f059fc5a7afb6e9d2ad64b5db02dfc8e846d4b713460f4b3c0b2468a4d52e1'
    marker=out.parent/'cache_complete.json'
    if marker.exists():
        completed=json.loads(marker.read_text())
        if completed['signature']==signature and completed['documents']==len(docs):
            # main() has already checked the complete corpus fingerprint/config.
            # A fully committed cache is independent of the extraction world size.
            if rank==0:print(json.dumps({'phase':'cache','state':'reusing_complete_cache','documents':len(docs)}),flush=True)
            barrier(world)
            return
    local_docs = sorted(docs)[rank::world]
    vision,processor = None,None
    attention_candidates=[]
    for i,doc in enumerate(local_docs):
        path = out / f'{doc}.pt'
        if path.exists():
            existing = torch.load(path,map_location='cpu',weights_only=True)
            if existing['signature']==signature:
                attention_candidates.append((existing['features'].shape[0],doc))
                continue
        if vision is None:
            vision,processor = load_vision(cfg)
        source = resolve(docs[doc]['image'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=doc:
            raise ValueError(f'Image content changed: {source}')
        media = processor(images=[Image.open(source).convert('RGB')],return_tensors='pt')
        pixels = media['pixel_values'].to(cfg['device'],dtype=vision.dtype)
        grid = media['image_grid_thw'].to(cfg['device'])
        with torch.inference_mode():
            features = vision(pixels,grid,bf16=vision.dtype==torch.bfloat16)
        assert features.ndim==2 and features.shape[1]==768
        attention_candidates.append((features.shape[0],doc))
        temp = path.with_suffix(f'.rank{rank}.tmp')
        stored=features.cpu().to(getattr(torch,cfg.get('feature_storage_dtype',cfg.get('vision_dtype',cfg['dtype']))))
        torch.save({'features':stored,'grid':grid.cpu(),'signature':signature},temp)
        temp.replace(path)
        if i%100==0:
            print(json.dumps({'phase':'cache','rank':rank,'documents_done':i+1,'documents_total':len(local_docs),'tokens':features.shape[0]}),flush=True)
        if STOP:
            raise InterruptedError('Cache stopped; rerun to resume')
    del vision,processor
    gc.collect()
    device=cfg['device'].split(':')[0]
    if device=='npu': torch.npu.empty_cache()
    if device=='cuda': torch.cuda.empty_cache()
    barrier(world)
    if rank==0:
        atomic_json(resolve(cfg['output_dir'])/'cache_complete.json',{'documents':len(docs),'signature':signature})
        if cfg.get('verify_vision_attention') and cfg['vision_attention']=='ascend_fa':
            verify_vision_attention(cfg,docs,out,attention_candidates)
    barrier(world)

def verify_vision_attention(cfg,docs,out,selected):
    """Compare the official NPU fused path to eager attention before a long run."""
    # Short and longest image expose both ordinary and long-sequence behavior.
    selected=[min(selected),max(selected)]
    eager_cfg={**cfg,'vision_attention':'eager'}
    vision,processor=load_vision(eager_cfg)
    comparisons=[]
    for n,doc in dict.fromkeys(selected):
        with Image.open(resolve(docs[doc]['image'])) as image:
            media=processor(images=[image.convert('RGB')],return_tensors='pt')
        with torch.inference_mode():
            reference=vision(media['pixel_values'].to(cfg['device'],dtype=vision.dtype),media['image_grid_thw'].to(cfg['device']),bf16=True).float().cpu()
        actual=torch.load(out/f'{doc}.pt',map_location='cpu',weights_only=True)['features'].float()
        ref64,act64=reference.double(),actual.double()
        cosine=nn.functional.cosine_similarity(ref64.reshape(1,-1),act64.reshape(1,-1)).item()
        relative_rmse=((act64-ref64).square().mean()/ref64.square().mean()).sqrt().item()
        comparisons.append({'document_id':doc,'tokens':n,'cosine_similarity':cosine,'relative_rmse':relative_rmse})
    atomic_json(resolve(cfg['output_dir'])/'vision_attention_verification.json',{'reference':'official eager BF16','candidate':'official ascend_fa BF16','comparisons':comparisons})
    if any(r['cosine_similarity']<.99 or r['relative_rmse']>.15 for r in comparisons):
        raise RuntimeError('NPU fused vision features differ substantially from eager; inspect vision_attention_verification.json and use eager before training')
    del vision
    torch.npu.empty_cache()

class TrainableVLM(VLM):
    def forward(self, samples):
        # Pad embeddings/masks and score answer positions only. Real micro batches are supported.
        inputs=[self.embeds(sample,sample['answers'][0]) for sample in samples]
        lengths=[x[0].shape[1] for x in inputs]
        if max(lengths)>self.settings['max_sequence_length']:
            raise ValueError(f'Sequence length {max(lengths)} exceeds explicit maximum; no silent truncation')
        padded=torch.nn.utils.rnn.pad_sequence([x[0][0] for x in inputs],batch_first=True)
        mask=torch.arange(padded.shape[1],device=padded.device)[None,:] < torch.tensor(lengths,device=padded.device)[:,None]
        states=self.lm.model(inputs_embeds=padded,attention_mask=mask.long(),use_cache=False,return_dict=True).last_hidden_state
        answer_states=torch.cat([states[i,prefix-1:prefix-1+ids.numel()] for i,(_,ids,prefix) in enumerate(inputs)])
        targets=torch.cat([ids.reshape(-1) for _,ids,_ in inputs])
        logits=self.lm.lm_head(answer_states).float()
        return nn.functional.cross_entropy(logits,targets)

def checkpoint(model, optimizer, scheduler, cfg, stage, step, rank, world, sampler):
    barrier(world)
    if rank==0:
        root=resolve(cfg['output_dir'])
        temp=root/'resume.tmp'
        temp.mkdir(exist_ok=True)
        model.lm.save_pretrained(temp/'language_model',safe_serialization=True,max_shard_size='4GB')
        model.tokenizer.save_pretrained(temp/'language_model')
        torch.save(cpu_tree(model.projector.state_dict()),temp/'projector.pt')
        torch.save(cpu_tree({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'stage':stage,'step':step,
                    'sampler':sampler,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()}),temp/'trainer.pt')
        atomic_json(temp/'config.json',{k:v for k,v in cfg.items() if k!='device'} )
        destination=root/'resume'
        backup=root/'resume.previous'
        # Remove only our previous checkpoint after a new complete checkpoint exists.
        import shutil
        if backup.exists(): shutil.rmtree(backup)
        if destination.exists(): destination.rename(backup)
        temp.rename(destination)
        atomic_json(root/'status.json',{'state':'training','stage':stage,'optimizer_step':step,'checkpoint':str(destination),'world_size':world})
    barrier(world)

@torch.no_grad()
def evaluate(model,cfg,split,label,rank,world,limit_per_task=0,blank=False):
    rows=read_split(cfg,split)
    if limit_per_task:
        selected=[]
        counts={}
        for row in sorted(rows,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()):
            t=row['task']
            if counts.get(t,0)<limit_per_task:
                selected.append(row); counts[t]=counts.get(t,0)+1
        rows=selected
    model.eval()
    out=resolve(cfg['output_dir'])
    dest=out/f'{label}.rank{rank}.jsonl'
    done=recover_predictions(dest)
    with dest.open('a') as handle:
        for i,row in enumerate(rows[rank::world]):
            if row['id'] in done: continue
            with amp(cfg): prediction=model.predict(row,blank_image=blank)
            record={**row,'prediction':prediction,'score':native_score(prediction,row),
                    'exact_match':exact_match(prediction,row['answers']),'anls':anls(prediction,row['answers'])}
            handle.write(json.dumps(record,ensure_ascii=False)+'\n'); handle.flush()
            if i%50==0: print(json.dumps({'phase':'eval','rank':rank,'label':label,'done':i+1,'total':len(rows[rank::world])}),flush=True)
            if STOP: raise InterruptedError('Evaluation interrupted; predictions can be resumed')
    barrier(world)
    metrics=None
    if rank==0:
        predictions={}
        for r in range(world):
            for line in (out/f'{label}.rank{r}.jsonl').read_text().splitlines():
                try:
                    row=json.loads(line); predictions[row['id']]=row
                except json.JSONDecodeError:
                    continue
        predictions=[predictions[r['id']] for r in rows]
        groups={}
        for row in predictions: groups.setdefault(row['task'],[]).append(row)
        metrics={'label':label,'split':split,'scope':cfg['eval_scope'],'limit_per_task':limit_per_task,'blank_image':blank,
                 'samples':len(predictions),'benchmarks':{t:{'samples':len(rr),'score_percent':100*sum(r['score'] for r in rr)/len(rr),
                              'metric': 'ANLS' if t in ('DocVQA','InfoVQA') else ('AccANLS' if t=='DT-VQA' else ('relaxed_accuracy' if t=='ChartQA' else ('WTQ denotation accuracy' if t=='WTQ' else 'OCRBench_accuracy')))} for t,rr in groups.items()}}
        for t,rr in groups.items():
            if t in ('DeepForm','KLC'):
                metrics['benchmarks'][t].update(due_field_f1(rr))
            if t=='ChartQA':
                sub={}
                for r in rr: sub.setdefault(str(r.get('question_type','unspecified')),[]).append(r)
                metrics['benchmarks'][t]['by_question_type']={k:{'samples':len(v),'score_percent':100*sum(r['score'] for r in v)/len(v)} for k,v in sub.items()}
            if t=='OCRBench':
                sub={}
                for r in rr: sub.setdefault(r.get('question_type','unspecified'),[]).append(r)
                metrics['benchmarks'][t]['official_score_out_of_1000']=sum(r['score'] for r in rr)
                metrics['benchmarks'][t]['by_type']={k:{'samples':len(v),'correct':sum(r['score'] for r in v)} for k,v in sub.items()}
            if t=='DT-VQA':
                sub={}
                for r in rr:sub.setdefault(r.get('dataset_name','unspecified'),[]).append(r)
                def dt_summary(records):
                    scores=[dtvqa_scores(r['prediction'],r['answers']) for r in records]
                    return {k+'_percent':100*sum(s[k] for s in scores)/len(scores) for k in ('ANLS','accuracy','accanls')}
                metrics['benchmarks'][t].update(dt_summary(rr))
                metrics['benchmarks'][t]['by_dataset']={k:{'samples':len(v),**dt_summary(v)} for k,v in sub.items()}
        atomic_json(out/f'{label}_metrics.json',metrics)
        print(json.dumps(metrics),flush=True)
    model.projector.train()
    return metrics

def train_stage(model,cfg,stage,rank,world,resume_state=None):
    rows=read_split(cfg,'train')
    batch=cfg['micro_batch_size']
    divisor=world*batch
    if cfg['effective_batch_size']%divisor:
        raise ValueError('Effective batch must divide world * microbatch')
    accum=cfg['effective_batch_size']//divisor
    epochs=cfg[f'stage{stage}_epochs']
    steps_per_epoch=math.ceil(len(rows)/cfg['effective_batch_size'])
    total=steps_per_epoch*epochs
    model.lm.requires_grad_(stage==2)
    model.projector.requires_grad_(True)
    model.lm.train()
    model.projector.train()
    if cfg['gradient_checkpointing']:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg[f'stage{stage}_lr'],weight_decay=cfg['weight_decay'],betas=(0.9,cfg['adam_beta2']),foreach=False)
    warmup=max(1,int(total*cfg['warmup_ratio']))
    def schedule(step):
        if step<warmup: return (step+1)/warmup
        return 0.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    start_step=0
    if resume_state and resume_state['stage']==stage:
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        start_step=resume_state['step']
    if rank==0:
        atomic_json(resolve(cfg['output_dir'])/'status.json',{'state':'training','stage':stage,'optimizer_step':start_step,'total_steps':total,'world_size':world})
    wrapped=DDP(model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False) if world>1 else model
    log=resolve(cfg['output_dir'])/'training.jsonl'
    started=time.monotonic()
    order,previous_epoch=[],None
    for step in range(start_step,total):
        epoch=step//steps_per_epoch
        if epoch!=previous_epoch:
            order=list(range(len(rows))); random.Random(cfg['seed']+stage*1000+epoch).shuffle(order)
            previous_epoch=epoch
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for micro in range(accum):
            base=(step%steps_per_epoch)*cfg['effective_batch_size']+micro*divisor+rank*batch
            samples=[rows[order[(base+i)%len(order)]] for i in range(batch)]
            sync=wrapped.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            with sync,amp(cfg):
                loss=wrapped(samples)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                (loss/accum).backward()
            losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],foreach=False)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite training gradient')
        optimizer.step(); scheduler.step()
        if rank==0:
            record={'stage':stage,'step':step+1,'total_steps':total,'epoch':epoch+1,'loss':sum(losses)/len(losses),
                    'gradient_norm':float(norm),'lr':scheduler.get_last_lr()[0],'effective_batch_size':cfg['effective_batch_size'],
                    'world_size':world,'trainable_parameters':sum(p.numel() for p in params),'elapsed_seconds':time.monotonic()-started}
            with log.open('a') as handle: handle.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        completed=step+1
        stop_tensor=torch.tensor(int(STOP),device=cfg['device'])
        if world>1: dist.all_reduce(stop_tensor,op=dist.ReduceOp.MAX)
        if completed%cfg['checkpoint_steps']==0 or completed==total or stop_tensor.item():
            checkpoint(model,optimizer,scheduler,cfg,stage,completed,rank,world,{'epoch':epoch,'seed':cfg['seed'],'corpus_size':len(rows)})
        if stop_tensor.item(): raise InterruptedError('Checkpoint saved on termination signal')
        if completed%cfg['validation_steps']==0 and completed!=total:
            evaluate(model,cfg,'val',f'stage{stage}_step{completed}_val',rank,world,cfg['validation_samples_per_task'])
            model.lm.train()
    del wrapped,optimizer,scheduler
    gc.collect()
    barrier(world)
    if rank==0:
        dest=resolve(cfg['output_dir'])/f'stage{stage}'
        dest.mkdir(exist_ok=True)
        torch.save(cpu_tree(model.projector.state_dict()),dest/'projector.pt')
        if stage==2:
            model.lm.save_pretrained(dest/'language_model',safe_serialization=True,max_shard_size='4GB')
            model.tokenizer.save_pretrained(dest/'language_model')
        atomic_json(dest/'complete.json',{'stage':stage,'steps':total,'epochs':epochs,'train_samples':len(rows)})
    barrier(world)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/full.json')
    p.add_argument('--phase',choices=['cache','train','eval','all'],default='all')
    args=p.parse_args()
    cfg=json.loads(resolve(args.config).read_text())
    rank,world,local=set_runtime(cfg)
    out=resolve(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    if rank==0:
        recorded={**cfg,'device':cfg['device'].split(':')[0]}
        config_path=out/'config.json'
        if config_path.exists():
            previous_config=json.loads(config_path.read_text())
            keys=['vision_model','language_model','seed','dtype','vision_dtype','mixed_precision','vision_attention',
                  'max_pixels','min_pixels','prompt_style','max_new_tokens','stage1_epochs','stage2_epochs',
                  'stage1_lr','stage2_lr','effective_batch_size','warmup_ratio','weight_decay','adam_beta2','gradient_clip','feature_storage_dtype']
            changed=[k for k in keys if previous_config.get(k)!=recorded.get(k)]
            if changed:raise ValueError(f'Run configuration changed ({changed}); use a new output_dir')
        fingerprint={s:hashlib.sha256((resolve(cfg['data_dir'])/f'{s}.jsonl').read_bytes()).hexdigest() for s in ('train','val','test')}
        previous=out/'corpus_fingerprint.json'
        if previous.exists() and json.loads(previous.read_text())!=fingerprint:
            raise ValueError('Corpus changed since this run; use a new output_dir')
        atomic_json(previous,fingerprint)
        atomic_json(out/'config.json',recorded)
        atomic_json(out/'execution.json',{'hostname':os.uname().nodename,'slurm_job':os.environ.get('SLURM_JOB_ID'),
                                          'world_size':world,'torch':torch.__version__,'effective_batch_size':cfg['effective_batch_size']})
    if args.phase in ('cache','all'):
        cache(cfg,rank,world)
    if args.phase=='cache':
        if rank==0:atomic_json(out/'status.json',{'state':'cached_waiting_for_training','world_size':world})
        if world>1:dist.destroy_process_group()
        return
    resume=out/'resume'
    if not (resume/'trainer.pt').exists() and (out/'resume.previous/trainer.pt').exists():
        # Recover if the process was killed between rotating the two directories.
        resume=out/'resume.previous'
    original=cfg['language_model']
    state=None
    use_final=args.phase=='eval' or (out/'stage2/complete.json').exists()
    if use_final:
        cfg['language_model']=str(out/'stage2/language_model')
    elif (resume/'trainer.pt').exists():
        cfg['language_model']=str(resume/'language_model')
        state=torch.load(resume/'trainer.pt',map_location='cpu',weights_only=False)
    model=TrainableVLM(cfg)
    cfg['language_model']=original
    if use_final:
        model.projector.load_state_dict(torch.load(out/'stage2/projector.pt',map_location='cpu',weights_only=True))
    elif state:
        model.projector.load_state_dict(torch.load(resume/'projector.pt',map_location='cpu',weights_only=True))
    elif (out/'stage1/complete.json').exists():
        model.projector.load_state_dict(torch.load(out/'stage1/projector.pt',map_location='cpu',weights_only=True))
    if args.phase in ('train','all'):
        for stage in (1,2):
            if (out/f'stage{stage}/complete.json').exists(): continue
            if state and state['stage']>stage: continue
            train_stage(model,cfg,stage,rank,world,state)
            state=None
            # Same fixed heldout questions quantify MLP-only alignment vs joint SFT.
            evaluate(model,cfg,'val',f'stage{stage}_alignment_val',rank,world,cfg['validation_samples_per_task'])
        # Gradient checkpointing must be off during autoregressive evaluation.
        model.lm.gradient_checkpointing_disable()
    for split in cfg['eval_splits']:
        if rank==0:
            atomic_json(out/'status.json',{'state':'evaluating','split':split,'world_size':world})
        evaluate(model,cfg,split,f'final_{split}',rank,world)
    if cfg.get('verify_completion',False):
        # Free the training model before exercising the public saved-model entry.
        del model
        gc.collect()
        backend=cfg['device'].split(':')[0]
        if backend=='npu':torch.npu.empty_cache()
        elif backend=='cuda':torch.cuda.empty_cache()
        barrier(world)
        if rank==0:
            import subprocess,sys
            atomic_json(out/'status.json',{'state':'verifying','world_size':world})
            env=os.environ.copy()
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK',
                        'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
                env.pop(key,None)
            subprocess.run([sys.executable,'-m','scripts.verify_full','--config',args.config,'--run-infer'],
                           cwd=resolve('.'),env=env,check=True)
        barrier(world)
    if rank==0:
        atomic_json(out/'status.json',{'state':'complete','stages':[1,2],'evaluation_splits':cfg['eval_splits'],
                                      'scope':cfg['eval_scope']})
    if world>1: dist.destroy_process_group()

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if int(os.environ.get('RANK',0))==0:
            import sys
            config_arg=sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/full.json'
            settings=json.loads(resolve(config_arg).read_text())
            root=resolve(settings['output_dir']);root.mkdir(parents=True,exist_ok=True)
            atomic_json(root/'status.json',{'state':'interrupted' if isinstance(error,InterruptedError) else 'failed',
                                          'error_type':type(error).__name__,'error':str(error)})
        raise
