"""Portable pilot -> dev-gated full training -> locked-checkpoint evaluation.

The frozen model and spatial routers are imported from the latest original
implementations. This runner removes dependence on historical output files.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from .config import sha, validate
from .io import save_json, save_torch, save_rows, read_json
from .runtime import barrier, seed, lr_factor, shard_batch, synchronize


def balanced(rows, limit):
    # Same hash order and task-stratified distinct-document pilot as native8B.
    if limit>=len(rows):return rows
    groups=collections.defaultdict(list)
    for row in rows:groups[row['task']].append(row)
    result=[]
    for i,task in enumerate(sorted(groups)):
        count=limit//len(groups)+(i<limit%len(groups))
        pool=sorted(groups[task],key=lambda r:hashlib.sha256(('pilot8b:'+r['id']).encode()).hexdigest())
        unique={}
        for row in pool:unique.setdefault(row['document_id'],row)
        chosen=list(unique.values())[:count];used={r['id'] for r in chosen}
        chosen += [r for r in pool if r['id'] not in used][:count-len(chosen)]
        result+=chosen
    if len(result)!=limit:
        raise ValueError('A task has too few rows for the configured balanced pilot')
    return sorted(result,key=lambda r:hashlib.sha256(('pilot-order:'+r['id']).encode()).hexdigest())


def summarize(rows, field='prediction'):
    from vqa.benchmark_metrics import native_score, due_field_f1
    by={}
    for task in sorted({r['task'] for r in rows}):
        rr=[dict(r,prediction=r[field]) for r in rows if r['task']==task]
        by[task]=due_field_f1(rr)['score_percent'] if task in {'KLC','DeepForm'} else 100*sum(native_score(r['prediction'],r) for r in rr)/len(rr)
    return dict(macro=sum(by.values())/len(by),by_task=by,questions=len(rows))


def eligible(initial,current,cfg):
    generation=(current['macro']>=initial['macro']+cfg['min_macro_gain'] or
                (initial['macro']>=initial['teacher']['macro']-cfg['min_macro_gain'] and
                 current['macro']>=initial['teacher']['macro']-cfg['min_macro_gain']))
    return current['kl']<=initial['kl']*(1-cfg['min_kl_reduction']) and generation


def source_digest():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):sha(p) for dirname in ('tessfold','vqa')
            for p in sorted((root/dirname).rglob('*')) if p.suffix in {'.py','.cpp'}}


def model_digest(cfg):
    # Pin actual local bytes, including weights and remote-code .py files.
    root=Path(cfg['model'])
    files=sorted(p for p in root.rglob('*') if p.is_file() and
                 p.suffix in {'.safetensors','.bin','.pt','.py','.json','.txt','.model'} and '.cache' not in p.parts)
    if not files:raise FileNotFoundError(f'No model files found in {root}')
    return {str(p.relative_to(root)):sha(p) for p in files}


class TeacherData:
    def __init__(self, engine, cfg, namespace, image_hashes):
        self.engine=engine;self.cfg=cfg
        self.root=Path(cfg['cache'])/namespace
        self.memory=collections.OrderedDict();self.image_hashes=image_hashes

    def get(self,row,need_logits=True):
        image_sha=self.image_hashes[row['image']]
        key=hashlib.sha256((image_sha+'\0'+row['question']).encode()).hexdigest()
        if key in self.memory:
            features,teacher=self.memory.pop(key)
            self.memory[key]=(features,teacher)
        else:
            fp=self.root/'features'/f'{image_sha}.pt'
            if self.cfg['cache_features'] and fp.exists():features=torch.load(fp,map_location='cpu',weights_only=True)
            else:
                features=self.engine.extract(row)
                if self.cfg['cache_features']:save_torch(fp,features)
            d=self.engine.prepare(row,features)
            tp=self.root/'teachers'/f'{key}.pt'
            if tp.exists():teacher=torch.load(tp,map_location='cpu',weights_only=True)
            else:
                ids=self.engine.greedy(d,cap=self.cfg['teacher_cap'])
                if not ids:raise ValueError('Empty teacher response')
                teacher=dict(response=ids,prediction=self.engine.decode(ids))
                save_torch(tp,teacher)
            if self.cfg['cache_memory_entries']:
                self.memory[key]=(features,teacher)
                while len(self.memory)>self.cfg['cache_memory_entries']:self.memory.popitem(last=False)
        d=self.engine.prepare(row,features)
        logp=None
        if need_logits:
            lp=self.root/'logp'/f'{key}.pt'
            if self.cfg['cache_logits'] and lp.exists():logp=torch.load(lp,map_location='cpu',weights_only=True).to(self.engine.device)
            else:
                with torch.no_grad():logp=self.engine.logits(d,teacher['response']).float().log_softmax(-1).detach()
                if self.cfg['cache_logits']:save_torch(lp,logp.cpu())
        return d,teacher,logp


def loss(engine,router,row,data,ratio):
    d,t,teacher=data.get(row)
    route=engine.route(router,d,ratio)
    student=engine.logits(d,t['response'],route).float().log_softmax(-1)
    kl=(teacher.exp()*(teacher-student)).sum(-1).mean().clamp_min(0)
    return kl,d,t,route


@torch.no_grad()
def development(engine,router,rows,data,cfg,out,label,rank,world):
    router.eval();local=[]
    for row in rows[rank::world]:
        kl,d,t,route=loss(engine,router,row,data,cfg['ratio'])
        ids=engine.greedy(d,route,cap=cfg['dev_cap'])
        local.append(dict(row,prediction=engine.decode(ids),generated_ids=ids,
                          teacher_prediction=t['prediction'],kl=float(kl),
                          n=engine.token_count(d),m=len(route[0])))
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,local)
        local=[r for rr in gathered for r in rr]
    local.sort(key=lambda r:r['id'])
    assert len(local)==len(rows) and len({r['id'] for r in local})==len(rows)
    result=dict(stage=label,kl=sum(r['kl'] for r in local)/len(local),
                **summarize(local),teacher=summarize(local,'teacher_prediction'))
    if rank==0:
        save_rows(out/f'{label}.predictions.jsonl',local)
        save_json(out/f'{label}.summary.json',result)
    router.train();return result


def run_train(cfg,runtime,pilot=False,resume=False,engine_factory=None):
    rank,world,local,device=runtime
    splits,datasets=validate(cfg)
    epochs=cfg['pilot_epochs'] if pilot else cfg['epochs']
    train=balanced(splits['train'],min(cfg['pilot_limit'],len(splits['train']))) if pilot else splits['train']
    out=Path(cfg['output'])/('pilot' if pilot else 'full')
    out.mkdir(parents=True,exist_ok=True)
    # Hash config, data, images, model bytes and actual method sources. Never
    # accept a stale cache/checkpoint merely because its output name matches.
    md=model_digest(cfg)
    signature=dict(config=cfg,datasets=datasets,model_sha256=md,source_sha256=source_digest(),
                   phase='pilot' if pilot else 'full',world=world,train_questions=len(train))
    if not pilot and cfg['require_pilot']:
        gate_path=Path(cfg['output'])/'pilot/complete.json'
        gate=read_json(gate_path)
        if not gate.get('gate_passed'):raise ValueError('Pilot failed the development gate; full training refused')
        pilot_sig=read_json(gate_path.parent/'manifest.json')
        for key in ('config','datasets','model_sha256','source_sha256','world'):
            if pilot_sig[key]!=signature[key]:raise ValueError(f'Pilot/full mismatch in {key}; run a fresh pilot')
    manifest=out/'manifest.json'
    # Only rank zero checks/creates the manifest. Otherwise another rank can
    # mistake a just-created manifest for a previous run and fail at startup.
    error=None
    if rank==0:
        try:
            if manifest.exists():
                if not resume:raise FileExistsError(f'{out} already has a run; use --resume or a new output')
                if read_json(manifest)!=signature:raise ValueError('Resume signature differs (code/model/data/config/device count)')
            elif resume:raise FileNotFoundError('No existing manifest to resume')
            else:save_json(manifest,signature)
        except Exception as exc:
            if world==1:raise
            error=f'{type(exc).__name__}: {exc}'
    if world>1:
        errors=[error];dist.broadcast_object_list(errors,src=0)
        if errors[0]:raise ValueError(errors[0])
        barrier(world)
        if read_json(manifest)!=signature:raise ValueError('Ranks disagree on source/data/model/config')
    if (out/'complete.json').exists():return read_json(out/'complete.json')
    from .engine import create,router as make_router
    engine=(engine_factory or create)(cfg,device)
    # Model loading must not influence the routing initialization.
    seed(cfg['seed']);router=make_router(cfg).to(device)
    optimizer=torch.optim.AdamW(router.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    state_path=out/'latest.pt';step=0
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        router.load_state_dict(state['router']);optimizer.load_state_dict(state['optimizer']);step=state['step']
    elif rank==0:save_torch(out/'initial.pt',dict(router=router.state_dict(),step=0,config=cfg))
    namespace=hashlib.sha256(json.dumps(dict(model=md,backend=cfg['backend'],teacher_cap=cfg['teacher_cap'],
                                source=signature['source_sha256']),sort_keys=True).encode()).hexdigest()
    data=TeacherData(engine,cfg,namespace,datasets['image_content_sha256'])
    wrapped=router
    if world>1:
        from torch.nn.parallel import DistributedDataParallel
        wrapped=DistributedDataParallel(router,device_ids=[local] if device.type!='cpu' else None,
                                        find_unused_parameters=True,broadcast_buffers=False)
    initial_path=out/'initial.summary.json'
    if initial_path.exists():initial=read_json(initial_path)
    else:
        if step:raise RuntimeError('Resume missing initial development evaluation')
        initial=development(engine,router,splits['dev'],data,cfg,out,'initial',rank,world)
    per_epoch=math.ceil(len(train)/cfg['batch_size']);total=per_epoch*epochs
    # If a crash occurred after committing an epoch checkpoint but before its
    # development evaluation, finish that evaluation before the next update.
    if step and step%per_epoch==0 and not (out/f'epoch{step//per_epoch}.summary.json').exists():
        development(engine,router,splits['dev'],data,cfg,out,f'epoch{step//per_epoch}',rank,world)
    log=out/'training.jsonl'
    if rank==0 and log.exists():
        valid=[]
        for line in log.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:break
            if r['step']<=step:valid.append(r)
        save_rows(log,valid)
    barrier(world);started=time.perf_counter()
    for s in range(step,total):
        epoch=s//per_epoch;order=list(range(len(train)));random.Random(cfg['seed']+epoch).shuffle(order)
        indices=order[(s%per_epoch)*cfg['batch_size']:(s%per_epoch+1)*cfg['batch_size']]
        samples=shard_batch(indices,rank,world)
        lr=cfg['lr']*lr_factor(s,total,cfg['warmup_fraction'],cfg['lr_floor'])
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);value=torch.zeros((),device=device)
        for pos,(ix,weight) in enumerate(samples):
            context=wrapped.no_sync() if world>1 and pos<len(samples)-1 else contextlib.nullcontext()
            with context:
                kl=loss(engine,wrapped,train[ix],data,cfg['ratio'])[0]
                if not torch.isfinite(kl):raise FloatingPointError(f'Nonfinite loss at update {s+1}')
                (kl*weight).backward();value+=kl.detach()*weight/world
        norm=float(torch.nn.utils.clip_grad_norm_(router.parameters(),cfg['clip']))
        if not math.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
        optimizer.step()
        if any(p.grad is not None or p.requires_grad for p in engine.model.parameters()):
            raise RuntimeError('Frozen backbone acquired trainable parameters or gradients')
        if world>1:dist.all_reduce(value)
        if rank==0:
            record=dict(step=s+1,epoch=epoch+1,kl=float(value),lr=lr,gradient_norm=norm,
                        clip_multiplier=min(1.,cfg['clip']/(norm+1e-6)),seconds=time.perf_counter()-started)
            with log.open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        end_epoch=(s+1)%per_epoch==0
        if (s+1)%cfg['checkpoint_every']==0 or end_epoch:
            if rank==0:
                state=dict(router=router.state_dict(),optimizer=optimizer.state_dict(),step=s+1,config=cfg)
                if end_epoch:save_torch(out/f'epoch{epoch+1}.pt',state)
                # Commit latest last: a recovered epoch boundary must always
                # have its selectable epoch checkpoint on disk already.
                save_torch(state_path,state)
            barrier(world)
        if end_epoch:development(engine,router,splits['dev'],data,cfg,out,f'epoch{epoch+1}',rank,world)
    result=None
    if rank==0:
        candidates=[read_json(out/f'epoch{i}.summary.json') for i in range(1,epochs+1)]
        passing=[r for r in candidates if eligible(initial,r,cfg)]
        best=max(passing,key=lambda r:(r['macro'],-r['kl'])) if passing else None
        chosen=(best['stage'] if best else 'initial')+'.pt'
        result=dict(gate_passed=bool(passing),initial=initial,best=best,selected=chosen,
                    selected_sha256=sha(out/chosen),epochs=epochs,updates=total,
                    selection='fixed development gate then highest dev macro; formal test never selects model',
                    backbone_trainable=0)
        save_json(out/'complete.json',result)
    barrier(world)
    return result


@torch.no_grad()
def run_evaluate(cfg,runtime,stage='full',engine_factory=None):
    rank,world,_,device=runtime;splits,datasets=validate(cfg)
    run=Path(cfg['output'])/stage;meta=read_json(run/'complete.json')
    manifest=read_json(run/'manifest.json')
    for key,actual in [('config',cfg),('datasets',datasets),('source_sha256',source_digest()),('model_sha256',model_digest(cfg))]:
        if manifest[key]!=actual:raise ValueError(f'Evaluation changed {key}; use the locked training protocol')
    checkpoint=run/meta['selected']
    if sha(checkpoint)!=meta['selected_sha256']:raise ValueError('Selected checkpoint changed')
    from .engine import create,router as make_router
    engine=(engine_factory or create)(cfg,device);router=make_router(cfg).to(device).eval()
    router.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['router'])
    out=run/'evaluation';out.mkdir(exist_ok=True)
    local=[]
    from vqa.benchmark_metrics import native_score
    for row in splits['eval'][rank::world]:
        # Fresh feature extraction, separate teacher/student generations, no
        # inter-question KV and no cached teacher text reused in evaluation.
        synchronize(device);start=time.perf_counter()
        features=engine.extract(row);d=engine.prepare(row,features)
        synchronize(device);front=time.perf_counter()-start
        synchronize(device);start=time.perf_counter()
        teacher_ids=engine.greedy(d,cap=cfg['eval_cap'])
        synchronize(device);teacher_s=time.perf_counter()-start
        synchronize(device);start=time.perf_counter()
        route=engine.route(router,d,cfg['ratio'])
        ids=engine.greedy(d,route,cap=cfg['eval_cap'])
        synchronize(device);student_s=time.perf_counter()-start
        prediction=engine.decode(ids);teacher=engine.decode(teacher_ids)
        local.append(dict(row,prediction=prediction,teacher_prediction=teacher,
                          score=native_score(prediction,row),teacher_score=native_score(teacher,row),
                          generated_ids=ids,teacher_generated_ids=teacher_ids,
                          n=engine.token_count(d),m=len(route[0]),
                          row_sum_error=float((route[0].sum(-1)-1).abs().max()),
                          frontend_seconds=front,teacher_decoder_seconds=teacher_s,
                          student_route_and_decoder_seconds=student_s,
                          timing_scope='shared cached frontend; diagnostic only, not full deployment benchmark',
                          teacher_hit_token_limit=len(teacher_ids)==cfg['eval_cap'],
                          hit_token_limit=len(ids)==cfg['eval_cap']))
    save_rows(out/f'predictions.rank{rank}.jsonl',local);barrier(world)
    result=None
    if rank==0:
        combined=[json.loads(line) for rank_id in range(world)
                  for line in (out/f'predictions.rank{rank_id}.jsonl').read_text().splitlines()]
        assert len(combined)==len(splits['eval']) and len({r['id'] for r in combined})==len(combined)
        student=summarize(combined);teacher=summarize(combined,'teacher_prediction')
        result=dict(student=student,teacher=teacher,macro_delta=student['macro']-teacher['macro'],
                    selected_checkpoint_sha256=meta['selected_sha256'],benchmark_used_for_selection=False)
        save_json(out/'summary.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    barrier(world);return result
