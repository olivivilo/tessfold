"""Only native8B P generator learns; fixed data, multi-image pilot gate, resume.

Several configurations train in independent HCCL subgroups on one allocation.
Pilot:512distinct-image-stratified trainQA,56fixed dev,3epochs. Full is gated,
uses the exact existing20kQA and4epochs. Gold is used only for dev evaluation.
"""
import argparse,collections,copy,hashlib,json,math,os,random,time,shutil
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
from scripts.train_meta_compress import init_runtime,atomic_save
from vqa.tessfold_qwen3vl import NativeQwen
from vqa.tessfold_qwen3vl_stable import StableColumn as NativeSpatialRouter
from vqa.benchmark_metrics import native_score,due_field_f1

BASE=ROOT/'artifacts/tessfold_acceleration_20261115'
CACHE=BASE/'qwen_cache'
FEATURE_SOURCE='Qwen3-VL-8B-Instruct_0c351dd01ed87e9c1b53cbc748cba10e6187ff3b_nativeBF16_short64_pixels1003520_v1'


def balanced(rows,limit):
    if limit>=len(rows):return rows
    groups=collections.defaultdict(list)
    for row in rows:groups[row['task']].append(row)
    result=[]
    for i,task in enumerate(sorted(groups)):
        count=limit//len(groups)+(i<limit%len(groups));pool=sorted(groups[task],key=lambda r:hashlib.sha256(('pilot8b:'+r['id']).encode()).hexdigest())
        unique={}
        for row in pool:unique.setdefault(row['document_id'],row)
        chosen=list(unique.values())[:count]
        used={r['id'] for r in chosen};chosen+=[r for r in pool if r['id'] not in used][:count-len(chosen)]
        result+=chosen
    return sorted(result,key=lambda r:hashlib.sha256(('pilot-order:'+r['id']).encode()).hexdigest())


class Data:
    def __init__(self,engine,cache_entries=64):
        self.engine=engine;self.memory=collections.OrderedDict();self.entries=cache_entries
        for name in ['features','teachers','logp']:(CACHE/name).mkdir(parents=True,exist_ok=True)
        self.historical={r['id']:r for p in (ROOT/'artifacts/tessfold_qwen3vl_short_extension_20261115').glob('predictions.rank*.jsonl') for r in lines(p)}

    def features(self,row):
        key=row['document_id'];path=CACHE/'features'/f'{key}.pt'
        if key in self.memory:
            f=self.memory.pop(key);self.memory[key]=f;return f
        if path.exists():f=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert f['source']==FEATURE_SOURCE
        else:
            original=self.engine.prepare(row)
            f=dict(x=original['x'].cpu(),deep=[v.cpu() for v in original['deep']],shape=original['shape'],source=FEATURE_SOURCE)
            check=self.engine.from_features(row,f)
            assert torch.equal(check['ids'],original['ids']) and torch.equal(check['positions'],original['positions']),'Cached prompt reconstruction differs'
            atomic_save(path,f)
        self.memory[key]=f
        while len(self.memory)>self.entries:self.memory.popitem(last=False)
        return f

    def get(self,row,need_logits=True):
        data=self.engine.from_features(row,self.features(row));key=hashlib.sha256(row['id'].encode()).hexdigest()
        path=CACHE/'teachers'/f'{key}.pt'
        if path.exists():
            t=torch.load(path,map_location='cpu',weights_only=True);assert t['source']==FEATURE_SOURCE and torch.equal(t['prompt'],data['ids'].cpu())
        else:
            ids=self.historical[row['id']]['generated_ids'] if row['id'] in self.historical else self.engine.greedy(data)
            t=dict(id=row['id'],prompt=data['ids'].cpu(),response=ids,prediction=self.engine.decode(ids),source=FEATURE_SOURCE)
            atomic_save(path,t)
        logp=None
        if need_logits:
            lp=CACHE/'logp'/f'{key}.pt'
            if lp.exists():logp=torch.load(lp,map_location='cpu',weights_only=True,mmap=True)['logp'].to(self.engine.device)
            else:
                with torch.no_grad():logp=self.engine.logits(data,t['response']).log_softmax(-1).detach()
                atomic_save(lp,dict(logp=logp.cpu(),source=FEATURE_SOURCE,response=t['response']))
        return data,t,logp


def summary(rows,prediction_key='prediction'):
    by={}
    for task in sorted({r['task'] for r in rows}):
        rr=[dict(r,prediction=r[prediction_key]) for r in rows if r['task']==task]
        by[task]=due_field_f1(rr)['score_percent'] if task in ['KLC','DeepForm'] else 100*sum(native_score(r['prediction'],r) for r in rr)/len(rr)
    return dict(macro=sum(by.values())/len(by),by_task=by)


def loss(engine,router,row,data,ratio):
    d,t,teacher=data.get(row)
    p,reps=router(d['x'],d['shape'],d['question'],ratio)
    logits=engine.logits(d,t['response'],p,reps)
    kl=(teacher.exp()*(teacher-logits.log_softmax(-1))).sum(-1).mean().clamp_min(0)
    return kl,d,t,p,reps


@torch.no_grad()
def evaluate(engine,router,valid,data,ratio,group,grank,gworld,out,label):
    module=router.module if hasattr(router,'module') else router;module.eval();rows=[]
    for row in valid[grank::gworld]:
        kl,d,t,p,reps=loss(engine,module,row,data,ratio)
        ids=engine.greedy(d,p,reps)
        rows.append(dict(row,prediction=engine.decode(ids),generated_ids=ids,teacher_prediction=t['prediction'],kl=float(kl),n=len(d['x']),m=len(p)))
    allrows=[None]*gworld;dist.all_gather_object(allrows,rows,group=group);rows=[r for rr in allrows for r in rr]
    assert len(rows)==len(valid) and len({r['id'] for r in rows})==len(rows)
    result=dict(stage=label,questions=len(rows),kl=sum(r['kl'] for r in rows)/len(rows),**summary(rows),teacher=summary(rows,'teacher_prediction'))
    if grank==0:
        save(out/f'{label}.summary.json',result)
        (out/f'{label}.predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda r:r['id'])))
        print(json.dumps(dict(evaluation=result,run=out.name)),flush=True)
    module.train();return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='qwen_pilot1');parser.add_argument('--limit',type=int,default=512);parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--configs',default='Column:.25,Tree:.25,Column:.5,Tree:.5');parser.add_argument('--lr',type=float,default=3e-4);parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--gate-root');a=parser.parse_args()
    rank,world,local,device=init_runtime('npu',4);configs=[(s.split(':')[0],float(s.split(':')[1])) for s in a.configs.split(',')]
    assert world%len(configs)==0;gworld=world//len(configs);assert a.batch%gworld==0
    groups=[dist.new_group(list(range(i*gworld,(i+1)*gworld))) for i in range(len(configs))]
    index=rank//gworld;group=groups[index];grank=rank%gworld;kind,ratio=configs[index]
    if a.limit>1024:
        assert a.gate_root,'Full training requires an explicit completed multi-image pilot'
        pilot=load(ROOT/a.gate_root/f'{kind}_r{ratio:g}'/'complete.json')
        assert pilot.get('pilot_gate_passed'),f'Pilot gate failed for {kind} {ratio}'
    template=load(ROOT/'artifacts/compression_lab_v2/configs/full_quadtree_question_pre-merger_r0.5.json')
    train=balanced(lines(ROOT/template['train_rows']),a.limit);valid=lines(ROOT/template['validation_rows'])
    assert not {r['document_id'] for r in train}&{r['document_id'] for r in valid}
    base=BASE/a.label;base.mkdir(parents=True,exist_ok=True);out=base/f'{kind}_r{ratio:g}';out.mkdir(exist_ok=True)
    sources=['vqa/tessfold_qwen3vl_stable.py','vqa/tessfold_qwen3vl.py','vqa/tessfold_native_partition.py','vqa/tessfold_partition_native.cpp',str(Path(__file__).relative_to(ROOT))]
    manifest=dict(kind=kind,ratio=ratio,seed=20261115,allocation='fixed_image_detail_only;learn_local_weights_and_temperature',train_questions=len(train),train_images=len({r['document_id'] for r in train}),validation_questions=len(valid),epochs=a.epochs,batch=a.batch,lr=a.lr,clip=5.,weight_decay=0.,
        scheduler='10percent linear warmup then cosine floor0.1',loss='full-vocabulary teacher greedy-prefix KL only; no gold training',
        insertion='native post-merger; same P on final features and all3DeepStack branches; representative MRoPE retained',
        source_sha256={s:sha(ROOT/s) for s in sources},data_sha256={k:sha(ROOT/template[k]) for k in ['train_rows','validation_rows']},world=world,subgroup=gworld,backbone_trainable=0,
        pilot_gate='Final/best dev KL <=0.95initial and dev macro >=initial+0.25pp, or >=teacher-0.25pp if initial already at teacher; never choose on formal test.',
        gate_root=a.gate_root,gate_sha256=sha(ROOT/a.gate_root/f'{kind}_r{ratio:g}'/'complete.json') if a.gate_root else None)
    if grank==0:
        if (out/'manifest.json').exists():assert load(out/'manifest.json')==manifest,'Resume signature changed'
        else:
            save(out/'manifest.json',manifest)
            for s in sources:
                p=out/'sources'/s;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/s,p)
        if index==0:(base/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train))
    engine=NativeQwen(device);data=Data(engine,384 if a.limit<=1024 else 64)
    for i,row in enumerate((train+valid)[rank::world]):
        data.get(row)
        if i%16==0:print(json.dumps(dict(cache_rank=rank,done=i+1)),flush=True)
    dist.barrier();torch.manual_seed(20261115)
    module=NativeSpatialRouter(kind).to(device);router=DDP(module,device_ids=[local],process_group=group,find_unused_parameters=True)
    optimizer=torch.optim.AdamW(module.parameters(),lr=a.lr,weight_decay=0.)
    steps_per_epoch=math.ceil(len(train)/a.batch);total=steps_per_epoch*a.epochs;first=0
    if (out/'latest.pt').exists():
        c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);module.load_state_dict(c['router']);optimizer.load_state_dict(c['optimizer']);first=c['step']
    else:
        if grank==0:atomic_save(out/'initial.pt',dict(router=module.state_dict(),step=0,manifest=manifest))
    initial=evaluate(engine,router,valid,data,ratio,group,grank,gworld,out,'initial') if first==0 else load(out/'initial.summary.json')
    if grank==0:
        previous=lines(out/'training.jsonl');(out/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in previous if r['step']<=first))
    started=time.perf_counter()
    for step in range(first,total):
        epoch=step//steps_per_epoch;order=list(range(len(train)));random.Random(20261115+epoch).shuffle(order)
        batch=order[(step%steps_per_epoch)*a.batch:(step%steps_per_epoch+1)*a.batch]
        indices=batch[grank::gworld];assert len(indices)>0
        warm=max(1,round(total*.1));factor=(step+1)/warm if step<warm else .1+.9*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2
        for g in optimizer.param_groups:g['lr']=a.lr*factor
        optimizer.zero_grad(set_to_none=True);values=[]
        for pos,ix in enumerate(indices):
            context=router.no_sync() if pos<len(indices)-1 else __import__('contextlib').nullcontext()
            with context:
                kl,d,t,p,reps=loss(engine,router,train[ix],data,ratio);assert torch.isfinite(kl)
                (kl/len(indices)).backward();values.append(float(kl.detach()));del kl,d,t,p,reps
        norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),5.));assert math.isfinite(norm)
        optimizer.step();assert all(p.grad is None for p in engine.model.parameters())
        value=torch.tensor(sum(values)/len(values),device=device);dist.all_reduce(value,group=group);value/=gworld
        r=dict(step=step+1,epoch=epoch+1,kl=float(value),lr=a.lr*factor,gradient_norm=norm,clip_multiplier=min(1.,5./(norm+1e-6)),seconds=time.perf_counter()-started)
        if grank==0:
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            print(json.dumps(dict(run=out.name,**r)),flush=True)
        if (step+1)%50==0 or (step+1)%steps_per_epoch==0:
            if grank==0:atomic_save(out/'latest.pt',dict(router=module.state_dict(),optimizer=optimizer.state_dict(),step=step+1,manifest=manifest))
        if (step+1)%steps_per_epoch==0:
            current=evaluate(engine,router,valid,data,ratio,group,grank,gworld,out,f'epoch{epoch+1}')
            if grank==0:atomic_save(out/f'epoch{epoch+1}.pt',dict(router=module.state_dict(),step=step+1,manifest=manifest))
    if grank==0:
        candidates=[load(p) for p in out.glob('epoch*.summary.json')]
        eligible=[r for r in candidates if r['kl']<=initial['kl']*.95 and (r['macro']>=initial['macro']+.25 or (initial['macro']>=initial['teacher']['macro']-.25 and r['macro']>=initial['teacher']['macro']-.25))]
        best=max(eligible,key=lambda r:(r['macro'],-r['kl'])) if eligible else None
        save(out/'complete.json',dict(time=now(),pilot_gate_passed=bool(eligible),initial=initial,best=best,final=current,backbone_trainable=0))
    dist.barrier()
    if rank==0:save(base/'complete.json',dict(time=now(),runs=[load(base/f'{k}_r{r:g}'/'complete.json') for k,r in configs]))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
