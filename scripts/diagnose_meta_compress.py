"""Isolated causal diagnostics, never modifies the completed MetaCompress run.

Entrypoints: --prepare; --variant N; --report (see Slurm wrapper).
Single-image overfit is a debugging experiment, not a benchmark or model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from torch import nn
from vqa.official_und import ROOT, OfficialUND
from vqa.meta_compress import MetaCompress, meta_losses
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score
from scripts.train_meta_compress import prepare_example, teacher_logits_for, atomic_json, atomic_save, sha
from scripts.eval_token_proxy import generate, shortened_inputs

OUT = ROOT/'artifacts/meta_compress_diagnosis'
CACHE = ROOT/'artifacts/meta_compress_cache'
VARIANTS = [
    dict(name='original_all_sgd_clip001', kind='generator', objective='total', optimizer='sgd', lr=.001, clip=.01, ratio=.1),
    dict(name='all_sgd_clip1', kind='generator', objective='total', optimizer='sgd', lr=.001, clip=1., ratio=.1),
    dict(name='all_sgd_no_clip', kind='generator', objective='total', optimizer='sgd', lr=.001, clip=None, ratio=.1),
    dict(name='kl_sgd_no_clip', kind='generator', objective='prediction', optimizer='sgd', lr=.001, clip=None, ratio=.1),
    dict(name='kl_adam_clip1', kind='generator', objective='prediction', optimizer='adam', lr=.0003, clip=1., ratio=.1),
    dict(name='all_adam_clip1', kind='generator', objective='total', optimizer='adam', lr=.0003, clip=1., ratio=.1),
    dict(name='free_dense_kl_adam', kind='free', objective='prediction', optimizer='adam', lr=.03, clip=1., ratio=.1),
    dict(name='local_bias_kl_adam', kind='local', objective='prediction', optimizer='adam', lr=.0003, clip=1., ratio=.1),
    dict(name='kl_adam_keep025', kind='generator', objective='prediction', optimizer='adam', lr=.0003, clip=1., ratio=.25),
    dict(name='kl_adam_keep050', kind='generator', objective='prediction', optimizer='adam', lr=.0003, clip=1., ratio=.5),
]


def cachepath(sub, row):
    return CACHE/sub/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Protocol already exists: preserve it; use existing cohort')
    rows = [json.loads(x) for x in (ROOT/'artifacts/meta_compress/samples.jsonl').read_text().splitlines()]
    # Metadata from completed run: avoid loading 20k large feature tensors.
    token_counts = {}
    for path in sorted((ROOT/'artifacts/meta_compress').glob('examples.rank*.jsonl')):
        for line in path.read_text().splitlines():
            item=json.loads(line); token_counts[item['id']]=item['tokens']
    selected=[]; counts={}; docs=set()
    for row in sorted(rows, key=lambda r: hashlib.sha256(('diagnosis42:'+r['id']).encode()).hexdigest()):
        task=row['task']; need=2 if task=='DocVQA' else 1
        if counts.get(task,0)>=need or row['document_id'] in docs: continue
        if token_counts[row['id']]>(5200 if task in ('KLC','DeepForm') else 4096): continue
        teacher=torch.load(cachepath('teachers',row),map_location='cpu',weights_only=True)
        if len(teacher['generated_ids'])>12: continue
        if teacher['prediction'].strip().lower() in ('none','null','not found','n/a',''): continue
        if native_score(teacher['prediction'],row)!=1: continue
        item={**row,'diagnostic_teacher_prediction':teacher['prediction'],
              'diagnostic_tokens':token_counts[row['id']],
              'role':'fit' if task=='DocVQA' and counts.get(task,0)==0 else 'holdout'}
        selected.append(item);docs.add(row['document_id']);counts[task]=counts.get(task,0)+1
        if len(selected)==8: break
    assert len(selected)==8 and len({r['task'] for r in selected})==7, counts
    fit=[r for r in selected if r['role']=='fit']; assert len(fit)==1
    assert not ({r['document_id'] for r in selected}&{json.loads(x)['document_id'] for s in ['val','test'] for x in (ROOT/f'data/full/{s}.jsonl').read_text().splitlines()})
    (OUT/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    sources=['scripts/diagnose_meta_compress.py','vqa/meta_compress.py','vqa/decoder_dependent_loss.py','scripts/eval_token_proxy.py','scripts/train_meta_compress.py','vqa/official_und.py','vqa/benchmark_metrics.py']
    protocol={'variants':VARIANTS,'seed':20261026,'steps':50,'model':'official MonkeyOCRv2-B-Und, frozen BF16',
        'insertion':'post-merger','fit_images':1,'holdout_images':7,
        'selection':'original 20k training rows only; hash diagnosis42; <=4096 tokens (KLC/DeepForm <=5200); <=12 response tokens; teacher score=1; exclude null answers; image-disjoint; not representative benchmark',
        'max_new_tokens':64,'training_target':'teacher greedy common-prefix full-vocabulary forward KL, EOS included',
        'original_initial_sha256':sha(ROOT/'artifacts/meta_compress/initial.pt'),
        'original_final_sha256':sha(ROOT/'artifacts/meta_compress/meta_compress.pt'),
        'samples_sha256':sha(OUT/'samples.jsonl'),'code_sha256':{s:sha(ROOT/s) for s in sources}}
    for s in sources:
        dest=OUT/'source_snapshots'/s;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/s).read_bytes())
    atomic_json(OUT/'protocol.json',protocol)
    print(json.dumps(selected,ensure_ascii=False,indent=2))


class DiagnosticModule(nn.Module):
    def __init__(self, cfg, source, grid):
        super().__init__(); self.cfg=cfg
        initial=torch.load(ROOT/'artifacts/meta_compress/initial.pt',map_location='cpu',weights_only=False)
        self.generator=MetaCompress();self.generator.load_state_dict(initial['generator'])
        if cfg['kind']=='free':
            with torch.no_grad(): _,p=self.generator(source.float(),grid,cfg['ratio'])
            self.logits=nn.Parameter(p.clamp_min(1e-30).log())
            self.generator.requires_grad_(False)

    def forward(self, x, grid):
        kind=self.cfg['kind']; ratio=self.cfg['ratio']
        if kind=='free': p=self.logits.softmax(-1)
        else:
            _,p=self.generator(x.float(),grid,ratio)
            if kind=='local':
                # Experimental modification: balanced contiguous support bias,
                # NOT the published MetaCompress generator. Log floor 1e-6.
                m,n=p.shape; groups=torch.arange(n)*m//n
                local=(groups[None,:]==torch.arange(m)[:,None])
                p=(p.clamp_min(1e-30).log()+(~local)*math.log(1e-6)).softmax(-1)
        return p@x.float(),p


def evaluate(engine, module, examples, stage, cfg, out):
    records=[]
    with torch.no_grad():
        for row,prompt,raw,grid,source,response,teacher in examples:
            if cfg['kind']=='free' and row['role']!='fit':continue
            c,p=module(source,grid)
            sp=c.to(raw.dtype);si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
            logits=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
            losses=meta_losses(p,teacher,logits)
            # Pixel values are unused by the patched tower, but kwargs trigger
            # the native image injection path. Fresh generate call each time.
            inp={**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid}
            prediction,rt=generate(engine,inp,64,sp,c.to(raw.dtype))
            h=float(losses['entropy'])
            item={'id':row['id'],'document_id':row['document_id'],'task':row['task'],'role':row['role'],
                'stage':stage,'question':row['question'],'answers':row['answers'],
                'teacher_prediction':row['diagnostic_teacher_prediction'],'teacher_score':1.,
                'student_prediction':prediction,'student_score':float(native_score(prediction,row)),
                'tokens':len(source),'compressed_tokens':len(c),'hit_token_limit':rt['hit_token_limit'],
                **{k:float(v) for k,v in losses.items()},'entropy_fraction':h/math.log(len(source)),
                'effective_support':math.exp(h),'row_max_mean':float(p.max(-1).values.mean()),
                'row_sum_error':float((p.sum(-1)-1).abs().max())}
            records.append(item)
            with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    return records


def gradients(engine,module,example):
    row,prompt,raw,grid,source,response,tl=example
    c,p=module(source,grid)
    si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
    sl=common_prefix_logits(engine.model,si['input_ids'],c.to(raw.dtype)[None],response)
    losses=meta_losses(p,tl,sl);params=[v for v in module.parameters() if v.requires_grad]
    grads={}
    for k in ['prediction','entropy','collapse']:
        g=torch.autograd.grad(losses[k],params,retain_graph=True,allow_unused=True)
        grads[k]=torch.cat([(torch.zeros_like(v) if a is None else a).reshape(-1).float() for a,v in zip(g,params)])
    result={'losses':{k:float(v.detach()) for k,v in losses.items()},
        'norms':{k:float(v.norm()) for k,v in grads.items()},
        'cosines':{a+'__'+b:float(torch.nn.functional.cosine_similarity(grads[a],grads[b],dim=0)) for a,b in [('prediction','entropy'),('prediction','collapse'),('entropy','collapse')]},
        'answer_tokens':response,'kl_by_answer_position':((tl.float().log_softmax(-1).exp())*(tl.float().log_softmax(-1)-sl.detach().float().log_softmax(-1))).sum(-1).flatten().tolist()}
    return result


def interface_audit(engine,example):
    row,prompt,raw,grid,source,response,tl=example
    with torch.no_grad():
        direct=common_prefix_logits(engine.model,prompt,source[None],response)
        # Actual P=I oracle; tests the same matrix multiply/cast/prompt path.
        eye=torch.eye(len(source)); ident=(eye@source.float()).to(raw.dtype)
        si=shortened_inputs({'input_ids':prompt},len(source),engine.model.config.image_token_id)
        il=common_prefix_logits(engine.model,si['input_ids'],ident[None],response)
        losses=meta_losses(eye,tl,il)
        inp=engine.inputs(row)
        native,nrt=generate(engine,inp,64)
        cached,crt=generate(engine,inp,64,source,raw)
        # Independent causal-prefix oracle on real model, first/last positions.
        positions=sorted({0,len(response)-1}); errors=[]
        for i in positions:
            prefix=torch.cat([prompt,torch.tensor([response[:i]],dtype=prompt.dtype)],-1)
            one=common_prefix_logits(engine.model,prefix,source[None],[response[i]])
            errors.append({'position':i,'max_abs':float((one[0,0]-direct[0,i]).abs().max()),
                'argmax_equal':int(one[0,0].argmax())==int(direct[0,i].argmax())})
        result={'cached_logits_max_abs':float((tl-direct).abs().max()),'identity_logits_max_abs':float((tl-il).abs().max()),
            'identity_KL':float(losses['prediction']),'native_prediction':native,'cached_prediction':cached,
            'native_cached_generated_ids_equal':nrt['generated_ids']==crt['generated_ids'],
            'native_cached_feature_max_abs':None,'causal_response_alignment':errors,
            'base_trainable_parameters':sum(v.numel() for v in engine.model.parameters() if v.requires_grad),
            'fit_image':row['id']}
        native_features,native_raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
        result['native_cached_feature_max_abs']=float((native_features-source).abs().max())
        result['native_cached_raw_max_abs']=float((native_raw-raw).abs().max())
        assert torch.equal(prompt,inp['input_ids'])
        assert result['identity_KL']==0 and result['cached_logits_max_abs']==0
        assert result['native_cached_generated_ids_equal']
    return result


def run(index, steps, threads):
    torch.set_num_threads(threads);torch.manual_seed(20261026)
    protocol=json.loads((OUT/'protocol.json').read_text());cfg=protocol['variants'][index]
    assert cfg==VARIANTS[index]
    assert sha(OUT/'samples.jsonl')==protocol['samples_sha256']
    for name,digest in protocol['code_sha256'].items():
        assert sha(ROOT/name)==digest, f'Code changed after protocol freeze: {name}'
    out=OUT/cfg['name'];out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():print('Already complete');return
    if (out/'training.jsonl').exists():raise RuntimeError('Partial run: preserve and choose explicit restart policy')
    atomic_json(out/'config.json',{**cfg,'steps':steps,'job':os.environ.get('SLURM_JOB_ID')})
    start=time.monotonic();engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    examples=[]
    rows=[json.loads(x) for x in (OUT/'samples.jsonl').read_text().splitlines()]
    for row in rows:
        prompt,raw,grid,source,response=prepare_example(engine,row,CACHE,64)
        tl=teacher_logits_for(engine,row,prompt,source,response,CACHE)
        examples.append((row,prompt,raw,grid,source,response,tl))
    fit=next(e for e in examples if e[0]['role']=='fit')
    module=DiagnosticModule(cfg,fit[4],fit[3])
    parameters=[v for v in module.parameters() if v.requires_grad]
    initial={n:v.detach().clone() for n,v in module.named_parameters() if v.requires_grad}
    optimizer=(torch.optim.SGD if cfg['optimizer']=='sgd' else torch.optim.Adam)(parameters,lr=cfg['lr'])
    if index==0:atomic_json(OUT/'interface_audit.json',interface_audit(engine,fit))
    ga=gradients(engine,module,fit);atomic_json(out/'gradients.initial.json',ga)
    if index==0:
        original=DiagnosticModule(cfg,fit[4],fit[3]); ck=torch.load(ROOT/'artifacts/meta_compress/meta_compress.pt',map_location='cpu',weights_only=False)
        original.generator.load_state_dict(ck['generator'])
        atomic_json(OUT/'gradients.original_final.json',gradients(engine,original,fit))
        del original
    evaluate(engine,module,examples,'initial',cfg,out)
    row,prompt,raw,grid,source,response,tl=fit
    for step in range(1,steps+1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        c,p=module(source,grid);sp=c.to(raw.dtype)
        si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
        sl=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
        losses=meta_losses(p,tl,sl);losses[cfg['objective']].backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['clip'] if cfg['clip'] is not None else float('inf'))
        assert math.isfinite(float(norm)) and all(math.isfinite(float(v.detach())) for v in losses.values())
        before=[v.detach().clone() for v in parameters];optimizer.step()
        delta=math.sqrt(sum(float((v.detach()-b).square().sum()) for v,b in zip(parameters,before)))
        item={'step':step,**{k:float(v.detach()) for k,v in losses.items()},'gradient_norm':float(norm),
            'update_norm':delta,'clip_multiplier':min(1.,cfg['clip']/(float(norm)+1e-6)) if cfg['clip'] is not None else 1.,
            'compressed_feature_norm':float(c.detach().norm()),'seconds':time.monotonic()-tick}
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps({'variant':cfg['name'],**item}),flush=True)
        del sl,losses,p,c,sp
        if step in (10,25,50):
            atomic_save(out/'checkpoint.pt',{'module':module.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':cfg})
    atomic_json(out/'gradients.final.json',gradients(engine,module,fit))
    evaluate(engine,module,examples,'final',cfg,out)
    changes={n:{'relative_change':float((v.detach()-initial[n]).norm()/initial[n].norm()),
                 'absolute_change':float((v.detach()-initial[n]).norm())} for n,v in module.named_parameters() if v.requires_grad}
    atomic_json(out/'parameter_changes.json',changes)
    atomic_json(out/'complete.json',{'steps':steps,'seconds':time.monotonic()-start,'fit_id':row['id'],
        'base_trainable_parameters':sum(v.numel() for v in engine.model.parameters() if v.requires_grad)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--variant',type=int)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--threads',type=int,default=16)
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.variant is not None:run(a.variant,a.steps,a.threads)
    else:p.error('Use --prepare or --variant N')


if __name__=='__main__':main()
