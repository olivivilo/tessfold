"""Frozen UND supplementary training math and diagnostic evaluation."""
import os
import torch
from vqa.official_und import ROOT
from vqa.supplement_generators import build
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score
from scripts.train_compression_lab import Data, macro
from scripts.train_compression_lab_v2 import read
from scripts.train_meta_compress import atomic_json
from scripts.eval_token_proxy import generate, shortened_inputs

BASE=ROOT/'artifacts/compression_supplement_20261111'
SOURCES=['vqa/supplement_generators.py','scripts/compression_supplement_math.py',
         'scripts/train_compression_supplement.py','vqa/meta_compress.py',
         'vqa/compression_generators_v2.py','vqa/compression_generators.py',
         'vqa/official_und.py','vqa/decoder_dependent_loss.py','vqa/benchmark_metrics.py',
         'scripts/train_compression_lab.py','scripts/train_meta_compress.py',
         'scripts/train_meta_compress_strict.py','scripts/eval_token_proxy.py',
         'scripts/compression_lab_v2_io.py','scripts/compression_lab_v2_io_retry.py']


def student(engine,module,row,data,cfg):
    prompt,raw,grid,x,response,tl,tp,q=data.get_all(row)
    if cfg['method']=='M':c,p=module(x.float(),grid,cfg['ratio']);sp=c.to(raw.dtype)
    elif cfg['insertion']=='pre-merger':
        _,p=module(torch.zeros_like(x,dtype=torch.float32),grid,cfg['ratio'],raw=raw,question=q)
        sp=engine.model.vision_tower.merger((p@raw.float()).to(raw.dtype))
    else:
        c,p=module(x.float(),grid,cfg['ratio'],raw=raw,question=q);sp=c.to(raw.dtype)
    ids=shortened_inputs({'input_ids':prompt},len(p),engine.model.config.image_token_id)['input_ids']
    sl=common_prefix_logits(engine.model,ids,sp[None],response)
    logt=tl.detach().float().log_softmax(-1);logs=sl.float().log_softmax(-1)
    kl=(logt.exp()*(logt-logs)).sum(-1).mean()
    entropy=-(p.float()*p.float().clamp_min(1e-12).log()).sum(-1).mean()
    collapse=p.float().sum(0).max()
    total=kl+entropy+collapse if cfg['method']=='M' else kl
    return dict(kl=kl,total=total,entropy=entropy,collapse=collapse),(p,sp,ids,raw,grid,tp,response)


def evaluate(engine,module,rows,data,cfg,out,stage,cohort='validation',do_generate=True):
    path=out/f'{cohort}.{stage}.jsonl';items=read(path);done={r['id'] for r in items}
    if path.exists():path.write_text(''.join(__import__('json').dumps(r,ensure_ascii=False)+'\n' for r in items))
    module.eval()
    with torch.no_grad(),path.open('a') as f:
        for row in rows:
            if row['id'] in done:continue
            losses,(p,sp,ids,raw,grid,tp,response)=student(engine,module,row,data,cfg)
            pred='';rt={}
            if do_generate:pred,rt=generate(engine,{'input_ids':ids,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,sp,raw)
            item=dict(id=row['id'],document_id=row['document_id'],task=row['task'],question=row['question'],answers=row['answers'],
                      stage=stage,cohort=cohort,student_prediction=pred if do_generate else None,teacher_prediction=tp,
                      student_score=float(native_score(pred,row)) if do_generate else None,teacher_score=float(native_score(tp,row)),
                      **{k:float(v) for k,v in losses.items()},tokens=p.shape[1],compressed_tokens=p.shape[0],
                      row_sum_error=float((p.sum(-1)-1).abs().max()),min_weight=float(p.min()),
                      hit_token_limit=rt.get('hit_token_limit'),student_ids=rt.get('generated_ids'))
            assert item['min_weight']>=0 and item['row_sum_error']<1e-4
            f.write(__import__('json').dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush();items.append(item)
    assert len(items)==len(rows) and len(done|{r['id'] for r in items})==len(rows)
    summary=dict(rows=len(items),teacher_macro=100*macro(items,'teacher_score'),
                 **{k:sum(x[k] for x in items)/len(items) for k in ['kl','total','entropy','collapse']})
    if do_generate:summary['student_macro']=100*macro(items,'student_score')
    atomic_json(out/f'{cohort}.{stage}.summary.json',summary);module.train();return summary
