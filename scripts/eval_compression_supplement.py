"""Independent full-cohort generation for feedback baselines and interventions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scripts.supplement_common import ROOT,BASE,verify,load,save,lines,save_lines,sha,benchmark_rows,summary
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.eval_token_proxy import generate,shortened_inputs
from vqa.official_und import OfficialUND
from vqa.benchmark_metrics import native_score
from vqa.supplement_generators import build,intervention,grid_star
from vqa.supplement_s2prune import s2_generate


def question_embedding(engine,row):
    ids=engine.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(engine.device)
    return engine.model.get_input_embeddings()(ids)


def resized_inputs(engine,row,original_n,ratio,image=None):
    from qwen_vl_utils.vision_process import smart_resize
    image=image or Image.open(ROOT/row['image']).convert('RGB')
    width,height=image.size;k=max(1,round(original_n*ratio));shapes=set()
    for target in {max(16,k+4*d) for d in range(-32,33)}:
        hh,ww=smart_resize(height,width,factor=28,min_pixels=3136,max_pixels=min(1003520,target*196))
        shapes.add((hh,ww))
    hh,ww=min(shapes,key=lambda x:(abs(x[0]*x[1]//196-k),abs(math.log((x[1]/x[0])/(width/height))),x[0]*x[1],x))
    resized=image.resize((ww,hh),Image.Resampling.BICUBIC)
    messages=[{'role':'user','content':[{'type':'image','image':resized,'max_pixels':1003520},{'type':'text','text':row['question']}]}]
    text=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    images,videos=process_vision_info(messages)
    batch=engine.processor(text=text,images=images,videos=videos,padding=True,return_tensors='pt')
    inputs={k:v.to(engine.device) for k,v in batch.items()}
    actual=int((inputs['input_ids']==engine.model.config.image_token_id).sum())
    assert actual==int(inputs['image_grid_thw'].prod())
    return inputs,dict(original_size=[width,height],resized_size=[ww,hh],target_tokens=k,
                       actual_tokens=actual,budget_error=actual-k,budget_abs_error=abs(actual-k),
                       actual_grid=inputs['image_grid_thw'].cpu().tolist())


def load_module(engine,cfg,checkpoint=None):
    spec=cfg.get('generator_config',cfg)
    module=build(spec).to(engine.device).eval()
    path=ROOT/(checkpoint or cfg['checkpoint'])
    state=torch.load(path,map_location='cpu',weights_only=False)
    module.load_state_dict(state['generator']);return module


def predict_cached(engine,module,row,cfg,variant,prompt,raw,grid,x,qwrong=None,cap=256):
    q=question_embedding(engine,row);details={};p=None
    method=cfg['method']
    if method=='R':
        inp,details=resized_inputs(engine,row,len(raw),cfg['ratio'])
        pred,rt=generate(engine,inp,cap);m=details['actual_tokens']
        return pred,rt,None,m,details
    if method=='S':
        with Image.open(ROOT/row['image']) as image:
            pred,rt=s2_generate(engine,prompt,x,grid,image.convert('RGB'),cfg['ratio'],cap)
        return pred,rt,None,rt['suffix_visual_tokens'],{k:v for k,v in rt.items() if k not in ['generated_ids']}
    if method=='Gstar':p,details=grid_star(module,raw,x,grid,q,cfg['ratio'])
    elif method in ['HA','QSWAP']:
        p,details=intervention(module,raw,x,grid,q,qwrong if qwrong is not None else q,cfg['ratio'],variant)
    elif method=='M':_,p=module(x.float(),grid,cfg['ratio'])
    else:
        source=torch.zeros_like(x,dtype=torch.float32) if cfg['insertion']=='pre-merger' else x.float()
        _,p=module(source,grid,cfg['ratio'],raw=raw,question=q)
    sp=(engine.model.vision_tower.merger((p@raw.float()).to(raw.dtype)) if cfg['insertion']=='pre-merger' else (p@x.float()).to(raw.dtype))
    ids=shortened_inputs({'input_ids':prompt},len(sp),engine.model.config.image_token_id)['input_ids']
    pred,rt=generate(engine,{'input_ids':ids,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},cap,sp,raw)
    return pred,rt,p,len(sp),details


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    reg=verify();cfg=load(ROOT/args.config);out=ROOT/cfg['output']/('smoke_eval' if args.smoke else 'benchmark');out.mkdir(parents=True,exist_ok=True)
    rank,world,_,device=init_runtime('npu',4)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    rows=lines(ROOT/cfg['eval_rows']) if cfg.get('eval_rows') else benchmark_rows()
    if args.smoke:rows=lines(BASE/'smoke_rows.jsonl')
    signature=dict(config=cfg,world=world,rows_sha256=sha(ROOT/(cfg.get('eval_rows') or reg['benchmark_rows'])),source_sha256=reg['source_sha256'],
                   checkpoint_sha256=sha(ROOT/cfg['checkpoint']) if cfg.get('checkpoint') else None,cap=16 if args.smoke else 256)
    manifest=out/f'manifest.rank{rank}.json'
    if manifest.exists():assert load(manifest)==signature
    else:save(manifest,signature)
    module=load_module(engine,cfg) if cfg['method'] not in ['R','S'] else None
    variants=cfg.get('variants',['trained']);path=out/f'predictions.rank{rank}.jsonl';items=lines(path)
    save_lines(path,items);done={(r['id'],r['variant']) for r in items}
    mapping=load(BASE/'same_page_mapping.json');byid={r['id']:r for r in benchmark_rows()}
    cache=ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache'
    routing=out/'routing';routing.mkdir(exist_ok=True)
    with torch.no_grad(),path.open('a') as handle:
        for count,row in enumerate(rows[rank::world]):
            if all((row['id'],v) in done for v in variants):continue
            prompt,raw,grid,x,response=prepare_example(engine,row,cache,256)
            tp=torch.load(cache/'teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True)['prediction']
            qwrong=None
            if cfg['method']=='QSWAP':qwrong=question_embedding(engine,byid[mapping[row['id']]])
            for variant in variants:
                if (row['id'],variant) in done:continue
                pred,rt,p,m,trace=predict_cached(engine,module,row,cfg,variant,prompt,raw,grid,x,qwrong,16 if args.smoke else 256)
                if 'leaf_ids' in trace:
                    filename=hashlib.sha256((row['id']+variant).encode()).hexdigest()+'.npz'
                    np.savez_compressed(routing/filename,leaf_ids=np.asarray(trace.pop('leaf_ids'),dtype=np.int32),leaf_order=np.asarray(trace.pop('leaf_order'),dtype=np.int32))
                    trace['routing_file']=str((routing/filename).relative_to(ROOT))
                record=dict(id=row['id'],qa_id=row['id'],document_id=row['document_id'],image_id=row['image'],task=row['task'],
                            question=row['question'],answers=row['answers'],variant=variant,method=cfg['method'],ratio=cfg['ratio'],seed=cfg.get('seed'),
                            student_prediction=pred,student_score=float(native_score(pred,row)),teacher_prediction=tp,teacher_score=float(native_score(tp,row)),
                            tokens=len(raw),target_tokens=max(1,round(len(raw)*cfg['ratio'])),compressed_tokens=m,
                            first_layer_visual_tokens=len(raw) if cfg['method']=='S' else m,suffix_visual_tokens=m,
                            student_ids=rt['generated_ids'],hit_token_limit=rt['hit_token_limit'],
                            row_sum_error=float((p.sum(-1)-1).abs().max()) if p is not None else None,
                            wrong_question_id=mapping.get(row['id']) if cfg['method']=='QSWAP' else None,trace=trace)
                assert all(v.grad is None and not v.requires_grad for v in engine.model.parameters())
                if cfg['method']!='R':assert m==record['target_tokens']
                if p is not None:assert record['row_sum_error']<1e-4 and float(p.min())>=0
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();del p
            if count%100==0:print(json.dumps(dict(method=cfg['method'],rank=rank,questions=count+1,total=len(rows[rank::world]))),flush=True)
    dist.barrier()
    if rank==0:
        allrows=[r for f in sorted(out.glob('predictions.rank*.jsonl')) for r in lines(f)]
        assert {(r['id'],r['variant']) for r in allrows}=={(r['id'],v) for r in rows for v in variants}
        assert len(allrows)==len(rows)*len(variants)
        for r in allrows:assert abs(native_score(r['student_prediction'],byid[r['id']])-r['student_score'])<1e-12
        stats={v:summary([r for r in allrows if r['variant']==v]) for v in variants}
        save(out/'summary.json',dict(questions=len(rows),records=len(allrows),variants=stats,config=cfg,benchmark_used_for_selection=False))
        with (out/'answers.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
        save(out/'complete.json',dict(questions=len(rows),records=len(allrows),world=world,job=os.environ.get('SLURM_JOB_ID')))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
