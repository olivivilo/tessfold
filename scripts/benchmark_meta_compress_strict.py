"""Final 23,565-QA evaluation only after frozen validation/confirmation selection."""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
import torch
import torch.distributed as dist
from scripts.train_meta_compress_strict import BASE,ROOT,read
from scripts.train_meta_compress import init_runtime,barrier,prepare_example,teacher_logits_for,atomic_json,sha
from scripts.baseline_meta_compress_strict import Fixed
from scripts.eval_token_proxy import generate,shortened_inputs
from vqa.official_und import OfficialUND
from vqa.meta_compress import meta_losses
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score

def build(cfg):
    kind=cfg['initialization']
    if kind=='pca-whitened':from vqa.meta_compress_whitened_initialization import initialize
    elif kind=='pca-pool-null':from vqa.meta_compress_pool_null_initialization import initialize
    elif kind=='fourier-dense-position':from vqa.meta_compress_dense_position_initialization import initialize
    elif kind=='multiscale-position':from vqa.meta_compress_multiscale_position_initialization import initialize
    else:from vqa.meta_compress_initialization import initialize
    return initialize(cfg)

def project(engine,module,raw,grid,x,cfg):
    merger=engine.model.vision_tower.merger;position=cfg['insertion']
    if position=='pre-merger':source=raw
    elif position=='post-merger':source=x
    else:
        source=merger.ln_q(raw)
        if position=='post-gelu':source=merger.mlp[1](merger.mlp[0](source))
        elif position!='post-norm':raise ValueError(position)
    c,p=module(source.float(),grid,cfg['ratio']);sp=c.to(raw.dtype)
    if position=='pre-merger':sp=merger(sp)
    elif position=='post-norm':sp=merger.mlp(sp)
    elif position=='post-gelu':sp=merger.mlp[2](sp)
    return sp,p

def final_report(out,rows,cfg):
    from scripts.report_meta_compress_strict import paired
    records=[r for f in out.glob('predictions.rank*.jsonl') for r in read(f)]
    expected={(r['id'],v) for r in rows for v in ['initial','trained','mean']}
    assert {(r['id'],r['variant']) for r in records}==expected
    assert len(records)==len(expected)
    grouped=defaultdict(list)
    for r in records:grouped[(r['variant'],r['task'])].append(r)
    mean=lambda values:sum(values)/len(values)
    tasks=sorted({r['task'] for r in rows});stats={}
    for task in tasks:
        variants={}
        for v in ['initial','trained','mean']:
            rr=grouped[(v,task)]
            variants[v]={'n':len(rr),'score':100*mean([r['student_score'] for r in rr]),
                'KL':mean([r['prediction'] for r in rr]),'total':mean([r['total'] for r in rr]),
                'truncated':sum(r['hit_token_limit'] for r in rr)}
        stats[task]={'teacher':100*mean([r['teacher_score'] for r in grouped[('trained',task)]]),**variants}
    macro={v:mean([stats[t][v]['score'] for t in tasks]) for v in ['initial','trained','mean']}
    macro['teacher']=mean([stats[t]['teacher'] for t in tasks])
    uncertainty=paired([r for r in records if r['variant']=='initial'],[r for r in records if r['variant']=='trained'])
    result={'questions':len(rows),'variant_records':len(records),'tasks':stats,'macro':macro,
            'paired_trained_vs_initial':uncertainty,'config':cfg,'benchmark_selection_used':False,
            'max_new_tokens':256,'metric_limitation':'DeepForm/KLC field exact, not corpus document F1; native mixed-task macro is descriptive.'}
    atomic_json(out/'summary.json',result)
    fields=['id','document_id','task','question','answers','variant','teacher_prediction','student_prediction',
            'teacher_score','student_score','prediction','entropy','collapse','total','tokens','compressed_tokens','hit_token_limit']
    with (out/'answers.csv').open('w') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    lines=['# 严格 MetaCompress：完整训练后的独立全量评测','',
        f'实际完成 {len(rows):,} QA，{len(records):,} 条压缩变体记录。未压缩 teacher、初始生成器、训练后生成器和 mean pooling 全部在本轮 NPU BF16 上重新独立生成，cap=256。',
        '评测只在家族/epoch 固定之后运行，不用本表选择 adapter、超参数或 checkpoint。训练/调参的 cap=64，与本表不同。','',
        'DeepForm/KLC 为逐字段 exact，DT-VQA 延用既有 scorer；不同任务宏平均不是新的官方统一榜单分数。','',
        '| 任务 | QA | teacher | 初始 | 训练后 | mean | 训练增益 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t,s in stats.items():lines.append(f"| {t} | {s['trained']['n']} | {s['teacher']:.2f} | {s['initial']['score']:.2f} | {s['trained']['score']:.2f} | {s['mean']['score']:.2f} | {s['trained']['score']-s['initial']['score']:+.2f} |")
    lines+=['',f"八任务宏平均：teacher **{macro['teacher']:.2f}**，初始 **{macro['initial']:.2f}**，训练后 **{macro['trained']:.2f}**，mean **{macro['mean']:.2f}**。",
        '',f"训练相对同预算初始化的配对增益：{uncertainty['gain_points']:+.3f} 点，图像聚类 bootstrap 95% CI：{uncertainty['cluster_bootstrap_95ci_points']}。",'',
        '[训练与验证实时结果](METACOMPRESS_STRICT_RESULTS.md) · [逐题作答](../'+str((out/'answers.csv').relative_to(ROOT))+') · [机器统计](../'+str((out/'summary.json').relative_to(ROOT))+')']
    (ROOT/'docs/METACOMPRESS_STRICT_FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=json.loads((ROOT/a.config).read_text());run=ROOT/cfg['output'];out=run/('benchmark_smoke' if a.smoke else 'benchmark');out.mkdir(parents=True,exist_ok=True)
    rank,world,_,device=init_runtime('npu',a.threads)
    rows=[r for r in read(ROOT/'data/full/val.jsonl') if r['task'] in ['DocVQA','InfoVQA']]+read(ROOT/'data/full/test.jsonl')
    assert len(rows)==23565 and len({r['id'] for r in rows})==len(rows)
    assert not ({r['document_id'] for r in rows}&{r['document_id'] for r in read(ROOT/cfg['train_rows'])})
    if a.smoke:
        seen=set();subset=[]
        for r in rows:
            if r['task'] not in seen:subset.append(r);seen.add(r['task'])
        rows=subset
    sources=['scripts/benchmark_meta_compress_strict.py','scripts/train_meta_compress_strict.py',
             'scripts/train_meta_compress.py','scripts/eval_token_proxy.py','vqa/meta_compress.py',
             'vqa/meta_compress_initialization.py','vqa/meta_compress_whitened_initialization.py',
             'vqa/meta_compress_pool_null_initialization.py','vqa/meta_compress_dense_position_initialization.py',
             'vqa/meta_compress_multiscale_position_initialization.py','vqa/official_und.py',
             'vqa/decoder_dependent_loss.py','vqa/benchmark_metrics.py','scripts/baseline_meta_compress_strict.py']
    signature={'config':cfg,'checkpoint_sha256':sha(ROOT/a.checkpoint),'initial_sha256':sha(run/'initial.pt'),
        'rows_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        'world':world,'device':'npu','max_new_tokens':256,'smoke':a.smoke,'code_sha256':{s:sha(ROOT/s) for s in sources}}
    path=out/f'manifest.rank{rank}.json'
    if path.exists():assert json.loads(path.read_text())==signature
    atomic_json(path,signature)
    if rank==0:
        for s in sources:
            dest=out/'source_snapshots'/s;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/s).read_bytes())
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    modules={}
    for v,checkpoint in [('initial',run/'initial.pt'),('trained',ROOT/a.checkpoint)]:
        module=build(cfg);module.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['generator'])
        modules[v]=module.to(device).eval()
    modules['mean']=Fixed('mean').to(device)
    cache=BASE/'benchmark_npu_cache'
    for name in ['features','teachers','logits']:(cache/name).mkdir(parents=True,exist_ok=True)
    output=out/f'predictions.rank{rank}.jsonl';committed=read(output)
    if output.exists():output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in committed))
    done={(r['id'],r['variant']) for r in committed}
    with torch.no_grad(),output.open('a') as f:
        for index,row in enumerate(rows[rank::world]):
            if all((row['id'],v) in done for v in modules):continue
            prompt,raw,grid,x,response=prepare_example(engine,row,cache,256)
            teacher=teacher_logits_for(engine,row,prompt,x,response,cache)
            record=torch.load(cache/'teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True)
            tp=record['prediction']
            for variant,module in modules.items():
                if (row['id'],variant) in done:continue
                sp,pmat=project(engine,module,raw,grid,x,cfg)
                si=shortened_inputs({'input_ids':prompt},len(sp),engine.model.config.image_token_id)
                pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},256,sp,raw)
                sl=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
                losses=meta_losses(pmat,teacher,sl)
                item={'id':row['id'],'document_id':row['document_id'],'task':row['task'],
                    'question':row['question'],'answers':row['answers'],'variant':variant,
                    'teacher_prediction':tp,'student_prediction':pred,'teacher_score':float(native_score(tp,row)),
                    'student_score':float(native_score(pred,row)),**{k:float(v) for k,v in losses.items()},
                    'tokens':len(raw),'compressed_tokens':len(sp),'ratio':cfg['ratio'],
                    'hit_token_limit':rt['hit_token_limit'],'teacher_hit_token_limit':len(response)==256,
                    'student_generated_ids':rt['generated_ids'],'teacher_generated_ids':response}
                f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            if index%25==0:print({'rank':rank,'done':index+1,'total':len(rows[rank::world])},flush=True)
    atomic_json(out/f'complete.rank{rank}.json',{'rows':len(rows[rank::world])})
    barrier(world)
    if rank==0:
        if not a.smoke:final_report(out,rows,cfg)
        atomic_json(out/'complete.json',{'questions':len(rows),'variants':list(modules),'world':world,'smoke':a.smoke})
    barrier(world)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
