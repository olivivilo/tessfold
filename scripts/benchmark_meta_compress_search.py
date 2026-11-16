"""Fixed original 512-question benchmark, no benchmark-based model selection."""
import argparse
import json
import os
import time
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
from scripts.train_meta_compress_search import read,load_module,ROOT
from scripts.train_meta_compress import feature_for,sha,atomic_json
from scripts.baseline_meta_compress_search import BalancedMean
from scripts.eval_token_proxy import generate,shortened_inputs
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.meta_compress import meta_losses
from vqa.benchmark_metrics import native_score
from vqa.official_und import OfficialUND


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--smoke',action='store_true',help='One question per task for execution audit only, separate output; not model selection')
    p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));torch.set_num_threads(a.threads)
    cfg=json.loads((ROOT/a.config).read_text());out=ROOT/cfg['output']/('benchmark_smoke' if a.smoke else 'benchmark');out.mkdir(parents=True,exist_ok=True)
    old=ROOT/'artifacts/meta_compress/eval';rows=read(old/'samples.benchmark.jsonl')
    teacher={}
    for f in old.glob('predictions.benchmark.rank*.jsonl'):
        for r in read(f):
            value={k:r[k] for k in ['teacher_prediction','teacher_generated_ids','teacher_score','teacher_hit_token_limit']}
            if r['id'] in teacher:assert teacher[r['id']]==value
            teacher[r['id']]=value
    assert {r['id'] for r in rows}==set(teacher)
    assert json.loads((old/'protocol.benchmark.json').read_text())['device']=='cpu'
    if a.smoke:
        chosen=[];seen=set()
        for row in rows:
            if row['task'] not in seen:chosen.append(row);seen.add(row['task'])
        rows=chosen
    sources=['scripts/benchmark_meta_compress_search.py','vqa/meta_compress_tuning.py','vqa/meta_compress.py',
             'vqa/official_und.py','vqa/decoder_dependent_loss.py','vqa/benchmark_metrics.py']
    signature={'config':cfg,'checkpoint':a.checkpoint,'checkpoint_sha256':sha(ROOT/a.checkpoint),
               'rows_sha256':sha(old/'samples.benchmark.jsonl'),'world':world,'threads':a.threads,
               'max_new_tokens':256,'smoke':a.smoke,'actual_samples':len(rows),
               'teacher_source':'fresh CPU BF16 official UND independent greedy generation per question; old full IDs retained only for audit',
               'teacher_protocol_sha256':sha(old/'protocol.benchmark.json'),
               'code_sha256':{f:sha(ROOT/f) for f in sources}}
    manifest=out/f'manifest.rank{rank}.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    atomic_json(manifest,signature)
    if rank==0:
        (out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        for f in sources:
            target=out/'source_snapshots'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/f).read_bytes())
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    modules={'initial':load_module(cfg),'trained':load_module(cfg,ROOT/a.checkpoint),'mean':BalancedMean()}
    for module in modules.values():module.eval()
    path=out/f'predictions.rank{rank}.jsonl';done={(r['id'],r['variant']) for r in read(path)} if path.exists() else set()
    tick=time.monotonic();teacher_checked=0;teacher_equal=0
    cache=ROOT/'artifacts/meta_compress_eval_cache'
    with torch.no_grad(),path.open('a') as h:
        for i,row in enumerate(rows[rank::world]):
            if all((row['id'],v) in done for v in modules):continue
            inputs=engine.inputs(row);raw,grid,x=feature_for(engine,inputs,row,cache)
            previous=teacher[row['id']]
            answer,rt=generate(engine,inputs,256,x,raw);response=rt['generated_ids']
            equal=response==previous['teacher_generated_ids'];teacher_checked+=1;teacher_equal+=int(equal)
            tr={'teacher_prediction':answer,'teacher_generated_ids':response,
                'teacher_score':float(native_score(answer,row)),'teacher_hit_token_limit':rt['hit_token_limit'],
                'original_teacher_ids_equal':equal,'original_teacher_score':previous['teacher_score']}
            tl=common_prefix_logits(engine.model,inputs['input_ids'],x[None],response)
            for variant,module in modules.items():
                if (row['id'],variant) in done:continue
                compressed,pmat=module(x.float(),grid,cfg['ratio'])
                si=shortened_inputs(inputs,len(compressed),engine.model.config.image_token_id);sp=compressed.to(raw.dtype)
                answer,rt=generate(engine,si,256,sp,raw)
                sl=common_prefix_logits(engine.model,si['input_ids'],sp[None],response);loss=meta_losses(pmat,tl,sl)
                record={'id':row['id'],'task':row['task'],'document_id':row['document_id'],'question':row['question'],
                    'answers':row['answers'],'variant':variant,**tr,'student_prediction':answer,
                    'student_score':float(native_score(answer,row)),'student_generated_ids':rt['generated_ids'],
                    'student_hit_token_limit':rt['hit_token_limit'],'tokens':len(x),'compressed_tokens':len(compressed),
                    'ratio':cfg['ratio'],**{k:float(v) for k,v in loss.items()},
                    'objective':float(loss['prediction']+cfg['entropy_weight']*loss['entropy']+cfg['collapse_weight']*loss['collapse']),
                    'column_mass_fraction':float(loss['collapse'])/len(pmat),'row_sum_error':float((pmat.sum(-1)-1).abs().max())}
                h.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');h.flush()
            print(json.dumps({'rank':rank,'done':i+1,'seconds':time.monotonic()-tick}),flush=True)
    atomic_json(out/f'teacher_check.rank{rank}.json',{'fresh_generation_questions_this_attempt':teacher_checked,'exact_old_ids_equal_this_attempt':teacher_equal,
        'note':'Resumed questions are checked using original_teacher_ids_equal in their preserved prediction record.'})
    atomic_json(out/f'complete.rank{rank}.json',{'samples':len(rows[rank::world]),'variants':list(modules),'world':world})


if __name__=='__main__':main()
