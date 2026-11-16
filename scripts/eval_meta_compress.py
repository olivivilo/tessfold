"""Independent UND teacher/student generations; fresh UND raw features only.

--split benchmark: DocVQA/InfoVQA official validation, other 6 official tests.
--variants trained,initial,mean: matched evaluation, without test model selection.
RANK/WORLD_SIZE shard deterministic unique-image samples across CPU workers.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
import time
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from vqa.meta_compress import MetaCompress, meta_losses
from vqa.official_und import ROOT, OfficialUND
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score
from scripts.eval_token_proxy import generate, shortened_inputs
from scripts.train_meta_compress import atomic_json, atomic_save, read_rows, feature_for, sha


def choose(rows, limit):
    counts=defaultdict(int); docs=defaultdict(set); result=[]
    for row in sorted(rows,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()):
        task=row['task']
        if row['document_id'] in docs[task] or (limit and counts[task]>=limit):continue
        docs[task].add(row['document_id']); counts[task]+=1; result.append(row)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='artifacts/meta_compress/meta_compress.pt')
    p.add_argument('--initial',default='artifacts/meta_compress/initial.pt')
    p.add_argument('--output',default='artifacts/meta_compress/eval')
    p.add_argument('--cache',default='artifacts/meta_compress_eval_cache')
    p.add_argument('--split',default='benchmark',choices=['val','test','benchmark'])
    p.add_argument('--limit-per-task',type=int,default=64)
    p.add_argument('--keep-ratio',type=float,default=None)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--device',choices=['cpu','npu'],default='npu')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--variants',default='trained,initial,mean')
    a=p.parse_args(); rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.set_num_threads(a.threads)
    if a.device=='npu':
        import torch_npu
        local=int(os.environ.get('LOCAL_RANK',0));torch.npu.set_device(local)
        torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
        device=f'npu:{local}'
    else:device='cpu'
    out,cache=ROOT/a.output,ROOT/a.cache
    out.mkdir(parents=True,exist_ok=True);(cache/'features').mkdir(parents=True,exist_ok=True)
    (out/'matrices').mkdir(exist_ok=True)
    state=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False)
    cfg=state['config'];ratio=a.keep_ratio if a.keep_ratio is not None else cfg['keep_ratio']
    if a.split=='benchmark':
        rows=[r for r in read_rows('val') if r['task'] in ['DocVQA','InfoVQA']]+read_rows('test')
    else:rows=read_rows(a.split)
    rows=choose(rows,a.limit_per_task)
    train_docs={r['document_id'] for r in read_rows('train')}
    assert not any(r['document_id'] in train_docs for r in rows)
    variants=a.variants.split(',')
    protocol={'checkpoint_sha256':sha(ROOT/a.checkpoint),'split':a.split,'samples':len(rows),
              'limit_per_task':a.limit_per_task,'keep_ratio':ratio,'max_new_tokens':a.max_new_tokens,
              'variants':variants,'device':a.device,'raw_source':'fresh official UND BF16',
              'code_sha256':{f:sha(ROOT/f) for f in ['scripts/eval_meta_compress.py','vqa/meta_compress.py',
                    'vqa/official_und.py','vqa/decoder_dependent_loss.py','vqa/benchmark_metrics.py']},
              'sample_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
    manifest=out/f'run.{a.split}.rank{rank}.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise RuntimeError('Evaluation restart changed checkpoint/protocol; choose a new output directory')
    atomic_json(manifest,protocol)
    if rank==0:
        atomic_json(out/f'protocol.{a.split}.json',protocol)
        (out/f'samples.{a.split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        for name in protocol['code_sha256']:
            snapshot=out/'source_snapshots'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes((ROOT/name).read_bytes())
    engine=OfficialUND(device=device);engine.model.eval().requires_grad_(False)
    generator=MetaCompress(feature_dim=cfg['feature_dim']).to(device);generator.load_state_dict(state['generator']);generator.eval()
    initial=None
    if 'initial' in variants:
        init=torch.load(ROOT/a.initial,map_location='cpu',weights_only=False)
        initial=MetaCompress(feature_dim=cfg['feature_dim']).to(device);initial.load_state_dict(init['generator']);initial.eval()
    path=out/f'predictions.{a.split}.rank{rank}.jsonl'
    done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():r=json.loads(line);done.add((r['id'],r['variant']))
    start=time.monotonic()
    with torch.no_grad(),path.open('a') as handle:
        for i,row in enumerate(rows[rank::world]):
            if all((row['id'],v) in done for v in variants):continue
            inputs=engine.inputs(row)
            raw,grid,projected=feature_for(engine,inputs,row,cache)
            teacher_prediction,trt=generate(engine,inputs,a.max_new_tokens,projected,raw)
            response=trt['generated_ids'];tscore=float(native_score(teacher_prediction,row))
            tl=common_prefix_logits(engine.model,inputs['input_ids'],projected[None],response)
            source=projected if cfg['insertion']=='post-merger' else raw
            for variant in variants:
                if (row['id'],variant) in done:continue
                if variant in ['trained','initial']:
                    module=generator if variant=='trained' else initial
                    compressed,pmat=module(source.float(),grid,ratio)
                elif variant=='mean':
                    n=len(raw);m=MetaCompress.output_length(n,ratio)
                    # Balanced contiguous mean pool, strictly row stochastic.
                    groups=(torch.arange(n,device=device)*m//n).long()
                    pmat=torch.nn.functional.one_hot(groups,m).T.float()
                    pmat=pmat/pmat.sum(-1,keepdim=True)
                    compressed=pmat@source.float()
                else:raise ValueError(variant)
                sp=(compressed.to(raw.dtype) if cfg['insertion']=='post-merger'
                    else engine.model.vision_tower.merger(compressed.to(raw.dtype)))
                si=shortened_inputs(inputs,len(compressed),engine.model.config.image_token_id)
                prediction,srt=generate(engine,si,a.max_new_tokens,sp,compressed.to(raw.dtype))
                sl=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
                losses=meta_losses(pmat,tl,sl);sscore=float(native_score(prediction,row))
                record={'id':row['id'],'variant':variant,'task':row['task'],'document_id':row['document_id'],
                        'question':row['question'],'answers':row['answers'],
                        'teacher_prediction':teacher_prediction,'student_prediction':prediction,
                        'teacher_score':tscore,'student_score':sscore,'score_drop':tscore-sscore,
                        **{k+'_loss':float(v) for k,v in losses.items()},
                        'tokens':len(raw),'compressed_tokens':len(compressed),
                        'teacher_generated_ids':response,'student_generated_ids':srt['generated_ids'],
                        'teacher_hit_token_limit':trt['hit_token_limit'],'student_hit_token_limit':srt['hit_token_limit'],
                        'p_row_sum_error':float((pmat.sum(-1)-1).abs().max())}
                if i==0 and variant=='trained':
                    matrix_path=out/'matrices'/f'{a.split}.rank{rank}.pt'
                    atomic_save(matrix_path,{'P':pmat.cpu(),'grid':grid.cpu(),'document_id':row['document_id'],
                                             'checkpoint_sha256':protocol['checkpoint_sha256'],'source_dim':cfg['feature_dim']})
                    record['matrix_file']=str(matrix_path.relative_to(ROOT))
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
                done.add((row['id'],variant))
            print(json.dumps({'rank':rank,'done':i+1,'total':len(rows[rank::world]),'elapsed_seconds':time.monotonic()-start}),flush=True)
    atomic_json(out/f'complete.{a.split}.rank{rank}.json',{'samples':len(rows[rank::world]),'variants':variants,'seconds':time.monotonic()-start})

if __name__=='__main__':main()
