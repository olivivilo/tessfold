"""Fixed pool/sample budget controls and BF16 update-resolution audit.

Uses the frozen, training-only cohort of diagnose_meta_compress.py.
"""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import json
import math
import torch
from scripts.diagnose_meta_compress import OUT,CACHE,cachepath
from scripts.train_meta_compress import prepare_example,teacher_logits_for,atomic_json,sha
from scripts.eval_token_proxy import shortened_inputs,generate
from vqa.official_und import OfficialUND,ROOT
from vqa.meta_compress import MetaCompress,meta_losses
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score


def main():
    torch.set_num_threads(16)
    out=OUT/'controls';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    assert not (out/'evaluations.jsonl').exists(),'Preserve partial run before explicit restart'
    atomic_json(out/'protocol.json',{'code_sha256':sha(__file__),'ratios':[.1,.25,.5],
        'methods':['mean','sample'],'sample':'middle token of balanced contiguous groups in native block sequence order',
        'teacher_response_max_new_tokens':64,'student_max_new_tokens':64,'job':os.environ.get('SLURM_JOB_ID')})
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    init=MetaCompress();init.load_state_dict(torch.load(ROOT/'artifacts/meta_compress/initial.pt',weights_only=False,map_location='cpu')['generator'])
    final=MetaCompress();final.load_state_dict(torch.load(ROOT/'artifacts/meta_compress/meta_compress.pt',weights_only=False,map_location='cpu')['generator'])
    audit=[]
    with torch.no_grad():
        for row in [json.loads(x) for x in (OUT/'samples.jsonl').read_text().splitlines()]:
            prompt,raw,grid,x,response=prepare_example(engine,row,CACHE,64)
            tl=teacher_logits_for(engine,row,prompt,x,response,CACHE)
            ci,pi=init(x.float(),grid,.1);cf,pf=final(x.float(),grid,.1)
            zi=ci.to(raw.dtype);zf=cf.to(raw.dtype)
            audit.append({'id':row['id'],'role':row['role'],'n':len(x),
                'original_final_vs_initial_compressed_relative_l2':float((cf-ci).norm()/ci.norm()),
                'bf16_compressed_changed_fraction':float((zi!=zf).float().mean()),
                'raw_mean_norm':float(raw.float().norm(dim=-1).mean()),
                'projected_mean_norm':float(x.float().norm(dim=-1).mean()),
                'dense_compressed_mean_norm':float(ci.norm(dim=-1).mean()),
                'initial_logits_std_mean':float(pi.clamp_min(1e-30).log().std(-1).mean()),
                'initial_logit_range_mean':float((pi.max(-1).values.log()-pi.min(-1).values.clamp_min(1e-30).log()).mean())})
            for ratio in [.1,.25,.5]:
                n=len(x);m=MetaCompress.output_length(n,ratio);groups=torch.arange(n)*m//n
                for method in ['mean','sample']:
                    if method=='mean':
                        p=torch.nn.functional.one_hot(groups,m).T.float();p/=p.sum(-1,keepdim=True)
                    else:
                        centers=((torch.arange(m)+.5)*n/m).long().clamp_max(n-1)
                        p=torch.nn.functional.one_hot(centers,n).float()
                    c=p@x.float();si=shortened_inputs({'input_ids':prompt},m,engine.model.config.image_token_id)
                    sl=common_prefix_logits(engine.model,si['input_ids'],c.to(raw.dtype)[None],response)
                    loss=meta_losses(p,tl,sl)
                    pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,c.to(raw.dtype),raw)
                    item={'id':row['id'],'role':row['role'],'task':row['task'],'method':method,'ratio':ratio,
                        'question':row['question'],'answers':row['answers'],'teacher_prediction':row['diagnostic_teacher_prediction'],
                        'teacher_score':1.,'student_prediction':pred,'student_score':float(native_score(pred,row)),
                        'tokens':n,'compressed_tokens':m,**{k:float(v) for k,v in loss.items()},
                        'hit_token_limit':rt['hit_token_limit']}
                    with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                    print(json.dumps(item,ensure_ascii=False),flush=True)
    atomic_json(out/'feature_precision_audit.json',audit)
    atomic_json(out/'complete.json',{'images':len(audit),'evaluations':len(audit)*6})


if __name__=='__main__':main()
