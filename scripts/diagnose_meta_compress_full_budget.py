"""Hold sequence length/positions fixed: dense generator at keep_ratio=1.

Identity teacher vs P@X has equal token budget. This isolates content mixing
from shortening, on both DocVQA cohort images and the ChartQA image.
"""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import json
import math
import torch
from scripts.diagnose_meta_compress import OUT,CACHE
from scripts.train_meta_compress import prepare_example,teacher_logits_for,atomic_json,sha
from scripts.eval_token_proxy import shortened_inputs,generate
from vqa.official_und import OfficialUND,ROOT
from vqa.meta_compress import MetaCompress,meta_losses
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.benchmark_metrics import native_score


def main():
    torch.set_num_threads(16)
    out=OUT/'full_budget';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    assert not (out/'records.jsonl').exists(),'Explicit restart needed'
    atomic_json(out/'protocol.json',{'code_sha256':sha(__file__),'job':os.environ.get('SLURM_JOB_ID'),
        'ratio':1.,'generator':'original initial','tasks':['DocVQA','ChartQA'],'training':False})
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    generator=MetaCompress();generator.load_state_dict(torch.load(ROOT/'artifacts/meta_compress/initial.pt',weights_only=False,map_location='cpu')['generator'])
    records=[]
    with torch.no_grad():
        for row in [json.loads(x) for x in (OUT/'samples.jsonl').read_text().splitlines()]:
            if row['task'] not in ['DocVQA','ChartQA']:continue
            prompt,raw,grid,x,response=prepare_example(engine,row,CACHE,64)
            tl=teacher_logits_for(engine,row,prompt,x,response,CACHE)
            c,p=generator(x.float(),grid,1.)
            si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
            assert torch.equal(si['input_ids'],prompt) and len(c)==len(x)
            sl=common_prefix_logits(engine.model,prompt,c.to(raw.dtype)[None],response)
            losses=meta_losses(p,tl,sl)
            pred,rt=generate(engine,{**si,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},64,c.to(raw.dtype),raw)
            item={'id':row['id'],'task':row['task'],'role':row['role'],'tokens':len(x),'compressed_tokens':len(c),
                'teacher_prediction':row['diagnostic_teacher_prediction'],'teacher_score':1.,
                'student_prediction':pred,'student_score':float(native_score(pred,row)),
                **{k:float(v) for k,v in losses.items()},'entropy_fraction':float(losses['entropy'])/math.log(len(x)),
                'prompt_and_position_length_unchanged':True,'hit_token_limit':rt['hit_token_limit']}
            records.append(item)
            with (out/'records.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
            print(json.dumps(item,ensure_ascii=False),flush=True)
    assert len(records)==3
    atomic_json(out/'complete.json',{'records':3})


if __name__=='__main__':main()
