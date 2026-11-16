"""Eight training-only images: per-component gradient audit across seven tasks."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import json
import torch
from scripts.diagnose_meta_compress import OUT,CACHE,VARIANTS,DiagnosticModule,gradients
from scripts.train_meta_compress import prepare_example,teacher_logits_for,atomic_json,sha
from vqa.official_und import OfficialUND,ROOT


def main():
    torch.set_num_threads(16)
    out=OUT/'task_gradients';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    assert not (out/'records.jsonl').exists(),'Preserve partial run'
    atomic_json(out/'protocol.json',{'code_sha256':sha(__file__),'job':os.environ.get('SLURM_JOB_ID'),
        'cohort':'same eight training-only teacher-correct images; no optimization',
        'states':['original_initial','original_20k_final'],'ratio':.1})
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    records=[]
    for row in [json.loads(x) for x in (OUT/'samples.jsonl').read_text().splitlines()]:
        prompt,raw,grid,x,response=prepare_example(engine,row,CACHE,64)
        tl=teacher_logits_for(engine,row,prompt,x,response,CACHE)
        example=(row,prompt,raw,grid,x,response,tl)
        for state in ['original_initial','original_20k_final']:
            module=DiagnosticModule(VARIANTS[0],x,grid)
            if state.endswith('final'):
                module.generator.load_state_dict(torch.load(ROOT/'artifacts/meta_compress/meta_compress.pt',weights_only=False,map_location='cpu')['generator'])
            record={'id':row['id'],'task':row['task'],'role':row['role'],'state':state,**gradients(engine,module,example)}
            records.append(record)
            with (out/'records.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            del module
    atomic_json(out/'complete.json',{'records':len(records)})


if __name__=='__main__':main()
