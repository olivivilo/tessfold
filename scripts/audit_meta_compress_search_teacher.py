"""Fresh common-prefix and greedy checks against frozen 20k teacher caches."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import hashlib
import json
import torch
from scripts.train_meta_compress_search import ROOT,CACHE,read
from scripts.train_meta_compress import prepare_example,teacher_logits_for,atomic_json
from scripts.eval_token_proxy import generate
from vqa.official_und import OfficialUND
from vqa.decoder_dependent_loss import common_prefix_logits


def main():
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));torch.set_num_threads(8)
    selected=[];seen=set()
    for row in read(ROOT/'artifacts/meta_compress_search/pilot.jsonl'):
        if row['task'] not in seen:seen.add(row['task']);selected.append(row)
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False);results=[]
    for row in selected[rank::world]:
        prompt,raw,grid,x,response=prepare_example(engine,row,CACHE,64)
        old=teacher_logits_for(engine,row,prompt,x,response,CACHE)
        with torch.no_grad():fresh=common_prefix_logits(engine.model,prompt,x[None],response)
        logp=old.log_softmax(-1);logq=fresh.log_softmax(-1)
        kl=float((logp.exp()*(logp-logq)).sum(-1).mean())
        inp={'input_ids':prompt,'attention_mask':torch.ones_like(prompt),'pixel_values':raw.new_empty(0),'image_grid_thw':grid}
        answer,rt=generate(engine,inp,64,x,raw)
        teacher=torch.load(CACHE/'teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt'),map_location='cpu',weights_only=True)
        results.append({'id':row['id'],'task':row['task'],'cached_answer':teacher['prediction'],'fresh_answer':answer,
            'cached_vs_fresh_logit_equal':bool(torch.equal(old,fresh)),
            'max_abs_logit_difference':float((old-fresh).abs().max()),'identity_teacher_kl':kl,
            'greedy_ids_equal':rt['generated_ids']==response,'current_cpu_threads':8})
    atomic_json(ROOT/f'artifacts/meta_compress_search/teacher_runtime_audit.rank{rank}.json',results)


if __name__=='__main__':main()
