"""Audit pure-selection commutation and BF16 batch-shape effects on real cases."""
import json
import torch
from scripts.train_meta_compress import init_runtime,atomic_json
from scripts.train_meta_compress_strict import BASE,Examples
from scripts.report_meta_compress_strict import read
from vqa.official_und import OfficialUND
from vqa.decoder_dependent_loss import common_prefix_logits
from scripts.eval_token_proxy import shortened_inputs

def main():
    rank,world,local,device=init_runtime('npu',8)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);examples=Examples(engine,'npu')
    controls=[]
    for pos in ['pre-merger','post-merger']:
        controls.append({r['id']:r for f in (BASE/'baselines'/f'{pos}_uniform-selection').glob('eval.*rank*.jsonl') for r in read(f)})
    aa,bb=controls
    targets=[r for r in read(BASE/'validation.jsonl') if aa[r['id']]['student_score']!=bb[r['id']]['student_score']]
    records=[]
    with torch.no_grad():
        for row in targets:
            prompt,raw,grid,x,response,teacher,tp=examples.get(row)
            m=round(len(raw)*.5);indices=torch.arange(m,device=device)*len(raw)//m
            pre=engine.model.vision_tower.merger(raw[indices]);post=x[indices]
            ids=shortened_inputs({'input_ids':prompt},m,engine.model.config.image_token_id)['input_ids']
            a=common_prefix_logits(engine.model,ids,pre[None],response)
            b=common_prefix_logits(engine.model,ids,post[None],response)
            repeat=common_prefix_logits(engine.model,ids,post[None],response)
            records.append({'id':row['id'],'pre_prediction':aa[row['id']]['student_prediction'],
                'post_prediction':bb[row['id']]['student_prediction'],
                'projected_max_abs_diff':float((pre.float()-post.float()).abs().max()),
                'projected_relative_l2':float((pre.float()-post.float()).norm()/post.float().norm()),
                'projected_equal_fraction':float((pre==post).float().mean()),
                'common_prefix_logits_max_abs_diff':float((a-b).abs().max()),
                'repeat_same_post_logits_max_abs_diff':float((repeat-b).abs().max())})
    atomic_json(BASE/'selection_placement_numerics.json',{'cases':records,
        'selection':'all three validation QA whose fixed-selection pre/post scores differ',
        'not_a_benchmark_or_model_selection':True})
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
