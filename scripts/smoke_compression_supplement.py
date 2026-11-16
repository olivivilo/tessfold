"""Real-model technical preflight, independently recorded before all full jobs."""
import argparse
import json
import math
import os
import torch
import torch.distributed as dist
from PIL import Image
from scripts.supplement_common import ROOT,BASE,verify,lines,load,save,legacy
from scripts.train_meta_compress import init_runtime,prepare_example
from scripts.compression_supplement_math import Data,student
from scripts.eval_token_proxy import generate
from scripts.eval_compression_supplement import question_embedding,predict_cached
from vqa.official_und import OfficialUND
from vqa.supplement_generators import build,score_parts,make_groups,weights,intervention,allocate
from vqa.supplement_s2prune import s2_generate


def main():
    verify();rank,world,_,device=init_runtime('npu',4)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);data=Data(engine)
    rows=lines(BASE/'smoke_rows.jsonl');mine=rows[rank::world];out=BASE/'smoke';out.mkdir(exist_ok=True)
    cache=ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache';results=[]
    for row in mine:
        prompt,raw,grid,x,response=prepare_example(engine,row,cache,256);q=question_embedding(engine,row)
        for method in ['T','C','V','J','M']:
            torch.manual_seed(20261105);cfg=dict(method=method,insertion='post-merger' if method=='M' else 'pre-merger',ratio=.25)
            module=build(cfg).to(device)
            # A train-row teacher uses cap64, distinct from formal cache.
            trainrow=lines(ROOT/'artifacts/compression_lab_v2/full_train.jsonl')[rank]
            loss,_=student(engine,module,trainrow,data,cfg)
            loss['total'].backward();norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.))
            assert math.isfinite(norm) and norm>0
            assert all(p.grad is None and not p.requires_grad for p in engine.model.parameters())
            if method in ['T','C']:
                source=torch.zeros_like(x);_,p=module(source,grid,.25,raw=raw,question=q)
                s,_,e,shape=score_parts(module,source,raw,grid,q)
                g=make_groups(e,shape,len(p),mode='tree' if method=='T' else 'col')
                pp,_=weights(module,s,g,len(p));assert torch.equal(p,pp),'Supplement helper differs from signed legacy'
                if method=='T':
                    pp,_=intervention(module,raw,x,grid,q,q,.25,'QQ');assert torch.equal(p,pp)
            results.append(dict(id=row['id'],method=method,gradient_norm=norm,loss=float(loss['total']),pass_gradient=True))
            del module,loss
        # Same-model no-prune decoder regression (16 independent generated tokens).
        pred,rt=generate(engine,{'input_ids':prompt,'pixel_values':raw.new_empty(0),'image_grid_thw':grid},16,x,raw)
        with Image.open(ROOT/row['image']) as im:
            full,manual=s2_generate(engine,prompt,x,grid,im.convert('RGB'),1.,16)
            short,pruned=s2_generate(engine,prompt,x,grid,im.convert('RGB'),.25,16)
        assert rt['generated_ids']==manual['generated_ids'],f'S2 full-path regression failed: {row["id"]}'
        assert pruned['suffix_visual_tokens']==max(1,round(len(raw)*.25))
        # Fixed original T produces identical QQ generation, not just similar logits.
        cfg=dict(method='QSWAP',insertion='pre-merger',ratio=.25)
        module=build(dict(method='T')).to(device).eval()
        ck=torch.load(ROOT/legacy('quadtree_question',.25)/'epoch4.pt',map_location='cpu',weights_only=False);module.load_state_dict(ck['generator'])
        cfg_ref=dict(method='T',insertion='pre-merger',ratio=.25)
        _,a,_,_,_=predict_cached(engine,module,row,cfg_ref,'trained',prompt,raw,grid,x,cap=16)
        _,b,_,_,_=predict_cached(engine,module,row,cfg,'QQ',prompt,raw,grid,x,qwrong=q,cap=16)
        assert a['generated_ids']==b['generated_ids']
        results.append(dict(id=row['id'],s2_full_regression=True,s2_pruned_decode_steps=len(pruned['generated_ids']),qq_production_regression=True))
        del module
    save(out/f'rank{rank}.json',results);dist.barrier()
    if rank==0:
        rr=[r for rank in range(world) for r in load(out/f'rank{rank}.json',[])]
        save(out/'complete.json',dict(passed=True,questions=len(rows),rows=rr,world=world,job=os.environ.get('SLURM_JOB_ID')))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
