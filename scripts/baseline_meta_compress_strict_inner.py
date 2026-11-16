"""Identity/gradient audit and fixed controls for frozen-merger inner boundaries."""
import torch
import torch.distributed as dist
from scripts import train_meta_compress_strict_inner as inner
from scripts.train_meta_compress_strict import BASE,Examples,evaluate,ROOT
from scripts.train_meta_compress import init_runtime,atomic_json
from scripts.report_meta_compress_strict import read
from scripts.baseline_meta_compress_strict import Fixed
from vqa.official_und import OfficialUND
from vqa.decoder_dependent_loss import common_prefix_logits

def main():
    rank,world,_,device=init_runtime('npu',8)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);examples=Examples(engine,'npu')
    row=read(BASE/'pilot.jsonl')[rank];prompt,raw,grid,x,response,teacher,tp=examples.get(row)
    merger=engine.model.vision_tower.merger;checks=[]
    for insertion in ['post-norm','post-gelu']:
        with torch.no_grad():
            source=merger.ln_q(raw)
            if insertion=='post-gelu':source=merger.mlp[1](merger.mlp[0](source))
            identical=merger.mlp(source) if insertion=='post-norm' else merger.mlp[2](source)
            logits=common_prefix_logits(engine.model,prompt,identical[None],response)
            direct=common_prefix_logits(engine.model,prompt,x[None],response)
            error=float((direct-logits).abs().max())
            assert error==0,(insertion,error)
        cfg={'seed':20261101,'insertion':insertion,'initialization':'fourier-dense-position',
             'position_strength':128.,'projection_gain':.125,'ratio':.5}
        module=inner.initialize(cfg).to(device)
        losses,details=inner.student(engine,module,row,examples,cfg)
        grads=torch.autograd.grad(losses['prediction'],tuple(module.parameters()),retain_graph=True)
        norm=float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
        assert norm>0 and all(torch.isfinite(g).all() for g in grads)
        losses['total'].backward();p=details[1]
        checks.append({'position':insertion,'id':row['id'],'identity_logits_max_abs_diff':error,
            'prediction_gradient_norm':norm,'full_loss':float(losses['total'].detach()),
            'row_sum_error':float((p.detach().sum(-1)-1).abs().max()),
            'backbone_trainable_parameters':sum(p.numel() for p in engine.model.parameters() if p.requires_grad)})
        del module,losses,details,grads
    atomic_json(BASE/f'inner_audit.rank{rank}.json',checks)
    rows=read(BASE/'validation.jsonl')
    for insertion in ['post-norm','post-gelu']:
        for ratio in [.5,.75]:
            for kind in ['mean','uniform-selection']:
                suffix='' if ratio==.5 else f'_ratio{ratio:g}'
                cfg={'insertion':insertion,'ratio':ratio};out=BASE/'baselines'/f'{insertion}_{kind}{suffix}';out.mkdir(parents=True,exist_ok=True)
                evaluate(engine,Fixed(kind),rows,examples,cfg,out,'fixed','validation',rank,world)
                if rank==0:atomic_json(out/'complete.json',{'rows':len(rows),'ratio':ratio,'insertion':insertion,'kind':kind})
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
