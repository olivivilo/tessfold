"""Native8B manual path equivalence and differentiable all-branch compression."""
import json,os
import torch
import torch.distributed as dist
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha
from vqa.tessfold_qwen3vl import NativeQwen,NativeSpatialRouter
from vqa.benchmark_metrics import native_score

def main():
    rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115/qwen_smoke1';base.mkdir(exist_ok=True)
    engine=NativeQwen(device);rows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')[:8][rank::world]
    historical={r['id']:r for p in (ROOT/'artifacts/tessfold_qwen3vl_short_extension_20261115').glob('predictions.rank*.jsonl') for r in lines(p)}
    results=[]
    for row in rows:
        data=engine.prepare(row);ids=engine.greedy(data);assert ids==historical[row['id']]['generated_ids'],'Native identity manual decode differs from stock HF'
        teacher=engine.logits(data,ids).detach().log_softmax(-1)
        for kind in ['Column','Tree']:
            router=NativeSpatialRouter(kind).to(device)
            for ratio in [.25,.5]:
                p,reps=router(data['x'],data['shape'],data['question'],ratio)
                student=engine.logits(data,ids,p,reps).log_softmax(-1)
                kl=(teacher.exp()*(teacher-student)).sum(-1).mean()
                kl.backward();norm=float(torch.nn.utils.clip_grad_norm_(router.parameters(),1.0));router.zero_grad(set_to_none=True)
                pred=engine.decode(engine.greedy(data,p,reps));r=dict(id=row['id'],kind=kind,ratio=ratio,n=len(data['x']),m=len(p),kl=float(kl),gradient_norm=norm,score=native_score(pred,row),prediction=pred,native_identity_equal=True,
                    all_backbone_frozen=all(not x.requires_grad for x in engine.model.parameters()))
                results.append(r);print(json.dumps(r),flush=True)
                del p,student,kl
    save(base/f'rows.rank{rank}.json',results);dist.barrier()
    if rank==0:save(base/'complete.json',dict(rows=sum(len(load(base/f'rows.rank{k}.json',[])) for k in range(world)),native_identity_passed=True))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
