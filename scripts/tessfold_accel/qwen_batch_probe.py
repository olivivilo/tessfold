"""Compare same objective/parameters/QA before adopting microbatch training."""
import json,os,time
import torch
import torch.distributed as dist
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha
from scripts.tessfold_accel.qwen_training import Data
from vqa.tessfold_qwen3vl import NativeQwen
from vqa.tessfold_qwen3vl_batch import BatchSpatialRouter,batch_kl


def gradient(module):
    return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in module.parameters()])

def main():
    rank,world,_,device=init_runtime('npu',4);meta_group=dist.new_group(backend='gloo');base=ROOT/'artifacts/tessfold_acceleration_20261115/qwen_microbatch_probe';base.mkdir(exist_ok=True)
    engine=NativeQwen(device);data=Data(engine);rows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl')
    pairs=[rows[rank*2:rank*2+2],rows[8+rank*2:10+rank*2]];results=[]
    torch.manual_seed(20261115);module=BatchSpatialRouter('Tree').to(device)
    for rr in pairs:
        samples=[data.get(r) for r in rr];examples=[x[0] for x in samples];teachers=[x[2] for x in samples];responses=[x[1]['response'] for x in samples]
        def sequential():
            module.zero_grad(set_to_none=True);losses=[]
            for d,t,teacher in samples:
                p,reps=module(d['x'],d['shape'],d['question'],.25)
                logp=engine.logits(d,t['response'],p,reps).log_softmax(-1)
                kl=(teacher.exp()*(teacher-logp)).sum(-1).mean().clamp_min(0)
                (kl/len(rr)).backward();losses.append(float(kl.detach()))
            return losses,gradient(module).clone()
        def batched():
            module.zero_grad(set_to_none=True);visuals=module(examples,.25)
            losses=batch_kl(engine,examples,visuals,teachers,responses);(sum(losses)/len(rr)).backward()
            return [float(x.detach()) for x in losses],gradient(module).clone()
        sequential();batched();torch.npu.synchronize()
        measurements=[]
        for fn in [sequential,batched]:
            torch.npu.reset_peak_memory_stats();torch.npu.synchronize();start=time.perf_counter()
            losses,g=fn();torch.npu.synchronize();elapsed=(time.perf_counter()-start)*1e3
            measurements.append((losses,g,elapsed,int(torch.npu.max_memory_allocated())))
        aa,bb=measurements
        cosine=float(torch.nn.functional.cosine_similarity(aa[1],bb[1],dim=0))
        result=dict(ids=[r['id'] for r in rr],sequence_lengths=[len(d['ids'][0]) for d in examples],sequential_kl=aa[0],batched_kl=bb[0],
            mean_kl_abs_difference=abs(sum(aa[0])-sum(bb[0]))/len(rr),gradient_cosine=cosine,
            sequential_ms=aa[2],batched_ms=bb[2],speedup=aa[2]/bb[2],sequential_peak_gib=aa[3]/2**30,batched_peak_gib=bb[3]/2**30,
            all_backbone_grad_none=all(p.grad is None for p in engine.model.parameters()))
        assert cosine>.98 and result['mean_kl_abs_difference']<.05 and result['all_backbone_grad_none'],result
        results.append(result);print(json.dumps(result),flush=True)
    save(base/f'rows.rank{rank}.json',results);dist.barrier(group=meta_group)
    if rank==0:save(base/'complete.json',dict(pairs=2*world,all_objective_checks_passed=True))
    dist.barrier(group=meta_group);dist.destroy_process_group()

if __name__=='__main__':main()
