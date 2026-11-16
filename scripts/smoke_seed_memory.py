"""Real UND instrumentation and seed/math equivalence before new full runs."""
import os,json
import numpy as np
import torch
import torch.distributed as dist
from scripts.seed_memory_common import *
from scripts.train_meta_compress import init_runtime
from scripts.profile_seed_memory import load_model,reference_generation,profile_request
from vqa.official_und import OfficialUND
from vqa.stage_memory import StageMemory,cache_nbytes

def main():
 reg=verify();rank,world,_,device=init_runtime('npu',4);assert world==8
 engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
 out=BASE/'smoke';out.mkdir(exist_ok=True);meter=StageMemory(device)
 # Each rank one of2kinds x4ratios; all are existing seed5 models.
 metas=[m for m in reg['models'] if m['seed']==SEEDS[0]];meta=metas[rank]
 module=load_model(engine,meta);rows=lines(BASE/'smoke_rows.jsonl')
 row=rows[rank];expected=reference_generation(engine,module,row,meta,16)
 result=profile_request(engine,module,row,meta,meter,out/f'trace.rank{rank}.npz',cap=16)
 assert result['generated_ids']==expected,'Instrumented generate changed output'
 assert result['decoder_state']['prefill_kv_storage_bytes']>0
 # Known allocation checks peak/average lower than peak and trace captures a plateau.
 meter.reset();meter.begin('allocation_check');v=torch.empty(16*1024**2,device=device,dtype=torch.float32);v.fill_(1)
 import time
 torch.npu.synchronize();time.sleep(.02);del v;torch.npu.synchronize();meter.end()
 values=meter.phases['allocation_check'][0];assert values['peak_allocated_bytes']-values['start_allocated_bytes']>=64*1024**2
 assert values['samples']>=3 and values['time_average_allocated_bytes']>values['start_allocated_bytes']
 # Actual frozen UND train gradient with a fresh random seed6 generator at each budget.
 from scripts.train_compression_lab_v2_repaired import build
 from scripts import train_compression_lab_v2 as lab
 torch.manual_seed(20261106);fresh=build(dict(kind=meta['kind'])).to(device)
 data=lab.Data(engine);trainrow=lines(ROOT/'artifacts/compression_lab_v2/full_train.jsonl')[rank]
 cfg=dict(kind=meta['kind'],ratio=meta['ratio'],insertion='pre-merger')
 loss,_=lab.student(engine,fresh,trainrow,data,cfg);loss['total'].backward()
 norm=float(torch.nn.utils.clip_grad_norm_(fresh.parameters(),1.));assert np.isfinite(norm) and norm>0
 opt=torch.optim.AdamW(fresh.parameters(),lr=3e-4,weight_decay=0)
 before=[v.detach().clone() for v in fresh.parameters()];opt.step()
 delta=sum(float((v.detach()-p).square().sum()) for v,p in zip(fresh.parameters(),before))**.5;assert delta>0
 assert all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
 save(out/f'rank{rank}.json',dict(passed=True,model=meta['id'],hook_tokens_match=True,
  gradient_norm=norm,update_l2=delta,allocated_plateau_check=values,profile=result))
 meter.close();dist.barrier()
 if rank==0:
  rr=[load(out/f'rank{k}.json') for k in range(world)];assert all(r['passed'] for r in rr)
  save(out/'complete.json',dict(passed=True,cases=8,job=os.environ.get('SLURM_JOB_ID'),results=rr))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
