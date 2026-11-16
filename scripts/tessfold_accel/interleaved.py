"""Additional paired temporal control: randomize methods WITHIN each QA block.

Prespecified90QA,7configs,natural/fixed64,2repeats. Mitigates drift from the
sequential method order in the large15configuration measurement.
"""
import json,os,random,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_accel.system_common import ROOT,BASE as SYSTEM,verify,load,lines,save,sha
from scripts.tessfold_repair.common import load_module
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_accel.system_benchmark import single_request
from vqa.official_und import OfficialUND

@torch.no_grad()
def main():
    verify();rank,world,_,device=init_runtime('npu',4)
    meta_group=dist.new_group(backend='gloo')
    base=ROOT/'artifacts/tessfold_acceleration_20261115/interleaved';base.mkdir(exist_ok=True)
    rows=lines(SYSTEM/'timing_rows.jsonl')[rank::world]
    configs=[('REF',1.),('C_DENSE',.25),('T_DENSE',.25),('C_FAST',.25),('T_FAST',.25),('C_FAST',.5),('T_FAST',.5)]
    manifest=dict(rank=rank,world=world,source_sha256=sha(__file__),configs=[list(x) for x in configs],repeats=2,questions=90,modes=['natural_eos','fixed64'],expected_records=2520,ordering='random methods inside each fixedQA block, randomizedQA blocks, all modules resident,2warmups/config')
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    modules={(kind,ratio):load_module(engine,kind,ratio).requires_grad_(False) for kind in ['col_adaptive','quadtree_question'] for ratio in [.25,.5]}
    def module(method,ratio):return None if method=='REF' else modules[('col_adaptive' if method.startswith('C') else 'quadtree_question',ratio)]
    for method,ratio in configs:
        for row in rows[:2]:single_request(engine,module(method,ratio),row,method,ratio,cap=16)
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={(r['id'],r['method'],r['ratio'],r['mode'],r['repeat']) for r in previous}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    blocks=[(r,fixed,k) for r in rows for fixed in [False,True] for k in range(2)];rng=random.Random(20261115+rank);rng.shuffle(blocks)
    with path.open('a') as f:
        for row,fixed,repeat in blocks:
            order=configs.copy();rng.shuffle(order)
            for method,ratio in order:
                key=(row['id'],method,ratio,'fixed64' if fixed else 'natural_eos',repeat)
                if key in done:continue
                result=single_request(engine,module(method,ratio),row,method,ratio,fixed=fixed);result.update(repeat=repeat,interleaved=True)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(key)
        print(json.dumps(dict(rank=rank,records=len(done))),flush=True)
    dist.barrier(group=meta_group)
    if rank==0:
        rr=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)];assert len(rr)==2520
        hist={(r['id'],r['method'],r['ratio'],r['mode'],r['repeat']):r['generated_ids'] for f in (SYSTEM/'single').glob('results.rank*.jsonl') for r in lines(f) if r['instrumentation']=='synchronized_wall_and_NPU_events'}
        assert all(r['generated_ids']==hist[(r['id'],r['method'],r['ratio'],r['mode'],r['repeat'])] for r in rr)
        save(base/'complete.json',dict(records=2520,questions=90,configs=7,repeats=2,all_generation_equal=True))
    dist.barrier(group=meta_group);dist.destroy_process_group()

if __name__=='__main__':main()
