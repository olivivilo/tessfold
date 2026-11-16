"""Within-question random eager/bidir timing to control kernel/order drift."""
import collections,json,random
from contextlib import nullcontext
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now,verify,load_module
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_accel.system_benchmark import single_request
from vqa.official_und import OfficialUND
from vqa.tessfold_bidirectional_vision import fused_vision
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


@torch.no_grad()
def main():
    verify();assert load(BASE/'bidirectional_vision_pilot/complete.json')['expansion_gate_passed']
    rank,world,_,device=init_runtime('npu',4);group=dist.new_group(backend='gloo');base=BASE/'bidir_interleaved';base.mkdir(exist_ok=True)
    qa=lines(ROOT/'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl');assert len(qa)==90
    configs=[(backend,method) for backend in ['EAGER','BIDIR'] for method in ['REF','C_FAST','T_FAST']]
    manifest=dict(rank=rank,world=world,questions=90,configs=configs,repeats=2,modes=['natural_eos','fixed64'],records=2160,
        sources={s:sha(ROOT/s) for s in ['scripts/tessfold_accel/bidir_interleaved.py','scripts/tessfold_accel/system_benchmark.py','vqa/tessfold_bidirectional_vision.py','vqa/tessfold_fast_runtime.py']},
        motivation='ObservedcompressedprefilldiffersbetweenoldblocktimingandnewPFAstudy; randomize6configswithineachidenticalQAblocktocontrolorder/platformdrift. No model math or selectionchanges.')
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    modules={'REF':None,'C_FAST':load_module(engine,'col_adaptive',.25).requires_grad_(False),'T_FAST':load_module(engine,'quadtree_question',.25).requires_grad_(False)}
    reference={}
    for backend,directory,pattern in [('EAGER',BASE/'system_exact/single','results.rank*.jsonl'),('BIDIR',BASE/'bidirectional_timing','rows.rank*.jsonl')]:
        for p in directory.glob(pattern):
            for r in lines(p):
                if r['method'] in modules and r['ratio']==(1. if r['method']=='REF' else .25):reference[(r['id'],backend,r['method'],r['mode'])]=r['generated_ids']
    assert len(reference)==90*6*2
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);key=lambda r:(r['id'],r['backend'],r['method'],r['mode'],r['repeat'])
    done={key(r) for r in previous};path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    jobs=[(row,fixed,k) for row in qa[rank::world] for fixed in [False,True] for k in range(2)];rng=random.Random(20261115+rank);rng.shuffle(jobs)
    for backend,method in configs:
        with (fused_vision(engine.model.vision_tower) if backend=='BIDIR' else nullcontext()):
            for row in qa[rank::world][:2]:single_request(engine,modules[method],row,method,1. if method=='REF' else .25,cap=4)
    with path.open('a') as f:
      for row,fixed,repeat in jobs:
        order=list(configs);rng.shuffle(order)
        for backend,method in order:
            mode='fixed64' if fixed else 'natural_eos';k=(row['id'],backend,method,mode,repeat)
            if k in done:continue
            with (fused_vision(engine.model.vision_tower) if backend=='BIDIR' else nullcontext()):r=single_request(engine,modules[method],row,method,1. if method=='REF' else .25,fixed=fixed)
            r.update(backend=backend,repeat=repeat,prior_generation_equal=r['generated_ids']==reference[(row['id'],backend,method,mode)],config_order=[list(c) for c in order])
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(k)
        if len(done)%60==0:print(json.dumps(dict(rank=rank,records=len(done))),flush=True)
    dist.barrier(group=group)
    if rank==0:
        rows=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)]
        expected={(r['id'],b,m,mode,k) for r in qa for b,m in configs for mode in ['natural_eos','fixed64'] for k in range(2)}
        assert len(rows)==len(expected)==2160 and {key(r) for r in rows}==expected
        mismatches=collections.Counter(r['backend'] for r in rows if not r['prior_generation_equal'])
        save(base/'complete.json',dict(time=now(),records=2160,questions=90,repeats=2,configs=6,prior_token_mismatches=dict(mismatches),
            note='EAGER and BIDIR quality remain differentvariants. Priortokenchecksreportedexplicitly; pairedcurrentlatencycannotinheritprioraccuracyifanychanges.'))
    dist.barrier(group=group);dist.destroy_process_group()

if __name__=='__main__':main()
