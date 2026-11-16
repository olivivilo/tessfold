"""Independent1ms allocator traces for native8B; never latency observations.

Same90QA, five dev-selected models, three natural-generation repeats. Keeps the
same tensor lifetimes as timing; visual encoder+native mergers measured jointly.
"""
import argparse,hashlib,json
import torch
import torch.distributed as dist
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
from vqa.tessfold_qwen3vl_timing import TimingQwen
from vqa.tessfold_qwen3vl import NativeSpatialRouter
from vqa.tessfold_qwen3vl_stable import StableColumn
from vqa.stage_memory import StageMemory,cache_nbytes
from vqa.tessfold_runtime import kv_visual_bytes
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


class MemoryQwen(TimingQwen):
    def visual(self,*args,**kwargs):
        if self.memory is None:return super().visual(*args,**kwargs)
        self.memory.begin('visual_setup')
        try:return super().visual(*args,**kwargs)
        finally:self.memory.end()


@torch.no_grad()
def request(engine,router,row,meta,trace):
    memory=StageMemory(engine.device);engine.memory=memory;state={};calls=[0]
    try:
        memory.begin('front_processor_vision_mergers');data=engine.prepare(row);memory.end()
        memory.begin('p_generation')
        p,reps=(None,None) if router is None else router(data['x'],data['shape'],data['question'],meta['ratio'])
        memory.end();m=len(data['x']) if p is None else len(p)
        def pre(mod,args,kwargs):
            if calls[0]==0:memory.begin('qwen_prefill')
        def post(mod,args,kwargs,out):
            calls[0]+=1
            if calls[0]==1:
                state['prefill_kv_bytes']=cache_nbytes(out.past_key_values)
                state['visual_kv_bytes']=kv_visual_bytes(out.past_key_values,m)
                memory.end();memory.begin('generation')
            state['final_kv_bytes']=cache_nbytes(out.past_key_values)
        h=engine.model.model.language_model.register_forward_pre_hook(pre,with_kwargs=True)
        hh=engine.model.model.language_model.register_forward_hook(post,with_kwargs=True)
        try:ids=engine.greedy(data,p,reps,cap=256)
        finally:h.remove();hh.remove()
        memory.end();stages=memory.export(trace)
        return dict(id=row['id'],document_id=row['document_id'],task=row['task'],model=meta['id'],ratio=meta['ratio'],
            original_visual_tokens=len(data['x']),visual_tokens=m,generated_ids=ids,prediction=engine.decode(ids),
            stages=stages,decoder_state=state,peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in stages.values()),
            downstream_peak_allocated_bytes=max(stages[k]['peak_allocated_bytes'] for k in ['visual_setup','qwen_prefill','generation']),
            instrumentation='independent1ms_memory_pass_not_latency',cap=256)
    finally:
        if memory.active:memory.end()
        memory.close();engine.memory=None


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    rank,world,_,device=init_runtime('npu',4);group=dist.new_group(backend='gloo')
    registry=load(ROOT/args.runs);base=BASE/('qwen_memory_smoke' if args.smoke else 'qwen_memory');base.mkdir(exist_ok=True);(base/'traces').mkdir(exist_ok=True)
    rows=lines(ROOT/'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl');assert len(rows)==90
    if args.smoke:
        ordered=sorted(rows,key=lambda r:(r['original_tokens'],r['id']));rows=[ordered[i] for i in [0,45,89]]
    repeats=1 if args.smoke else 3
    methods=[dict(id='REF',kind='REF',ratio=1.)]+registry['runs'];assert len(methods)==5
    manifest=dict(rank=rank,world=world,run_registry_sha256=sha(ROOT/args.runs),questions=len(rows),repeats=repeats,models=5,
        sources={s:sha(ROOT/s) for s in ['scripts/tessfold_accel/qwen_memory.py','vqa/tessfold_qwen3vl_timing.py','vqa/tessfold_qwen3vl.py','vqa/tessfold_qwen3vl_stable.py','vqa/stage_memory.py']},
        rule='Independent memory observations only. Joint native front, P generation,4branch visual setup,prefill,generation. Allocator peak+time-weighted1ms sample, logical visual KV separately. Compare token IDs against uninstrumented timing before combining.')
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    timing_gate=load(BASE/'qwen_timing_identity/complete.json');assert timing_gate['natural_equal']==56 and timing_gate['fixed64_equal']==7
    assert timing_gate['timing_source_sha256']==sha(ROOT/'vqa/tessfold_qwen3vl_timing.py')
    if not args.smoke:
        gate=load(BASE/'qwen_memory_smoke/complete.json');assert gate['records']==15
        assert gate['memory_source_sha256']==sha(__file__)
    engine=MemoryQwen(device);engine.memory=None;modules={'REF':None}
    for meta in registry['runs']:
        assert sha(ROOT/meta['checkpoint'])==meta['checkpoint_sha256']
        router=(StableColumn if meta.get('allocation') else NativeSpatialRouter)(meta['kind']).to(device).eval().requires_grad_(False)
        router.load_state_dict(torch.load(ROOT/meta['checkpoint'],map_location='cpu',weights_only=False)['router']);modules[meta['id']]=router
    for row in rows[rank::world][:1]:
        data=engine.prepare(row)
        for meta in methods:
            p,reps=(None,None) if meta['id']=='REF' else modules[meta['id']](data['x'],data['shape'],data['question'],meta['ratio'])
            engine.greedy(data,p,reps,cap=4)
        del data,p,reps
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={(r['id'],r['model'],r['repeat']) for r in previous}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    with path.open('a') as f:
      for row in rows[rank::world]:
        for meta in methods:
          for repeat in range(repeats):
            key=(row['id'],meta['id'],repeat)
            if key in done:continue
            trace=base/'traces'/(hashlib.sha256(str(key).encode()).hexdigest()+'.npz')
            result=request(engine,modules[meta['id']],row,meta,trace)
            result.update(repeat=repeat,trace=str(trace.relative_to(ROOT)),trace_sha256=sha(trace))
            f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add(key)
        print(json.dumps(dict(rank=rank,records=len(done))),flush=True)
    dist.barrier(group=group)
    if rank==0:
        records=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)]
        expected={(r['id'],m['id'],k) for r in rows for m in methods for k in range(repeats)}
        assert len(records)==len(expected)==len(rows)*len(methods)*repeats
        assert {(r['id'],r['model'],r['repeat']) for r in records}==expected
        for r in records:assert sha(ROOT/r['trace'])==r['trace_sha256']
        save(base/'complete.json',dict(time=now(),records=len(records),questions=len(rows),models=5,repeats=repeats,memory_source_sha256=sha(__file__),independent_memory_only=True,
            joint_front_note='Native visual mergers and DeepStack are included in front, P is after all native mergers. Full encoder and downstream peak remain separate. Token equality with timing is audited by final report.'))
    dist.barrier(group=group);dist.destroy_process_group()

if __name__=='__main__':main()
