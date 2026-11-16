"""Separate full quality and efficiency for noncausal-PFA numerical candidate."""
import collections,json
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now,verify
from scripts.tessfold_repair.report import speed_rows
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def compute():
    reg=verify();result=dict(time=now(),not_bit_exact=True);formal=load(BASE/'bidirectional_formal/complete.json')
    timing=load(BASE/'bidirectional_timing/complete.json')
    if formal:
        rows=[r for p in (BASE/'bidirectional_formal').glob('rows.rank*.jsonl') for r in lines(p)];assert len(rows)==117825
        current={(r['id'],r['model']):r for r in rows};historical={}
        for meta in [m for m in reg['models'] if m['ratio'] in [.25,.5]]:
            count=0
            for p in (ROOT/meta['output']/'benchmark').glob('predictions.rank*.jsonl'):
                with p.open() as f:
                    for line in f:
                        old=json.loads(line)
                        if old['variant']!='trained':continue
                        historical[(old['id'],meta['id'])]=dict(score=old['student_score'],prediction=old['student_prediction'],generated_ids=old['student_ids']);count+=1
                        if meta['kind']=='col_adaptive' and meta['ratio']==.25:
                            historical[(old['id'],'REF')]=dict(score=old['teacher_score'],prediction=old['teacher_prediction'])
            assert count==23565
        assert set(current)==set(historical)
        models=sorted({r['model'] for r in rows});comparisons={}
        for model in models:
            ss=[r for r in rows if r['model']==model];by={};boots=[];rng=np.random.default_rng(20261115)
            for task in sorted({r['task'] for r in ss}):
                rr=[r for r in ss if r['task']==task];pages=collections.defaultdict(lambda:np.zeros(2))
                for r in rr:
                    old=historical[(r['id'],model)];pages[r['document_id']]+=np.array([r['score']-old['score'],1.])
                arr=np.asarray([pages[k] for k in sorted(pages)]);boot=[]
                for _ in range(100):
                    total=arr[rng.integers(len(arr),size=(100,len(arr)))].sum(1);boot.extend(100*total[:,0]/total[:,1])
                boots.append(np.asarray(boot));by[task]=dict(questions=len(rr),eager=100*np.mean([historical[(r['id'],model)]['score'] for r in rr]),
                    bidirectional=100*np.mean([r['score'] for r in rr]),delta=100*np.mean([r['score']-historical[(r['id'],model)]['score'] for r in rr]),
                    document_bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist())
            comparisons[model]=dict(shared_fieldEM_macro_delta=float(np.mean([v['delta'] for v in by.values()])),
                shared_fieldEM_macro_delta_task_stratified_document_bootstrap_95_ci=np.quantile(np.mean(boots,axis=0),[.025,.975]).tolist(),by_task=by,
                decoded_prediction_changes=sum(r['prediction']!=historical[(r['id'],model)]['prediction'] for r in ss),
                token_changes=None if model=='REF' else sum(r['generated_ids']!=historical[(r['id'],model)]['generated_ids'] for r in ss))
        result['quality_vs_original_eager']=comparisons
    if timing:
        rows=[r for p in (BASE/'bidirectional_timing').glob('rows.rank*.jsonl') for r in lines(p)];assert len(rows)==3150
        index={(r['id'],r['method'],r['ratio'],r['mode'],r['repeat'],r['instrumentation']):r for r in rows};assert len(index)==3150
        time_rows=[r for r in rows if r['instrumentation']=='synchronized_wall_and_NPU_events']
        memory=[r for r in rows if r['instrumentation']=='memory1ms'];assert len(time_rows)==2700 and len(memory)==450
        for r in memory:
            assert sha(ROOT/r['trace'])==r['trace_sha256']
            for k in range(3):assert r['generated_ids']==index[(r['id'],r['method'],r['ratio'],'natural_eos',k,'synchronized_wall_and_NPU_events')]['generated_ids']
        formal_matches=0
        if formal:
            mapping={('REF',1.):'REF'}
            for m in [m for m in reg['models'] if m['ratio'] in [.25,.5]]:mapping[('C_FAST' if m['kind']=='col_adaptive' else 'T_FAST',m['ratio'])]=m['id']
            for r in rows:
                if r['mode']=='natural_eos':
                    assert r['generated_ids']==current[(r['id'],mapping[(r['method'],r['ratio'])])]['generated_ids'],'Bidir formal/timing changed generation'
                    formal_matches+=1
        result['timing_summary']=speed_rows(rows);result['formal_natural_token_equal_records']=formal_matches
        result['memory_summary']=[dict(method=method,ratio=ratio,full_max_gib=max(r['peak_allocated_bytes'] for r in rr)/2**30,
            encoder_max_gib=max(r['stages']['encoder']['peak_allocated_bytes'] for r in rr)/2**30,
            downstream_max_gib=max(max(r['stages'][s]['peak_allocated_bytes'] for s in ['compression_and_merger','qwen_prefill','generation']) for r in rr)/2**30,
            visual_kv_median_mib=float(np.median([r['decoder_state']['visual_kv_bytes'] for r in rr]))/2**20)
            for (method,ratio),rr in sorted(_groups(memory).items())]
        comparisons=[];rng=np.random.default_rng(20261115);qa=lines(ROOT/'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl')
        for method in ['C_FAST','T_FAST']:
          for ratio in [.25,.5]:
           for mode in ['natural_eos','fixed64']:
            pages=collections.defaultdict(lambda:np.zeros(4))
            for q in qa:
                aa=[index[(q['id'],'REF',1.,mode,k,'synchronized_wall_and_NPU_events')] for k in range(3)]
                bb=[index[(q['id'],method,ratio,mode,k,'synchronized_wall_and_NPU_events')] for k in range(3)]
                pages[q['document_id']]+=np.array([np.mean([r['end_to_end_ms'] for r in aa]),np.mean([r['end_to_end_ms'] for r in bb]),np.mean([r['ttft_ms'] for r in aa]),np.mean([r['ttft_ms'] for r in bb])])
            arr=np.asarray([pages[k] for k in sorted(pages)]);total=arr.sum(0);boot=arr[rng.integers(len(arr),size=(10000,len(arr)))].sum(1)
            comparisons.append(dict(method=method,ratio=ratio,mode=mode,e2e_speedup=float(total[0]/total[1]),ttft_speedup=float(total[2]/total[3]),
                e2e_document_bootstrap_95_ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist(),ttft_document_bootstrap_95_ci=np.quantile(boot[:,2]/boot[:,3],[.025,.975]).tolist()))
        result['paired_speed_vs_same_bidirectional_REF']=comparisons
    save(BASE/'bidirectional_statistics.json',result);return result


def _groups(rows):
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['method'],r['ratio'])].append(r)
    return grouped

if __name__=='__main__':print(json.dumps(compute()))
