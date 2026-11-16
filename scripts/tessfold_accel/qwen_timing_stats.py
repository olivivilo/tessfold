"""Native8B paired latency statistics by resolution; no inferred speedups."""
import collections
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def compute():
    base=BASE/'qwen_timing';assert (base/'complete.json').exists()
    rows=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)]
    qa=lines(ROOT/'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl')
    assert len(rows)==2700 and len(qa)==90
    strata={r['id']:['short','medium','long'][i//30] for i,r in enumerate(sorted(qa,key=lambda r:(r['original_tokens'],r['id'])))}
    by_id={r['id']:r for r in qa};index={(r['id'],r['model'],r['mode'],r['repeat']):r for r in rows}
    assert len(index)==2700
    # Formal and timing must agree for identical natural-generation requests.
    formal=BASE/'qwen_formal/complete.json'
    matches=0
    if formal.exists():
        expected={(r['id'],r['model']):r['generated_ids'] for p in (BASE/'qwen_formal').glob('rows.rank*.jsonl') for r in lines(p) if r['id'] in by_id and r['role']=='primary'}
        assert len(expected)==450
        for r in rows:
            if r['mode']=='natural_eos':
                assert r['generated_ids']==expected[(r['id'],r['model'])],f'Formal/timing generation differs: {r["id"]}/{r["model"]}'
                matches+=1
    comparisons=[];rng=np.random.default_rng(20261115)
    for model in sorted({r['model'] for r in rows}-{'REF'}):
      for mode in ['natural_eos','fixed64']:
       for stratum in ['all','short','medium','long']:
        cohort=[r for r in qa if stratum=='all' or strata[r['id']]==stratum]
        pages=collections.defaultdict(lambda:np.zeros(6))
        for q in cohort:
            aa=[index[(q['id'],'REF',mode,k)] for k in range(3)]
            bb=[index[(q['id'],model,mode,k)] for k in range(3)]
            values=[]
            for field in ['end_to_end_ms','ttft_ms','post_front_ttft_ms']:
                values.extend([np.mean([r[field] for r in aa]),np.mean([r[field] for r in bb])])
            pages[q['document_id']]+=np.asarray(values)
        arr=np.asarray(list(pages.values()));tot=arr.sum(0)
        boot=arr[rng.integers(len(arr),size=(10000,len(arr)))].sum(1)
        result=dict(model=model,mode=mode,stratum=stratum,questions=len(cohort))
        for label,col in [('e2e',0),('ttft',2),('post_front_ttft',4)]:
            result[label+'_speedup']=float(tot[col]/tot[col+1])
            result[label+'_bootstrap_95_ci']=np.quantile(boot[:,col]/boot[:,col+1],[.025,.975]).tolist()
        comparisons.append(result)
    result=dict(time=now(),records=2700,comparisons=comparisons,formal_natural_token_matches=matches,
        scope='Mean of three repeats perQA, paired latency sum ratio, document bootstrap10000. Strata fixed from original UND visual-token cohort; native8B resolutions can differ. Postfront excludes its complete native vision+mergers. Timing source identity gate checked before measurement.')
    save(base/'statistics.json',result);return result
