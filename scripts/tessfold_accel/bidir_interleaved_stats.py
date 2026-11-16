"""SameQA eager/bidir speed comparisons, with explicit prior-output checks."""
import collections
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def compute():
    base=BASE/'bidir_interleaved';assert (base/'complete.json').exists()
    rows=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)];assert len(rows)==2160
    index={(r['id'],r['backend'],r['method'],r['mode'],r['repeat']):r for r in rows};assert len(index)==2160
    qa=lines(ROOT/'artifacts/tessfold_system_repair_20261115/timing_rows.jsonl');rng=np.random.default_rng(20261115)
    pairs=[('EAGER','REF','EAGER','C_FAST'),('EAGER','REF','EAGER','T_FAST'),('BIDIR','REF','BIDIR','C_FAST'),('BIDIR','REF','BIDIR','T_FAST'),
        ('EAGER','REF','BIDIR','REF'),('EAGER','C_FAST','BIDIR','C_FAST'),('EAGER','T_FAST','BIDIR','T_FAST'),('EAGER','REF','BIDIR','C_FAST'),('EAGER','REF','BIDIR','T_FAST')]
    comparisons=[]
    for ab,am,bb,bm in pairs:
      for mode in ['natural_eos','fixed64']:
        pages=collections.defaultdict(lambda:np.zeros(8))
        for q in qa:
            a=[index[(q['id'],ab,am,mode,k)] for k in range(2)];b=[index[(q['id'],bb,bm,mode,k)] for k in range(2)]
            def avg(rr,field):
                if field in ['encoder','qwen_prefill']:return np.mean([r['stages'][field]['wall_ms'] for r in rr])
                return np.mean([r[field] for r in rr])
            pages[q['document_id']]+=np.asarray([v for field in ['end_to_end_ms','ttft_ms','encoder','qwen_prefill'] for v in [avg(a,field),avg(b,field)]])
        arr=np.asarray([pages[k] for k in sorted(pages)]);total=arr.sum(0);boot=arr[rng.integers(len(arr),size=(10000,len(arr)))].sum(1)
        r=dict(reference_backend=ab,reference_method=am,candidate_backend=bb,candidate_method=bm,mode=mode)
        for field,col in [('e2e',0),('ttft',2),('encoder',4),('prefill',6)]:
            r[field+'_speedup']=float(total[col]/total[col+1]);r[field+'_document_bootstrap_95_ci']=np.quantile(boot[:,col]/boot[:,col+1],[.025,.975]).tolist()
        comparisons.append(r)
    phase=[]
    for backend in ['EAGER','BIDIR']:
      for method in ['REF','C_FAST','T_FAST']:
        rr=[r for r in rows if r['backend']==backend and r['method']==method and r['mode']=='natural_eos']
        phase.append(dict(backend=backend,method=method,encoder_mean_ms=float(np.mean([r['stages']['encoder']['wall_ms'] for r in rr])),
            prefill_mean_ms=float(np.mean([r['stages']['qwen_prefill']['wall_ms'] for r in rr])),
            ttft_mean_ms=float(np.mean([r['ttft_ms'] for r in rr])),e2e_mean_ms=float(np.mean([r['end_to_end_ms'] for r in rr])),
            prior_generation_mismatches=sum(not r['prior_generation_equal'] for r in rows if r['backend']==backend and r['method']==method)))
    result=dict(time=now(),records=2160,comparisons=comparisons,natural_phase_means=phase,scope='WithinQA6randomconfigs,90QA,2repeats each natural/fixed64. Samephysicalprocessesandweights; pairedpagebootstrap10000. Crossbackend qualitymustbeviewedwithfullindependentVQAresults, noaccuracyinheritanceifpriorgenerationchanges.')
    save(base/'statistics.json',result);return result
