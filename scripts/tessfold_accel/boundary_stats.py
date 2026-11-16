"""Actual token scales and equal-boundary strictUND post-tower comparisons."""
import collections
import numpy as np
from scripts.tessfold_repair.common import ROOT,lines,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def compute():
    original=[r for p in (BASE/'system_exact/single').glob('results.rank*.jsonl') for r in lines(p)]
    native=[r for p in (BASE/'qwen_timing').glob('rows.rank*.jsonl') for r in lines(p)]
    old=[r for r in original if r['method']=='REF' and r['mode']=='natural_eos' and r['repeat']==0 and r['instrumentation']=='synchronized_wall_and_NPU_events']
    new=[r for r in native if r['model']=='REF' and r['mode']=='natural_eos' and r['repeat']==0]
    assert len(old)==len(new)==90
    scales={name:dict(questions=90,input_visual_token_quantiles=np.quantile([r[key] for r in rr],[0,.5,.9,1]).tolist())
        for name,rr,key in [('UND',old,'visual_tokens'),('Native8B',new,'original_visual_tokens')]}
    ix={(r['id'],r['method'],r['ratio'],r['mode'],r['repeat']):r for r in original if r['instrumentation']=='synchronized_wall_and_NPU_events'}
    rng=np.random.default_rng(20261115);pairs=[]
    for method in ['C_FAST','T_FAST']:
      for ratio in [.25,.5]:
       for mode in ['natural_eos','fixed64']:
        pages=collections.defaultdict(lambda:np.zeros(2))
        for q in old:
            a=np.mean([ix[(q['id'],'REF',1.,mode,k)]['post_encoder_ttft_ms'] for k in range(3)])
            b=np.mean([ix[(q['id'],method,ratio,mode,k)]['post_encoder_ttft_ms'] for k in range(3)])
            pages[q['document_id']]+=np.array([a,b])
        arr=np.asarray([pages[k] for k in sorted(pages)]);total=arr.sum(0);boot=arr[rng.integers(len(arr),size=(10000,len(arr)))].sum(1)
        pairs.append(dict(method=method,ratio=ratio,mode=mode,post_tower_ttft_speedup=float(total[0]/total[1]),document_bootstrap_95_ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()))
    result=dict(time=now(),source_sha256=sha(__file__),token_scales=scales,strict_und_post_tower_pairs=pairs,
        scope='REFandFASTbothreturnrawtowerbeforemerger,equalboundary. Throughfirsttoken, notE2E. NoS/DENSEboundarymix. Native8BmergersprecedeP/DeepStackdiffers,90fixedrequests.')
    save(BASE/'paper/token_boundary_and_posttower.json',result);return result

if __name__=='__main__':print(compute())
