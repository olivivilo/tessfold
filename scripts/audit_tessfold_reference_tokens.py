"""Independent historical native-generate audit for repaired timed requests.

Includes REF, which has no compressor, and four trained method/budget settings.
Compares token IDs, including fixed64 continuations, rather than decoded strings.
"""
import argparse,json
from collections import Counter
from scripts.tessfold_repair.common import ROOT,BASE,lines,save,verify,now


def audit(require_complete=False):
    verify()
    olddir=ROOT/'artifacts/column_tree_seed_memory_20261113/memory'
    native={}
    for path in sorted(olddir.glob('memory.rank*.jsonl')):
        for r in lines(path):
            if r['model']=='REF':method='REF'
            elif r['seed']==20261105 and r['ratio'] in [.25,.5]:
                method={'col_adaptive':'C','quadtree_question':'T'}[r['model'].split('_pre-merger')[0]]
            else:continue
            key=(method,r['ratio'],r['mode'],r['id'])
            if key in native:assert native[key]==r['generated_ids'],'Historical deterministic repetitions differ'
            native[key]=r['generated_ids']
    counts=Counter();mismatches=[]
    for path in sorted((BASE/'single').glob('results.rank*.jsonl')):
        for r in lines(path):
            method=r['method']
            if method=='REF':oldmethod='REF'
            elif method in ['C_STRICT','T_STRICT','C_DENSE','T_DENSE']:oldmethod=method[0]
            else:continue
            key=(oldmethod,r['ratio'],r['mode'],r['id'])
            assert key in native,key
            counts[method]+=1
            if native[key]!=r['generated_ids']:
                mismatches.append(dict(method=method,ratio=r['ratio'],id=r['id'],mode=r['mode'],repeat=r['repeat'],instrumentation=r['instrumentation']))
    if require_complete:
        assert counts==Counter(REF=630,C_STRICT=1260,T_STRICT=1260,C_DENSE=1260,T_DENSE=1260),counts
    result=dict(time=now(),complete=require_complete,compared=sum(counts.values()),by_method=dict(counts),
        mismatches=mismatches,passed=not mismatches,
        reference='Independent completed native-HF-generate memory experiment20261113, seed20261105; natural EOS and fixed64, all repeated histories checked.')
    save(BASE/'historical_native_generation_audit.json',result)
    print(json.dumps(result,ensure_ascii=False))
    assert not mismatches,'Optimized generation differs from historical native generation'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    audit(a.require_complete)
