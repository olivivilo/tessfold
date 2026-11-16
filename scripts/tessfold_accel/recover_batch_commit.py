"""CPU-only recovery: all144requests committed before final HCCL barrier failed.

No rerun, no edits to recorded measurements. Independently verify signatures,
full Cartesian coverage, nested distinctQA cohorts, metrics and batch1 tokens.
"""
import itertools,json
from scripts.tessfold_repair.common import ROOT,lines,load,save,sha,now
from scripts.tessfold_accel.system_common import verify
from vqa.benchmark_metrics import native_score

def main():
    verify();base=ROOT/'artifacts/tessfold_acceleration_20261115/batched_serving'
    rows=[r for f in base.glob('rows.rank*.jsonl') for r in lines(f)]
    manifests=[load(base/f'manifest.rank{k}.json') for k in range(2)];assert all(manifests)
    assert all(m['source_sha256']==sha(ROOT/'scripts/tessfold_accel/batched_serving.py') for m in manifests)
    assert manifests[0]['cohorts']==manifests[1]['cohorts']
    keys={(r['stratum'],r['method'],r['batch_size'],r['mode'],r['repeat']) for r in rows}
    expected=set(itertools.product(['short','medium','long'],['REF','Column'],[1,2,4,8],['natural_eos','fixed64'],range(3)))
    assert len(rows)==len(keys)==144 and keys==expected
    reference={}
    for f in (ROOT/'artifacts/tessfold_system_repair_20261115/single').glob('results.rank*.jsonl'):
        for r in lines(f):
            if r['method']=='REF' or (r['method']=='C_STRICT' and r['ratio']==.25):reference[(r['id'],'REF' if r['method']=='REF' else 'Column',r['mode'])]=r['generated_ids']
    batch1=0
    for r in rows:
        assert len(r['queries'])==r['batch_size'] and r['batch_ms']>0 and r['peak_allocated_bytes']>0
        assert [q['id'] for q in r['queries']]==manifests[0]['cohorts'][r['stratum']][:r['batch_size']]
        for q in r['queries']:
            assert abs(q['score']-native_score(q['prediction'],q))<1e-10
            assert q['visual_tokens']==(q['original_tokens'] if r['method']=='REF' else round(q['original_tokens']*.25))
            if r['mode']=='fixed64':assert len(q['generated_ids'])==64
            if r['batch_size']==1:
                assert q['generated_ids']==reference[(q['id'],r['method'],r['mode'])],'Batch1 differs from historical native/strict'
                batch1+=1
    save(base/'complete.json',dict(time=now(),requests=144,batch_sizes=[1,2,4,8],strata=3,repeats=3,models=2,
        cpu_commit_recovery=True,reason='Job165880 committed72rows on eachrank; failed only final lazy HCCL barrier. Scientific rows preserved, independentCPU audit passed.',
        historical_batch1_token_equal_records=batch1,source_sha256=sha(ROOT/'scripts/tessfold_accel/batched_serving.py'),
        result_sha256={str(f.relative_to(ROOT)):sha(f) for f in base.glob('rows.rank*.jsonl')}))
    print(json.dumps(dict(recovered=True,requests=144,historical_batch1_equal=batch1)))

if __name__=='__main__':main()
