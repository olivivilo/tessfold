"""CPU audit/commit of completed fused pilot after numpy.bool_ serialization."""
import json
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
from scripts.tessfold_accel.system_common import verify
from vqa.benchmark_metrics import native_score
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    verify();base=BASE/'fused_vision_pilot'
    wanted=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl');rows=[]
    for rank in range(2):
        m=load(base/f'manifest.rank{rank}.json');assert m['world']==2 and m['rank']==rank
        for path,digest in m['source_sha256'].items():assert sha(ROOT/path)==digest
        path=base/f'rows.rank{rank}.jsonl'
        with path.open() as f:rr=[json.loads(line) for line in f if line.strip()]
        assert {r['id'] for r in rr}=={r['id'] for r in wanted[rank::2]}
        rows.extend(rr)
    assert len(rows)==len({r['id'] for r in rows})==56
    qa={r['id']:r for r in wanted};methods=set(rows[0]['methods']);assert len(methods)==5
    for r in rows:
        assert set(r['methods'])==methods
        assert r['eager_encoder_ms']>0 and r['fused_encoder_ms']>0
        for m,v in r['methods'].items():
            assert abs(v['score']-native_score(v['prediction'],qa[r['id']]))<1e-10
            assert abs(v['original_score']-native_score(v['original_prediction'],qa[r['id']]))<1e-10
    ratio=float(np.median([r['eager_encoder_ms']/r['fused_encoder_ms'] for r in rows]))
    delta=float(100*np.mean([r['methods']['REF']['score']-r['methods']['REF']['original_score'] for r in rows]))
    result=dict(time=now(),cases=56,median_encoder_speedup=ratio,ref_mean_qa_delta_pp=delta,
        expansion_gate_passed=bool(ratio>=1.5 and delta>=-2.),
        token_changes={m:sum(not r['methods'][m]['token_equal'] for r in rows) for m in methods},
        score_delta_pp={m:float(100*np.mean([r['methods'][m]['score']-r['methods'][m]['original_score'] for r in rows])) for m in methods},
        not_bit_exact=True,cpu_commit_recovery=True,
        reason='All56records committed in job165900; rank0 failed serializing numpy.bool_ in final metadata. IndependentCPU source/coverage/metric audit, no measurements rewritten.',
        result_sha256={str(p.relative_to(ROOT)):sha(p) for p in base.glob('rows.rank*.jsonl')})
    save(base/'complete.json',result);print(json.dumps(result))

if __name__=='__main__':main()
