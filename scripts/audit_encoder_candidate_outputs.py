"""Verify full trial coverage and supplementary readout invariance before publishing."""
import hashlib,json,math
from pathlib import Path
import torch
from vqa.encoder_evidence import EvidenceHead
from scripts.train_encoder_relation import LineHead
from scripts.eval_encoder_relation import read
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1';BASE=ROOT/'artifacts/token_proxy/random_v1_20x'
def main():
    torch.set_num_threads(1)
    samples=list(map(json.loads,(BASE/'samples.jsonl').read_text().splitlines()));expected={s['document_id']:set() for s in samples}
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for r in map(json.loads,path.read_text().splitlines()):
            assert r['id'] not in expected[r['document_id']]
            expected[r['document_id']].add(r['id'])
    records={}
    for directory in ['losses','line_losses','exact_ot']:
        seen=set();digest=hashlib.sha256();fields=None
        for doc,ids in expected.items():
            path=OUT/directory/f'{doc}.json';blob=path.read_bytes();d=json.loads(blob);trials=d['trials']
            assert d['document_id']==doc and len(trials)==len(ids)==26
            assert {r['id'] for r in trials}==ids
            digest.update(path.name.encode());digest.update(hashlib.sha256(blob).digest())
            for r in trials:
                assert r['id'] not in seen;seen.add(r['id'])
                if fields is None:fields=set(r['loss'])
                assert set(r['loss'])==fields
                assert all(math.isfinite(v) for v in r['loss'].values())
        assert len(seen)==93626
        records[directory]={'documents':len(expected),'trials':len(seen),'fields':sorted(fields),'aggregate_sha256':digest.hexdigest()}
    a=json.loads((OUT/'analysis.json').read_text());assert a['images']==3601 and a['random_trials']==86424 and a['controls']==7202
    assert a['has_full_token_OT'] and a['has_line_probe']
    for path,value in a['code_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    for filename,key in [('evidence_head.pt','head_sha256'),('line_head.pt','line_head_sha256')]:
        assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest()==a[key]
    # A genuine supplementary invariant: jointly permuting feature/position rows
    # must preserve the region-addressed relation readout even on real tokens.
    ck=torch.load(OUT/'evidence_head.pt',weights_only=True);head=EvidenceHead(ck['rms']);head.load_state_dict(ck['state_dict']);head.eval().requires_grad_(False)
    line=LineHead();line.load_state_dict(torch.load(OUT/'line_head.pt',weights_only=True));line.eval().requires_grad_(False)
    f=torch.load(BASE/'features'/f"{samples[0]['document_id']}.pt",weights_only=True);x=f['x'].float();u=f['u'];g=torch.Generator().manual_seed(24)
    centers=torch.rand(128,2,generator=g);perm=torch.randperm(len(x),generator=g)
    with torch.no_grad():
        h,_=read(x,u,centers,head);hp,_=read(x[perm],u[perm],centers,head)
        p=line(h[:-1],h[1:],centers[1:]-centers[:-1]).sigmoid();pp=line(hp[:-1],hp[1:],centers[1:]-centers[:-1]).sigmoid()
    error=float((p-pp).abs().max());assert error<1e-6,error
    result={'passed':True,'directories':records,'line_readout_joint_permutation_max_error':error,'analysis_code_and_head_hashes_match':True,'source_matrix_and_X_and_PX_hashes':'verified for all trials by eval_encoder_candidates worker before writing losses','scope':'output completeness, finite values, provenance and relation readout invariant; does not establish semantic evidence accuracy'}
    (OUT/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
