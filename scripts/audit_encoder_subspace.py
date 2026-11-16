"""Spectrum/condition/coefficient audit on hash-selected images, without VQA scores."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from vqa.encoder_evidence import EvidenceHead
from vqa.token_proxy import BalancedMatrix,stable_seed
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/token_proxy/random_v1_20x';OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'
def work(item):
    sample,rows=item;torch.set_num_threads(2);f=torch.load(BASE/'features'/f"{sample['document_id']}.pt",weights_only=True);x=f['x'].double()
    ck=torch.load(OUT/'evidence_head.pt',weights_only=True);lam=.001*ck['rms']**2;g=torch.Generator().manual_seed(stable_seed('encoder-candidate-anchors',sample['document_id']));idx=torch.randperm(len(x),generator=g)[:256];ax=x[idx];out=[]
    for r in rows:
        m=np.load(ROOT/r['matrix_file']);P=BalancedMatrix(torch.from_numpy(m['group']),torch.from_numpy(m['weight']).double(),int(m['m']),float(m['floor']));c=P.compress(x);s=torch.linalg.svdvals(c)
        cov=c.T@c/len(c)+lam*torch.eye(768,dtype=torch.float64);solve=torch.linalg.solve(cov,ax.T).T;coef=solve@c.T/len(c)
        residual=ax-coef@c
        q={'id':r['id'],'m':len(c),'d':768,'rank_at_relative_1e_4':int((s>s[0]*1e-4).sum()),'rank_at_float32_default_tol':int((s>s[0]*max(c.shape)*torch.finfo(torch.float32).eps).sum()),'largest_singular':float(s[0]),'smallest_returned_singular':float(s[-1]),'nonzero_spectrum_condition':float(s[0]/s[-1]),'ridge_cov_condition':float((s[0]**2/len(c)+lam)/(lam+(s[-1]**2/len(c) if len(c)>=768 else 0))), 'coefficient_abs_row_sum_p95':float(torch.quantile(coef.abs().sum(1),.95)),'coefficient_l2_row_norm_p95':float(torch.quantile(coef.norm(dim=1),.95)),'coefficient_negative_fraction':float((coef<0).double().mean()),'unweighted_anchor_residual_fraction':float(residual.square().sum()/ax.square().sum())}
        out.append(q)
    return {'task':sample['task'],'document_id':sample['document_id'],'n':len(x),'trials':out}
def main():
    chosen={d['document_id'] for d in json.loads((OUT/'transport_resolution_audit.json').read_text())['documents']};samples=[r for r in map(json.loads,(BASE/'samples.jsonl').read_text().splitlines()) if r['document_id'] in chosen];bydoc={r['document_id']:[] for r in samples}
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for r in map(json.loads,path.read_text().splitlines()):
            if r['document_id'] in chosen and r['kind']=='random' and r['ratio'] in [.125,.875]:bydoc[r['document_id']].append({k:r[k] for k in ['id','matrix_file']})
    with ProcessPoolExecutor(max_workers=8) as pool:docs=list(pool.map(work,[(s,bydoc[s['document_id']]) for s in samples]))
    for doc in docs:
        prior={r['id']:r['loss']['subspace_uniform'] for r in json.loads((OUT/'losses'/f"{doc['document_id']}.json").read_text())['trials']}
        for r in doc['trials']:
            r['float32_uniform_absolute_error']=abs(prior[r['id']]-r['unweighted_anchor_residual_fraction'])
            r['float32_uniform_relative_error']=r['float32_uniform_absolute_error']/max(1e-30,r['unweighted_anchor_residual_fraction'])
    rows=[r for d in docs for r in d['trials']];keys=[k for k,v in rows[0].items() if isinstance(v,(int,float))]
    result={'selection':'one image/task by hash, both families/seeds at .125/.875; no downstream scores used','images':len(docs),'trials':len(rows),'computation':'float64 spectral diagnostics; actual primary ridge losses use float32','summary':{k:dict(zip(['min','median','max'],map(float,np.quantile([r[k] for r in rows],[0,.5,1])))) for k in keys},'documents':docs}
    (OUT/'subspace_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':
    import multiprocessing;multiprocessing.set_start_method('spawn');main()
