"""Independent math/interface checks; no decoder invocation."""
import json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from vqa.encoder_evidence import EvidenceHead,source_weights
from vqa.encoder_candidate_losses import EncoderCandidates,entropic_ot,sqdist
from vqa.token_proxy import BalancedMatrix,DenseMatrix
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def main():
    torch.set_num_threads(1);torch.manual_seed(24)
    # Tiny entropic transport checked against constrained optimization.
    x=torch.randn(4,3,dtype=torch.float64)*.2;y=torch.randn(3,3,dtype=torch.float64)*.2
    a=torch.tensor([.1,.2,.3,.4],dtype=torch.float64);b=torch.tensor([.2,.5,.3],dtype=torch.float64);eps=.05;d=sqdist(x,y).numpy();ab=(a[:,None]*b).numpy()
    def fun(v):
        p=v.reshape(4,3);return float((p*d).sum()+eps*(p*np.log(np.maximum(p,1e-300)/ab)-p+ab).sum())
    constraints=[{'type':'eq','fun':lambda p:p.reshape(4,3).sum(1)-a.numpy()}, {'type':'eq','fun':lambda p:p.reshape(4,3).sum(0)[:2]-b.numpy()[:2]}]
    sol=minimize(fun,ab.ravel(),method='SLSQP',bounds=[(1e-12,1)]*12,constraints=constraints,options={'ftol':1e-12,'maxiter':1000})
    ot,diag=entropic_ot(x,a,y,b,eps);reverse,_=entropic_ot(y,b,x,a,eps)
    assert sol.success and abs(float(ot)-sol.fun)<2e-6
    assert abs(float(ot-reverse))<2e-6
    ckpt=torch.load(OUT/'evidence_head.pt',weights_only=True);head=EvidenceHead(ckpt['rms']);head.load_state_dict(ckpt['state_dict']);head.eval().requires_grad_(False)
    raw=torch.randn(120,768)*ckpt['rms'];u=torch.rand(120,2);R=torch.linalg.qr(torch.randn(768,64),mode='reduced').Q
    proxy=EncoderCandidates(raw,u,'what is the total in 2019?',head,R,torch.arange(120))
    identity=BalancedMatrix(torch.arange(120),torch.ones(120),120,0.)
    _,loss,diagnostic=proxy(identity)
    zero_fields=['coverage_uniform','coverage_pos_question','sinkhorn_uniform','sinkhorn_question_excess','subspace_question_excess','evidence_combined_question']
    assert max(abs(loss[k]) for k in zero_fields)<2e-6,{k:loss[k] for k in zero_fields}
    perm=torch.randperm(120);dense=torch.eye(120)[perm];_,permloss,_=proxy(DenseMatrix(dense))
    assert max(abs(loss[k]-permloss[k]) for k in loss)<2e-5
    # Full-rank span can become almost exact even for mixed tokens: record evidence.
    smallx=torch.randn(80,12,dtype=torch.float64);p=torch.rand(30,80,dtype=torch.float64);p=p/p.sum(1,keepdim=True);c=p@smallx
    residual=smallx-smallx@torch.linalg.pinv(c)@c
    assert float(residual.norm()/smallx.norm())<1e-10
    # Region-addressed evidence should detect geometry changes even when set stays identical.
    changed=EncoderCandidates(raw,u[perm],'what is the total in 2019?',head,R,torch.arange(120))
    result={'passed':True,'transport_oracle_abs_error':abs(float(ot)-sol.fun),'transport_symmetry_abs_error':abs(float(ot-reverse)),
      'transport_diagnostics':diag,'identity_loss':loss,'identity_zero_fields':zero_fields,'row_permutation_max_loss_error':max(abs(loss[k]-permloss[k]) for k in loss),
      'full_rank_mixed_span_relative_residual':float(residual.norm()/smallx.norm()),
      'note':'set/span/geometry-addressed losses invariant to jointly permuted C and PU; target causal decoder need not be invariant; cannot assert universal decoder-agnostic guarantee'}
    (OUT/'math_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
