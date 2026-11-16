"""Full-token Sinkhorn on NPU; no spatial coarsening, projector, or decoder."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from vqa.encoder_evidence import EvidenceHead,source_weights
from vqa.token_proxy import BalancedMatrix
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/token_proxy/random_v1_20x';OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def ot(x,a,y,b,eps=.05,maxiter=1500,tol=1e-6):
    d=(x.square().sum(-1)[:,None]+y.square().sum(-1)[None]-2*x@y.T).clamp_min(0)
    # Row shifts change scaling factors, never the optimal transport plan.
    k=torch.exp(-(d-d.min(1).values[:,None])/eps)
    v=torch.ones_like(b);err=float('inf');mode='scaling'
    for it in range(maxiter):
        u=a/(k@v).clamp_min(1e-35);v=b/(k.T@u).clamp_min(1e-35)
        norm=v.mean().clamp_min(1e-30);v=v/norm;u=u*norm
        if (it+1)%25==0 and it+1>=150:
            err=float((u*(k@v)-a).abs().max().cpu())
            if np.isfinite(err) and err<tol:break
            if not np.isfinite(err):break
    plan=u[:,None]*k*v[None]
    if not torch.isfinite(plan).all() or not np.isfinite(err) or err>=tol:
        mode='log';lk=-d/eps;lu=torch.zeros_like(a);lv=torch.zeros_like(b)
        for it in range(maxiter):
            lu=a.log()-torch.logsumexp(lk+lv[None],dim=1)
            lv=b.log()-torch.logsumexp(lk+lu[:,None],dim=0)
            if (it+1)%25==0 and it+1>=150:
                rows=torch.exp(lu+torch.logsumexp(lk+lv[None],dim=1));err=float((rows-a).abs().max().cpu())
                if err<tol:break
        plan=(lu[:,None]+lk+lv[None]).exp()
    logratio=plan.clamp_min(1e-35).log()-a.log()[:,None]-b.log()[None]
    value=(plan*d).sum()+eps*((plan*logratio).sum()-plan.sum()+1)
    err=max(float((plan.sum(1)-a).abs().max().cpu()),float((plan.sum(0)-b).abs().max().cpu()))
    return float(value.cpu()),{'marginal_error':err,'iterations':it+1,'mode':mode}

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');a=p.parse_args()
    import torch_npu
    local=int(os.environ.get('LOCAL_RANK',0));rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=f'npu:{local}'
    torch.npu.set_device(local);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False;torch.set_num_threads(2)
    ck=torch.load(OUT/'evidence_head.pt',weights_only=True);head=EvidenceHead(ck['rms']);head.load_state_dict(ck['state_dict']);head.eval().requires_grad_(False)
    g=torch.Generator().manual_seed(20261022);R=torch.linalg.qr(torch.randn(768,64,generator=g),mode='reduced').Q
    samples=[json.loads(s) for s in (BASE/'samples.jsonl').read_text().splitlines()];bydoc={r['document_id']:[] for r in samples}
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for r in map(json.loads,path.read_text().splitlines()):bydoc[r['document_id']].append({k:r[k] for k in ['id','matrix_file','matrix_file_sha256','compressed_feature_sha256']})
    destdir=OUT/('exact_ot_benchmark' if a.benchmark else 'exact_ot');destdir.mkdir(exist_ok=True)
    if a.benchmark:
        audit=json.loads((OUT/'transport_resolution_audit.json').read_text());docs={r['document_id'] for r in audit['documents']};samples=[r for r in samples if r['document_id'] in docs]
        ids={r['id'] for d in audit['documents'] for r in d['trials']}
        bydoc={doc:[r for r in rows if r['id'] in ids] for doc,rows in bydoc.items()}
    for i,row in enumerate(samples[rank::world]):
        doc=row['document_id'];dest=destdir/f'{doc}.json'
        if dest.exists():continue
        start=time.time();f=torch.load(BASE/'features'/f'{doc}.pt',weights_only=True);x=f['x'].float();u=f['u'];z=x@R/(ck['rms']*8)
        with torch.no_grad():weights=source_weights(head(x).sigmoid(),row['question'])
        mu=torch.cat([z,.5*u],-1).to(device);unif=torch.full((len(x),),1/len(x),device=device);weights=weights.to(device)
        tu,du=ot(mu,unif,mu,unif);tq,dq=ot(mu,weights,mu,weights);tqu,dqu=ot(mu,weights,mu,unif);identity=tqu-.5*tu-.5*tq
        outputs=[]
        for r in bydoc[doc]:
            path=ROOT/r['matrix_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['matrix_file_sha256']
            m=np.load(path);P=BalancedMatrix(torch.from_numpy(m['group']),torch.from_numpy(m['weight']),int(m['m']),float(m['floor']))
            c=P.compress(x);cu=P.compress(u)
            assert hashlib.sha256(c.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==r['compressed_feature_sha256']
            cz=c@R/(ck['rms']*8);nu=torch.cat([cz,.5*cu],-1).to(device);b=torch.full((len(c),),1/len(c),device=device)
            ou,eu=ot(mu,unif,nu,b);oq,eq=ot(mu,weights,nu,b);ss,es=ot(nu,b,nu,b)
            outputs.append({'id':r['id'],'loss':{'sinkhorn_full_uniform':ou-.5*tu-.5*ss,'sinkhorn_full_question_raw':oq-.5*tq-.5*ss,'sinkhorn_full_question_excess':oq-.5*tq-.5*ss-identity},
                            'max_marginal_error':max(e['marginal_error'] for e in [du,dq,dqu,eu,eq,es]),'max_iterations':max(e['iterations'] for e in [du,dq,dqu,eu,eq,es]),'log_fallback':sum(e['mode']=='log' for e in [eu,eq,es])})
        result={'document_id':doc,'n':len(x),'seconds':time.time()-start,'trials':outputs,'identity_q_baseline':identity}
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(dest)
        print(json.dumps({'rank':rank,'completed':i+1,'total':len(samples[rank::world]),'document_id':doc,'n':len(x),'seconds':time.time()-start}),flush=True)
    (destdir/f'rank{rank}.complete').write_text('complete\n')
if __name__=='__main__':main()
