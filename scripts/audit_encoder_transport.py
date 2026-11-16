"""Transport resolution audit, selected by image hash, not by task scores."""
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from vqa.token_proxy import BalancedMatrix
from vqa.encoder_evidence import EvidenceHead,source_weights
from vqa.encoder_candidate_losses import coreset,entropic_ot
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/token_proxy/random_v1_20x';OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def work(item):
    sample,rows=item;torch.set_num_threads(1);doc=sample['document_id'];dest=OUT/'transport_audit'/f'{doc}.json'
    if dest.exists():return json.loads(dest.read_text())
    start=time.time();f=torch.load(BASE/'features'/f'{doc}.pt',weights_only=True);x=f['x'].float();u=f['u']
    ck=torch.load(OUT/'evidence_head.pt',weights_only=True);h=EvidenceHead(ck['rms']);h.load_state_dict(ck['state_dict']);h.eval().requires_grad_(False)
    g=torch.Generator().manual_seed(20261022);R=torch.linalg.qr(torch.randn(768,64,generator=g),mode='reduced').Q
    z=x@R/(float(h.rms)*8)
    with torch.no_grad():a=source_weights(h(x).sigmoid(),sample['question'])
    unif=torch.full((len(x),),1/len(x));outputs=[]
    for resolution in ['512','2048','full']:
        def measure(z,u,a):
            if resolution=='full':return torch.cat([z,.5*u],-1),a
            return coreset(z,u,a,(32,16) if resolution=='512' else (64,32))[:2]
        mu=measure(z,u,unif);mq=measure(z,u,a)
        tu,du=entropic_ot(*mu,*mu);tq,dq=entropic_ot(*mq,*mq);tqu,dqu=entropic_ot(*mq,*mu);identity=tqu-.5*tq-.5*tu
        for r in rows:
            m=np.load(ROOT/r['matrix_file']);P=BalancedMatrix(torch.from_numpy(m['group']),torch.from_numpy(m['weight']),int(m['m']),float(m['floor']))
            c=P.compress(x);cu=P.compress(u);cz=c@R/(float(h.rms)*8);nu=measure(cz,cu,torch.full((len(c),),1/len(c)))
            ou,erru=entropic_ot(*mu,*nu);oq,errq=entropic_ot(*mq,*nu);ss,errs=entropic_ot(*nu,*nu)
            outputs.append({'id':r['id'],'resolution':resolution,'sinkhorn_uniform':float(ou-.5*tu-.5*ss),'sinkhorn_question_excess':float(oq-.5*tq-.5*ss-identity),
                            'max_marginal_error':max(d['marginal_max_error'] for d in [du,dq,dqu,erru,errq,errs])})
        print(json.dumps({'audit_doc':doc,'task':sample['task'],'resolution':resolution,'seconds':time.time()-start}),flush=True)
    result={'document_id':doc,'task':sample['task'],'n':len(x),'seconds':time.time()-start,'trials':outputs};dest.write_text(json.dumps(result));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int,default=0);a=p.parse_args();(OUT/'transport_audit').mkdir(exist_ok=True)
    samples=[json.loads(s) for s in (BASE/'samples.jsonl').read_text().splitlines()];chosen={}
    for r in sorted(samples,key=lambda r:hashlib.sha256(('ot-resolution'+r['document_id']).encode()).hexdigest()):chosen.setdefault(r['task'],r)
    if a.limit:chosen=dict(list(chosen.items())[:a.limit])
    bydoc={r['document_id']:[] for r in chosen.values()}
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for r in map(json.loads,path.read_text().splitlines()):
            if r['document_id'] in bydoc and r['kind']=='random' and r['ratio'] in [.125,.875]:bydoc[r['document_id']].append({k:r[k] for k in ['id','matrix_file']})
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(work,[(r,bydoc[r['document_id']]) for r in chosen.values()]))
    (OUT/'transport_resolution_audit.json').write_text(json.dumps({'selection':'one image/task by hash; both families/seeds at .125 and .875; no downstream scores used','documents':results},indent=2)+'\n')
if __name__=='__main__':
    import multiprocessing;multiprocessing.set_start_method('spawn');main()
