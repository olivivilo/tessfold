"""Evaluate supplementary OCR-line preservation on all frozen compression trials."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from vqa.encoder_evidence import EvidenceHead,source_weights,bernoulli_js
from vqa.token_proxy import BalancedMatrix
from scripts.train_encoder_relation import LineHead
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/token_proxy/random_v1_20x';OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'
def read(x,u,centers,head):
    d=((centers[:,None]-u[None])**2).sum(-1);v,idx=d.topk(min(8,len(u)),largest=False)
    pooled=(x[idx]*torch.softmax(-v/(2*.04**2),-1)[:,:,None]).sum(1)
    confidence=torch.exp(-v[:,0]/(2*.04**2));pooled=confidence[:,None]*pooled
    return F.gelu(head.net[0](pooled/head.rms)),pooled

def work(item):
    sample,rows=item;torch.set_num_threads(1);doc=sample['document_id'];dest=OUT/'line_losses'/f'{doc}.json'
    if dest.exists():return doc
    ck=torch.load(OUT/'evidence_head.pt',weights_only=True);head=EvidenceHead(ck['rms']);head.load_state_dict(ck['state_dict']);head.eval().requires_grad_(False)
    line=LineHead();line.load_state_dict(torch.load(OUT/'line_head.pt',weights_only=True));line.eval().requires_grad_(False)
    f=torch.load(BASE/'features'/f'{doc}.pt',weights_only=True);x=f['x'].float();u=f['u'];yy,xx=torch.meshgrid((torch.arange(8)+.5)/8,(torch.arange(16)+.5)/16,indexing='ij');centers=torch.stack([xx.flatten(),yy.flatten()],-1);pairs=[]
    for j in range(8):
        for i in range(16):
            k=j*16+i
            if i<15:pairs.append((k,k+1))
            if j<7:pairs.append((k,k+16))
    pairs=torch.tensor(pairs);a,b=pairs.T;delta=centers[b]-centers[a]
    with torch.no_grad():
        hx,rx=read(x,u,centers,head);teacher=line(hx[a],hx[b],delta).sigmoid();weights=source_weights(head(rx).sigmoid(),sample['question']);pw=(weights[a]+weights[b])/2;pw=pw/pw.sum();outputs=[]
        for r in rows:
            m=np.load(ROOT/r['matrix_file']);P=BalancedMatrix(torch.from_numpy(m['group']),torch.from_numpy(m['weight']),int(m['m']),float(m['floor']));c=P.compress(x);cu=P.compress(u);hc,_=read(c,cu,centers,head);student=line(hc[a],hc[b],delta).sigmoid();loss=bernoulli_js(teacher,student)
            outputs.append({'id':r['id'],'loss':{'evidence_line_question':float((loss*pw).sum()),'evidence_line_uniform':float(loss.mean())}})
    dest.write_text(json.dumps({'document_id':doc,'trials':outputs}));return doc

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=32);a=p.parse_args();(OUT/'line_losses').mkdir(exist_ok=True)
    samples=list(map(json.loads,(BASE/'samples.jsonl').read_text().splitlines()));bydoc={s['document_id']:[] for s in samples}
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for r in map(json.loads,path.read_text().splitlines()):bydoc[r['document_id']].append({k:r[k] for k in ['id','matrix_file']})
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,doc in enumerate(pool.map(work,[(s,bydoc[s['document_id']]) for s in samples])):
            if i%100==0:print(json.dumps({'completed':i+1,'total':len(samples)}),flush=True)
if __name__=='__main__':
    import multiprocessing;multiprocessing.set_start_method('spawn');main()
