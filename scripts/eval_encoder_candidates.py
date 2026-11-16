"""Replay verified raw features and identical P; join to cached VQA only afterwards."""
import argparse,hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from vqa.token_proxy import BalancedMatrix,stable_seed,random_matrix
from vqa.encoder_evidence import EvidenceHead
from vqa.encoder_candidate_losses import EncoderCandidates
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/token_proxy/random_v1_20x';OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def digest(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def work(item):
    doc,row,rows=item;torch.set_num_threads(1)
    dest=OUT/'losses'/f'{doc}.json'
    if dest.exists():return {'document_id':doc,'cached':True}
    start=time.time();f=torch.load(BASE/'features'/f'{doc}.pt',weights_only=True)
    assert digest(f['x'])==f['sha256']
    ckpt=torch.load(OUT/'evidence_head.pt',weights_only=True);head=EvidenceHead(ckpt['rms']);head.load_state_dict(ckpt['state_dict']);head.eval().requires_grad_(False)
    gen=torch.Generator().manual_seed(20261022)
    projection=torch.linalg.qr(torch.randn(768,64,generator=gen),mode='reduced').Q
    gen=torch.Generator().manual_seed(stable_seed('encoder-candidate-anchors',doc));idx=torch.randperm(len(f['x']),generator=gen)[:256]
    with torch.no_grad():
        proxy=EncoderCandidates(f['x'].float(),f['u'],row['question'],head,projection,idx)
        outputs=[]
        for i,r in enumerate(rows):
            path=ROOT/r['matrix_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['matrix_file_sha256']
            z=np.load(path);matrix=BalancedMatrix(torch.from_numpy(z['group']),torch.from_numpy(z['weight']),int(z['m']),float(z['floor']))
            # Rebuild one matrix per document independently, without touching old generator.
            if i==0:
                regenerated=random_matrix(f['u'],r['ratio'],r['family'],r['matrix_seed'],logit_bound=r['logit_bound'])
                assert torch.equal(matrix.group,regenerated.group) and torch.equal(matrix.weight,regenerated.weight)
            c,losses,diag=proxy(matrix)
            assert digest(c.to(torch.bfloat16))==r['compressed_feature_sha256']
            assert all(np.isfinite(v) for v in losses.values()),r['id']
            outputs.append({'id':r['id'],'loss':losses,'diagnostics':diag})
        payload={'document_id':doc,'source_feature_sha256':f['sha256'],'trials':outputs,'seconds':time.time()-start}
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(payload));temp.replace(dest)
    return {'document_id':doc,'seconds':time.time()-start,'trials':len(outputs)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=32);p.add_argument('--limit',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    (OUT/'losses').mkdir(exist_ok=True)
    samples={r['document_id']:r for r in map(json.loads,(BASE/'samples.jsonl').read_text().splitlines())};bydoc={doc:[] for doc in samples}
    # Only identifiers, P/checksums, and compression configuration enter workers.
    fields=['id','matrix_file','matrix_file_sha256','compressed_feature_sha256','ratio','family','matrix_seed','logit_bound']
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for r in map(json.loads,path.read_text().splitlines()):bydoc[r['document_id']].append({k:r[k] for k in fields})
    items=[(doc,{'question':samples[doc]['question']},rr) for doc,rr in bydoc.items()][a.shard::a.shards]
    if a.limit:items=items[:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(work,items)):
            if i%10==0:print(json.dumps({'completed':i+1,'total':len(items),'shard':a.shard,**r}),flush=True)
    (OUT/f'losses.shard{a.shard}.complete').write_text(json.dumps({'documents':len(items),'shards':a.shards}))
if __name__=='__main__':
    import multiprocessing;multiprocessing.set_start_method('spawn');main()
