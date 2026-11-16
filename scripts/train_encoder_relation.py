"""Supplementary structural evidence probe trained on OCR same-line labels."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vqa.encoder_evidence import EvidenceHead
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'
class LineHead(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(578,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,a,b,delta):return self.net(torch.cat([a,b,(a-b).abs(),delta],-1)).squeeze(-1)
def main():
    torch.set_num_threads(8);torch.manual_seed(20261023);ck=torch.load(OUT/'evidence_head.pt',weights_only=True);head=EvidenceHead(ck['rms']);head.load_state_dict(ck['state_dict']);head.eval().requires_grad_(False)
    rows=list(map(json.loads,(OUT/'probe_documents.jsonl').read_text().splitlines()));data={s:[[],[]] for s in ['train','holdout']}
    for row in rows:
        f=torch.load(OUT/'probe_features'/f"{row['document_id']}.pt",weights_only=True);u=f['u'];x=f['x'].float();features=[];centers=[];lines=[]
        for w in row['words']:
            if w['line']<0:continue
            xx,yy,bw,bh=w['box'];center=torch.tensor([xx+bw/2,yy+bh/2]);mask=(u[:,0]>=xx)&(u[:,0]<=xx+bw)&(u[:,1]>=yy)&(u[:,1]<=yy+bh)
            if not mask.any():mask[((u-center)**2).sum(-1).argmin()]=True
            features.append(x[mask].mean(0));centers.append(center);lines.append(w['line'])
        if len(features)<4:continue
        with torch.no_grad():h=F.gelu(head.net[0](torch.stack(features)/head.rms))
        centers=torch.stack(centers);lines=torch.tensor(lines);dist=torch.cdist(centers,centers)
        idx=dist.topk(min(7,len(h)),largest=False).indices[:,1:];a=torch.arange(len(h))[:,None].expand_as(idx).flatten();b=idx.flatten();labels=(lines[a]==lines[b]).float()
        pos=(labels>0).nonzero().flatten();neg=(labels==0).nonzero().flatten();ix=torch.cat([pos[torch.randperm(len(pos))[:128]],neg[torch.randperm(len(neg))[:128]]])
        inputs=torch.cat([h[a],h[b],(h[a]-h[b]).abs(),centers[b]-centers[a]],-1)[ix]
        data[row['split']][0].append(inputs);data[row['split']][1].append(labels[ix])
    data={s:[torch.cat(v) for v in vs] for s,vs in data.items()};tx,ty=data['train'];vx,vy=data['holdout'];model=LineHead();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[]
    for epoch in range(1,21):
        total=0
        for ix in torch.randperm(len(tx)).split(2048):
            logits=model.net(tx[ix]).squeeze(-1);loss=F.binary_cross_entropy_with_logits(logits,ty[ix]);opt.zero_grad();loss.backward();opt.step();total+=float(loss)*len(ix)
        with torch.no_grad():pred=model.net(vx).squeeze(-1);vl=float(F.binary_cross_entropy_with_logits(pred,vy))
        history.append({'epoch':epoch,'training_BCE':total/len(tx),'holdout_BCE':vl});print(json.dumps(history[-1]),flush=True)
    from scripts.train_encoder_evidence import ap
    prob=pred.sigmoid().numpy();y=vy.numpy();delta=vx[:,-2:].numpy();geo=np.exp(-(delta[:,1]/.015)**2)
    metrics={'parameters':sum(p.numel() for p in model.parameters()),'train_pairs':len(ty),'holdout_pairs':len(vy),'positive_fraction':float(vy.mean()),'same_line_AP':ap(y,prob),'same_line_accuracy':float(((prob>=.5)==y).mean()),'geometry_only_AP':ap(y,geo),'labels':'InfoVQA training OCR WORD -> LINE relationships; word ROI means; image-disjoint 720/240 images','history':history,'scope':'same OCR line, not table header/value or arithmetic relation semantics','selection':'supplementary structure probe; fixed 20 epochs, no VQA-supervised selection'}
    torch.save(model.state_dict(),OUT/'line_head.pt');(OUT/'line_probe_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps({k:v for k,v in metrics.items() if k!='history'}))
if __name__=='__main__':main()
