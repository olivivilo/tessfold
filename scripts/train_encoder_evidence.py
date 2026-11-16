"""Train OCR evidence probes on training images only, with image-held-out diagnostics."""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from vqa.encoder_evidence import EvidenceHead,token_targets,NC,NB,NT,DIM,relevance
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def ap(y,p):
    order=np.argsort(-p);yy=y[order];return float((np.cumsum(yy)/(np.arange(len(yy))+1)*yy).sum()/max(1,yy.sum()))

def main():
    torch.set_num_threads(8);torch.manual_seed(20261022)
    rows=[json.loads(s) for s in (OUT/'probe_documents.jsonl').read_text().splitlines()]
    data={s:[[],[]] for s in ['train','holdout']};rms_sum=count=0
    for i,r in enumerate(rows):
        f=torch.load(OUT/'probe_features'/f"{r['document_id']}.pt",weights_only=True);x=f['x'].float()
        target,_=token_targets(f['u'],r['words'],f['grid'])
        g=torch.Generator().manual_seed(i+20261022);pos=(target[:,-NT]>0).nonzero().flatten();neg=(target[:,-NT]==0).nonzero().flatten()
        ix=torch.cat([pos[torch.randperm(len(pos),generator=g)[:512]],neg[torch.randperm(len(neg),generator=g)[:256]]])
        data[r['split']][0].append(x[ix]);data[r['split']][1].append(target[ix])
        if r['split']=='train':rms_sum+=float(x.square().sum());count+=x.numel()
    data={s:[torch.cat(v) for v in vs] for s,vs in data.items()};rms=(rms_sum/count)**.5
    trainx,trainy=data['train'];valx,valy=data['holdout']
    head=EvidenceHead(rms);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
    # Capped class weights avoid trivial all-background predictions.
    posweight=((len(trainy)-trainy.sum(0))/(trainy.sum(0)+1)).clamp(1,15)
    history=[];start=time.time()
    for epoch in range(1,21):
        order=torch.randperm(len(trainx));total=0
        head.train()
        for ix in order.split(2048):
            logits=head(trainx[ix]);loss=F.binary_cross_entropy_with_logits(logits,trainy[ix],pos_weight=posweight)
            opt.zero_grad();loss.backward();opt.step();total+=float(loss)*len(ix)
        head.eval()
        with torch.no_grad():
            vl=sum(float(F.binary_cross_entropy_with_logits(head(x),y,pos_weight=posweight,reduction='sum')) for x,y in zip(valx.split(4096),valy.split(4096)))/valy.numel()
        r={'epoch':epoch,'training_weighted_BCE':total/len(trainx),'holdout_weighted_BCE':vl,'seconds':time.time()-start};history.append(r);print(json.dumps(r),flush=True)
    torch.save({'state_dict':head.state_dict(),'rms':rms,'parameters':sum(p.numel() for p in head.parameters()),'epochs':20,'posweight':posweight},OUT/'evidence_head.pt')
    with torch.no_grad():p=torch.cat([head(x).sigmoid() for x in valx.split(4096)]).numpy()
    y=valy.numpy();positive=y[:,-NT]>0
    metrics={'training_images':720,'holdout_images':len(rows)-720,'train_tokens':len(trainx),'holdout_tokens':len(valx),'parameters':sum(p.numel() for p in head.parameters()),'rms':rms,'epochs':20,'selection':'fixed final epoch; not selected on VQA scores',
       'text_AP':ap(y[:,-NT],p[:,-NT]),'text_positive_fraction':float(y[:,-NT].mean()),
       'digit_AP':ap(y[:,-NT+1],p[:,-NT+1]),'digit_positive_fraction':float(y[:,-NT+1].mean()),
       'character_macro_AP_text_tokens':float(np.mean([ap(y[positive,j],p[positive,j]) for j in range(NC)])),
       'bigram_macro_AP_text_tokens':float(np.mean([ap(y[positive,j],p[positive,j]) for j in range(NC,NC+NB)])),
       'type_AP':{name:ap(y[:,-NT+j],p[:,-NT+j]) for j,name in enumerate(['text','digit','decimal','percent','currency','negative','date','alphabet'])},'history':history}
    # Probe localization: independent OCR words, no VQA question/answer supervision.
    hits=base=queries=0
    for r in [r for r in rows if r['split']=='holdout'][:80]:
        f=torch.load(OUT/'probe_features'/f"{r['document_id']}.pt",weights_only=True);x=f['x'].float();u=f['u']
        with torch.no_grad():prob=head(x).sigmoid()
        candidates=[w for w in r['words'] if len(w['text'])>=4][:12]
        for w in candidates:
            xx,yy,bw,bh=w['box'];dx,dy=.5/int(f['grid'][2]),.5/int(f['grid'][1]);mask=(u[:,0]>=xx-dx)&(u[:,0]<=xx+bw+dx)&(u[:,1]>=yy-dy)&(u[:,1]<=yy+bh+dy)
            if not mask.any():continue
            idx=relevance(prob,w['text']).topk(min(10,len(u))).indices
            hits+=int(mask[idx].any());base+=float(1-(1-float(mask.float().mean()))**min(10,len(u)));queries+=1
    metrics['OCR_word_query_localization']={'queries':queries,'top10_hit_rate':hits/max(1,queries),'uniform_approx_top10_hit_rate':base/max(1,queries),'note':'word-in-question lexical localization diagnostic; not VQA evidence grounding accuracy'}
    (OUT/'probe_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k!='history'},indent=2))
if __name__=='__main__':main()
