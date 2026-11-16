"""Train-image-only PCA/mean-null parameter initialization, no auxiliary loss."""
import json
import torch
from scripts.train_meta_compress import init_runtime,atomic_save,atomic_json,sha
from scripts.report_meta_compress_strict import BASE,read
from vqa.official_und import OfficialUND

def main():
    rank,world,local,device=init_runtime('npu',8);assert world==1
    out=BASE/'calibration';out.mkdir(exist_ok=True)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    rows=read(BASE/'pilot.jsonl');docs=sorted({r['document_id'] for r in rows});values={'pre-merger':[],'post-merger':[]}
    with torch.no_grad():
        for doc in docs:
            f=torch.load(BASE/'npu_cache/features'/f'{doc}.pt',map_location='cpu',weights_only=True)
            assert f['source']=='MonkeyOCRv2-B-Und-native-bf16-v1'
            raw=f['features'].to(device);x=engine.model.vision_tower.merger(raw)
            idx=torch.linspace(0,len(raw)-1,min(64,len(raw))).round().long()
            values['pre-merger'].append(raw.cpu()[idx].float())
            values['post-merger'].append(x.cpu()[idx].float())
    for insertion,parts in values.items():
        x=torch.cat(parts).double();mu=x.mean(0);center=x-mu
        u=mu/mu.norm().clamp_min(1e-12)
        center=center-(center@u)[:,None]*u[None]
        cov=center.T@center/len(center)
        eigenvalues,vectors=torch.linalg.eigh(cov)
        val=eigenvalues[-64:].flip(0).clamp_min(1e-6);basis=vectors[:,-64:].flip(1).T
        weight=(basis/val.sqrt()[:,None]).float()
        atomic_save(out/f'{insertion}.pt',{'weight':weight,'mean':mu.float(),'eigenvalues':val.float(),
            'images':len(docs),'tokens':len(x),'train_sha256':sha(BASE/'pilot.jsonl'),
            'source':'official UND NPU BF16, pilot training images only; mean-null top64 PCA whitening'})
        print(insertion,'images',len(docs),'tokens',len(x),'eig',val.min().item(),val.max().item(),
              'mean_residual',float((weight@mu.float()).norm()),flush=True)
    atomic_json(out/'complete.json',{'images':len(docs),'pilot_sha256':sha(BASE/'pilot.jsonl'),
        'no_answers_or_validation_used':True})

if __name__=='__main__':main()
