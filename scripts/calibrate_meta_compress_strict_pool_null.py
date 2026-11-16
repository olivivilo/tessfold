"""Remove max-pool's shared query mean using only the pilot training images."""
import torch
from scripts.train_meta_compress import init_runtime,atomic_save,atomic_json,sha
from scripts.report_meta_compress_strict import BASE,read
from vqa.official_und import OfficialUND

def main():
    _,world,_,device=init_runtime('npu',8);assert world==1
    out=BASE/'pool_null_calibration';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    docs=sorted({r['document_id'] for r in read(BASE/'pilot.jsonl')})
    samples={p:[] for p in ['pre-merger','post-merger']};pooled={p:[] for p in samples}
    with torch.no_grad():
        for doc in docs:
            f=torch.load(BASE/'npu_cache/features'/f'{doc}.pt',map_location='cpu',weights_only=True)
            assert f['source']=='MonkeyOCRv2-B-Und-native-bf16-v1'
            raw=f['features'].to(device);projected=engine.model.vision_tower.merger(raw)
            idx=torch.linspace(0,len(raw)-3,64,device=device).round().long()
            for insertion,x in [('pre-merger',raw),('post-merger',projected)]:
                samples[insertion].append(x[idx].float().cpu())
                pooled[insertion].append(torch.stack([x[idx+j].float() for j in range(3)]).max(0).values.cpu())
    measurements=[]
    for insertion in samples:
        original=torch.load(BASE/'calibration'/f'{insertion}.pt',map_location='cpu',weights_only=True)
        weight=original['weight'].double();mu=original['mean'].double();u=mu/mu.norm()
        pool_mu=torch.cat(pooled[insertion]).double().mean(0)
        v=pool_mu-(pool_mu@u)*u;v=v/v.norm()
        # Existing PCA weights annihilate mu. Projecting another orthogonal
        # direction removes query bias while preserving the first constraint.
        weight=weight-(weight@v)[:,None]*v[None]
        x=torch.cat(samples[insertion]).double();variance=((x-x.mean(0))@weight.T).square().mean(0)
        weight=weight/variance.clamp_min(1e-6).sqrt()[:,None]
        atomic_save(out/f'{insertion}.pt',{'weight':weight.float(),'raw_mean':mu.float(),
            'pool_mean':pool_mu.float(),'pilot_sha256':sha(BASE/'pilot.jsonl'),'images':len(docs)})
        item={'insertion':insertion,'initial_pool_mean_projection_norm':float((original['weight'].double()@pool_mu).norm()),
              'new_pool_mean_projection_norm':float((weight@pool_mu).norm()),
              'raw_mean_projection_norm':float((weight@mu).norm())}
        measurements.append(item);print(item,flush=True)
    atomic_json(out/'complete.json',{'training_only':True,'measurements':measurements})

if __name__=='__main__':main()
