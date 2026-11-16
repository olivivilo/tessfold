"""Export learned matrices on the fit image and plot their distributions."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import json
import math
import torch
from scripts.diagnose_meta_compress import OUT,CACHE,DiagnosticModule,VARIANTS
from scripts.diagnose_meta_compress_factorial import VARIANTS as FACTORIAL
from scripts.train_meta_compress import prepare_example,atomic_save,atomic_json,sha
from vqa.official_und import OfficialUND,ROOT


def main():
    torch.set_num_threads(16)
    out=OUT/'matrices';out.mkdir(exist_ok=True)
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    row=next(json.loads(x) for x in (OUT/'samples.jsonl').read_text().splitlines() if json.loads(x)['role']=='fit')
    prompt,raw,grid,x,response=prepare_example(engine,row,CACHE,64)
    choices=[('original_initial',VARIANTS[0],ROOT/'artifacts/meta_compress/initial.pt'),
        ('original_20k_final',VARIANTS[0],ROOT/'artifacts/meta_compress/meta_compress.pt'),
        ('kl_sgd_no_clip',VARIANTS[3],OUT/'kl_sgd_no_clip/checkpoint.pt'),
        ('local_bias_all_adam',FACTORIAL[1],OUT/'factorial/local_bias_all_adam/checkpoint.pt'),
        ('kl_adam_keep025',VARIANTS[8],OUT/'kl_adam_keep025/checkpoint.pt')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(len(choices),1,figsize=(13,12),layout='constrained')
    audits=[]
    with torch.no_grad():
        for ax,(name,cfg,path) in zip(axes,choices):
            module=DiagnosticModule(cfg,x,grid);ck=torch.load(path,weights_only=False,map_location='cpu')
            if 'module' in ck:module.load_state_dict(ck['module'])
            else:module.generator.load_state_dict(ck['generator'])
            c,p=module(x,grid);h=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean())
            bin_width=math.ceil(p.shape[1]/1200)
            item={'name':name,'id':row['id'],'P':p,'grid':grid,'config':cfg,'checkpoint_sha256':sha(path),
                'code_sha256':sha(__file__),'entropy':h,'entropy_fraction':h/math.log(len(x)),
                'collapse':float(p.sum(0).max()),'row_sum_error':float((p.sum(-1)-1).abs().max()),
                'row_max_weight_mean':float(p.max(-1).values.mean()),'display_column_max_bin_width':bin_width}
            atomic_save(out/f'{name}.pt',item)
            audits.append({k:v for k,v in item.items() if k not in ['P','grid']})
            # Max-bin only for display, avoiding loss of one-column peaks
            # when a 3036-column matrix is rasterized into a narrower figure.
            display=torch.nn.functional.max_pool1d(p[None],bin_width,bin_width,ceil_mode=True)[0]
            im=ax.imshow(np.log10(display.clamp_min(1e-8).numpy()),aspect='auto',vmin=-6,vmax=0,
                extent=(0,p.shape[1],p.shape[0],0),cmap='magma',interpolation='nearest')
            ax.set_title(f"{name}: m={len(p)}, H/log(n)={item['entropy_fraction']:.4f}, max column mass={item['collapse']:.3f}")
            ax.set_ylabel('Output row');ax.set_xlabel('Source token in native 2x2 block order')
    fig.colorbar(im,ax=list(axes),label='log10(P)',shrink=.7)
    fig.savefig(out/'matrix_comparison.png',dpi=160);fig.savefig(out/'matrix_comparison.pdf')
    atomic_json(out/'complete.json',{'matrices':audits,'fit_id':row['id']})


if __name__=='__main__':main()
