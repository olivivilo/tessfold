"""Read-only diagnosis of the completed run; never changes its checkpoint."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import json,math,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'artifacts/meta_compress'
torch.set_num_threads(2)
initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)['generator']
final=torch.load(out/'meta_compress.pt',map_location='cpu',weights_only=False)['generator']
parameters={k:{'initial_norm':float(v.norm()),'delta_norm':float((final[k]-v).norm()),'relative_change':float((final[k]-v).norm()/v.norm())} for k,v in initial.items()}
rows=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines() if x.strip()]
clip=[min(1.,.01/(r['grad_norm']+1e-6)) for r in rows]
matrices=[]
for path in sorted((out/'eval/matrices').glob('*.pt')):
 item=torch.load(path,map_location='cpu',weights_only=True);p=item['P'].float();m,n=p.shape
 h=-(p*p.clamp_min(1e-30).log()).sum(-1)
 normed=torch.nn.functional.normalize(p,dim=-1)
 if m>1:
  pair=(normed[:-1]*normed[1:]).sum(-1).mean()
 else:pair=torch.tensor(float('nan'))
 centroid=p.mean(0)
 matrices.append({'document_id':item['document_id'],'n':n,'m':m,'row_entropy':float(h.mean()),
  'entropy_fraction_of_uniform':float((h/math.log(n)).mean()) if n>1 else 0.,
  'effective_support':float(h.exp().mean()),'effective_support_fraction':float(h.exp().mean()/n),
  'mean_row_max_weight':float(p.max(-1).values.mean()),
  'neighbor_row_cosine':float(pair),'row_diversity_relative_l2':float((p-centroid).norm()/p.norm())})
summary={'parameter_changes':parameters,
 'clip':{'all_steps_clipped':all(x<1 for x in clip),'median_gradient_multiplier':statistics.median(clip),
         'median_effective_lr_along_original_gradient':.001*statistics.median(clip),
         'max_update_norm_per_step':.001*.01,'path_length_upper_bound':len(rows)*.001*.01},
 'matrix_examples':len(matrices),
 'matrix_summary':{k:{'mean':statistics.mean(r[k] for r in matrices),'min':min(r[k] for r in matrices),'max':max(r[k] for r in matrices)} for k in ['entropy_fraction_of_uniform','effective_support','effective_support_fraction','mean_row_max_weight','neighbor_row_cosine','row_diversity_relative_l2']},
 'matrices':matrices}
(out/'failure_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='matrices'},indent=2))
