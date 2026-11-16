"""Backend-order check only; no model or optimizer execution."""
import json
from pathlib import Path
from scripts.compression_lab_v2_io_retry import install, visible_load
install()
from scripts.train_meta_compress import init_runtime
rank,world,_,_=init_runtime('npu',1)
import torch
assert torch.load is visible_load, 'NPU backend replaced the cache guard'
p=Path('artifacts/compression_lab_v2/infrastructure/recovery_20261106')
if rank==0:
 (p/'npu_backend_order.json').write_text(json.dumps({'torch_load_guard_survives_backend_init':True,'torch_load':str(torch.load)})+'\n')
 run=Path('artifacts/compression_lab_v2/full/full_quadtree_question_pre-merger_r0.1')
 old=torch.load(p/'initial.pt',map_location='cpu',weights_only=False)
 new=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
 assert old['config']==new['config']
 assert old['generator'].keys()==new['generator'].keys()
 assert all(torch.equal(old['generator'][k],new['generator'][k]) for k in old['generator'])
 latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
 assert latest['config']==old['config'] and latest['optimizer']['state']
 (p/'checkpoint_audit.json').write_text(json.dumps({'initialization_bitwise_identical':True,
  'latest_step_observed':latest['step'],'optimizer_state_present':True,'config_identical':True})+'\n')
print('NPU backend order verified')
if world>1:
 import torch.distributed as dist
 dist.barrier();dist.destroy_process_group()
