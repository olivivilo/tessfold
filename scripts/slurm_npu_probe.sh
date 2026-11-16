#!/usr/bin/env bash
#SBATCH --job-name=monkey-npu-probe
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/npu-probe-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
.venv-npu/bin/python - <<'PY'
import os, torch, torch_npu
print('HOST', os.uname().nodename, 'VISIBLE', os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), flush=True)
print('VERSIONS', torch.__version__, torch_npu.__version__, flush=True)
print('DEVICES', torch.npu.device_count(), flush=True)
torch.npu.set_device(0)
x = torch.randn(32,32,device='npu',dtype=torch.bfloat16,requires_grad=True)
(x@x).float().sum().backward()
print('BACKWARD_OK', x.grad.abs().mean().item(), torch.npu.get_device_properties(0), flush=True)
from vqa.full_train import TrainableVLM, amp
from vqa.model import load_vision
import json
from PIL import Image
cfg=json.load(open('configs/full.json'))
cfg.update(device='npu:0',output_dir='artifacts/full_probe')
vision, processor=load_vision(cfg)
sample=json.loads(open('data/full_probe/train.jsonl').readline())
media=processor(images=[Image.open(sample['image']).convert('RGB')],return_tensors='pt')
with torch.no_grad():
 features=vision(media['pixel_values'].to('npu',dtype=torch.bfloat16),media['image_grid_thw'].to('npu'),bf16=True)
print('VISION_NATIVE_RESOLUTION_OK',list(features.shape),flush=True)
del vision
torch.npu.empty_cache()
model=TrainableVLM(cfg)
model.features[sample['document_id']]=features.float()
model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
for stage in (1,2):
 model.lm.requires_grad_(stage==2)
 model.lm.train()
 model.zero_grad(set_to_none=True)
 with amp(cfg):loss=model([sample])
 loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 print('REAL_VLM_BACKWARD_OK',stage,float(loss),torch.npu.max_memory_allocated()/1e9,flush=True)
from pathlib import Path
Path('artifacts/NPU_PROBE_PASSED').write_text('Native resolution + both stages backward passed\n')
PY
