"""Compare a native-resolution CPU FP32 reference to cached NPU eager features."""
import json
from pathlib import Path
import torch
from PIL import Image
from vqa.model import load_vision,resolve
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'configs/full.json').read_text())
cfg.update(device='cpu',threads=4,vision_dtype='float32',vision_attention='eager')
torch.set_num_threads(4)
sample=json.loads((root/'data/full_probe/train.jsonl').read_text().splitlines()[0])
vision,processor=load_vision(cfg)
with Image.open(resolve(sample['image'])) as img:
 media=processor(images=[img.convert('RGB')],return_tensors='pt')
with torch.inference_mode():reference=vision(media['pixel_values'].float(),media['image_grid_thw'],bf16=False)
actual=torch.load(root/'artifacts/npu_ddp_probe_eager/features'/f'{sample["document_id"]}.pt',map_location='cpu',weights_only=True)['features'].float()
# Float64 reductions avoid float32 norm drift over millions of feature values.
ref64,act64=reference.double(),actual.double()
cosine=torch.nn.functional.cosine_similarity(ref64.reshape(1,-1),act64.reshape(1,-1)).item()
relative_rmse=((act64-ref64).square().mean()/ref64.square().mean()).sqrt().item()
result={'reference':'original CPU eager FP32','candidate':'NPU eager FP32, BF16 feature storage','document_id':sample['document_id'],'tokens':reference.shape[0],'cosine_similarity':cosine,'relative_rmse':relative_rmse}
(root/'artifacts/vision_cpu_npu_reference.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
if cosine<.999 or relative_rmse>.03:raise ValueError('CPU/NPU eager vision reference differs substantially')
