"""Choose CPU threading from forward time only; never looks at VQA scores."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import json,time,torch
from vqa.official_und import OfficialUND
from vqa.decoder_dependent_loss import common_prefix_logits
from scripts.eval_decoder_dependent import BASE
engine=OfficialUND(device='cpu');engine.model.requires_grad_(False)
row=json.loads((BASE/'samples.jsonl').read_text().splitlines()[0]);inputs=engine.inputs(row)
feature=torch.load(BASE/'features'/f"{row['document_id']}.pt",weights_only=True)
with torch.inference_mode():
 projected=engine.model.vision_tower.merger(feature['x'])
 for threads in [8,16,32,4]:
  torch.set_num_threads(threads);start=time.monotonic()
  logits=common_prefix_logits(engine.model,inputs['input_ids'],projected[None],[13175,151645])
  print(json.dumps({'threads':threads,'seconds':time.monotonic()-start,'shape':list(logits.shape)}),flush=True)
