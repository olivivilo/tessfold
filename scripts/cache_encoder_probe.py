"""Extract only official UND encoder weights; alignment MLP is removed before load."""
import argparse,json,os
from pathlib import Path
import torch
from safetensors import safe_open
from PIL import Image
from transformers import AutoConfig,AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from vqa.official_und import ROOT,MODEL,use_eager_vision,set_device
from vqa.token_proxy import patch_geometry

def main():
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    out=ROOT/'artifacts/token_proxy/encoder_candidates_v1';(out/'probe_features').mkdir(exist_ok=True)
    device=set_device();use_eager_vision()
    cls=get_class_from_dynamic_module('modeling_monkeyocrv2_vision.MonkeyOCRv2VisionTransformer',str(MODEL))
    cfg=AutoConfig.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True).vision_config
    cfg.attn_implementation='eager'
    model=cls(cfg);model.merger=torch.nn.Identity()
    with safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as f:
        state={k[len('vision_tower.'):]:f.get_tensor(k) for k in f.keys() if k.startswith('vision_tower.') and not k.startswith('vision_tower.merger.')}
    model.load_state_dict(state,strict=True);model=model.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
    proc=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True).image_processor
    rows=[json.loads(s) for s in (out/'probe_documents.jsonl').read_text().splitlines()]
    for i,row in enumerate(rows[rank::world]):
        dest=out/'probe_features'/f"{row['document_id']}.pt"
        if dest.exists():continue
        image=Image.open(ROOT/row['image']).convert('RGB')
        inputs=proc(images=[image],return_tensors='pt',max_pixels=1003520,min_pixels=3136)
        grid=inputs['image_grid_thw']
        with torch.no_grad(): _,x=model(inputs['pixel_values'].to(device),grid.to(device))
        u,_,_=patch_geometry(grid[0].tolist())
        torch.save({'x':x.cpu(),'u':u,'grid':grid[0].tolist(),'source':'official UND raw encoder; merger replaced by Identity; no LLM weights loaded'},dest)
        if i%20==0:print(json.dumps({'rank':rank,'done':i+1,'total':len(rows[rank::world])}),flush=True)
    (out/f'probe_cache.rank{rank}.complete').write_text('complete\n')
if __name__=='__main__':main()
