"""Independent-image inference for one trained compression generator.
Run on an allocated NPU using .venv-und, never load the full model on a login node.
"""
import argparse,json,os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
from vqa.compression_generators import ConvexGenerator
from vqa.official_und import OfficialUND
from scripts.train_meta_compress import init_runtime
from scripts.eval_token_proxy import generate,shortened_inputs

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--question',required=True);p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--with-teacher',action='store_true');a=p.parse_args()
 _,_,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.requires_grad_(False)
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];module=ConvexGenerator(cfg['kind']).to(device).eval();module.load_state_dict(saved['generator'])
 with torch.no_grad():
  inp=engine.inputs({'image':a.image,'question':a.question});x,raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
  ids=engine.processor.tokenizer(a.question,add_special_tokens=False,return_tensors='pt')['input_ids'][0,:64].to(device)
  q=engine.model.get_input_embeddings()(ids)
  if cfg.get('insertion')=='pre-merger':
   _,p=module(torch.zeros_like(x,dtype=torch.float32),inp['image_grid_thw'],cfg['ratio'],raw=raw,question=q)
   c=engine.model.vision_tower.merger((p@raw.float()).to(raw.dtype))
  else:c,p=module(x.float(),inp['image_grid_thw'],cfg['ratio'],raw=raw,question=q)
  si=shortened_inputs(inp,len(c),engine.model.config.image_token_id);pred,rt=generate(engine,si,a.max_new_tokens,c.to(raw.dtype),raw)
  out={'answer':pred,'kind':cfg['kind'],'ratio':cfg['ratio'],'tokens':len(raw),'compressed_tokens':len(c),'row_sum_error':float((p.sum(-1)-1).abs().max()),'generation':rt}
  if a.with_teacher:out['teacher_answer']=generate(engine,inp,a.max_new_tokens,x,raw)[0]
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
