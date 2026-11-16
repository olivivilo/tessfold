"""Entry: python -m vqa.full_infer --image IMAGE --question QUESTION (on a compute node)."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from .full_train import amp, set_runtime
from .model import VLM, load_vision, resolve

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/full.json')
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--stage',type=int,choices=[1,2],default=2)
    p.add_argument('--image',required=True)
    p.add_argument('--question',required=True)
    args=p.parse_args()
    cfg=json.loads(resolve(args.config).read_text())
    set_runtime(cfg)
    checkpoint=resolve(args.checkpoint) if args.checkpoint else resolve(cfg['output_dir'])/f'stage{args.stage}'
    if not (checkpoint/'projector.pt').exists():raise FileNotFoundError(f'No trained MLP checkpoint: {checkpoint}')
    if args.stage==2 and not (checkpoint/'language_model/config.json').exists():raise FileNotFoundError(f'No trained stage-2 language model: {checkpoint}')
    image=resolve(args.image)
    sample={'document_id':hashlib.sha256(image.read_bytes()).hexdigest(),'image':str(image),'question':args.question}
    vision,processor=load_vision(cfg)
    media=processor(images=[Image.open(image).convert('RGB')],return_tensors='pt')
    with torch.inference_mode():
        features=vision(media['pixel_values'].to(cfg['device'],dtype=vision.dtype),media['image_grid_thw'].to(cfg['device']),bf16=vision.dtype==torch.bfloat16)
    # Match feature-cache quantization exactly, including the CPU casting backend.
    stored=features.cpu().to(dtype=getattr(torch,cfg.get('feature_storage_dtype',cfg['vision_dtype'])))
    del features
    del vision
    if args.stage==2:cfg['language_model']=str(checkpoint/'language_model')
    model=VLM(cfg)
    model.projector.load_state_dict(torch.load(checkpoint/'projector.pt',map_location='cpu',weights_only=True))
    model.features[sample['document_id']]=stored.to(model.device_name,dtype=model.dtype)
    model.eval()
    with amp(cfg):answer=model.predict(sample)
    print(json.dumps({'image':str(image),'question':args.question,'answer':answer,'checkpoint':str(checkpoint),'stage':args.stage},ensure_ascii=False))

if __name__=='__main__':main()
