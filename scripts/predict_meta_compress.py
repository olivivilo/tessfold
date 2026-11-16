"""Explicit inference entry: trained MetaCompress inserted after official merger."""
import argparse
import json
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from vqa.meta_compress import MetaCompress
from vqa.official_und import ROOT,OfficialUND
from scripts.eval_token_proxy import generate,shortened_inputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--question',required=True)
    p.add_argument('--checkpoint',default='artifacts/meta_compress/meta_compress.pt')
    p.add_argument('--device',choices=['cpu','npu'],default='cpu');p.add_argument('--keep-ratio',type=float)
    p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--save-matrix');p.add_argument('--teacher',action='store_true')
    a=p.parse_args();torch.set_num_threads(16)
    state=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);cfg=state['config']
    engine=OfficialUND(device='cpu' if a.device=='cpu' else None)
    module=MetaCompress(feature_dim=cfg['feature_dim']).to(engine.device).eval();module.load_state_dict(state['generator'])
    row={'image':a.image,'question':a.question}
    with torch.no_grad():
        inputs=engine.inputs(row);projected,raw=engine.model.vision_tower(inputs['pixel_values'],inputs['image_grid_thw'])
        source=projected if cfg['insertion']=='post-merger' else raw
        ratio=a.keep_ratio if a.keep_ratio is not None else cfg['keep_ratio']
        compressed,pmat=module(source.float(),inputs['image_grid_thw'],ratio)
        sp=compressed.to(raw.dtype) if cfg['insertion']=='post-merger' else engine.model.vision_tower.merger(compressed.to(raw.dtype))
        si=shortened_inputs(inputs,len(compressed),engine.model.config.image_token_id)
        answer,runtime=generate(engine,si,a.max_new_tokens,sp,compressed.to(raw.dtype))
        result={'answer':answer,'original_tokens':len(source),'compressed_tokens':len(compressed),
                'row_sum_max_error':float((pmat.sum(-1)-1).abs().max()),'insertion':cfg['insertion']}
        if a.teacher:result['teacher_answer']=generate(engine,inputs,a.max_new_tokens,projected,raw)[0]
        if a.save_matrix:
            torch.save({'P':pmat.cpu(),'grid':inputs['image_grid_thw'].cpu(),'image':a.image},a.save_matrix)
            result['matrix_file']=a.save_matrix
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
