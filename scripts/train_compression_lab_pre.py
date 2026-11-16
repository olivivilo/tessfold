"""Registered pre-merger placement ablation; no post-insertion conditioning features."""
import sys,os
import torch
from scripts import train_compression_lab as lab
from vqa.compression_generators import FixedGenerator
from vqa.decoder_dependent_loss import common_prefix_logits
from scripts.eval_token_proxy import shortened_inputs
from scripts.train_meta_compress import init_runtime
from vqa.official_und import OfficialUND

def student(engine,module,row,data,cfg):
 prompt,raw,grid,x,response,tl,tp,q=data.get_all(row)
 # Do not give post-merger features to the pre-merger generator. Fixed zeros
 # occupy the unused 2048-channel input adapter; only H and q carry information.
 dummy=raw.new_zeros((len(raw),2048),dtype=torch.float32)
 _,p=module(dummy,grid,cfg['ratio'],raw=raw,question=q)
 compressed=p@raw.float();sp=engine.model.vision_tower.merger(compressed.to(raw.dtype))
 ids=shortened_inputs({'input_ids':prompt},len(compressed),engine.model.config.image_token_id)['input_ids']
 sl=common_prefix_logits(engine.model,ids,sp[None],response)
 logt=tl.detach().float().log_softmax(-1);logs=sl.float().log_softmax(-1)
 kl=(logt.exp()*(logt-logs)).sum(-1).mean()
 col=p.sum(0)/(p.shape[0]/p.shape[1]);load=(col-1).square().mean()
 return {'kl':kl,'load_variance':load,'total':kl},(p,sp,ids,raw,grid,tp,response)

lab.student=student
lab.SOURCES=lab.SOURCES+['scripts/train_compression_lab_pre.py']
if __name__=='__main__':
 if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1]=='baseline':
  _,_,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);data=lab.Data(engine)
  for ratio in [.1,.25,.5]:
   for kind in ['mean','selection']:
    cfg={'kind':kind,'ratio':ratio,'insertion':'pre-merger'};out=lab.BASE/'baselines_pre'/f'{kind}_r{ratio:g}';out.mkdir(parents=True,exist_ok=True)
    lab.evaluate(engine,FixedGenerator(kind),lab.read(lab.BASE/'validation.jsonl'),data,cfg,out,'fixed')
 else:lab.main()
