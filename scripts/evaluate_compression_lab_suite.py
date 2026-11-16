"""Locked confirmation suite; no model selection on these results."""
import argparse,json
from scripts import train_compression_lab as lab
from scripts.train_meta_compress import init_runtime
from vqa.official_und import OfficialUND,ROOT
from vqa.compression_generators import ConvexGenerator,FixedGenerator
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--list',required=True);a=p.parse_args()
 _,_,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);data=lab.Data(engine)
 original=lab.student
 from scripts.train_compression_lab_pre import student as pre_student
 for case in json.loads((ROOT/a.list).read_text()):
  if 'checkpoint' in case:
   state=torch.load(ROOT/case['checkpoint'],map_location='cpu',weights_only=False);cfg=state['config'];module=ConvexGenerator(cfg['kind']).to(device);module.load_state_dict(state['generator'])
  else:cfg=case['config'];module=FixedGenerator(cfg['kind'])
  lab.student=pre_student if cfg.get('insertion')=='pre-merger' else original
  out=ROOT/case['output'];out.mkdir(parents=True,exist_ok=True)
  print(json.dumps({'case':case}),flush=True)
  lab.evaluate(engine,module,lab.read(ROOT/case['rows']),data,cfg,out,case['stage'],case.get('cohort','confirmation'))
if __name__=='__main__':main()
