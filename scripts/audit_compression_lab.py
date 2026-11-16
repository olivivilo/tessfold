"""No-compression identity and question-order controls on real official UND."""
import os,json
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
from scripts.train_compression_lab import Data,read,BASE,evaluate
from scripts.train_meta_compress import init_runtime,atomic_json
from vqa.official_und import OfficialUND
from vqa.compression_generators import FixedGenerator
from scripts.eval_token_proxy import generate

def main():
 _,_,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False);data=Data(engine)
 rr=read(BASE/'validation.jsonl');seen=set();rows=[]
 for r in rr:
  if r['task'] not in seen:rows.append(r);seen.add(r['task'])
 out=BASE/'infrastructure';out.mkdir(exist_ok=True)
 cfg={'kind':'selection','ratio':1.};s=evaluate(engine,FixedGenerator('selection'),rows,data,cfg,out,'identity','audit')
 results=read(out/'audit.identity.jsonl')
 assert max(abs(r['kl']) for r in results)<1e-5
 assert all(r['student_prediction']==r['teacher_prediction'] for r in results),'Identity generation mismatch'
 row=rows[0];prompt,raw,grid,x,response,tl,tp,q=data.get_all(row)
 # A fresh inference after six unrelated questions must match the teacher cache.
 pred,rt=generate(engine,{'input_ids':prompt,'pixel_values':raw.new_empty(0),'image_grid_thw':grid,'attention_mask':torch.ones_like(prompt)},64,x,raw)
 assert rt['generated_ids']==response
 atomic_json(out/'audit.json',{'identity_rows':len(rows),'identity_max_abs_KL':max(abs(r['kl']) for r in results),'identity_answers_equal_teacher':True,'order_after_unrelated_questions_ids_equal':True,'trainable_backbone_parameters':sum(p.numel() for p in engine.model.parameters() if p.requires_grad),'summary':s,'job':os.environ.get('SLURM_JOB_ID')})
 print('AUDIT PASS',flush=True)
if __name__=='__main__':main()
