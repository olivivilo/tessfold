"""Independent no-retraining system study; fixed requests, checkpoints and gates."""
from __future__ import annotations
import collections,datetime,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/tessfold_system_20261114';REG=BASE/'registration.json'
SOURCES=['vqa/tessfold_runtime.py','scripts/tessfold_system_common.py','scripts/tessfold_system_suite.py',
 'scripts/validate_tessfold_npu.py','scripts/benchmark_tessfold_system.py','scripts/report_tessfold_system.py',
 'scripts/run_tessfold_system_stage.py','scripts/slurm_tessfold_system_stage.sh','scripts/slurm_tessfold_system_controller.sh',
 'scripts/test_tessfold_runtime.py','vqa/stage_memory.py','scripts/repair_supplement_resize.py',
 'vqa/compression_generators_v2.py','vqa/compression_generators.py','vqa/supplement_generators.py',
 'vqa/supplement_s2prune.py','vqa/official_und.py','scripts/eval_token_proxy.py',
 'scripts/benchmark_compression_supplement.py','scripts/eval_compression_supplement.py',
 'scripts/seed_memory_common.py','scripts/profile_seed_memory.py','scripts/train_meta_compress.py',
 'models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2.py','models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2_vision.py',
 'models/MonkeyOCRv2-B-Und/config.json','tessfold.md']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return {} if default is None else default
def lines(p):
 try:ss=Path(p).read_text().splitlines()
 except FileNotFoundError:return []
 rr=[]
 for i,s in enumerate(ss):
  if not s.strip():continue
  try:rr.append(json.loads(s))
  except json.JSONDecodeError:
   if i!=len(ss)-1:raise
 return rr
def records(p):
 for f in sorted(Path(p).glob('predictions.rank*.jsonl')):
  for r in lines(f):
   if r.get('variant','trained')=='trained':yield r

def verify():
 reg=load(REG);assert reg
 for p,h in reg['source_sha256'].items():assert sha(ROOT/p)==h,f'Changed {p}'
 for p,h in reg['data_sha256'].items():assert sha(ROOT/p)==h,f'Changed data {p}'
 for m in reg['models']:assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256']
 return reg

def load_module(engine,kind,ratio):
 import torch
 from vqa.compression_generators_v2 import SpatialGenerator
 m=next(m for m in load(REG)['models'] if m['kind']==kind and m['ratio']==ratio)
 module=SpatialGenerator(kind).to(engine.device).eval();module.load_state_dict(torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=False)['generator'])
 return module
