"""Independent no-retraining system study; fixed requests, checkpoints and gates."""
from __future__ import annotations
import collections,datetime,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/tessfold_system_repair_20261115';REG=BASE/'registration.json'
SOURCES=['scripts/tessfold_system_common.py', 'scripts/tessfold_system_suite.py', 'scripts/benchmark_tessfold_system.py', 'scripts/validate_tessfold_npu.py', 'scripts/report_tessfold_system.py', 'scripts/run_tessfold_system_stage.py', 'scripts/slurm_tessfold_system_stage.sh', 'scripts/slurm_tessfold_system_controller.sh', 'scripts/tessfold_repair/common.py', 'scripts/tessfold_repair/suite.py', 'scripts/tessfold_repair/benchmark.py', 'scripts/tessfold_repair/validate.py', 'scripts/tessfold_repair/report.py', 'scripts/tessfold_repair/run_stage.py', 'scripts/tessfold_repair/stage.sh', 'scripts/tessfold_repair/controller.sh', 'scripts/tessfold_repair/__init__.py', 'vqa/tessfold_generation_compat.py', 'scripts/test_tessfold_generation_compat.py']
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
 from scripts.tessfold_system_common import verify as verify_parent
 verify_parent()
 reg=load(REG);assert reg
 assert sha(ROOT/reg['parent_registration'])==reg['parent_registration_sha256']
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
