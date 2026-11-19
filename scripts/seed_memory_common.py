"""Independent fixed three-seed/four-budget registry and audit utilities."""
from __future__ import annotations
import collections,datetime,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/column_tree_seed_memory_20261113'
REG=BASE/'registration.json'
SEEDS=[20261105,20261106,20261107]
RATIOS=[.1,.25,.5,.75]
KINDS=['col_adaptive','quadtree_question']
LABEL={'col_adaptive':'Adaptive column partitioning','quadtree_question':'Question-conditioned spatial tree'}
NEW_SOURCES=['scripts/seed_memory_common.py','scripts/column_tree_seed_memory_suite.py',
 'scripts/run_seed_memory_stage.py','scripts/eval_seed_memory.py','scripts/profile_seed_memory.py',
 'vqa/stage_memory.py','scripts/report_seed_memory.py','scripts/smoke_seed_memory.py',
 'scripts/slurm_seed_memory_stage.sh','scripts/slurm_seed_memory_controller.sh']
OLD_SOURCES=['vqa/compression_generators_v2.py','vqa/compression_generators.py',
 'vqa/compression_generators_v2_repair.py','vqa/official_und.py','vqa/decoder_dependent_loss.py',
 'vqa/benchmark_metrics.py','vqa/metrics.py','scripts/train_compression_lab_v2.py',
 'scripts/train_compression_lab_v2_distributed.py','scripts/train_compression_lab_v2_repaired.py',
 'scripts/train_compression_lab.py','scripts/train_meta_compress.py','scripts/train_meta_compress_strict.py',
 'scripts/eval_token_proxy.py','scripts/compression_lab_v2_io.py','scripts/compression_lab_v2_io_retry.py',
 'scripts/benchmark_compression_lab_v2_io.py','models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2.py',
 'models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2_vision.py','models/MonkeyOCRv2-B-Und/config.json']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return {} if default is None else default
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
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
def save_lines(p,rr):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr))
def model_id(kind,ratio,seed):return f'{kind}_pre-merger_r{ratio:g}_seed{seed}'
def verify():
 reg=load(REG);assert reg and len(reg['models'])==24
 for p,h in reg['source_sha256'].items():assert sha(ROOT/p)==h,f'Changed source {p}'
 for p,h in reg['data_sha256'].items():assert sha(ROOT/p)==h,f'Changed data {p}'
 for m in reg['models']:
  assert sha(ROOT/m['config'])==m['config_sha256']
  if m['reuse']:assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256']
 return reg
def model_records(m):
 rr=[]
 for p in sorted((ROOT/m['output']/'benchmark').glob('predictions.rank*.jsonl')):
  for r in lines(p):
   if r.get('variant','trained')=='trained':rr.append(r)
 return rr
def summarize(rr):
 groups=collections.defaultdict(list)
 for r in rr:groups[r['task']].append(r['student_score'])
 return dict(questions=len(rr),images=len({r['document_id'] for r in rr}),tasks={t:100*sum(v)/len(v) for t,v in sorted(groups.items())},macro=sum(100*sum(v)/len(v) for v in groups.values())/len(groups))
