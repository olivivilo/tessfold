"""Freeze data/protocol/configs before screening; no outcome-based filtering."""
import hashlib,json
from pathlib import Path
from collections import defaultdict,Counter
from vqa.compression_generators import NAMES
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/compression_lab'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def lines(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def main():
 if (BASE/'protocol.json').exists():print('Existing protocol retained');return
 tasks=['ChartQA','DT-VQA','DeepForm','DocVQA','InfoVQA','KLC','WTQ'];forbidden=set()
 paths=['data/full/val.jsonl','data/full/test.jsonl','artifacts/meta_compress/samples.jsonl','artifacts/meta_compress_search/full_train.jsonl','artifacts/meta_compress_search/search.jsonl','artifacts/meta_compress_search/confirm.jsonl','artifacts/meta_compress_strict/train.jsonl','artifacts/meta_compress_strict/validation.jsonl','artifacts/meta_compress_strict/confirm.jsonl','artifacts/meta_compress_diagnosis/samples.jsonl']
 for p in paths:forbidden.update(r['document_id'] for r in read(ROOT/p))
 groups=defaultdict(list)
 for r in read(ROOT/'data/full/train.jsonl'):
  if r['task'] in tasks and r['document_id'] not in forbidden:groups[(r['task'],r['document_id'])].append(r)
 used=set();meta={}
 for name,images,qa in [('validation',4,1),('confirmation',8,2)]:
  rr=[];counts=Counter()
  keys=sorted(groups,key=lambda k:hashlib.sha256(f'compression-lab-20261105:{name}:{k}'.encode()).hexdigest())
  for task,doc in keys:
   if counts[task]>=images or doc in used:continue
   counts[task]+=1;used.add(doc);rr+=sorted(groups[(task,doc)],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:qa]
  assert all(counts[t]==images for t in tasks),counts
  lines(BASE/f'{name}.jsonl',rr)
 groups=defaultdict(list)
 for r in read(ROOT/'artifacts/meta_compress_strict/pilot.jsonl'):groups[(r['task'],r['document_id'])].append(r)
 counts=Counter();rr=[]
 for task,doc in sorted(groups,key=lambda k:hashlib.sha256(str(k).encode()).hexdigest()):
  if counts[task]>=4:continue
  counts[task]+=1;rr+=groups[(task,doc)][:2]
 lines(BASE/'train.jsonl',rr)
 audit=[];counts=Counter()
 for r in rr:
  if counts[r['task']]<2:audit.append(r);counts[r['task']]+=1
 lines(BASE/'train_audit.jsonl',audit)
 lines(BASE/'expanded_train.jsonl',read(ROOT/'artifacts/meta_compress_strict/train.jsonl'))
 for name in ['train','validation','confirmation','expanded_train']:
  p=BASE/f'{name}.jsonl';r=read(p);meta[name]={'images':len({x['document_id'] for x in r}),'questions':len(r),'task_questions':dict(Counter(x['task'] for x in r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 protocol={'created':'2026-11-05','backbone':'models/MonkeyOCRv2-B-Und; ALL parameters frozen','insertion':'post-merger','design_document':'docs/COMPRESSION_GENERATOR_DESIGNS.md','schemes':NAMES,'ratios':[.1,.25,.5],
 'cohorts':meta,'question_input':'frozen input embedding of question only; <=64 tokens; no decoder hidden state or answer','teacher_and_student_max_new_tokens':64,
 'optimizer':{'name':'AdamW','lr':1e-4,'weight_decay':0.,'clip':1.,'batch':4,'epochs':3,'warmup_fraction':.1,'min_lr_fraction':.1},
 'loss':'full vocabulary KL T=1 on generated teacher common prefix; overlapping-support schemes add .01 relative column-load variance; no entropy loss',
 'gate':{'validation_KL_relative_drop':.05,'validation_gain_vs_initial_points':2.,'must_beat_both_fixed_controls':True,'confirmation_two_seeds':True,'confirmation_cluster_bootstrap_lower_bound':0.,'full_training_only_after_confirmation':True},
 'selection':'screen on new validation only; chosen family per ratio, at most once confirm; no full benchmark selection; initial shared screen is exploratory',
 'prior_data_exclusions':paths,'seed':20261105,'metrics':'task native macro; DeepForm/KLC field exact not document F1; no teacher-correct filtering'}
 dump(BASE/'protocol.json',protocol)
 configs=[]
 for ratio in protocol['ratios']:
  for i,name in enumerate(NAMES):
   tag=f'{i+1:02d}_{name}_r{ratio:g}';cfg={'name':tag,'kind':name,'ratio':ratio,'seed':20261105,'epochs':3,'lr':1e-4,'clip':1.,'batch_size':4,'train_rows':'artifacts/compression_lab/train.jsonl','validation_rows':'artifacts/compression_lab/validation.jsonl','output':f'artifacts/compression_lab/runs/{tag}'}
   path=BASE/'configs'/f'{tag}.json';dump(path,cfg);configs.append(str(path.relative_to(ROOT)))
 dump(BASE/'screen_configs.json',configs);print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
