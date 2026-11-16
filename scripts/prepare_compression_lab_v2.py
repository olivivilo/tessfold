"""Register a second, isolated search with fresh image-disjoint selection/confirmation."""
import hashlib,json
from pathlib import Path
from collections import defaultdict,Counter
from vqa.compression_generators_v2 import NAMES
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/compression_lab_v2'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def lines(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if (BASE/'protocol.json').exists():print('Existing v2 registration retained');return
 tasks=['ChartQA','DT-VQA','DeepForm','DocVQA','InfoVQA','KLC','WTQ']
 excluded=['data/full/val.jsonl','data/full/test.jsonl',
 'artifacts/meta_compress/samples.jsonl','artifacts/meta_compress_search/full_train.jsonl',
 'artifacts/meta_compress_search/search.jsonl','artifacts/meta_compress_search/confirm.jsonl',
 'artifacts/meta_compress_strict/train.jsonl','artifacts/meta_compress_strict/validation.jsonl',
 'artifacts/meta_compress_strict/confirm.jsonl','artifacts/meta_compress_diagnosis/samples.jsonl',
 'artifacts/compression_lab/validation.jsonl','artifacts/compression_lab/confirmation.jsonl']
 forbidden={r['document_id'] for p in excluded for r in read(ROOT/p)}
 groups=defaultdict(list)
 for r in read(ROOT/'data/full/train.jsonl'):
  if r['task'] in tasks and r['document_id'] not in forbidden:groups[(r['task'],r['document_id'])].append(r)
 available=Counter(t for t,d in groups);used=set();cohorts={}
 for cohort,images_per_task,questions_per_image in [('validation',8,1),('confirmation',16,2)]:
  selected=[];counts=Counter()
  keys=sorted(groups,key=lambda k:hashlib.sha256(f'lab-v2-20261105:{cohort}:{k}'.encode()).hexdigest())
  for task,doc in keys:
   if doc in used or counts[task]>=images_per_task:continue
   counts[task]+=1;used.add(doc)
   selected+=sorted(groups[(task,doc)],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:questions_per_image]
  assert all(counts[t]==images_per_task for t in tasks),(cohort,counts,available)
  cohorts[cohort]=selected
 expanded=read(ROOT/'artifacts/compression_lab/expanded_train.jsonl');cohorts['expanded_train']=expanded
 gg=defaultdict(list)
 for row in expanded:gg[(row['task'],row['document_id'])].append(row)
 small=[];counts=Counter()
 for task,doc in sorted(gg,key=lambda k:hashlib.sha256(f'v2-pilot:{k}'.encode()).hexdigest()):
  if counts[task]>=8:continue
  counts[task]+=1;small+=gg[(task,doc)][:2]
 cohorts['train']=small
 audit=[];counts=Counter()
 for row in small:
  if counts[row['task']]<2:audit.append(row);counts[row['task']]+=1
 cohorts['train_audit']=audit
 meta={}
 for cohort,rows in cohorts.items():
  p=BASE/f'{cohort}.jsonl';lines(p,rows)
  meta[cohort]={'images':len({r['document_id'] for r in rows}),'questions':len(rows),'tasks':dict(Counter(r['task'] for r in rows)),'sha256':digest(p)}
 for a,b in [('train','validation'),('train','confirmation'),('validation','confirmation'),('expanded_train','validation'),('expanded_train','confirmation')]:
  assert not {r['document_id'] for r in cohorts[a]}&{r['document_id'] for r in cohorts[b]}
 assert not used&forbidden
 for rows in cohorts.values():
  assert all((ROOT/r['image']).exists() for r in rows)
 protocol={'created':'2026-11-05','schemes':NAMES,'ratios':[.1,.25,.5],
 'positions':['pre-merger','post-merger'],'screen_runs':len(NAMES)*6,'cohorts':meta,
 'backbone':'models/MonkeyOCRv2-B-Und; vision/merger/Qwen all frozen',
 'loss':'full-vocabulary KL(teacher||student), T=1; own teacher greedy prefix; no entropy/column penalty; gold for eval only',
 'optimizer':{'name':'AdamW','lr':.001,'weight_decay':0.,'clip':1.,'batch_size':8,'epochs':6,'warmup_fraction':.1,'min_lr_fraction':.1},
 'generation':{'max_new_tokens':64,'greedy':True,'generation_epochs':[0,3,6],'KL_every_epoch':True},
 'gates':{'screen_validation_KL_drop':.05,'screen_train_audit_KL_drop':.05,'screen_score_gain_points':2.,'must_beat_all_fixed_controls_both_positions':True,
 'expanded_train_images':672,'expanded_epochs':8,'expanded_lr':.0003,'expanded_batch_size':32,'two_seeds':[20261105,20261106],
 'confirmation_images':112,'confirmation_at_most_once_per_budget':True,'paired_cluster_bootstrap_95CI_lower_must_exceed':0.,'full_training_only_after_confirmation':True},
 'selection':'144 exploratory pilots; freeze one architecture/position per budget before 2-seed expansion; freeze final epoch; no test-set selection',
 'excluded_prior_cohorts':excluded,'new_heldout_unused_pool_by_task':dict(available),
 'limitations':'Tiny exploratory pilot; fixed support changes often detached; ST is biased. Novel combinations are hypotheses, not a claim of publication novelty.'}
 dump(BASE/'protocol.json',protocol)
 configs=[]
 # Spread expensive and light variants across shards. All jobs use the same registered optimizer.
 for ratio in protocol['ratios']:
  for i,kind in enumerate(NAMES):
   for position in protocol['positions']:
    tag=f'{i+1:02d}_{kind}_{position}_r{ratio:g}'
    cfg=dict(name=tag,kind=kind,ratio=ratio,insertion=position,phase='v2_screen',seed=20261105,
     epochs=6,lr=.001,clip=1.,batch_size=8,train_rows='artifacts/compression_lab_v2/train.jsonl',
     validation_rows='artifacts/compression_lab_v2/validation.jsonl',audit_rows='artifacts/compression_lab_v2/train_audit.jsonl',
     generation_epochs=[0,3,6],output=f'artifacts/compression_lab_v2/runs/{tag}')
    path=BASE/'configs'/f'{tag}.json';dump(path,cfg);configs.append(str(path.relative_to(ROOT)))
 dump(BASE/'screen_configs.json',configs)
 dump(BASE/'data_integrity.json',{'image_disjoint':True,'all_images_exist':True,'excluded_image_ids':len(forbidden),'fresh_heldout_image_ids':len(used),'cohorts':meta})
 print(json.dumps(meta,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
