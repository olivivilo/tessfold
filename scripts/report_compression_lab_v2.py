"""Generate second-round tables exclusively from committed experiment artifacts."""
import csv,json,datetime,os
from pathlib import Path
from collections import Counter
from scripts.prepare_compression_lab_v2 import ROOT,BASE
from scripts.train_compression_lab_v2 import read
CONTROL_CACHE={}
def refresh(path):
 # The shared FS can cache negative lookups on CPU/NPU nodes independently.
 # Readdir the parent and directory before checking newly committed result files.
 try:list(path.parent.iterdir());list(path.iterdir())
 except FileNotFoundError:pass
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def fmt(v):return '—' if v is None else f'{v:.3f}'
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def controls(ratio,cohort='validation'):
 key=(ratio,cohort)
 if key in CONTROL_CACHE:return CONTROL_CACHE[key]
 folder=BASE/'baselines' if cohort=='validation' else BASE/'confirmation_baselines'
 result=[load(folder/f'{position}_{kind}_r{ratio:g}'/f'{cohort}.fixed.summary.json').get('student_macro')
 for position in ['pre-merger','post-merger'] for kind in ['mean','selection','cell_mean','cell_selection']]
 CONTROL_CACHE[key]=result;return result
def export_answers():
 """Include only actual generation, with explicit run/stage/cohort; no seed averaging."""
 fields=['run','kind','insertion','ratio','id','document_id','task','cohort','stage','question','answers',
 'teacher_prediction','student_prediction','teacher_score','student_score','kl','tokens','compressed_tokens',
 'entropy','row_max_mean','distinct_row_argmax','hit_token_limit','teacher_hit_token_limit']
 configs=[load(ROOT/p) for name in ['screen_configs.json','expanded_configs.json','full_configs.json'] for p in load(BASE/name)]
 answer_tmp=BASE/f'answers.{os.getpid()}.tmp'
 with answer_tmp.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for cfg in configs:
   out=ROOT/cfg['output']
   for pattern in ['validation.*.jsonl','confirmation.*.jsonl']:
    for path in sorted(out.glob(pattern)):
     for row in read(path):
      if row.get('student_score') is None:continue
      writer.writerow({'run':cfg['name'],'kind':cfg['kind'],'insertion':cfg['insertion'],'ratio':cfg['ratio'],**row,'answers':json.dumps(row['answers'],ensure_ascii=False)})
 answer_tmp.replace(BASE/'answers.csv')
def gate(cfg):
 out=ROOT/cfg['output'];a=load(out/'validation.initial.summary.json');b=load(out/f'validation.epoch{cfg["epochs"]}.summary.json')
 ta=load(out/'train_audit.initial.summary.json');tb=load(out/f'train_audit.epoch{cfg["epochs"]}.summary.json');cc=controls(cfg['ratio']);reasons=[]
 if not all([a,b,ta,tb]) or None in cc or 'student_macro' not in b:return {'pass':False,'reasons':['incomplete']}
 if b['kl']>a['kl']*.95:reasons.append('validation KL decrease <5%')
 if tb['kl']>ta['kl']*.95:reasons.append('train audit KL decrease <5%')
 if b['student_macro']<a['student_macro']+2:reasons.append('score gain <2points')
 if b['student_macro']<=max(cc):reasons.append('not above all eight fixed controls')
 return {'pass':not reasons,'reasons':reasons,'initial':a,'final':b,'fixed_controls':cc}
def main():
 CONTROL_CACHE.clear()
 refresh(BASE)
 for folder in ['runs','expanded','full','confirmation_baselines']:refresh(BASE/folder)
 configs=[load(ROOT/p) for listname in ['screen_configs.json','expanded_configs.json','full_configs.json'] for p in load(BASE/listname)]
 stages=load(BASE/'stages.json').get('ratios',{})
 from scripts.compression_lab_v2_jobs import live_jobs
 queued=live_jobs()
 records=[]
 for cfg in configs:
  out=ROOT/cfg['output'];refresh(out);a=load(out/'validation.initial.summary.json');b=load(out/f'validation.epoch{cfg["epochs"]}.summary.json');training=read(out/'training.jsonl');gg=gate(cfg)
  status='Completed' if (out/'complete.json').exists() else 'Infrastructure failure' if (out/'failure.txt').exists() else 'Running' if a or training else 'Not started'
  if cfg['phase']=='v2_full_20k' and status!='Completed':
   job=stages.get(f'{cfg["ratio"]:g}',{}).get('full_job')
   status='Training / pending allocation' if str(job) in queued else 'Interrupted / awaiting resume'
  cc=controls(cfg['ratio']);best=max(cc) if None not in cc else None
  records.append({'name':cfg['name'],'kind':cfg['kind'],'insertion':cfg['insertion'],'ratio':cfg['ratio'],'phase':cfg['phase'],'seed':cfg['seed'],
   'status':status,'initial_score':a.get('student_macro'),'final_score':b.get('student_macro'),'teacher_score':a.get('teacher_macro'),
   'gain':b['student_macro']-a['student_macro'] if 'student_macro' in a and 'student_macro' in b else None,
   'initial_kl':a.get('kl'),'final_kl':b.get('kl'),'kl_drop_percent':100*(1-b['kl']/a['kl']) if a and b else None,
   'fixed_best':best,'gate':gg['pass'],'gate_reasons':'; '.join(gg['reasons']),'steps':len(training),
   'max_gradient':max((r['grad_norm'] for r in training),default=None),
   'median_clip':sorted(r['clip_multiplier'] for r in training)[len(training)//2] if training else None,
   'total_update_l2':sum(r['update_l2'] for r in training)})
 if records:
  csv_tmp=BASE/f'summary.{os.getpid()}.csv.tmp'
  with csv_tmp.open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
  csv_tmp.replace(BASE/'summary.csv')
 summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned':len(records),'status':dict(Counter(r['status'] for r in records)),
 'pilot_passed':[r['name'] for r in records if r['gate'] and r['phase']=='v2_screen'],'records':records}
 save(BASE/'summary.json',summary)
 stages=load(BASE/'stages.json').get('ratios',{});protocol=load(BASE/'protocol.json')
 doc=['# Second-Round Compression Generators: 24 Designs, Protocol, Procedure, and Results','',f'Updated (UTC): {summary["updated_utc"]}; training-configuration status: `{json.dumps(summary["status"],ensure_ascii=False)}`. Registration: {len(records)} runs; unexecuted results remain blank.',
 '', '**Entry point: `bash scripts/run_compression_lab_v2.sh status`**. Architectures: [COMPRESSION_GENERATOR_V2_DESIGNS.md](COMPRESSION_GENERATOR_V2_DESIGNS.md). Generator source:  `vqa/compression_generators_v2.py`and its numerical-repair module; actual training entry `scripts/train_compression_lab_v2_repaired.py`, shared training loop in `train_compression_lab_v2.py`; use the repaired sparse_relu entry rather than bypass it. All experiments use venv; NPU uses `.venv-und`; CPU reports use `.venv`. Earlier experiments are not overwritten.','',
 '## Current Stage','', '|Retention / token-compression factor|Stage|Selected family / insertion|','|---|---|---|']
 for ratio in [.1,.25,.5]:
  state=stages.get(f'{ratio:g}',{});chosen=state.get('selected_pilot',{})
  doc.append(f'|{ratio:.0%} / {1/ratio:g}×|{state.get("stage","Screening / not yet selected")}|{chosen.get("kind","—")} / {chosen.get("insertion","—")}|')
 doc+=['','Completed full-screening runs: '+str(sum(r['status']=='Completed' and r['phase']=='v2_screen' for r in records))+'/144; eligible configurations: '+str(len(summary['pilot_passed']))+'/144. Passing counts per budget: '+', '.join(f'{ratio:.0%}: '+str(sum(r['gate'] and r['phase']=='v2_screen' and r['ratio']==ratio for r in records))+'/48' for ratio in [.1,.25,.5])+'. Passing means satisfying this round’s criteria, not near-teacher quality.']
 doc+=['','|Full-training retention fraction|Recorded / planned updates|Latest complete epoch|Selection-set generation score at that epoch|','|---|---:|---:|---:|']
 for ratio,state in stages.items():
  if not state.get('full_config'):continue
  cfg=load(ROOT/state['full_config']);directory=ROOT/cfg['output'];logs=read(directory/'training.jsonl')
  epoch=max((e for e in range(cfg['epochs']+1) if load(directory/f'validation.{"initial" if e==0 else "epoch"+str(e)}.summary.json').get('student_macro') is not None),default=0)
  result=load(directory/f'validation.{"initial" if epoch==0 else "epoch"+str(epoch)}.summary.json')
  doc.append(f'|{float(ratio):.0%}|{max((r["step"] for r in logs),default=0)} / 2500|{epoch} / {cfg["epochs"]}|'+fmt(result.get('student_macro'))+'|')
 doc+=['','Epoch 0 is initialization. Logged steps need not be committed checkpoint steps; resume from the designated checkpoint. These are progress records, not final formal results.']
 doc+=['','## Fixed Screening Candidates (Model-Selection Scores)','', 'Display completed candidates by final-epoch score only. Freeze no model before all same-budget runs finish, and do not equate the maximum with eligibility. Budgets share the new screening set, not the first round’s set.','',
 '|Retention fraction|Candidate / insertion point|Initialization → final epoch|Gain|Uncompressed teacher|Strongest fixed control|Complete screening criterion|','|---|---|---|---:|---:|---:|---|']
 for ratio in [.1,.25,.5]:
  done=[r for r in records if r['phase']=='v2_screen' and r['ratio']==ratio and r['status']=='Completed']
  if not done:continue
  best=max(done,key=lambda r:r['final_score'])
  doc.append(f'|{ratio:.0%}|{best["kind"]} / {best["insertion"]}|{fmt(best["initial_score"])} → {fmt(best["final_score"])}|{fmt(best["gain"])}|{fmt(best["teacher_score"])}|{fmt(best["fixed_best"])}|'+('Passed' if best['gate'] else best['gate_reasons'])+'|')
 doc+=['','## Comparison of All 24 Methods at a Fixed Final Epoch','',
 'Same new screening set, fixed epoch 6, no seed averaging. ✓ requires training/validation KL and answer criteria. Screening maxima select only as registered, not as independent test results.','',
 '|Method|10% pre|10% post|25% pre|25% post|50% pre|50% post|','|---|---:|---:|---:|---:|---:|---:|']
 names=list(dict.fromkeys(r['kind'] for r in records if r['phase']=='v2_screen'))
 lookup={(r['kind'],r['ratio'],r['insertion']):r for r in records if r['phase']=='v2_screen'}
 for name in names:
  cells=[]
  for ratio in [.1,.25,.5]:
   for position in ['pre-merger','post-merger']:
    r=lookup.get((name,ratio,position),{});cells.append(fmt(r.get('final_score'))+(' ✓' if r.get('gate') else ''))
  doc.append('|'+name+'|'+'|'.join(cells)+'|')
 doc+=['','[Measured 24×6 score heatmap PDF](../artifacts/compression_lab_v2/plots/screen_score_matrix.pdf) · [PNG](../artifacts/compression_lab_v2/plots/screen_score_matrix.png). ']
 doc+=['','## Data and Freezing Protocol','', 'The official MonkeyOCRv2-B-Und encoder, merger, and Qwen3-1.7B are frozen; only P is trained. pre outputs merger(PH), post outputs PX; pre does not read X. Teacher is uncompressed; teacher/student generate independently per question, greedily, with cap=64. Scores are task-native macro percentages; DeepForm/KLC use field exact match, **not official document-level F1**. Teacher-prefix KL is not substituted for independent-answer scores.', '',
 'Scoring: raw-character ANLS for DocVQA/InfoVQA; 5%-tolerance relaxed accuracy for ChartQA; max(answer-containment accuracy, ANLS) for DT-VQA; DUE denotation accuracy for WTQ; field exact match for DeepForm/KLC. Screening/confirmation equally weight seven tasks. Formal evaluation adds OCRBench with native answer-containment scoring and equally weights eight tasks. Absolute means across different sets/task scopes are not training improvements or regressions.','',
 '|Cohort|Images|QA|Purpose|','|---|---:|---:|---|']
 for name,meta in protocol.get('cohorts',{}).items():doc.append(f'|{name}|{meta["images"]}|{meta["questions"]}|'+{'train':'144 screening runs','validation':'New model-selection set','confirmation':'New independent confirmation, used only after candidates are fixed','expanded_train':'Two-seed expanded training','train_audit':'Fixed training-loss audit'}[name]+'|')
 doc+=['','Data come from existing DocVQA/InfoVQA/ChartQA/DT-VQA/WTQ/DeepForm/KLC training sources, without MonkeyDocV2 OCR-only objectives. New held-out images exclude all first-round held-outs, earlier tuning/confirmation, old 20k, and formal val/test. Split SHA values and image existence/disjointness checks are in  `protocol.json`, `data_integrity.json`. This does not guarantee that official pretraining never saw the source data.', '',
 '## Training and Selection Protocol','', 'Screen 24 architectures × 2 positions × 3 budgets =144 fresh initializations. AdamW LR=1e-3, wd=0, batch=8, 6 epochs, 10% warmup/cosine to 0.1 LR, global clip=1. Full-vocabulary KL(teacher ||student), T=1, with teacher-greedy response as shared prefix; no entropy/column-collapse terms. Log validation/train-audit KL each epoch and independently generate at epochs 0/3/6. Epoch 6 is fixed, not retrospectively selected as the best intermediate epoch.', '',
 'Screening requires both KL measures to fall ≥5%, generation macro to improve ≥2 points from initial, and performance above all 8 fixed controls: 1D mean/selection and 2D cell_mean/cell_selection at both pre/post positions. Select one architecture/position per budget, then train fresh on 672 images with seeds 20261105/20261106 for 8 epochs, LR=3e-4, batch=32, and 4-NPU DDP, retaining scheduler/clipping. Only after both seeds pass is the new 112-image confirmation used. Enter original-20k training only if the paired image-cluster 95% CI for gain over each initialization has lower bound >0 and scores beat fixed controls. Failure blocks automatic full training. Separately authorized extensions are registered independently and retain the failed original verdict.','',
 'Before first use of the new confirmation set, full_training_addendum.json registers fresh full training of each confirmed family: original 20k, seed=20261105, 4 epochs, LR=3e-4, global batch=32, clip=1, unchanged warmup/cosine. The 20,000 QA / 16,731 images comprise DT-VQA 9883, DocVQA 3204, ChartQA 2337, InfoVQA 2021, WTQ 1153, KLC 1109, DeepForm 293. Original shuffling is retained without task balancing, so training mixture differs from macro-metric weighting. Only fixed epoch 4 receives 23,565-QA formal evaluation at cap=256, including teacher/initial/trained and same-position mean/selection/cell_mean/cell_selection. Formal scores do not select models. Formal controls are four at the selected position; eligibility always compares eight across positions.', '',
 'Discrete partitions/order are detached; hard selection uses a biased straight-through estimator, as documented in the design table. Step logs retain gradients, clipping multipliers, and actual updates. Screening begins only after every candidate/position passes real-UND gradient checks; infrastructure errors are not scientific negative results. Screening moved from 8 to 14 independent NPU processes, still one NPU/run, resuming committed epoch checkpoints without optimizer/data/order changes and replaying uncommitted epochs. Job history is in initial_submissions.json.','',
 'Resources vary by stage: 8 NPUs for 10%, 4 for 25%, and 4 for formal evaluation; global batch=32, ordering, objective, LR, seed, and fixed epoch remain unchanged. Requests for 16 NPUs were withdrawn before execution because no full nodes were available. A 25% eight-NPU run was interrupted by I/O before its first epoch; insufficient resources then required returning to the planned four ranks with same-seed replay. Uncommitted logs, initialization, and source manifests remain in `infrastructure/`, not restarted according to scores. The 2,500 global batches cover identical samples across 4/8/16 ranks; audit: `infrastructure/full_compute_repartition.json`; 10%-budget initialization is tensorwise bit-exact across 4/8 NPUs; see `full_initialization_identity.json`. Gradient accumulation order differs, so bitwise training reproduction is not claimed. Expanded training still uses 4 NPUs.','',
 '## Numerical Repair and Preservation of Original Failures','', 'Original sparse_relu adds 1e-12 to denominators of tiny positive-weight rows, violating unit row sums; evaluation assertions caught this. Repair module `vqa/compression_generators_v2_repair.py`Exactly renormalize nonempty P rows; empty rows still fall back to 2D means. Rerun all 6 configurations from scratch without old-checkpoint mixing. Original two failures/logs remain in `infrastructure/failed_sparse_original/`, details `sparse_repair.json`. This is an implementation error, not a negative architecture result. The other 23 source signatures/checkpoints remain unchanged. The training entry point is `train_compression_lab_v2_repaired.py`explicitly dispatches the repair and includes its source in affected run manifests.', '',
 'Full stages encounter concurrent cache EEXIST: jobs compute the same uncached teacher entry, another process commits first, and shared-FS rename fails. Add a pure-I/O entry point `scripts/compression_lab_v2_io.py`(formal evaluation uses `benchmark_compression_lab_v2_io.py`): only for immutable features/teachers/logits EEXIST, accept an already committed file when every tensor/metadata field is bit-identical to the pending object. Content mismatches and checkpoint errors remain failures. Independent boundary checks: `infrastructure/cache_eexist_guard_test.json`, failure/replay logs: `cache_eexist_repair.json`and original Slurm logs. Actual wrapper SHA is additionally recorded per run in `io_runtime.rank*.json`, scientific-source SHA unchanged; replay uncommitted updates from fixed checkpoints, never by score.','',
 'Another shared-FS issue is ENOENT immediately after a successful existence check; versioned entry `compression_lab_v2_io_retry.py`Only immutable caches receive bounded directory-refresh/retry; genuine missing files still fail and checkpoints are not retried. Records: `cache_read_visibility_repair.json`, wrapper SHA saved separately `io_retry_runtime.rank*.json`, without editing signed scientific sources.','',
 'Spare NPUs precomputed original-20k frozen teachers in reverse first-epoch order, then stopped for resource contention while retaining committed cache. Entry point: `scripts/slurm_compression_lab_v2_prefetch.sh`, recorded in `infrastructure/full_cache_prefetch/`and`full_cache_prefetch_submission.json`. This makes zero model updates, computes no candidate scores, and changes neither samples/order nor KL.','',
 '## Fixed Controls on the New Screening Set','', '|Retention fraction|pre mean|pre selection|pre cell mean|pre cell selection|post mean|post selection|post cell mean|post cell selection|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for ratio in [.1,.25,.5]:doc.append(f'|{ratio:.0%}|'+ '|'.join(fmt(v) for v in controls(ratio))+'|')
 doc+=['','## Previously Trained Models on the Same New Screening Set','', 'Evaluate only the four previously completed expanded models at their fixed epoch 8, without reselecting old checkpoints. This same-set reference does not change current preregistered criteria or reselect models on old confirmation.','',
 '|Previous-round model|New-screening-set macro|KL|','|---|---:|---:|']
 for path in sorted((BASE/'legacy_controls').glob('*/validation.epoch8.summary.json')):
  val=load(path);doc.append(f'|{path.parent.name}|{fmt(val["student_macro"])}|{fmt(val["kl"])}|')
 doc+=['','## Training Results for Each Method','', 'Scores/gains are percentage points; KL reduction is percent. Passing below means only stage-level continuous-loss/generation eligibility; independent confirmation is in the stage table.','',
 '|Method / stage / seed|Insertion point|Retention fraction|Status|Initial score|Final-epoch score|Gain|KL: initial → final|KL reduction (%)|Strong fixed controls|Stage eligibility criterion|','|---|---|---:|---|---:|---:|---:|---|---:|---:|---|']
 for r in records:doc.append(f'|{r["kind"]} / {r["phase"]} / {r["seed"]}|{r["insertion"]}|{r["ratio"]:.0%}|{r["status"]}|{fmt(r["initial_score"])}|{fmt(r["final_score"])}|{fmt(r["gain"])}|{fmt(r["initial_kl"])} → {fmt(r["final_kl"])}|{fmt(r["kl_drop_percent"])}|{fmt(r["fixed_best"])}|'+('Passed' if r['gate'] else r['gate_reasons'])+'|')
 doc+=['','## Independent Confirmation','', 'The CI tests improvement over same-seed initialization. Beating fixed controls is a point-estimate criterion, not a tested significant difference from them. Report both seeds separately rather than averaging away a failure.', '',
 '|Retention fraction|seed|Initialization|Trained|Uncompressed teacher|Gain over initialization|Image-cluster 95% CI|Strongest fixed control|Passed|','|---|---:|---:|---:|---:|---:|---|---:|---|']
 anyconfirmation=False
 for ratio,state in stages.items():
  for check in state.get('confirmation_checks',[]):
   anyconfirmation=True;paired=check['paired_vs_initial'];ci=paired['cluster_bootstrap_95ci_points']
   doc.append(f'|{float(ratio):.0%}|{check["seed"]}|{fmt(check.get("initial_score"))}|{fmt(check["score"])}|{fmt(check.get("teacher_score"))}|{fmt(paired["gain_points"])}|[{fmt(ci[0])}, {fmt(ci[1])}]|{fmt(max(check["fixed_controls"]))}|'+('Yes' if check['pass'] else 'No')+'|')
 if not anyconfirmation:doc+=['','No completed independent confirmation; validation success is not established.']
 if anyconfirmation:
  doc+=['','### Claims Supported by the Current Evidence','',
   'Adaptive row/column/rectangular coverage and global selection lead screening. More complex prototypes, Sinkhorn, or deformable routing do not automatically score higher. This compares common settings, not each family after exhaustive tuning.','']
  for ratio,state in stages.items():
   checks=state.get('confirmation_checks',[])
   if not checks:continue
   for c in checks:
    teacher=c.get('teacher_score');retention=100*c['score']/teacher if teacher else None
    doc.append(f'- Retain {float(ratio):.0%} / seed{c["seed"]}: confirmation score {fmt(c["score"])}, drop from teacher {fmt(teacher-c["score"])} points; score retention {fmt(retention)}%; gain over initialization {fmt(c["paired_vs_initial"]["gain_points"])} points.')
   if not all(c['pass'] for c in checks):
    doc+=['',f'Retain{float(ratio):.0%}Both seeds do not satisfy preregistered eligibility, so automatic full training stops for that budget. A zero-crossing CI means insufficient evidence, not complete ineffectiveness; do not discard the failed seed.']
  doc+=['','Learning gains at 10%/25% do not establish near-teacher quality under high compression; inspect absolute gaps and full benchmarks. Confirmation uses 112 images/209 QA and seven-task macros. With this small sample, formal evaluation must be reported separately.']
 if anyconfirmation:
  doc+=['','### Per-Task Results on the Independent Confirmation Set','',
   'Retain each seed separately. Task scores explain locked-model behavior and do not reselect architecture, epoch, or seed.','',
   '|Retention fraction|seed|Task|Questions|teacher|Initial|Trained|Improved / tied / degraded question counts|','|---|---:|---|---:|---:|---:|---:|---|']
  for ratio,state in stages.items():
   if not state.get('confirmation_checks'):continue
   for cfgpath in state['expanded_configs']:
    cfg=load(ROOT/cfgpath);out=ROOT/cfg['output'];before={r['id']:r for r in read(out/'confirmation.initial.jsonl')};after=read(out/'confirmation.epoch8.jsonl')
    for task in sorted({r['task'] for r in after}):
     rr=[r for r in after if r['task']==task];n=len(rr)
     t=100*sum(r['teacher_score'] for r in rr)/n;a=100*sum(before[r['id']]['student_score'] for r in rr)/n;b=100*sum(r['student_score'] for r in rr)/n
     differences=[r['student_score']-before[r['id']]['student_score'] for r in rr]
     counts=[sum(d>1e-12 for d in differences),sum(abs(d)<=1e-12 for d in differences),sum(d< -1e-12 for d in differences)]
     doc.append(f'|{float(ratio):.0%}|{cfg["seed"]}|{task}|{n}|{fmt(t)}|{fmt(a)}|{fmt(b)}|'+ ' / '.join(map(str,counts))+'|')
 doc+=['','## Actual Matrix Statistics for the Selected Generator','',
  'On the same 56-question screening set, report actual P only for runs with complete initial/fixed-final results. Entropy is mean row entropy; maximum weight is mean row maximum. These are post-training descriptions, not reselection criteria. Do not average the two seeds.','',
  '|Stage / retention / seed|Mean row entropy: initial → final|Mean maximum row weight: initial → final|Final minimum distinct-argmax row fraction|Final maximum row-sum error|','|---|---:|---:|---:|---:|']
 for state in stages.values():
  paths=list(state.get('expanded_configs',[]))
  if state.get('full_config'):paths.append(state['full_config'])
  for path in paths:
   cfg=load(ROOT/path);out=ROOT/cfg['output']
   if not (out/'complete.json').exists():continue
   aa=read(out/'validation.initial.jsonl');bb=read(out/f'validation.epoch{cfg["epochs"]}.jsonl')
   if len(aa)!=56 or len(bb)!=56:continue
   def avg(rr,key):return sum(r[key] for r in rr)/len(rr)
   unique=min(r['distinct_row_argmax']/r['compressed_tokens'] for r in bb)
   error=max(r['row_sum_error'] for r in bb)
   doc.append(f'|{cfg["phase"]} / {cfg["ratio"]:.0%} / {cfg["seed"]}|{avg(aa,"entropy"):.4f} → {avg(bb,"entropy"):.4f}|{avg(aa,"row_max_mean"):.4f} → {avg(bb,"row_max_mean"):.4f}|{unique:.4f}|{error:.2e}|')
 doc+=['','global_topk uses one-hot forward/evaluation, so row entropy is always 0. Training improves which original tokens are selected, not entropy. Larger within-region maxima indicate sharper adaptive-region weights, but no entropy regularizer is used. P statistics alone do not establish VQA quality.']
 doc+=['','## Full Training and Formal Evaluation','', 'Original workflow registers 20k only after confirmation. The separately requested 50% extension retains the failed confirmation verdict; see [50% extension protocol](COMPRESSION_GENERATOR_V2_KEEP50_FULL.md). Formal evaluation fixes 23,565 QA and cap=256 and never selects models. Blank entries mean no complete formal result; screening or 672-image scores are not substitutes.', '',
 '|Retention fraction|teacher|initial|trained|mean|selection|cell_mean|cell_selection|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for ratio,state in stages.items():
  full=state.get('benchmark_summary',{})
  if full:
   doc.append(f'|{float(ratio):.0%}|'+ '|'.join(fmt(full['macro'].get(v)) for v in ['teacher','initial','trained','mean','selection','cell_mean','cell_selection'])+'|')
 for ratio,state in stages.items():
  full=state.get('benchmark_summary',{})
  if full:
   mm=full['macro'];retention=100*mm['trained']/mm['teacher'] if mm['teacher'] else None
   doc+=['',f'Retention fraction{float(ratio):.0%}: trained drop from teacher {fmt(mm["teacher"]-mm["trained"])} points; score retention {fmt(retention)}%; improvement over the strongest fixed control at this insertion point: {fmt(mm["trained"]-max(mm[k] for k in ["mean","selection","cell_mean","cell_selection"]))} points. Score retention is not execution speed.']
 for ratio,state in stages.items():
  full=state.get('benchmark_summary',{})
  if full:
   doc+=['',f'Retention fraction{float(ratio):.0%}Per-task scores: ','', '|Task|teacher|initial|trained|mean|selection|cell_mean|cell_selection|','|---|---:|---:|---:|---:|---:|---:|---:|']
   for task,values in sorted(full['tasks'].items()):doc.append(f'|{task}|'+ '|'.join(fmt(values.get(v)) for v in ['teacher','initial','trained','mean','selection','cell_mean','cell_selection'])+'|')
   cfg=load(ROOT/state['full_config']);rel='../'+cfg['output']+'/benchmark'
   doc+=['',f'Paired trained-versus-initial: `{json.dumps(full["paired_trained_vs_initial"],ensure_ascii=False)}`. Formal per-question answers: [CSV]({rel}/answers.csv) · [Image/gold/teacher/initial/trained comparison]({rel}/review.html). ','']
   if 'student_token_limit_counts' in full:
    doc+=['',f'Questions reaching cap=256 by student variant: `{json.dumps(full["student_token_limit_counts"],ensure_ascii=False)}`, teacher {full["teacher_token_limit_count"]} questions. All remain scored, without deletion or shorter-output substitution.','']
 recovery=load(BASE/'infrastructure/recovery_20261106/recovery.json')
 if recovery:
  doc+=['','## 2026-11-06 Recovery from Interruption','',
   'Query the live user queue and treat stale job IDs as exited, while reporting scheduler errors. Persist transitions before auxiliary-cache cleanup so cleanup failures do not lose evaluation job IDs. Permit at most 3 full-stage recoveries without architecture/seed/epoch reselection.','',
   'For immutable features/teachers/logits caches, ENOENT triggers at most 12 directory refreshes and bounded backoff, followed by loading through an open handle; genuinely missing files still fail. Concurrent EEXIST requires tensor and metadata equality. Checkpoints are not treated as caches, and read/content errors are not suppressed.','',
   'The original 10% run logged 463 steps without committing an epoch checkpoint, so it is replayed from initialization with the same seed and ordering. Earlier logs, initial weights, manifest, and sources remain in infrastructure/recovery_20261106/. The only training-loop addition is committing latest.pt every 25 steps; fixed epoch-4 selection, data, KL, optimizer, and scheduler remain unchanged. Completed 25% checkpoints are not retrained.','',
   'Ascend replaces torch.load during import, so the cache patch is installed after importing the backend and verified to remain active after initialization on a real NPU. Replayed initial tensors match exactly; periodic checkpoints contain optimizer state, and recovery from step 150 succeeded. The first replayed update matches the original log, but later NPU trajectories are not bit-exact; complete determinism is not claimed.','',
   '[Recovery records and source hashes](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/recovery.json) · [Failure-boundary validation](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/checks.json) · [NPU backend installation-order check](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/npu_backend_order.json) · [Checkpoint checks](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/checkpoint_audit.json) · [Numerical replay comparisons](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/replay_trajectory_audit.json). Formal evaluation uses 23,565 questions/budget and cap=256, with four controls at the selected insertion point; small-screening scores never replace formal results.']
 smoke=load(BASE/'smoke/complete.json')
 doc+=['','## Audits, Per-Question Answers, and Curves','',f'Real-UND gradient checks: {smoke.get("cases",0)}/48 (24 architectures × 2 positions); details:  `smoke/results.json`; zero trainable backbone parameters/all backbone grads absent, nonnegative unit-sum P, and actual generator updates.', '',
 'Retain initial/epoch*/latest checkpoints, source/data hashes, training.jsonl, and fixed train-audit/validation KL per epoch. Per-question answers:  `runs/<name>/validation.{initial,epoch3,epoch6}.jsonl`, expanded experiment in  `expanded/<name>/`, independent confirmation in  `confirmation.*.jsonl`; all include question/gold/teacher/student/score/KL/P statistics.', '',
 'Summary CSV: [summary.csv](../artifacts/compression_lab_v2/summary.csv). Per-question summary: [answers.csv](../artifacts/compression_lab_v2/answers.csv) (updated by report execution; actual generations only). [Browse image/gold/teacher/student](../artifacts/compression_lab_v2/review.html)Filter budget, insertion, method, task, and cohort. Per-epoch training/validation curves:  `artifacts/compression_lab_v2/plots/`. ','',
 '|Screening retention fraction|Pre-merger training/validation KL|Post-merger training/validation KL|Pre-merger answer score|Post-merger answer score|','|---|---|---|---|---|',
 *[f'|{ratio:.0%}|'+ '|'.join(f'[{metric} {position}](../artifacts/compression_lab_v2/plots/{metric}_v2_screen_{position}_keep{ratio:g}.pdf)' for metric,position in [('loss','pre-merger'),('loss','post-merger'),('score','pre-merger'),('score','post-merger')])+'|' for ratio in [.1,.25,.5]], '',
 '```bash','bash scripts/run_compression_lab_v2.sh status','bash scripts/run_compression_lab_v2.sh report','bash scripts/run_compression_lab_v2.sh plot','bash scripts/run_compression_lab_v2.sh resume','```','',
 'Relative to the earlier round, this round changes support, parts of the network, head-initialization scale, and screening data/update count together. Cross-round improvements cannot be attributed solely to 2D support. Evaluating old models on the same set reduces dataset confounding but is not a single-factor ablation. Screening maxima have multiple-candidate selection bias. Confirmation intervals are task-stratified/image-clustered without adjustment across three budgets; formal tests do not support repeated model selection.', '',
 'Report completed evidence only. Independent confirmation/final benchmarks determine success at high compression, not design count or falling loss.']
 target=ROOT/'docs/COMPRESSION_GENERATOR_V2_EXPERIMENTS.md';tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
 print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
if __name__=='__main__':
 main()
 if '--answers' in __import__('sys').argv:export_answers()
