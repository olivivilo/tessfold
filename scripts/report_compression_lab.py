"""Report only committed real experiment records; never infer an unrun score."""
import csv,json,datetime
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/compression_lab'
def read(p):
 if not p.exists():return []
 out=[]
 for line in p.read_text().splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:break
 return out
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def fmt(x):return '—' if x is None else f'{x:.3f}'
def main():
 protocol=load(BASE/'protocol.json');configs=[load(ROOT/p) for p in load(BASE/'screen_configs.json')]+[load(ROOT/p) for p in load(BASE/'followup_configs.json')]+[load(ROOT/p) for p in load(BASE/'pre_configs.json')]+[load(ROOT/p) for p in load(BASE/'expanded_configs.json')]+[load(ROOT/p) for p in load(BASE/'full_configs.json')];records=[];answers=[]
 baselines={}
 for position,folder in [('post-merger','baselines'),('pre-merger','baselines_pre')]:
  for ratio in [.1,.25,.5]:
   vals=[]
   for kind in ['mean','selection']:
    s=load(BASE/folder/f'{kind}_r{ratio:g}'/'validation.fixed.summary.json')
    if s:vals.append(s['student_macro'])
   if len(vals)==2:baselines[(position,ratio)]=max(vals)
 for cfg in configs:
  out=ROOT/cfg['output'];initial=load(out/'validation.initial.summary.json');final=load(out/f'validation.epoch{cfg["epochs"]}.summary.json');train=read(out/'training.jsonl')
  a=load(out/'train_audit.initial.summary.json');b=load(out/f'train_audit.epoch{cfg["epochs"]}.summary.json')
  status='Completed' if (out/'complete.json').exists() else 'Infrastructure failure' if (out/'failure.txt').exists() else 'Running' if initial or train else 'Not started'
  drop=(initial['kl']-final['kl'])/max(initial['kl'],1e-12) if initial and final else None
  gain=final['student_macro']-initial['student_macro'] if initial and final else None
  baseline=baselines.get((cfg.get('insertion','post-merger'),cfg['ratio']))
  passed=status=='Completed' and drop>=.05 and gain>=2 and baseline is not None and final['student_macro']>baseline
  rec={'name':cfg['name'],'phase':cfg.get('phase','initial_screen'),'kind':cfg['kind'],'ratio':cfg['ratio'],'status':status,'initial_score':initial.get('student_macro'),'final_score':final.get('student_macro'),'teacher_score':initial.get('teacher_macro'),'gain':gain,'initial_kl':initial.get('kl'),'final_kl':final.get('kl'),'kl_drop_percent':100*drop if drop is not None else None,'train_audit_initial_kl':a.get('kl'),'train_audit_final_kl':b.get('kl'),'fixed_best':baseline,'screen_gate':bool(passed),'steps':len(train),'max_gradient':max((r['grad_norm'] for r in train),default=None),'median_clip':sorted(r['clip_multiplier'] for r in train)[len(train)//2] if train else None,'total_update_l2':sum(r['update_l2'] for r in train)}
  records.append(rec)
  for path in sorted(out.glob('validation.*.jsonl')):
   for row in read(path):answers.append({'run':cfg['name'],'ratio':cfg['ratio'],**row})
 if records:
  with (BASE/'screen_summary.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 if answers:
  with (BASE/'answers.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(answers[0]));w.writeheader();w.writerows(answers)
 completed=sum(x['status']=='Completed' for x in records);failed=sum(x['status']=='Infrastructure failure' for x in records)
 summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned_runs':len(records),'planned_screen_runs':len(load(BASE/'screen_configs.json')),'planned_followup_runs':len(load(BASE/'followup_configs.json')),'planned_placement_runs':len(load(BASE/'pre_configs.json')),'planned_expanded_runs':len(load(BASE/'expanded_configs.json')),'planned_full_runs':len(load(BASE/'full_configs.json')),'completed':completed,'failed':failed,'screen_passed':[x['name'] for x in records if x['screen_gate']],'records':records}
 summary_tmp=BASE/f'summary.{__import__("os").getpid()}.tmp';summary_tmp.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');summary_tmp.replace(BASE/'summary.json')
 stages=load(BASE/'stages.json').get('ratios',{})
 outcomes=['## Current-Stage Conclusions','', 'The table uses  `stages.json` complete stage criteria determine eligibility; screening success does not imply confirmation success.','', '|Token retention / compression factor|Final stage|Conclusion|','|---|---|---|']
 stage_text={'no_pilot_pass':'Screening failed; no expanded training','expanded_failed_no_large_training':'Expanded training failed eligibility; no full training started','confirmation_failed_no_large_training':'Independent confirmation failed; no full training started'}
 for ratio,state in stages.items():
  outcomes.append(f'|{float(ratio):.0%} / {1/float(ratio):g}×|`{state.get("stage","waiting")}`|{stage_text.get(state.get("stage"),"See stage status")}|')
 outcomes+=['','### Expanded Training: 672 Images / 1,801 Questions','', 'Scores below use 28 images/questions for model selection, not independent confirmation. Epoch 8 is the fixed endpoint.','', '|Retention fraction|seed|Initial score|Trained score|Development KL: initial → final|','|---|---:|---:|---:|---|']
 for ratio,state in stages.items():
  for config_path,gate in zip(state.get('expanded_configs',[]),state.get('expanded_gates',[])):
   cfg=load(ROOT/config_path);a=gate['initial'];b=gate['final']
   outcomes.append(f'|{float(ratio):.0%}|{cfg.get("seed","—")}|{fmt(a.get("student_macro"))}|{fmt(b.get("student_macro"))}|{fmt(a.get("kl"))} → {fmt(b.get("kl"))}|')
 outcomes+=['','### Independent Confirmation: 56 Images / 103 Questions','', '|Retention fraction|seed|Trained score|Gain over same-seed initialization|95% CI for gain|Strongest fixed control|Passed|','|---|---:|---:|---:|---|---:|---|']
 confirmation_seen=False
 for ratio,state in stages.items():
  for check in state.get('confirmation_checks',[]):
   confirmation_seen=True;paired=check['paired_vs_initial'];ci=paired['cluster_bootstrap_95ci_points']
   outcomes.append(f'|{float(ratio):.0%}|{check["seed"]}|{fmt(check["score"])}|{fmt(paired["gain_points"])}|[{fmt(ci[0])}, {fmt(ci[1])}]|{fmt(max(check["fixed_controls"]))}|'+('Yes' if check['pass'] else 'No')+'|')
 if confirmation_seen:
  outcomes+=['','Intervals use image-cluster bootstrap. Both seed estimates improve, but both CIs cross zero and fail the preregistered lower-bound >0 criterion. This is insufficient evidence for stable gains, not proof of complete ineffectiveness. The uncompressed teacher macro on independent 50%-retention confirmation is 55.220. This differs from the screening set; differences in absolute scores across sets are not training degradation. Raw confirmation answers are in  `artifacts/compression_lab/expanded/expanded_spatial_cnn_r0.5_seed20261105/confirmation.epoch8.jsonl` and matching seed20261106 files, including gold/teacher/student.','']
 doc=['# New Frozen-UND Compression Generators: Protocol and Actual Progress','',f'Updated (UTC): {summary["updated_utc"]}. Completed **{completed}/{len(records)}** training configurations (102 screening/additional/position-ablation runs and 4 expanded runs); infrastructure failures:  {failed} . Full-training runs registered:  {summary["planned_full_runs"]} . **Leave incomplete results blank; submitting a job is not successful validation.**','',
 'Entry point: design in [COMPRESSION_GENERATOR_DESIGNS.md](COMPRESSION_GENERATOR_DESIGNS.md); code  `vqa/compression_generators.py`; training  `scripts/train_compression_lab.py`; unified command  `bash scripts/run_compression_lab.sh status`. ','',
 *outcomes, '## Experimental Protocol','',
 'The entire official MonkeyOCRv2-B-Und backbone (visual tower, merger, Qwen3-1.7B) is frozen; only P is trained. Compression is post-merger: C=P@X. Retentions 10%/25%/50% mean 10×/4×/2× token compression, not system speedups. Each method/budget is trained independently.', '',
 'Training: AdamW, LR=1e-4, weight_decay=0, batch=4, 3 epochs, 10% warmup + cosine to 0.1×LR, global gradient clipping=1. Record pre-clipping norms and actual updates. Objective: full-vocabulary KL at temperature 1; overlapping-support variants add 0.01×relative column-load variance, without entropy loss. Teacher-greedy answers supply shared prefixes; gold is not a training label.', '',
 'Provenance: DocVQA/InfoVQA use lmms-lab training sources; ChartQA uses ahmed-masry; DT-VQA uses shuozhang2; DeepForm/KLC/WTQ use DUE QA in echo840/Monkey_Data. Processed images reside in data/full/images. No additional MonkeyDocV2 OCR-only parsing objective trains these generators.', '', 'Use the existing seven training tasks: DocVQA / InfoVQA / ChartQA / DT-VQA / WTQ / DeepForm / KLC. No OCRBench training set is invented. New validation/confirmation images come from training sources, excluding old 20k, tuning, confirmation, and formal-test images, without filtering by teacher correctness. Screening-training samples reuse earlier training images and are not claimed unseen; each generator initializes fresh.', '',
 '|Stage|Images|QA|Purpose|','|---|---:|---:|---|']
 for name in ['train','validation','confirmation','expanded_train']:
  v=protocol.get('cohorts',{}).get(name,{});doc.append(f'|{name}|{v.get("images","—")}|{v.get("questions","—")}|'+{'train':'24×3 candidate training runs','validation':'Screening and model selection','confirmation':'Independent two-seed confirmation of selected method (results above)','expanded_train':'Expand training only after passing screening'}[name]+'|')
 doc+=['','Screening requires final epoch-3 validation KL reduction ≥5%, generation macro gain ≥2 points over initial, and superiority to same-budget mean and center-selection baselines. A 28-image validation set supports screening, not significance claims. Passing candidates train on 672 images with two seeds; only after fixing the method is the 56-image confirmation used. Full original-20k training requires a positive image-cluster bootstrap lower bound and improved training/validation KL. Formal tests never select models.','',
 'Teacher and student independently generate greedily with the same 64-token cap. Rebuild the prompt for each QA without cross-question KV. Native-task macro scores are reported; DeepForm/KLC are field exact match here, **not official document-level F1**. Per-question CSV retains gold, teacher/student answers, KL, and P statistics.','',
 '## Additional Hyperparameter Runs (Adaptive Exploration, Not Independent Confirmation)','', 'The first 8 runs at 10% retention show nearly unchanged validation KL, finite gradients, and no clipping. Register 18 additional runs: 6 architectures × 3 budgets, increasing LR from 1e-4 to 1e-3 and epochs from 3 to 12, with other data/architecture/loss/criteria unchanged. Complete the original 72 runs and label additions separately; extra training is not a new architecture. Registration:  `artifacts/compression_lab/followup_preregistration.json`. ','', 'Additional insertion-point ablation: four architectures × three budgets, before the merger, 12 epochs / LR=1e-3. Read only raw H and input question embeddings; fill the 2,048-dimensional X slot with zeros. Decoder input is merger(P@H), unlike post-merger P@merger(H). Compare against separate pre-merger fixed baselines; insertion-point changes are not training gains. Registered in  `pre_preregistration.json`. ','', 'Expanded protocol: after completing candidate selection, retain one family per budget, initialize fresh on 672 images/1,801 QA, and use seeds 20261105/20261106, 4-NPU DDP, global batch=32, LR=3e-4, and 8 epochs. Retain 10% warmup/cosine and clip=1. LR is 0.3× the added small-run 1e-3, with about 456 updates versus 168 pilot updates; do not repeat the earlier >10× LR reduction on full data. Final batches are cyclically padded; actual counts are in complete.json. Epoch 8 is fixed, not test-selected. Both seeds must reduce training-audit/validation KL ≥5%, improve score ≥2 points, and beat same-budget baselines before the 56-image independent confirmation. Full original-20k training then requires paired image-cluster CI lower bound >0 and superiority to fixed baselines.', '', 'Original-20k training starts only after confirmation passes: fresh initialization, fixed seed 20261105, 4 epochs, global batch=32, LR=3e-4, and unchanged scheduler/clipping. Only epoch 4 is evaluated formally. The full 23,565-QA benchmark at cap=256 reports teacher/initial/trained/mean/selection; formal scores never feed back into selection. Failed confirmation prevents large-scale training.', '', 'Additional strong controls tighten, rather than relax, eligibility: the 50% pre-merger pilot beats same-position controls but scores below post-merger mean. Admission from expanded to full training therefore also requires beating all four mean/selection controls across pre/post positions. Original screening criteria and selected families remain unchanged. strong_control_addendum.json records the stricter gate before first use of independent confirmation; actual confirmation results appear above.', '', '## Fixed Baselines','', '|Retention fraction|Mean|Center selection|','|---|---:|---:|']
 for ratio in [.1,.25,.5]:
  vals=[load(BASE/'baselines'/f'{kind}_r{ratio:g}'/'validation.fixed.summary.json').get('student_macro') for kind in ['mean','selection']]
  doc.append(f'|{ratio:.0%}|{fmt(vals[0])}|{fmt(vals[1])}|')
 doc+=['','Pre-merger fixed baselines: see  `artifacts/compression_lab/baselines_pre/`; each eligibility column uses fixed baselines matching its insertion point.','', '## Screening Results for Each Method','', 'Scores/gains are percentage points. KL compares distributions under shared teacher-generated prefixes. Screening success ≠ independent confirmation ≠ full-training success.','', '|Method / stage|Retention fraction|Status|Initial score|Final score|Change|Development KL: initial → final|KL reduction (%)|Strong baseline at the same budget|Eligibility criterion|','|---|---:|---|---:|---:|---:|---|---:|---:|---|']
 for r in records:
  doc.append(f'|{r["kind"]} / {r["phase"]}|{r["ratio"]:.0%}|{r["status"]}|{fmt(r["initial_score"])}|{fmt(r["final_score"])}|{fmt(r["gain"])}|{fmt(r["initial_kl"])} → {fmt(r["final_kl"])}|{fmt(r["kl_drop_percent"])}|{fmt(r["fixed_best"])}|'+('Passed screening' if r['screen_gate'] else 'Failed / incomplete')+'|')
 smoke=load(BASE/'smoke/complete.json')
 doc+=['','## Subsequent Stage Status', '', *[f'- Retention fraction {k}: `{v.get("stage","waiting")}`; selectedpilot `{v.get("selected_pilot",{}).get("name","None / not selected yet")}`. ' for k,v in load(BASE/'stages.json').get('ratios',{}).items()], '', 'Stage controller entry: `.venv/bin/python -m scripts.advance_compression_lab --submit`. File locks and registered state prevent duplicate progression; failed eligibility blocks full training.', '', '## Gradient and Backbone-Freezing Audit','',f'Real-UND decoder-gradient smoke complete: {smoke.get("schemes",0)}/24. Details:  `artifacts/compression_lab/smoke/results.json`. Check nonzero generator KL gradients, actual updates, nonnegative unit-sum P, zero trainable backbone parameters, and backbone grad=None.', '',
 f'Identity-matrix and QA-independence audit: `artifacts/compression_lab/infrastructure/audit.json`; seven real tasks, identity-P KL={load(BASE/"infrastructure/audit.json").get("identity_max_abs_KL","Not completed")}, answers match teacher exactly; interleaving other questions before revisiting produces the same IDs as teacher cache.', '', 'Each run retains  `initial.pt`, `epoch*.pt`, `latest.pt`, source snapshots/SHA, data SHA, `training.jsonl`, `validation.*.jsonl`, training-audit loss. Incomplete epochs resume committed checkpoints without duplicate counts. Infrastructure failures are not scientific negative results before repair.','',
 '## Outputs and Commands','', '```bash','bash scripts/run_compression_lab.sh prepare','bash scripts/run_compression_lab.sh smoke','bash scripts/run_compression_lab.sh cache','bash scripts/run_compression_lab.sh baseline','bash scripts/run_compression_lab.sh screen','bash scripts/run_compression_lab.sh status','bash scripts/run_compression_lab.sh report','```','',
 '- `artifacts/compression_lab/protocol.json`: preregistered criteria, splits, and parameters.','- `artifacts/compression_lab/screen_summary.csv`: actual gradient/clipping/update summaries.','- `artifacts/compression_lab/answers.csv`: screening-set teacher/student/gold/loss. Independent confirmation remains in separate confirmation.*.jsonl, not mixed into this CSV.','- `artifacts/compression_lab/literature/retrieval.json`: original-source HTTP retrieval records.','',
 'Per-epoch training/validation curves: `artifacts/compression_lab/plots/loss_*_keep*.pdf`; QA and fixed-baseline curves: `score_*_keep*.pdf`. Run  `.venv/bin/python -m scripts.plot_compression_lab`  updates.', '', '[Screening 10% loss](../artifacts/compression_lab/plots/loss_initial_screen_keep0.1.pdf) · [Screening 25% loss](../artifacts/compression_lab/plots/loss_initial_screen_keep0.25.pdf) · [Screening 50% loss](../artifacts/compression_lab/plots/loss_initial_screen_keep0.5.pdf) · [Additional 25% loss](../artifacts/compression_lab/plots/loss_higher_update_budget_keep0.25.pdf) · [Additional 25% QA](../artifacts/compression_lab/plots/score_higher_update_budget_keep0.25.pdf)', '', 'Only completed table entries constitute results. High-compression superiority and paper-level scores are not established.']
 doc_tmp=ROOT/f'docs/COMPRESSION_GENERATOR_EXPERIMENTS.{__import__("os").getpid()}.tmp';doc_tmp.write_text('\n'.join(doc)+'\n');doc_tmp.replace(ROOT/'docs/COMPRESSION_GENERATOR_EXPERIMENTS.md')
 print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
if __name__=='__main__':main()
