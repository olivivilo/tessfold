"""Audited24model report, paired3seed statistics, stage memory and raw CSV."""
from __future__ import annotations
import argparse,collections,csv,json,math,statistics
import numpy as np
from scripts.seed_memory_common import *
from vqa.benchmark_metrics import native_score
REPORT=ROOT/'docs/COLUMN_TREE_THREE_SEED_AND_MEMORY.md'

def collect_model(m,expected,teachers):
 out=ROOT/m['output'];train=lines(out/'training.jsonl')
 assert [r['step'] for r in train]==list(range(1,len(train)+1))
 for r in train:assert all(math.isfinite(r[k]) for k in ['kl','grad_norm','update_l2']) and r['update_l2']>0
 progress=dict(steps=len(train),training_complete=(out/'complete.json').exists(),epochs=[])
 for e in range(5):
  stage='initial' if e==0 else f'epoch{e}';vv=load(out/f'validation.{stage}.summary.json');aa=load(out/f'train_audit.{stage}.summary.json');tr=[r['kl'] for r in train if r['epoch']==e]
  progress['epochs'].append(dict(epoch=e,updates=len(tr),training_kl=statistics.mean(tr) if tr else None,validation_kl=vv.get('kl'),validation_score=vv.get('student_macro'),audit_kl=aa.get('kl')))
 if progress['training_complete']:
  done=load(out/'complete.json');assert done['steps']==2500 and done['epochs']==4 and done['trainable_backbone']==0
  assert len(train)==2500 and all(x['updates']==625 for x in progress['epochs'][1:])
  assert all(x['validation_kl'] is not None and x['audit_kl'] is not None for x in progress['epochs'])
 if not (out/'benchmark/complete.json').exists():return progress,None
 assert progress['training_complete']
 files=sorted((out/'benchmark').glob('predictions.rank*.jsonl'))
 signature=dict(checkpoint_sha256=sha(ROOT/m['checkpoint']),summary_sha256=sha(out/'benchmark/summary.json'),
  predictions={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in files})
 cached_path=BASE/'audits'/f'{m["id"]}.json';cached=load(cached_path)
 if cached.get('signature')==signature and cached.get('passed'):return progress,cached['summary']
 rr=model_records(m);assert len(rr)==len(expected)==23565
 seen=set()
 for r in rr:
  assert r['id'] not in seen;seen.add(r['id']);s=expected[r['id']]
  assert all(r[k]==s[k] for k in ['task','question','document_id','answers'])
  assert r['compressed_tokens']==max(1,round(m['ratio']*r['tokens']))
  assert r['row_sum_error']<1e-4 and len(r['student_ids'])<=256
  assert abs(native_score(r['student_prediction'],s)-r['student_score'])<1e-12
  assert (r['teacher_prediction'],r['teacher_score'])==teachers[r['id']]
 summary=summarize(rr);recorded=load(out/'benchmark/summary.json')
 orig=recorded['variants']['trained']['macro'] if 'variants' in recorded else recorded['macro']['trained'] if isinstance(recorded['macro'],dict) else recorded['macro']
 assert abs(summary['macro']-orig)<1e-8
 save(cached_path,dict(passed=True,audited_utc=now(),signature=signature,summary=summary,
  unique_coverage=True,scores_gold_teacher_checked=True,checkpoint_fixedepoch4=True,backbone_frozen=True))
 return progress,summary

def plot_training(m,progress):
 signature=progress['epochs'];stamp=BASE/'plots'/f'{m["id"]}.json'
 if load(stamp)==signature:return
 if not any(x['validation_kl'] is not None for x in signature):return
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,3.5))
 for key,label in [('training_kl','full train KL'),('validation_kl','validation KL'),('audit_kl','historical audit KL')]:
  pp=[x for x in signature if x[key] is not None];ax[0].plot([x['epoch'] for x in pp],[x[key] for x in pp],'o-',label=label)
 pp=[x for x in signature if x['validation_score'] is not None];ax[1].plot([x['epoch'] for x in pp],[x['validation_score'] for x in pp],'o-',label='validation generation')
 for a in ax:a.set_xlabel('epoch');a.grid(alpha=.2);a.legend(fontsize=7)
 ax[0].set_ylabel('KL');ax[1].set_ylabel('score %');fig.suptitle(m['id']);fig.tight_layout()
 stamp.parent.mkdir(exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(stamp.with_suffix('.'+ext),dpi=130)
 plt.close(fig);save(stamp,signature)

def final_statistics(reg,results):
 cached=load(BASE/'score_statistics.json')
 signature={m['id']:sha(ROOT/m['checkpoint']) for m in reg['models']}
 if cached.get('signature')==signature:return cached
 from scripts.analyze_compression_supplement import paired_maps,average_seed_maps
 maps={m['id']:{r['id']:r for r in model_records(m)} for m in reg['models']}
 pairs={};seeds={}
 for ratio in RATIOS:
  values=[]
  for seed in SEEDS:
   a=maps[model_id('quadtree_question',ratio,seed)];b=maps[model_id('col_adaptive',ratio,seed)]
   pairs[f'T-C_r{ratio:g}_seed{seed}']=paired_maps(a,b,10000);values.append((a,b))
  aa,bb=average_seed_maps(values);pairs[f'T-C_r{ratio:g}_three_seed_mean']=paired_maps(aa,bb,10000)
  tt=[results[model_id('quadtree_question',ratio,z)]['macro'] for z in SEEDS];cc=[results[model_id('col_adaptive',ratio,z)]['macro'] for z in SEEDS]
  seeds[str(ratio)]=dict(T_mean=statistics.mean(tt),T_sample_sd=statistics.stdev(tt),C_mean=statistics.mean(cc),C_sample_sd=statistics.stdev(cc),
    paired_seed_gains=[a-b for a,b in zip(tt,cc)],gain_mean=statistics.mean(a-b for a,b in zip(tt,cc)),gain_sample_sd=statistics.stdev(a-b for a,b in zip(tt,cc)))
 result=dict(signature=signature,seeds=seeds,paired=pairs,notes='Image bootstrap conditional on these3trained seeds; seed sample SD distinct,3seeds insufficient for strong population claims; viewed benchmark exploratory.')
 save(BASE/'score_statistics.json',result);return result

def memory_statistics(reg):
 out=BASE/'memory'
 if not (out/'complete.json').exists():return None
 cached=load(out/'statistics.json')
 manifests=sorted(out.glob('manifest.rank*.json'));parts=sorted(out.glob('memory.rank*.jsonl'))
 signature=dict(manifests={p.name:sha(p) for p in manifests},parts={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in parts})
 if cached.get('signature')==signature and cached.get('audited'):return cached
 rr=[r for p in parts for r in lines(p)]
 models=reg['models']+[dict(id='REF',kind='REF',ratio=1.,seed=None)]
 timing={r['id']:r for r in lines(BASE/'timing_rows.jsonl')}
 expected={(m['id'],key,mode,k) for m in models for key in timing for mode in ['natural_eos','fixed64'] for k in range(3)}
 assert len(rr)==13500 and {(r['model'],r['id'],r['mode'],r['repeat']) for r in rr}==expected
 predictions={m['id']:{r['id']:r for r in model_records(m) if r['id'] in timing} for m in reg['models']}
 teacher={}
 for r in next(iter(predictions.values())).values():teacher[r['id']]=r['teacher_prediction']
 groups=collections.defaultdict(list);flat=[]
 for r in rr:
  assert r['tokens']==timing[r['id']]['original_tokens'] and r['compressed_tokens']==max(1,round(r['ratio']*r['tokens']))
  assert sha(ROOT/r['trace'])==r['trace_sha256']
  if r['mode']=='natural_eos':
   if r['kind']=='REF':assert r['prediction']==teacher[r['id']]
   else:assert r['generated_ids']==predictions[r['model']][r['id']]['student_ids'],'Phase instrumentation must preserve native generation'
  else:assert len(r['generated_ids'])==64
  assert r['decoder_state']['prefill_visual_tokens']==r['compressed_tokens']
  for phase,v in r['phases'].items():
   assert v['peak_allocated_bytes']>=v['time_average_allocated_bytes']>=0
   assert v['seconds']>0 and v['samples']>=2
   key=(r['kind'],r['ratio'],r['seed'],r['mode'],phase);groups[key].append(v)
   flat.append({k:r[k] for k in ['model','kind','ratio','seed','mode','id','repeat','tokens','compressed_tokens']}|dict(phase=phase,**v))
 summaries=[]
 for (kind,ratio,seed,mode,phase),vv in sorted(groups.items(),key=lambda x:str(x[0])):
  summaries.append(dict(kind=kind,ratio=ratio,seed=seed,mode=mode,phase=phase,requests=len(vv),
   peak_allocated_max_gib=max(v['peak_allocated_bytes'] for v in vv)/1024**3,
   peak_allocated_mean_gib=statistics.mean(v['peak_allocated_bytes'] for v in vv)/1024**3,
   average_allocated_mean_gib=statistics.mean(v['time_average_allocated_bytes'] for v in vv)/1024**3,
   pooled_time_average_allocated_gib=sum(v['time_average_allocated_bytes']*v['seconds'] for v in vv)/sum(v['seconds'] for v in vv)/1024**3,
   start_allocated_mean_gib=statistics.mean(v['start_allocated_bytes'] for v in vv)/1024**3,
   incremental_peak_mean_gib=statistics.mean(v['incremental_peak_allocated_bytes'] for v in vv)/1024**3,
   peak_reserved_max_gib=max(v['peak_reserved_bytes'] for v in vv)/1024**3,
   average_reserved_mean_gib=statistics.mean(v['time_average_reserved_bytes'] for v in vv)/1024**3,
   max_sample_gap_seconds=max(v['max_sample_gap_seconds'] for v in vv)))
 with (out/'stage_memory_per_request.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 with (out/'stage_memory_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 result=dict(signature=signature,audited=True,records=len(rr),summaries=summaries,native_generation_regression_passed=True,traces_sha256_verified=True)
 save(out/'statistics.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();reg=verify();state=load(BASE/'state.json')
 expected={r['id']:r for r in lines(BASE/'formal_rows.jsonl')};teacherpath=BASE/'audits/teacher_reference.json';teachers=load(teacherpath)
 if not teachers:
  first=next(m for m in reg['models'] if m['reuse']);rr=model_records(first);teachers={r['id']:[r['teacher_prediction'],r['teacher_score']] for r in rr};save(teacherpath,teachers)
 teachers={k:tuple(v) for k,v in teachers.items()}
 results={};progress={}
 for m in reg['models']:
  pp,res=collect_model(m,expected,teachers);progress[m['id']]=pp
  if res:results[m['id']]=res
  plot_training(m,pp)
 stats=final_statistics(reg,results) if len(results)==24 else None
 memory=memory_statistics(reg)
 fmt=lambda x,n=2:'—' if x is None else f'{x:.{n}f}'
 doc=['# Column and Question-Conditioned Spatial Trees: Four Budgets, Three Shared Seeds, and Stagewise Device Memory','',f'Updated (UTC): {now()}; audited formal models {len(results)}/24, with 14 fresh full runs and 10 completed runs reused.','',
 '**Entry point**: `.venv/bin/python -m scripts.column_tree_seed_memory_suite status`; interrupt `resume`; fixed-registration check `verify`; final `.venv/bin/python -m scripts.report_seed_memory --require-complete`. ','',
 '## Fixed Design and Hyperparameters','',
 '10%/25%/50%/75% are token retention fractions (approximately 10×/4×/2×/1.33× token compression, not speedups). Compression is uniformly before the merger. The official MonkeyOCRv2-B-Und encoder/merger/Qwen3-1.7B are frozen; only P generators are trained. Both methods use the shared seeds 20261105, 20261106, and 20261107.','',
 'Reuse eight runs at seed 20261105 across two methods/four budgets, plus T25 seeds 20261106/07 from the supplement; train the remaining 14 afresh without seed selection. Fixed files/order: 20,000 training QA / 16,731 images, 56 validation QA, 14 historical audit QA, 23,565 formal QA / 6,494 images. AdamW LR=3e-4, wd=0, clip=1, batch=32; 8 ranks accumulate 4 questions each; 4 epochs / 2,500 updates, 250-step warmup and cosine to 10% peak LR. KL-only at T=1 under shared teacher-generated prefixes; train/validation cap=64, formal greedy cap=256, fixed epoch 4.','',
 'New formal evaluation runs the trained variant only, 23,565 records/model, without repeating six earlier controls. Earlier initial/control outputs remain; trained scoring is unchanged. Eight tasks are equally weighted; DeepForm/KLC use field exact match, not a unified official leaderboard. The formal benchmark was already inspected, so this is a fixed repeated extension.','',
 '## Formal Results by Seed','', '|Retention fraction|Method|seed|Source|Training steps|Eight-task macro|DocVQA|Formal audit|','|---|---|---:|---|---:|---:|---:|---|']
 for m in reg['models']:
  res=results.get(m['id'],{});pp=progress[m['id']]
  doc.append(f'|{m["ratio"]:.0%}|{LABEL[m["kind"]]}|{m["seed"]}|{"Reuse completed experiment" if m["reuse"] else "Newly trained in this round"}|{pp["steps"]}/2500|{fmt(res.get("macro"))}|{fmt(res.get("tasks",{}).get("DocVQA"))}|{"Passed" if res else "Pending"}|')
 doc+=['','## Three-Seed Aggregation and Paired Comparisons','','Means and sample SD describe training-seed variation; image-cluster bootstrap describes evaluation-image uncertainty conditional on these three trained models. Each model scores each question independently. Three-seed averaging estimates repeated-training statistics; it is not averaging random-compression downstream scores before loss-correlation analysis. Secondary comparisons are unadjusted, and three seeds do not precisely estimate population variance.','']
 if stats:
  doc+=['|Retention fraction|Column mean ± sample SD|Tree mean ± sample SD|Mean Tree−Column difference|SD of paired seed differences|95% image CI for the three-seed mean difference|','|---|---:|---:|---:|---:|---|']
  for ratio in RATIOS:
   s=stats['seeds'][str(ratio)];pair=stats['paired'][f'T-C_r{ratio:g}_three_seed_mean'];ci=pair['image_cluster_95ci_points']
   doc.append(f'|{ratio:.0%}|{s["C_mean"]:.4f}±{s["C_sample_sd"]:.4f}|{s["T_mean"]:.4f}±{s["T_sample_sd"]:.4f}|{s["gain_mean"]:+.4f}|{s["gain_sample_sd"]:.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|')
 else:doc+=['Not all 24 runs are available; no missing-seed imputation or premature three-seed conclusions.']
 doc+=['','## Stagewise Memory Definitions and Limitations','',
 'Independent memory diagnostics cover 24 checkpoints plus uncompressed REF: 90 fixed QA (30 per original-N stratum), 3 repetitions, natural EOS/cap=256 and fixed64, totaling 13,500 requests. Batch=1, resident weights, no visual-cache reads or cross-question KV, and one resident generator at a time; 10 warmup requests/model.','',
 'Six stages: processor; encoder (visual trunk ending before the official merger); native_merger (original full-length merger); compression (P generation, PH, the official merger on compressed features, and question embeddings); Qwen prefill (the first full-prompt forward in generate); generation (first-token selection after prefill and decoding to termination). REF\'s nearly empty compression stage is still reported. Original full-length X is retained after encoding to reproduce the existing pipeline without silently adding tensor-release optimization.','',
 'Synchronize and reset allocator peaks at each stage boundary. Peaks use torch.npu.max_memory_allocated/reserved. Means are sampling estimates: a CPU thread targets 1 ms intervals and trapezoidally time-weights actual memory_allocated/reserved samples, including CPU gaps within stages, not exact kernel-time averages. Save raw NPZ traces, sample counts, and maximum intervals. Narrow peaks can bias sampled means, while peak values independently come from the allocator.','',
 'Report absolute stage-start/end allocation, absolute peak, incremental peak above the start, and sampled mean. Distinguish mean/max request peaks and the mean of request averages from a duration-weighted aggregate. Actual generation-KV storage is separate; fewer tokens do not imply lower resident-weight or encoder peaks. Device allocated/reserved values exclude other processes and total driver usage. Synchronization/thread overhead is included; these durations alone do not establish speedup.','',
 'First regress native generation for 8 architecture/budget cases, visual-prefill/KV counts, known 64-MiB allocation sampling/peaks, and fresh-generator real-UND backprop/updates. Submit full training only after passing. Natural-EOS memory passes also match each formal checkpoint’s answers.','']
 if memory:
  doc+=['All 13,500 memory records/trace hashes are audited; native-generation regression passes. See [Per-request CSV](../artifacts/column_tree_seed_memory_20261113/memory/stage_memory_per_request.csv)and [Method/budget/seed/mode/stage summary CSV](../artifacts/column_tree_seed_memory_20261113/memory/stage_memory_summary.csv). ','',
   '|Retention fraction|Method|Stage|Mean request peak (GiB)|Mean per-request average (GiB)|Maximum request peak (GiB)|Mean incremental peak (GiB)|','|---|---|---|---:|---:|---:|---:|']
  for ratio in RATIOS:
   for kind in KINDS:
    for phase in ['encoder','native_merger','compression','qwen_prefill','generation']:
     pp=[x for x in memory['summaries'] if x['kind']==kind and x['ratio']==ratio and x['phase']==phase and x['mode']=='natural_eos'];assert len(pp)==3
     doc.append(f'|{ratio:.0%}|{LABEL[kind]}|{phase}|{statistics.mean(x["peak_allocated_mean_gib"] for x in pp):.4f}|{statistics.mean(x["average_allocated_mean_gib"] for x in pp):.4f}|{max(x["peak_allocated_max_gib"] for x in pp):.4f}|{statistics.mean(x["incremental_peak_mean_gib"] for x in pp):.4f}|')
 else:doc+=['Formal stagewise memory is incomplete; old E2E peaks do not fill stage tables.']
 doc+=['','## Training/Validation Loss and Original Entry Points','']
 for m in reg['models']:
  pp=progress[m['id']]
  doc+=[f'### {m["id"]}','',f'[Configuration](../{m["config"]}) · [Weights](../{m["checkpoint"]}) · [Curves](../artifacts/column_tree_seed_memory_20261113/plots/{m["id"]}.pdf) · [Per-question results](../{m["output"]}/benchmark/). ','',
   '|epoch|Recorded updates|Full-training KL|Development KL|Development generation macro|Historical audit KL|','|---|---:|---:|---:|---:|---:|']
  for e in pp['epochs']:doc.append(f'|{e["epoch"]}|{e["updates"]}|{fmt(e["training_kl"],5)}|{fmt(e["validation_kl"],5)}|{fmt(e["validation_score"])}|{fmt(e["audit_kl"],5)}|')
 doc+=['','## Scheduling and Recovery','',
 'A separate CPU controller holds a file lock and permits at most two 8-NPU stages queued/running concurrently. Data/source/configuration signatures remain fixed. Each stage allows up to 3 infrastructure recoveries using optimizer/latest saved every 25 updates; no success is reported without a completion marker. Exhaustion leaves attention_required without changing the protocol. Earlier weights and migration archives are not overwritten.','',
 f'Current stage status: {dict(collections.Counter(s["status"] for s in state.get("stages",{}).values()))}. ',
 '[Immutable registration](../artifacts/column_tree_seed_memory_20261113/registration.json) · [Live status](../artifacts/column_tree_seed_memory_20261113/state.json) · [Audit](../artifacts/column_tree_seed_memory_20261113/audits/). ']
 tmp=REPORT.with_suffix('.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(REPORT)
 save(BASE/'merged_results.json',dict(updated=now(),completed_models=len(results),results=results,progress=progress,score_statistics_ready=stats is not None,memory_ready=memory is not None))
 if a.require_complete:assert len(results)==24 and stats is not None and memory is not None
 print(json.dumps(dict(report=str(REPORT),completed_models=len(results),memory=memory is not None)),flush=True)
if __name__=='__main__':main()
