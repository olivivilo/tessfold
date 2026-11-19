"""Evidence-only report and paper-ready tables; no invented performance targets."""
import argparse,collections,csv,json,statistics
import numpy as np
from scripts.tessfold_repair.common import *
REPORT=ROOT/'docs/TESSFOLD_SYSTEM_REPAIR_20261115.md'

def speed_rows(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  if r['instrumentation']=='synchronized_wall_and_NPU_events':groups[(r['method'],r['ratio'],r['mode'])].append(r)
 result=[]
 for (method,ratio,mode),rr in sorted(groups.items()):
  phases=sorted(set(p for r in rr for p in r['stages']))
  row=dict(method=method,ratio=ratio,mode=mode,requests=len(rr),ttft_p50_ms=float(np.median([r['ttft_ms'] for r in rr])),ttft_p95_ms=float(np.quantile([r['ttft_ms'] for r in rr],.95)),
    post_encoder_ttft_p50_ms=float(np.median([r['post_encoder_ttft_ms'] for r in rr])),e2e_p50_ms=float(np.median([r['end_to_end_ms'] for r in rr])),
    peak_allocated_max_gib=max(r['peak_allocated_bytes'] for r in rr)/1024**3,
    visual_kv_mib=(float(np.median([r['decoder_state']['visual_kv_bytes'] for r in rr]))/1024**2 if all('visual_kv_bytes' in r['decoder_state'] for r in rr) else None))
  row['phase_wall_medians_ms']={p:float(np.median([r['stages'][p]['wall_ms'] for r in rr if p in r['stages']])) for p in phases}
  row['phase_event_medians_ms']={p:float(np.median([r['stages'][p]['device_event_ms'] for r in rr if p in r['stages']])) for p in phases}
  result.append(row)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();reg=verify();state=load(BASE/'state.json');cpu=load(BASE/'cpu_equivalence.json');val=load(BASE/'validation/complete.json')
 single_done=(BASE/'single/complete.json').exists();multi_done=(BASE/'multi/complete.json').exists()
 single=[r for p in (BASE/'single').glob('results.rank*.jsonl') for r in lines(p)] if single_done else []
 multi=[r for p in (BASE/'multi').glob('results.rank*.jsonl') for r in lines(p)] if multi_done else []
 doc=['# TessFold Inference-Only Refactoring: Accuracy Validation and Efficiency Measurements','',f'Updated (UTC): {now()}. Actual validation completed: {bool(val)}; single-request measurements completed: {single_done}; multi-question measurements completed: {multi_done}. ','',
 '**Entry point**: `.venv/bin/python -m scripts.tessfold_repair.suite status|resume|verify`; final `.venv/bin/python -m scripts.tessfold_repair.report --require-complete`. ','',
 '## Interface Repairs in This Version','',
 'The separate scripts/tessfold_repair version retains the failed 20261114 experiments and official sources. It restores real input_ids only after HF prepares the embedding-prefill inputs, without fabricated IDs, position/mask/KV changes, or weight updates. Tiny CPU Qwen reproduces the original failure and verifies repaired embedding generation, native generation, two independent prefixes, and exception restoration. Duplicate installation of the resize patch is also repaired.','',
 '## Scope of the Work','',
 'Follow tessfold.md with an inference-only refactor and no retraining. Eight Column/Tree checkpoints at seed 20261105 / epoch 4 are fixed, with all weights frozen. The historical 16 runs, three-seed studies, and efficiency data remain; signed sources and negative results are not overwritten. Only inference work and evaluation procedures are optimized: no retraining, Qwen replacement, or assumed submission/performance outcome.','',
 '## Necessary Corrections to the Reference Proposal','',
 '1. Equivalence over real arithmetic does not imply bitwise floating-point equality. Replacing FP32 softmax/GEMM with BF16 multiplication and accumulation changes rounding; FP32 scatter also changes reduction order. Default strict_dense retains the original FP32 partition/softmax/GEMM and removes only unused work. Experimental segmented_fp32/BF16 paths are reported separately and cannot silently inherit earlier scores.','',
 '2. The original implementation already uses an FP32 2D integral image and a max-heap; split locations still use local NumPy sum/cumsum. An FP64 integral image with binary search changes rounding, mass ties, and cut positions. Candidates are compared independently against the original; reference and candidate must not share the new partitioner. JIT latency below 0.8 ms has neither been proved nor promised.','',
 '3. Hard readout is a separate variant, not bit-exact to soft. A soft−hard CI crossing zero does not prove equivalence. Historical Hard25% score 38.24 applies only to its setting and cannot be assigned to new budgets.','',
 '4. Post-ViT compression cannot save an already incurred encoder peak; report full E2E peak. The new path skips unused full-length merger work, releases raw/P/input-image tensors after compression, then supplies identical visual embeddings to original generate. Independently verify that lifecycle changes preserve answers.','',
 '5. A 25% visual-KV retention fraction describes the logical visual segment at actual K/N, not fourfold improvement in all KV, device memory, or throughput. Per-page K is rounded; text/answer KV and resident weights are unchanged. Maximum concurrency was not stress-tested, so cache ratios cannot replace measured throughput.','',
 '6. Same-page prefix caching must reuse exactly the fixed causal prefix of the original prompt up to the visual boundary; it must not include a previous question or answer\'s KV. Qwen uses decoder self-attention, not a separate visual cross-attention module. REF receives the same feature/KV reuse permissions. Clone the read-only cache and include this cost for each question. Splitting prefix execution may change floating-point kernels, so generation must be regressed separately.','',
 '7. Keep official UND/Qwen3-1.7B in the main experiment for independent regression against historical answers. The separately requested newer ~7B-class Qwen extension has its own architecture/environment report and does not share main scores. No TeX source is provided; measured tables are separate.','',
 '8. The earlier fixed-64 baseline was 2532.14 ms versus T at 2502.28 ms, a difference of about 29.86 ms. The reference document\'s claims of “only 4.5 ms saved” and “250–300 ms for 64 steps” do not match the actual units and cannot identify the latency source precisely. Fixed-64 resize results and T−M/T−A CIs already exist; cite the exact records instead of inventing duplicate measurements.','',
 '## Completed Independent CPU Tests','',
 'A separate near-uniform-energy audit found an FP64 partition counterexample (fp64_partition_audit.json). Zero differences in these 350 ordinary cases therefore do not prove strict equivalence.','',f'Partition-candidate cases{cpu.get("partition_cases",0)}, including FP64-versus-original-FP32 mismatches {cpu.get("integral_candidate_mismatches",0)} cases; absence of a counterexample in finite tests does not prove universal bitwise equivalence.','',
 '|Method|Retention fraction|Strict-path bit-exact|FP32 segmented bit-exact|BF16 segmented bit-exact|BF16 maximum absolute difference|','|---|---:|---|---|---|---:|']
 for r in cpu.get('feature_tests',[]):doc.append(f'|{r["kind"]}|{r["ratio"]:.0%}|{r["strict_bit_exact"]}|{r["segmented_fp32_equal"]}|{r["segmented_bf16_equal"]}|{r["segmented_bf16_max_error"]:.8f}|')
 doc+=['','Unlike printing errors alone, strict tests assert torch.equal on original P/outputs. Retain group ordering, official 2×2 block/inverse permutation, smallest-index argmax ties, stable sorting, ties-to-even length, and energy-operation grouping.','',
 '## Independent Validation Gates on the Actual UND Backbone','',
 '56 fixed validation questions × 2 methods × 4 budgets give 448 cases. For every case, original-generator and strict P/PH/merger outputs must be bit-exact, with generated IDs also compared against historical student_ids. A clean path independently removes redundant merger work and releases intermediates, then compares generation. Segmented and FP64-partition candidates are diagnostic only; failures do not modify old weights or parameters. Each rank also tests two real same-page questions under REF/Column/Tree, comparing independent prefix-cache and full-prompt output and checking that questions/answers do not contaminate the base cache.','']
 if val:doc += [f'Actually completed: {val["cases"]} cases; strict checks passed: {val["strict_passed"]}. Segmented candidate mismatches: `{val["candidate_mismatches"]}`; FP64 partition mismatches: {val["integral_partition_mismatches"]}; prefix-cache token mismatches {val["prefix_reuse_token_mismatches"]}. ','']
 else:doc+=['Actual validation incomplete; no advance pass claim.','']
 doc+=['## Single-Request Efficiency Protocol','',
 'Same 90 questions, 30 in each of three strata; primary budget 25%, secondary 50%. Fifteen method/budget configurations include REF and T/C original dense, new strict, TreeHard, R, S at each budget. Natural EOS/cap=256 and fixed64 each have 3 repeats, giving 8,100 timing records. A separate 1 ms memory pass runs once per configuration/question, 1,350 records; sampling-thread overhead is excluded from primary timing. Warm up with two cap=16 questions/model, including JIT warmup and excluding first compilation.','',
 'Actual hardware is Ascend, with torch.npu.Event and synchronized wall timing. Wall time captures CPU partition/transfer costs; events can include stream idle waits and are not pure kernel times. Record processor, encoder, compression+merger, prefill, decode, TTFT, post-encoder TTFT, E2E latency, logical KV bytes, and stage allocated/reserved memory. Synchronization/hooks are identical across methods, not unperturbed serving throughput. S follows its internal-pruning path and reports total decoder time without inventing a separate prefill measurement.','']
 summaries=[];regression=[]
 if single_done:
  assert len(single)==9450
  bykey={(r['method'],r['ratio'],r['mode'],r['id'],r['repeat'],r['instrumentation']):r for r in single}
  for r in single:
   if r['method'] not in ['T_STRICT','C_STRICT']:continue
   old=bykey[(r['method'].replace('_STRICT','_DENSE'),r['ratio'],r['mode'],r['id'],r['repeat'],r['instrumentation'])]
   assert r['generated_ids']==old['generated_ids'],'Strict optimization changed benchmark generation'
  summaries=speed_rows(single)
  save(BASE/'single/summary.json',dict(rows=summaries,strict_generation_regression_passed=True))
  with (BASE/'single/latency_summary.csv').open('w') as f:
   fields=[k for k in summaries[0] if not k.startswith('phase_')];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summaries)
  doc+=['|Method|Retention fraction|Mode|TTFT p50 ms|Post-encoder TTFT p50 ms|E2E p50 ms|Compression+merger wall p50 (ms)|Peak (GiB)|Visual KV (MiB)|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
  for r in summaries:
   kv='Not measured independently' if r['visual_kv_mib'] is None else f'{r["visual_kv_mib"]:.2f}'
   doc.append(f'|{r["method"]}|{r["ratio"]:.0%}|{r["mode"]}|{r["ttft_p50_ms"]:.2f}|{r["post_encoder_ttft_p50_ms"]:.2f}|{r["e2e_p50_ms"]:.2f}|{r["phase_wall_medians_ms"]["compression_and_merger"]:.2f}|{r["peak_allocated_max_gib"]:.4f}|{kv}|')
  # Exactstrata are fixed from originaltiming sample lengths, not chosen by speed.
  timing=lines(BASE/'timing_rows.jsonl');ordered=sorted(timing,key=lambda r:(r['original_tokens'],r['id']));strata={r['id']:['short','medium','long'][i//30] for i,r in enumerate(ordered)}
  bins=[]
  for label in ['short','medium','long']:
   for row in speed_rows([r for r in single if strata[r['id']]==label]):bins.append(dict(stratum=label,**row))
  save(BASE/'single/stratified.json',bins)
  memory=[r for r in single if r['instrumentation']=='memory1ms']
  for r in memory:assert sha(ROOT/r['trace'])==r['trace_sha256']
  save(BASE/'single/audit.json',dict(passed=True,records=len(single),strict_tokens_checked=True,trace_sha_checked=True))
  # Paper tablecontains measured rows only, no asserted submillisecondcaption.
  tex=['% Automatically generated from completed measured results. Historical tables unchanged.',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Keep & TTFT (ms) & Post-encoder TTFT (ms) & E2E (ms) \\',r'\midrule']
  for r in summaries:
   if r['mode']=='natural_eos':tex.append(f"{r['method'].replace('_', chr(92)+'_')} & {100*r['ratio']:.0f}\\% & {r['ttft_p50_ms']:.2f} & {r['post_encoder_ttft_p50_ms']:.2f} & {r['e2e_p50_ms']:.2f} \\\\")
  tex += [r'\bottomrule',r'\end{tabular}'];(BASE/'single/measured_table.tex').write_text('\n'.join(tex)+'\n')
 else:doc+=['Formal single-request timing incomplete; do not report expected 0.XX ms or assumed speedups.']
 doc+=['','## Fair Caching Protocol for Same-Page Questions','',
 'Use 24 fixed real pages with at least 10 distinct questions each. K=1/2/5/10 uses prefixes of the same ten-question sequence, not duplicate questions. Only DocVQA/InfoVQA/WTQ/OCRBench provide enough such pages, so this shifted subset is not full eight-task generalization. REF, Column, and Tree each receive cold, feature_cache, and prefix_cache scenarios. A prefix-token mismatch invalidates that entire prefix scenario and is reported; feature_cache evaluation continues.','',
 'cold runs an independent image encoder per question. featurecache caches full-page H, with REF/Column also caching aligned features; Tree recomputes question-dependent compression. prefixcache reuses the visual causal prefix for REF/Column, while Tree reconstructs its question-dependent prefix. Prefix cloning and per-question processing are timed; initial ingestion is amortized as (ingestion + time for first K questions)/K. Cross-question answer KV is never treated as reusable visual prefix. All methods receive equal frontend-cache permissions.','']
 if multi_done:
  grouped=collections.defaultdict(list)
  for r in multi:
   for k,v in r['amortization'].items():grouped[(r['method'],r['scenario'],int(k))].append(v)
  stats=[dict(method=m,scenario=s,k=k,measurements=len(v),mean_ms_per_query=statistics.mean(v),median_ms_per_query=statistics.median(v),queries_per_second=1000/statistics.mean(v)) for (m,s,k),v in sorted(grouped.items())]
  # Regression uses samepage/method/repeat, compares alternatecache scenarios.
  reference={(r['page'],r['method'],r['repeat']):r for r in multi if r['scenario']=='cold'};mismatch=0
  for r in multi:
   orig=reference[(r['page'],r['method'],r['repeat'])]
   mismatch+=sum(a['generated_ids']!=b['generated_ids'] for a,b in zip(orig['queries'],r['queries']))
  save(BASE/'multi/summary.json',dict(rows=stats,cache_generation_mismatches=mismatch,exact_on_measured_cohort=mismatch==0))
  doc+=['|Method|Caching scenario|K|Mean ms/question (including initial ingestion)|Median ms/question|Measured queries/s|','|---|---|---:|---:|---:|---:|']
  for r in stats:doc.append(f'|{r["method"]}|{r["scenario"]}|{r["k"]}|{r["mean_ms_per_query"]:.2f}|{r["median_ms_per_query"]:.2f}|{r["queries_per_second"]:.2f}|')
  doc += ['',f'Token-regression mismatches against cold execution: {mismatch}. Nonzero counts prohibit a bit-exact cache claim.','']
  try:
   import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
   fig,ax=plt.subplots(figsize=(6,3.5))
   for m in ['REF','Column','Tree']:
    rr=[r for r in stats if r['method']==m and r['scenario']=='feature_cache'];ax.plot([r['k'] for r in rr],[r['mean_ms_per_query'] for r in rr],'o-',label=m)
   ax.set_xlabel('Questions per real page');ax.set_ylabel('Amortized ms/query incl ingestion');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
   for ext in ['pdf','png']:fig.savefig(BASE/'multi'/f'amortization.{ext}',dpi=170)
   plt.close(fig)
  except ImportError:pass
 else:doc+=['Formal multi-question timing incomplete; near-fourfold throughput is not established.']
 doc+=['','## FLOP Estimates and Manuscript Claim Boundaries','',
 'FLOPs are analytical estimates only. The original merger\'s linear terms are approximately 2N(768²+768×2048), scaled by actual M/N after compression. Qwen projection/MLP costs scale with total length L=M+text prefix; the causal attention term is approximately L², but text, layers, masks, kernels, and generation prevent all computation from scaling exactly as r². S uses original N in its first layer and K in the remaining 27. No hardware kernel-FLOP counter is used; analytical values are not measurements.','',
 'Retain all negative history and specify deployment scenarios for new optimization/caching. Report missed sub-ms/cache targets rather than deleting old dense-P or 6.414-GiB observations; show versioned comparisons. Registrations pin sources/data/weights; plans are not outcomes.','',
 f'Current stage: {state.get("stages",{})}. ','[Registration](../artifacts/tessfold_system_repair_20261115/registration.json) · [CPU tests](../artifacts/tessfold_system_repair_20261115/cpu_equivalence.json) · [Actual validation](../artifacts/tessfold_system_repair_20261115/validation/) · [Previous exact statistics](COMPRESSION_SUPPLEMENT_STATISTICS.md). ']
 REPORT.write_text('\n'.join(doc)+'\n')
 save(BASE/'summary.json',dict(time=now(),validation=bool(val),single=single_done,multi=multi_done))
 if a.require_complete:assert val and single_done and multi_done
 print(json.dumps(dict(report=str(REPORT),validation=bool(val),single=single_done,multi=multi_done)))
if __name__=='__main__':main()
