"""Feedback supplementary protocol, live progress, audited results and tables."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from scripts.supplement_common import ROOT,BASE,REG,load,save,lines,sha,now,verify,benchmark_rows,summary,legacy,teacher_records
from vqa.benchmark_metrics import native_score


def audited(record,expected,teacher):
    out=ROOT/record['output']/'benchmark'
    for parent in [out.parent,out]:
        try:list(parent.iterdir())
        except FileNotFoundError:pass
    if not (out/'complete.json').exists():return None
    signature=dict(summary_sha256=sha(out/'summary.json'),manifests={p.name:sha(p) for p in out.glob('manifest.rank*.json')},
                   predictions={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in out.glob('predictions.rank*.jsonl')})
    cache=out/'supplement_audit.json';old=load(cache)
    if old.get('signature')==signature and old.get('passed'):return load(out/'summary.json')
    cfg=load(ROOT/record['config']);wanted=lines(ROOT/cfg['eval_rows']) if cfg.get('eval_rows') else list(expected.values())
    variants=cfg.get('variants',['trained']);items=[r for p in sorted(out.glob('predictions.rank*.jsonl')) for r in lines(p)]
    assert len(items)==len(wanted)*len(variants)
    assert {(r['id'],r['variant']) for r in items}=={(r['id'],v) for r in wanted for v in variants}
    manifests=[load(p) for p in out.glob('manifest.rank*.json')]
    assert manifests and all(m==manifests[0] for m in manifests)
    assert len(manifests)==manifests[0]['world']
    if cfg.get('checkpoint'):assert manifests[0]['checkpoint_sha256']==sha(ROOT/cfg['checkpoint'])
    for r in items:
        source=expected[r['id']]
        for k in ['task','document_id','question','answers']:assert r[k]==source[k]
        assert r['teacher_prediction']==teacher[r['id']]['teacher_prediction']
        assert abs(r['teacher_score']-teacher[r['id']]['teacher_score'])<1e-12
        assert abs(native_score(r['student_prediction'],source)-r['student_score'])<1e-12
        assert len(r['student_ids'])<=256
        if cfg['method']!='R':assert r['compressed_tokens']==max(1,round(cfg['ratio']*r['tokens']))
        if cfg['method']=='S':assert r['first_layer_visual_tokens']==r['tokens']
        else:assert r['first_layer_visual_tokens']==r['compressed_tokens']
    ss=load(out/'summary.json')
    for v in variants:
        direct=summary([r for r in items if r['variant']==v]);assert direct==ss['variants'][v]
    if cfg['method']=='QSWAP':
        for r in items:
            if r['variant']=='QQ':
                assert r['student_ids']==teacher[r['id']]['student_ids'],'QQ no longer reproduces original production answer'
    if cfg['method']=='R':
        amendment=load(BASE/'resize_recovery/registration.json')
        recovery=[load(p) for p in out.glob('resize_recovery.rank*.json')]
        assert len(recovery)==manifests[0]['world'] and amendment
        assert all(r['amendment_sha256']==sha(BASE/'resize_recovery/registration.json') for r in recovery)
    save(cache,dict(passed=True,audited_utc=now(),signature=signature,questions=len(wanted),records=len(items),scores_recomputed=True,teacher_gold_identity=True,unique_coverage=True,qq_production_regression=cfg['method']=='QSWAP'))
    return ss


def training(record):
    out=ROOT/record['output']
    try:list(out.iterdir())
    except FileNotFoundError:pass
    rr=lines(out/'training.jsonl')
    if not rr:return dict(updates=0)
    assert [r['step'] for r in rr]==list(range(1,len(rr)+1))
    assert all(math.isfinite(r[k]) for r in rr for k in ['kl','total','grad_norm','update_l2'])
    cfg=load(out/'effective_config.json') or load(ROOT/record['config'])
    if (out/'complete.json').exists():
        done=load(out/'complete.json');assert done['trainable_backbone']==0
        assert done['steps']==cfg.get('max_steps',2500)
    return dict(updates=len(rr),latest=rr[-1],config=cfg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    reg=verify();state=load(BASE/'state.json');expected={r['id']:r for r in benchmark_rows()};teacher=teacher_records()
    results={};progress={}
    for record in reg['runs']:
        if record['phase']=='eval':
            result=audited(record,expected,teacher)
            if result:results[record['name']]=result
        if record['phase']=='train':progress[record['name']]=training(record)
    fmt=lambda v:'—' if v is None else f'{v:.2f}'
    complete=load(BASE/'complete.json');smoke=load(BASE/'smoke/complete.json');selection=load(BASE/'development_selection.json')
    doc=['# Document Visual Compression: Supplementary Design, Procedure, and Results','',f'Updated (UTC): {now()}. Additional formal evaluation audited: {len(results)} runs; complete-pipeline marker: {bool(complete)}. ','',
         '**Entry point**: `.venv/bin/python -m scripts.compression_supplement_suite status`see; `resume`Resume the independent background controller; final `.venv/bin/python -m scripts.report_compression_supplement --require-complete`verification.','',
         '## Experimental Design and Freezing Protocol','',
         'Following [feedback analysis](COMPRESSION_SCHEME_FEEDBACK_ACTIONS.md) , feedback/experiment_runbook.docx, and paper_experiment_tables.xlsx. Primary budget: 25%; secondary: 50%. Fixed training: 20,000 QA / 16,731 images; development: 56 QA; formal evaluation: 23,565 QA / 6,494 images. The formal set has already been inspected, so this is a fixed-benchmark extension, not a blind test. Official encoder, merger, and Qwen3-1.7B remain frozen.','',
         '- Eight full runs: M25/M50; V25/J25; T25 seeds 20261106/07; M25 seeds 20261106/07. Global batch=32, 8 NPUs, 625 updates/epoch, 4 epochs / 2,500 updates. Epoch 4 is fixed; formal scores never select models.',
         '- T/V/J use KL only. T retains AdamW at 3e-4, clip=1, 250-step warmup, and cosine decay to 10%. M retains the global query/key matrices and KL + row entropy + maximum column mass. UND adaptations are documented in vqa/meta_compress.py; this is not claimed to be the authors\' official implementation.',
         '- Six optimizer development pilots: three candidates each for M and T, using 224 QA from current full_train and 56 fixed validation QA, with 100 updates each. Candidates are original SGD 1e-3/clip 1e-2/constant, AdamW 3e-4/clip 1/cosine, and AdamW 3e-5/clip 1/cosine. M selects one setting by endpoint development generation macro, then KL, then registration order, shared across budgets/seeds. T retains its historical setting; all pilots are reported. These 600 additional updates are separate from the 20,000 formal updates across eight runs.',
         '- If the author-style M optimizer is not selected, retain its full pilot and curves. Optimization failure is not evidence of superiority over a strong baseline. M’s formal manifest/effective_config records the actual selection. Development selection does not guarantee formal-set optimality; no formal-score tuning follows.',
         '- V changes only C’s split axis to the longer axis. J changes only T’s energy-median cut to the floored geometric midpoint, still selecting leaves by energy. Both initialize afresh; other training settings are unchanged.',
         '- H/A reuse T25’s actual leaves and production ordering, taking the highest-weight original token or mean respectively. They are inference interventions, not separately trained hard/mean methods. G* uses existing T scores with legal fixed 4×4 cells, softmax quotas over mean rho, exact capacity-constrained rounding, and within-cell s top-k; it is not original GridPrune.',
         '- QQ/QW/WQ/WW change only the compressor question; Qwen always receives the current question. Fix each partition’s ordering using current-question representatives and save leaf IDs/order and boundary changes. Same-page wrong-question mapping is independent of answers/scores. QQ must reproduce historical production answers question by question.',
         f'- After deduplicating pages, lock {reg["cohorts"]["same_page_questions"]}QA/{reg["cohorts"]["same_page_images"]} images. Compare matched subsets/tasks only, not their macros against the full set.',
         '- S uses the official S²Prune allocation.py at a fixed commit. The first Qwen block runs at full length; ERC=L2(block output−input). Visual hidden states, masks, original 1D RoPE positions, and layer-0 KV are then pruned while text is retained; the remaining 27 blocks use K tokens. UND does not use native Qwen2.5-VL preprocessing or M-RoPE, so this is explicitly an UND adaptation.',
         '- S uses coarse-cell side min(8 at 25% or 9 at 50%, h,w,isqrt(K)), global image Laplacian complexity, official nonempty rectangular allocation, and local representative selection. Correctly permute 2×2 block order and restore representatives to sequence order.',
         '- R searches legal processor dimensions approximating the original aspect ratio, selects actual K′ closest to target K, and re-encodes the resized image. Record K′ error rather than treating target r as achieved retention.',
         '- Formal methods independently generate with greedy cap=256; teacher training/validation cap=64, T=1, full-vocabulary per-question mean KL. Teacher answers come from the original model, not gold. DeepForm/KLC use field EM; the eight-task mean is not a unified official leaderboard.',
         '- Efficiency: 90 fixed page-question pairs, 30 in each of three original-N strata; batch=1, cold visual caches, resident backbone weights. Methods are loaded individually to avoid memory bias from multiple resident generators. Each method has 10 warmup requests and 3 interleaved repetitions, with natural EOS and fixed-64 generation measured separately. Timing includes processor, visual encoder/original merger, compression/CPU work, and generation; a first-token callback measures TTFT. Report p50/p95 and allocated/reserved peaks. No inter-question KV reuse.','',
         '[Complete registration](../artifacts/compression_supplement_20261111/registration.json) · [Machine-readable status](../artifacts/compression_supplement_20261111/state.json) · [Original paper / official code](../artifacts/compression_supplement_20261111/sources/). ','',
         '## Technical Checks and Development-Set Selection','',f'Real-UND technical checks passed: {bool(smoke)}. Selection complete: {bool(selection)}. Incomplete stages are not reported as successes.','']
    if selection:
        doc += [f'Selected M optimizer configuration: `{selection["settings"]}`; fixed development data only, never formal tests.','', '|Method|Candidate|Endpoint development KL|Endpoint development macro|','|---|---|---:|---:|']
        for r in selection['all_trials']:doc.append(f'|{r["method"]}|{r["candidate"]}|{r["validation"]["kl"]:.5f}|{r["validation"]["student_macro"]:.2f}|')
    recovery=load(BASE/'resize_recovery/registration.json')
    if recovery:
        doc += ['', '### Resize-Baseline Error and Versioned Recovery', '',
                'On some elongated pages, resize candidates yielded zero height or width in smart_resize, causing division by zero during ranking. Three attempts failed at each of 25%/50%. The repair only filters nonpositive dimensions, preserving ranking, target budget, processor, generation settings, and signed evaluation sources. All image/budget candidates were checked; selected dimensions of committed earlier answers are unchanged. Resumption retains those answers and changes neither training nor other methods.', '',
                '[Recovery registration, original failures, and all-image candidate checks](../artifacts/compression_supplement_20261111/resize_recovery/). Retain recovery versions alongside original manifests; do not report incomplete R scores.', '']
    doc+=['','## Actual Progress of All Runs','', '|Run|Stage|Status|Update|Recorded formal outputs|','|---|---|---|---:|---:|']
    for r in reg['runs']:
        st=state['runs'].get(r['name'],{});count=0
        if r['phase']=='eval':
            for f in (ROOT/r['output']/'benchmark').glob('predictions.rank*.jsonl'):
                with f.open('rb') as h:count+=sum(line.endswith(b'\n') for line in h)
        doc.append(f'|{r["name"]}|{r["phase"]}|{st.get("stage","registered")}|{progress.get(r["name"],{}).get("updates",0)}|{count}|')
    doc+=['','## T1: Formal Results on the Same Backbone','', '|Budget|Method|Source|Macro score|DocVQA|ChartQA|DeepForm|OCRBench|','|---|---|---|---:|---:|---:|---:|---:|']
    for ratio in [.25,.5]:
        for label,kind,pos in [('T','quadtree_question','pre-merger'),('C','col_adaptive','pre-merger'),('TP','quadtree_question','post-merger')]:
            s=load(ROOT/legacy(kind,ratio,pos)/'benchmark/summary.json')
            doc.append(f'|{ratio:.0%}|{label}|Historical measurements|{s["macro"]["trained"]:.2f}|{s["tasks"]["DocVQA"]["trained"]:.2f}|{s["tasks"]["ChartQA"]["trained"]:.2f}|{s["tasks"]["DeepForm"]["trained"]:.2f}|{s["tasks"]["OCRBench"]["trained"]:.2f}|')
        for name,ss in results.items():
            c=ss['config']
            if c['ratio']!=ratio or c.get('seed') not in [None,20261105] or c['method'] in ['HA','QSWAP']:continue
            m=ss['variants']['trained'];t=m['by_task']
            doc.append(f'|{ratio:.0%}|{c["method"]}|New formal evaluation completed and audited|{m["macro"]:.2f}|{t["DocVQA"]:.2f}|{t["ChartQA"]:.2f}|{t["DeepForm"]:.2f}|{t["OCRBench"]:.2f}|')
    doc += ['', 'Blank methods are running or unaudited; small validation is not a substitute. Full task scores: benchmark/summary.json and answers.csv. REF macro=57.00409130135, DocVQA=76.9342.','', '## T3/T4: Mechanism Controls and Same-Page Question Swaps','']
    for name in ['HA_r0.25','QSWAP_r0.25']:
        if name not in results:doc+=[f'`{name}`: not yet completed/audited.',''];continue
        ss=results[name];doc+=['|Variant|QA|Images|Tasks represented|Subset macro|','|---|---:|---:|---:|---:|']
        for v,s in ss['variants'].items():doc.append(f'|{v}|{s["rows"]}|{s["images"]}|{s["tasks"]}|{s["macro"]:.2f}|')
        doc.append('')
    doc += ['## Training and Development Trajectory','', '|Run|epoch|Training KL|Training total|Development KL|Development generation macro|','|---|---:|---:|---:|---:|---:|']
    for r in reg['runs']:
        if r['category']!='full_training':continue
        rr=lines(ROOT/r['output']/'training.jsonl')
        for epoch in range(5):
            val=load(ROOT/r['output']/f'validation.{"initial" if epoch==0 else "epoch"+str(epoch)}.summary.json');batch=[x for x in rr if x['epoch']==epoch]
            doc.append(f'|{r["name"]}|{epoch}|{fmt(sum(x["kl"] for x in batch)/len(batch) if batch else None)}|{fmt(sum(x["total"] for x in batch)/len(batch) if batch else None)}|{fmt(val.get("kl"))}|{fmt(val.get("student_macro"))}|')
    doc+=['','## Statistics, Efficiency, and Final Conclusions','', 'Primary formal comparison is T_pre−M_post over three shared seeds: average same-seed question differences, then image-cluster bootstrap 10,000 times. Report SD across seed differences; trained−initial CIs do not replace cross-method CIs. No superiority claim before full coverage.','',
          'Final statistics generate T5 timing/memory, T6 paired intervals, and workbook entries only when all required outputs exist. Pending is not zero or success. Retain negative results and narrow claims if question-conditioning, soft weighting, or speed benefits lack support.', '',
          'Add development pilots to avoid known optimization mismatch. Document eight formal budgets, already inspected benchmarks, and historical-data limitations. Store implementations/logs/checkpoints/train-validation/answers/region diagnostics/timing separately, without overwriting the original 16 runs.']
    target=ROOT/'docs/COMPRESSION_SUPPLEMENT_RESULTS.md';tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
    save(BASE/'merged_results.json',dict(updated_utc=now(),results=results,progress=progress,planned=len(reg['runs'])))
    if args.require_complete:
        assert len(results)==sum(r['phase']=='eval' for r in reg['runs'])
        assert (BASE/'efficiency/complete.json').exists()
        import subprocess
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.analyze_compression_supplement'],cwd=ROOT,check=True)
        final=load(BASE/'analysis_complete.json')
        assert final['three_seed_complete'] and final['timing_complete'] and final['xlsx']
        target.write_text(target.read_text().replace('Full-pipeline completion marker: False.','Full-pipeline completion marker: True.'))
        with target.open('a') as handle:
            handle.write('\n## Final Paired Statistics, Efficiency, and Completed Workbook\n\nAll formal results are covered/audited; three seeds and efficiency records are complete. See [Full statistics and results](COMPRESSION_SUPPLEMENT_STATISTICS.md)and [Completed Excel workbook](../artifacts/compression_supplement_20261111/paper_experiment_tables_completed.xlsx). Conclusions follow actual paired intervals, task scores, and latency measurements; method superiority is not assumed.\n')
    print(json.dumps(dict(formal_audited=len(results),registered=len(reg['runs']),report=str(target))),flush=True)


if __name__=='__main__':main()
