"""Measured progress against tessfold.md; never fill targets as results."""
import argparse,json
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    p=argparse.ArgumentParser();p.add_argument('--publish-complete',action='store_true');a=p.parse_args()
    core=['system_exact','formal_equivalence','prefix_formal','interleaved','multi_deployment','batched_serving','qwen_formal','qwen_timing','qwen_memory','hard_formal','bidirectional_formal','bidirectional_timing','bidir_interleaved']
    stages={name:(BASE/name/'complete.json').exists() for name in core};complete=all(stages.values()) and bool(load(BASE/'final_audit.json').get('all_checks_passed'))
    if a.publish_complete:assert complete
    status='All planned core measurements and final audits completed' if complete else 'Still running; completed and incomplete stages are listed separately below'
    rows=[('No retraining of original UND or trained P','All original weights remain frozen; only added P is trained for native 8B. Source/data/checkpoint signatures pass.','Implemented'),
        ('100% old/new numerical and generation equivalence','94,260 records: 23,565 QA × two methods × 25%/50%; all P, PH, and generated tokens match.','Implemented' if stages['formal_equivalence'] else 'Running'),
        ('Full soft compression <0.8 ms and O(N)','Strict implementation retains FP32 dense aggregation. Median is 2.20 ms over 448 cases, and graph-replay median 1.57 ms over 8 cases, missing the target. O(N) segmented candidates change features/generation or regress latency.','Not achieved'),
        ('Hard fast path','No dense P is constructed. Across 24 graph-replay cases, median latency is 1.04 ms, minimum 0.687 ms. This is a separate hard model and inherits no soft losslessness; the minimum is not typical latency. Full 23,565-QA evaluation is separate.','Full evaluation completed' if stages['hard_formal'] else 'Full evaluation running'),
        ('Four-part efficiency measurement','All 9,450 UND records: stage timing/actual TTFT/independent memory/logical KV/same-page QA; another 2,520 interleaved measurements control drift.','Implemented' if stages['system_exact'] and stages['interleaved'] else 'Running'),
        ('Actual same-page QA amortization at K=1/2/5/10','24 real multi-question pages × 3 methods × 4 scenarios × 3 repeats = 864 page runs / 8,640 QA executions. REF has equal cache access. Feature-only results are identical; prefix reuse is a separate numerical variant.','Implemented' if stages['multi_deployment'] else 'Running'),
        ('Visual KV caching is necessarily lossless','Both causally valid prefix boundaries can be affected by chunked-prefill numerics; this is not answer leakage across questions. Full REF/Column evaluations each cover 23,565 questions.','Task-impact evaluation completed' if stages['prefix_formal'] else 'Task-impact evaluation running'),
        ('75% less visual KV implies fourfold serving throughput','That inference is invalid. Real synchronous batch1/2/4/8 is measured; long fixed64 batch8 gives REF 1.385 q/s versus Column25% 1.712 q/s, about 1.236×, not maximum asynchronous capacity.','Fourfold target not achieved; measurements reported'),
        ('Compression transfer to a newer ~7B-class model','Use newer complete native Qwen3-VL-8B-Instruct, frozen except post-merger P applied to all 3 DeepStack branches. Not an UND-vision graft. Four 20k×4-epoch runs use development-only checkpoint selection.','Formal training and evaluation completed' if stages['qwen_formal'] else 'Training or formal evaluation running'),
        ('8B cold-request TTFT speedup of 2.2–3.1×','Require measured 90-QA timings/CIs rather than copying expectations into results. Native 8B and UND have different insertion boundaries/frontends.','Measured; see below' if stages['qwen_timing'] else 'Pending formal timing'),
        ('8B compressor training improves generalization','Besides full teacher/four-model evaluation, a fixed 512-image/2,404-QA subset compares initial/dev-selected P per model, adding 9,616 generations without selection.','Completed' if stages['qwen_formal'] else 'Integrated; evaluation pending training completion'),
        ('Actual downstream memory and manuscript tables','Full UND stagewise memory is measured. Another 1,350 independent 1 ms native-8B measurements are token-checked against timing, retain encoder peaks, and generate separate LaTeX without changing old PDFs.','Completed' if stages['qwen_memory'] else 'UND completed; 8B pending')]
    doc=['# Progress Against tessfold.md: Implemented Improvements and Remaining Gaps','',f'UTC: {now()}. **{status}**. ','',
        '**Execution entry point**: `.venv/bin/python -m scripts.tessfold_accel.controller status|resume`. Rebuild results: `.venv/bin/python -m scripts.tessfold_accel.finish_report`; final audit: `.venv/bin/python -m scripts.tessfold_accel.final_audit`. ','',
        '[Full experiment report](TESSFOLD_ACCELERATION_AND_8B_RESULTS.md) · [UND efficiency details](TESSFOLD_COMPILED_SYSTEM_RESULTS.md) · [Measured manuscript tables](TESSFOLD_PAPER_EFFICIENCY_TABLE.md). ','',
        '## Comparison Against Each Target','', '|Reference-proposal target|Implemented architecture, evidence, and scope|Status|','|---|---|---|']
    doc += [f'|{goal}|{actual}|{state}|' for goal,actual,state in rows]
    fused=load(BASE/'fused_vision_pilot/complete.json')
    if fused:
        doc+=['','## Why Faster Visual Attention Is Not Enabled Directly','',
            f'Author-provided Ascend fused-attention median encoder speedup on 56 fixed development questions: {fused["median_encoder_speedup"]:.3f}×, but REF per-question mean change {fused["ref_mean_qa_delta_pp"]:+.3f} percentage points, failing the prespecified quality criterion. The original call\'s median final-feature cosine is about 0.9515. A subsequent audit identified a risk from supplying a mask without next_tokens, so explicit bidirectional attention was validated independently. The original call cannot be assumed mathematically equivalent with rounding-only differences. All records remain and the candidate is not promoted. Query-chunked eager attention preserving original rounding boundaries is also tested; expansion requires feature and generation checks.']
    bidir=load(BASE/'bidirectional_vision_pilot/complete.json')
    if bidir:
        doc+=['','## Explicit Bidirectional Fusion: A Separate Candidate Passing the Pilot','',
            f'Local PFA documentation states that pre/next constrain attention when a mask is provided. Original default next_tokens=0; the independent bidirectional-INTMAX version has median encoder speedup on 56 questions of {bidir["median_encoder_speedup"]:.3f}×; REF macro change {bidir["ref_mean_qa_delta_pp"]:+.3f} pp, passing median speedup≥1.5× and score loss≤2 pp. Expansion covers 23,565 QA×5 configurations and 3,150 independent timing/memory measurements. It remains non-bit-exact and does not overwrite strict eager scores. Full results/paired CIs are in the complete report.']
    selected=load(BASE/'qwen_selected_runs.json')
    if selected:
        doc+=['','## Locked Development-Set Selection for 8B','', '|Configuration|Selected epoch|checkpoint|','|---|---|---|']
        for r in selected['runs']:doc.append(f'|{r["id"]}|{r["stage"]}|`{r["checkpoint"]}`|')
    bidir_stats=load(BASE/'bidir_interleaved/statistics.json')
    if bidir_stats:
        doc+=['','## Interleaved Same-Question Comparison: Original Eager and Explicit Bidirectional Deployment','',
            'For each question/mode/repeat, randomly interleave six configurations over two repeats on 90 questions, counting all frontend work. Ratios below compare paired means. Cross-backend speedups require full quality reporting and are not bit-exact gains.','',
            '|Reference|Candidate|Mode|E2E speedup|95% page-level CI|TTFT speedup|','|---|---|---|---:|---|---:|']
        for v in bidir_stats['comparisons']:
            if v['mode']=='natural_eos' and v['reference_backend']=='EAGER' and v['reference_method']=='REF' and v['candidate_backend']=='BIDIR':
                ci=v['e2e_document_bootstrap_95_ci'];doc.append(f'|EAGER/REF|BIDIR/{v["candidate_method"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|')
    timing=load(BASE/'qwen_timing/statistics.json')
    if timing:
        doc+=['','## Does Measured 8B Speed Meet the Targets?','', '|Model|Mode|Same-question E2E speedup|Same-question TTFT speedup|TTFT 95% page-level CI|','|---|---|---:|---:|---|']
        for v in timing['comparisons']:
            if v['stratum']=='all':
                ci=v['ttft_bootstrap_95_ci'];doc.append(f'|{v["model"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|{v["ttft_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|')
    full_bidir=load(BASE/'bidirectional_formal/complete.json')
    if full_bidir:
        bs=load(BASE/'bidirectional_statistics.json')
        doc+=['','## Full-Evaluation Quality Changes with Bidirectional Fusion','',
            'Same frozen UND weights, independently generated 23,565 questions. Shared field-EM macro differences use task-stratified document-bootstrap CIs versus eager. Zero-crossing CIs do not imply numerical/statistical equivalence.','',
            '|Configuration|Full macro|Difference from eager (pp)|95% page-level CI|','|---|---:|---:|---|']
        for model,value in full_bidir['shared_fieldEM_results'].items():
            delta=bs['quality_vs_original_eager'][model];ci=delta['shared_fieldEM_macro_delta_task_stratified_document_bootstrap_95_ci']
            doc.append(f'|{model}|{value["macro"]:.4f}|{delta["shared_fieldEM_macro_delta"]:+.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|')
    doc+=['','## Current Core Stage','', '```json',json.dumps(stages,ensure_ascii=False,indent=2),'```','',
        'Incomplete stages have no final scores; submission is not a pass. Unsupported sub-ms soft, 75% request-memory reduction, fourfold concurrency, and target 8B speedups are not manuscript claims. Historical failures and model versions remain.']
    path=ROOT/'docs/TESSFOLD_TARGET_GAP.md';path.write_text('\n'.join(doc)+'\n')
    save(BASE/'target_gap.json',dict(time=now(),all_core_and_final_audit_complete=complete,stages=stages,tessfold_source_sha256=sha(ROOT/'tessfold.md'),report=str(path.relative_to(ROOT))))
    if a.publish_complete:
        note='**2026-11-15 TessFold round two: all core measurements and the final audit are complete.** [Full experimental results](docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md) · [Remaining gaps against tessfold.md](docs/TESSFOLD_TARGET_GAP.md) · [Measured manuscript tables](docs/TESSFOLD_PAPER_EFFICIENCY_TABLE.md). Missed submillisecond-soft, fourfold-concurrency, and expected-8B-speed targets are reported as measured. Original models/results remain unchanged.'
        readme=ROOT/'README.md';text=readme.read_text();parts=text.split('\n\n',2)
        if len(parts)>2 and parts[1].startswith('**Current continuation task'):text=parts[0]+'\n\n'+note+'\n\n'+parts[2]
        else:
            first=text.find('\n\n');text=text[:first]+'\n\n'+note+text[first:]
        readme.write_text(text)
        current=ROOT/'docs/CURRENT_PROGRESS.md';text=current.read_text();first=text.find('\n\n');local=note.replace('(docs/','(')
        text=text.replace('## 2026-11-15 Round-Two TessFold Acceleration and Native 8B Compression: In Progress','## History: Execution Records Before Round-Two Completion',1)
        text=text[:first]+'\n\n'+local+'\n\nRunning-state descriptions below are preserved history; final status is determined by this paragraph and the final report.'+text[first:]
        current.write_text(text)
    print(json.dumps(dict(report=str(path),all_complete=complete)))

if __name__=='__main__':main()
