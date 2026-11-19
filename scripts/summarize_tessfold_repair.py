"""Post-hoc measured summary; never launches, changes, or selects experiments.

Entry: .venv/bin/python -m scripts.summarize_tessfold_repair
Requires the signed repaired suite's three stages and its report audit to pass.
"""
from collections import defaultdict
import json
import numpy as np
from scripts.tessfold_repair.common import BASE, ROOT, lines, load, save, sha, verify, now


def paired_speed(rows, old, new, ratio, field, mode):
    """Mean latency ratio with paired document-cluster bootstrap, fixed 10k draws."""
    grouped = defaultdict(list)
    for row in rows:
        if row['instrumentation'] != 'synchronized_wall_and_NPU_events' or row['mode'] != mode:
            continue
        if row['method'] in [old, new] and row['ratio'] == (1. if row['method'] == 'REF' else ratio):
            grouped[(row['id'], row['method'])].append(row)
    pages = defaultdict(lambda: [0., 0., 0])
    ids = sorted({key[0] for key in grouped})
    for id_ in ids:
        aa, bb = grouped[(id_, old)], grouped[(id_, new)]
        assert len(aa) == len(bb) == 3
        page = aa[0]['document_id']
        pages[page][0] += float(np.mean([r[field] for r in aa]))
        pages[page][1] += float(np.mean([r[field] for r in bb]))
        pages[page][2] += 1
    array = np.asarray([pages[p] for p in sorted(pages)])
    rng = np.random.default_rng(20261115)
    samples = array[rng.integers(len(array), size=(10000, len(array)))].sum(axis=1)
    bootstrap = samples[:, 0] / samples[:, 1]
    return dict(reference=old, candidate=new, ratio=ratio, mode=mode, field=field,
        questions=len(ids), documents=len(pages), repeats_per_question=3,
        reference_mean_ms=float(array[:, 0].sum()/array[:, 2].sum()),
        candidate_mean_ms=float(array[:, 1].sum()/array[:, 2].sum()),
        speedup=float(array[:, 0].sum()/array[:, 1].sum()),
        document_bootstrap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist())


def main():
    verify()
    from scripts.audit_tessfold_reference_tokens import audit as audit_native
    native_audit = audit_native(require_complete=True)
    val = load(BASE/'validation/complete.json')
    assert val.get('strict_passed') and val['cases'] == 448
    assert load(BASE/'single/audit.json').get('passed')
    assert load(BASE/'multi/complete.json')
    single = [row for path in (BASE/'single').glob('results.rank*.jsonl') for row in lines(path)]
    assert len(single) == 9450
    latency = [r for r in single if r['instrumentation'] == 'synchronized_wall_and_NPU_events']
    memory = [r for r in single if r['instrumentation'] == 'memory1ms']
    assert len(latency) == 8100 and len(memory) == 1350
    paired = []
    for ratio in [.25, .5]:
        for method in ['C_STRICT', 'T_STRICT']:
            for old in [method.replace('_STRICT', '_DENSE'), 'REF']:
                for mode in ['natural_eos', 'fixed64']:
                    for field in ['end_to_end_ms', 'ttft_ms', 'post_encoder_ttft_ms']:
                        paired.append(paired_speed(single, old, method, ratio, field, mode))
    grouped = defaultdict(list)
    for row in memory:
        assert sha(ROOT/row['trace']) == row['trace_sha256']
        grouped[(row['method'], row['ratio'])].append(row)
    memstats = []
    for (method, ratio), rows in sorted(grouped.items()):
        assert len(rows) == 90
        stages = sorted({k for r in rows for k in r['stages']})
        qwen = [max(r['stages'][k]['peak_allocated_bytes'] for k in r['stages']
                    if k in ['qwen_prefill', 'generation', 'decoder_total']) for r in rows]
        memstats.append(dict(method=method, ratio=ratio, requests=90,
            whole_request_peak_max_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
            decoder_peak_max_gib=max(qwen)/2**30,
            decoder_peak_median_gib=float(np.median(qwen))/2**30,
            phase_peak_max_gib={k:max(r['stages'][k]['peak_allocated_bytes'] for r in rows if k in r['stages'])/2**30 for k in stages},
            phase_average_median_gib={k:float(np.median([r['stages'][k]['time_average_allocated_bytes'] for r in rows if k in r['stages']]))/2**30 for k in stages}))
    multi = load(BASE/'multi/summary.json')
    extension = load(ROOT/'artifacts/tessfold_qwen3vl_extension_20261115/complete.json')
    short_extension = load(ROOT/'artifacts/tessfold_qwen3vl_short_extension_20261115/complete.json')
    audit = dict(time=now(),latency_records=8100,memory_records=1350,
        validation_cases=448,historical_native_generation_audit=native_audit,prefix_gate_mismatches=val['prefix_reuse_token_mismatches'],
        actual_multi_generation_mismatches=multi['cache_generation_mismatches'],
        paired_comparisons=paired,memory=memstats,
        native8b_extension=dict(raw_macro=extension.get('macro_score_percent'),short_macro=short_extension.get('macro_score_percent'),
            scope='Separate native56developmentQA7task baseline and uniform short-answer control; no UND encoder transplant or compressor migration.'),
        phase_boundary='encoder denotes the timed tower call: old DENSE/S includes native full-length merger; clean REF/STRICT uses Identity merger and aligns in compression_and_merger. post_encoder_ttft is literal post-tower, not a uniform physical post-ViT boundary across all methods. E2E includes all work.',
        interpretation='Paired ratio of QA mean latencies, with pages jointly resampled; finite 90-QA cohort, fixed model order, shared Slurm nodes. Not production concurrent throughput.')
    save(BASE/'measured_interpretation.json',audit)
    doc=['# TessFold Repair: Interpretation of Measured Results','',f'Generated (UTC):{now()}. ','',
         '**Entry point**: `.venv/bin/python -m scripts.summarize_tessfold_repair`. Uses only completed audited results, without changing experiments or selecting models.','',
         '## Correctness Findings','',
         f'448/448 real-UND regressions pass: P, PH, and merger outputs are bit-exact; strict generation matches historical and newly cleaned paths. Prefix-KV checks have  {val["prefix_reuse_token_mismatches"]}/48 token mismatches; prefix KV is disabled in this round and is not presented as lossless optimization.','',
         f'Additional comparison against completed native-generate memory tests covers REF and both methods/budgets, old/new paths, totaling  {native_audit["compared"]}  natural/fixed64 records have identical generated tokens across all 90 timing questions.','',
         f'FP32 segmented candidate {val["candidate_mismatches"]["segmented_fp32"]}/448 non-bit-exact feature cases; BF16 candidate  {val["candidate_mismatches"]["segmented_bf16"]}/448 unequal cases; FP64 partitions  {val["integral_partition_mismatches"]}/448 changed partitions. Neither was promoted to the default.','',
         '## Latency: Strict Path Versus Original Path and REF','',
         '90 fixed questions, three repeats/question/mode. Below are natural-EOS mean E2E latency ratios: average repeats within question, then bootstrap page clusters 10,000 times. Ratios >1 favor the candidate. Median TTFT, stage timings, fixed64, and full configurations are in the main report/JSON.','',
         '|Reference|Candidate|Retention fraction|Reference mean (ms)|Candidate mean (ms)|Mean-latency speedup|Page-bootstrap 95% CI|',
         '|---|---|---:|---:|---:|---:|---|']
    for r in paired:
        if r['mode']=='natural_eos' and r['field']=='end_to_end_ms':
            lo,hi=r['document_bootstrap_95_ci']
            doc.append(f'|{r["reference"]}|{r["candidate"]}|{r["ratio"]:.0%}|{r["reference_mean_ms"]:.2f}|{r["candidate_mean_ms"]:.2f}|{r["speedup"]:.3f}×|[{lo:.3f}, {hi:.3f}]|')
    doc+=['','## Memory: Full-Pipeline and Decoder Peaks Reported Separately','',
          'The independent 1 ms memory pass has 1,350 measurements separate from 8,100 timing records. The table takes the maximum allocator peak across 90 requests; decoder peak is the maximum of prefill/generation. These are neither npu-smi device-wide memory nor mean occupancy.','',
          '|Method|Retention fraction|Maximum full-pipeline peak (GiB)|Maximum decoder peak (GiB)|Median request decoder peak (GiB)|',
          '|---|---:|---:|---:|---:|']
    for r in memstats:
        if r['method'] in ['REF','C_DENSE','C_STRICT','T_DENSE','T_STRICT']:
            doc.append(f'|{r["method"]}|{r["ratio"]:.0%}|{r["whole_request_peak_max_gib"]:.4f}|{r["decoder_peak_max_gib"]:.4f}|{r["decoder_peak_median_gib"]:.4f}|')
    doc+=['','Compression follows encoding and cannot remove an already incurred encoder peak. Reducing decoder working memory and reducing the maximum E2E peak are distinct claims.','',
          '## Multiple Questions on the Same Page','',
          f'24 real pages, 10 distinct questions/page, REF/Column/Tree, three repeats. Enabled scenarios: {load(BASE/"multi/complete.json")["scenarios"]}. Questions with cache-versus-cold generated-token mismatches: {multi["cache_generation_mismatches"]}. ','',
          'All methods have equal feature-cache access. Column reuses page compression; Tree recomputes question-dependent compression. Initial ingestion is included. K=1/2/5/10 uses prefixes of one ten-question sequence without duplicating questions. See the main report for amortized latency. This page subset covers DocVQA/InfoVQA/WTQ/OCRBench only, not overall eight-task accuracy.','',
          '## Scope and the Newer Qwen Backbone','',
          '`encoder`Timing boundary is tower return: old DENSE/S include full native merger inside tower, while new REF/STRICT skip it there and run it later. Therefore `post_encoder_ttft_ms`Across methods, interpret this strictly as TTFT after the tower call, not a common pure-ViT boundary. New REF/STRICT boundaries match. Compare old/new DENSE by full E2E rather than inventing independent full-merger latency from stage differences.','',
          'The official UND backbone is neither retrained nor replaced; performance is interpreted only after strict-path checks pass. Candidates are separate; hard selection is not a lossless implementation of soft aggregation. Batch=1 timings include synchronization/hooks, fixed model order, and Slurm-shared nodes. queries/s is serial amortized rate, not maximum concurrent throughput.','',
          'The main UND LLM is Qwen3-1.7B. Qwen2.5-VL-7B was never attached. Official Qwen3-VL-8B/Qwen3.5-9B configs were checked, but cannot inherit 2,048-dimensional alignment/question bridging directly. See [New-Qwen compatibility](TESSFOLD_QWEN_UPGRADE_FEASIBILITY.md). ','',
          (f'Separately completed [56-question native/short-answer Qwen3-VL-8B comparison](TESSFOLD_QWEN3VL_8B_PROMPT_COMPARISON.md), exploratory seven-task macro: original prompt {extension["macro_score_percent"]:.2f}, uniform short answers {short_extension["macro_score_percent"]:.2f}. It uses its own visual encoder/alignment, not an UND LLM replacement or transfer of the earlier compressor. It is not directly comparable to 23,565-QA/eight-task formal scores.' if short_extension else 'Independent full Qwen3-VL-8B comparison incomplete; unexecuted scores remain blank.'),' ',
          '[Full measured report](TESSFOLD_SYSTEM_REPAIR_20261115.md) · [Original result directory](../artifacts/tessfold_system_repair_20261115/) · [Machine-readable interpretation](../artifacts/tessfold_system_repair_20261115/measured_interpretation.json)']
    path=ROOT/'docs/TESSFOLD_REPAIR_MEASURED_RESULTS.md'
    path.write_text('\n'.join(doc)+'\n')
    primary=ROOT/'docs/TESSFOLD_SYSTEM_REPAIR_20261115.md'
    content=primary.read_text()
    marker='\n## Post-Completion Independent Audit and Stage Boundaries\n'
    content=content.split(marker)[0]
    content+=marker+'\nComplete tokenwise comparison with historical native generate, page bootstraps, and memory interpretation: [Measured-results interpretation](TESSFOLD_REPAIR_MEASURED_RESULTS.md). \n\n'+audit['phase_boundary']+'\n'
    primary.write_text(content)
    print(json.dumps(dict(report=str(path),comparisons=len(paired),memory_configs=len(memstats))))


if __name__=='__main__':
    main()
