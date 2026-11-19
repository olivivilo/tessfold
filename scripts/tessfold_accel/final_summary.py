"""Concise final entry, populated only from audited completed experiments."""
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    audit=load(BASE/'final_audit.json');assert audit['all_checks_passed']
    for stage,digest in audit['stage_complete_sha256'].items():assert sha(BASE/stage/'complete.json')==digest
    assert (BASE/'all_stages_complete.json').exists()
    native=load(BASE/'qwen_formal/complete.json');registry=load(BASE/'qwen_selected_runs.json');stats=load(BASE/'bidir_interleaved/statistics.json')
    names={'REF':'Uncompressed native 8B','Column_r0.25':'Column 25%','Tree_r0.25':'Tree 25%','Column_r0.5':'StableColumn 50%','Tree_r0.5':'Tree 50%'}
    methods=['REF','Column_r0.25','Tree_r0.25','Column_r0.5','Tree_r0.5'];tasks=['DocVQA','InfoVQA','ChartQA','OCRBench','DeepForm','KLC','WTQ','DT-VQA']
    doc=['# TessFold Final Results and Reading Guide','',f'Results compiled (UTC):{now()}. **All 13 stages and the final audit are complete; training/evaluation jobs have exited.**','',
        'Completed inference-only UND optimization and four native-Qwen3-VL-8B 20k×4-epoch compressor runs with full independent evaluation. Not all tessfold.md targets are reached; measured outcomes/limitations and original models/history/negative results remain.','',
        '## Recommended Reading Order','',
        '- [Full design, procedure, scores, latency, memory, and curves](TESSFOLD_ACCELERATION_AND_8B_RESULTS.md)','- [Progress and remaining gaps against tessfold.md](TESSFOLD_TARGET_GAP.md)',
        '- [Visual-attention range: direct operator evidence and bidirectional repair](TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md)',
        '- [Actual native 8B architecture and training method](TESSFOLD_NATIVE8B_METHOD.md) · [Validation protocol](TESSFOLD_FINAL_VALIDATION_PROTOCOL.md) · [Environment and entry points](TESSFOLD_RUNTIME_REPRODUCTION.md)',
        '- [Measured UND manuscript tables](TESSFOLD_PAPER_EFFICIENCY_TABLE.md) · [8B/fusion manuscript tables and figures](TESSFOLD_PAPER_EXTENSION_TABLES.md)','',
        '## 1. Strict UND Path: Exact Preservation of Compression Outputs','',
        'Compile partitioning, cache geometry, avoid integer-sort CPU fallback, and remove redundant merger. All 94,260 P/PH/generation checks across 23,565 QA×4 configurations match. Lossless means new-versus-original compressed implementation, not teacher-equivalent accuracy.',
        'Original FP32 softmax/GEMM are retained, so no bit-exact O(N) soft replacement has been obtained. Median compressor-only latency across 448 cases is about 2.20 ms. On the full 90-question natural-EOS deployment set, compression+merger medians are 3.42 ms for Column25% and 4.10 ms for Tree25%, versus about 26 ms previously. Completed tests comprise 9,450 full-efficiency records, 2,520 additional interleaved records, 864 real same-page runs / 8,640 question executions, and 144 real batches.','',
        'Feature caching and avoiding repeated processing match cold-start tokens exactly. Fair K=10 caching gives REF 313.86 ms/question and Column 235.09 ms/question, speedup 1.335× with 95% page CI [1.202,1.469], including initial ingestion. Prefix KV still changes some tokens on full evaluation: Column shared-macro difference −0.0044 pp, 95% CI [−0.1307,+0.1244]. It is a separate numerical path, not assumed lossless.','',
        'Hard without dense-P readout scores macro 38.2395 on all 23,565 QA, with tokens matching historical Hard. Soft Tree25% scores 38.4705; close point estimates do not constitute an equivalence test.','',
        '## 2. Explicit Bidirectional Fusion: A Key Repair','',
        'The author\'s Ascend branch supplies a mask but omits next_tokens, whose installed default is 0. Direct Q/K/V tests show that the original length-1024 call resembles restricted block attention, while explicit pre/next=INTMAX resembles full bidirectional attention. Rounding differences from eager remain; this cannot replace strict-path bit-exact evidence. The initial −16.68 pp result is retained. After a 56-question pilot passed, the corrected variant received independent evaluation on 23,565 QA × 5 configurations, plus 2,700 timing, 450 independent memory, and 2,160 same-question interleaved timing records.','',
        'Shared-macro changes versus eager: REF +0.1133, Column25% +0.2638, Tree25% −0.1956, Column50% −0.1947, Tree50% +0.0983 pp. All macro CIs cross zero, which does not establish equivalence. Full task changes are in the report.','',
        '|Interleaved same-question comparison, natural EOS|End-to-end speedup|End-to-end 95% page CI|TTFT speedup|TTFT 95% page-level CI|','|---|---:|---|---:|---|']
    for r in stats['comparisons']:
        if r['mode']=='natural_eos' and ((r['reference_backend']=='EAGER' and r['reference_method']=='REF' and r['candidate_backend']=='BIDIR') or (r['reference_backend']=='BIDIR' and r['reference_method']=='REF')):
            ci=r['e2e_document_bootstrap_95_ci'];ti=r['ttft_document_bootstrap_95_ci']
            doc.append(f'|{r["reference_backend"]}/{r["reference_method"]}→{r["candidate_backend"]}/{r["candidate_method"]}|{r["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{r["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
    doc+=['','Independent memory on the same fused backend: REF full peak 4.279 GiB, 25% students about 3.736 GiB. Old eager REF is about 6.410 GiB, but these are separate measured numerical paths; not all fusion-related memory gains belong to P.','',
        '## 3. Native 8B: Four Training Runs Improve Independent Generalization','',
        'Use complete Qwen3-VL-8B-Instruct with its own vision/merger, all frozen except new spatial P. Post-merger P aggregates final features plus 3 DeepStack branches and retains representative MRoPE. This is not UND-vision/8B-LLM grafting.','',
        'Each configuration passes a multi-image pilot before fresh 20k×4 training. Select only on image-disjoint 56-QA development. Formal native8B uses 23,565 QA, DUE corpus F1 for DeepForm/KLC, native others, task-equal macro. Old shared field-EM is saved separately.','',
        '|Model|DocVQA ANLS|InfoVQA ANLS|ChartQA|OCRBench|DeepForm F1|KLC F1|WTQ|DT-VQA|Native macro|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        v=native['native_metric_results'][m];doc.append('|'+names[m]+'|'+'|'.join(f'{v["by_task"][t]:.2f}' for t in tasks)+f'|{v["macro"]:.2f}|')
    doc+=['','Initial/trained P are paired on a fixed independent 512-image/2,404-QA subset that does not select models: ','',
        '|Model|Initial native macro|Trained native macro|Improvement (pp)|','|---|---:|---:|---:|']
    for m in methods[1:]:
        v=native['initial_control_results']['comparisons'][m];doc.append(f'|{names[m]}|{v["initial"]["native_macro"]:.2f}|{v["selected"]["native_macro"]:.2f}|{v["native_macro_delta"]:+.2f}|')
    doc+=['','All trained macros improve on the independent subset, but 25% still loses substantial quality. Among prespecified configurations, Tree50% is closest to teacher: full macro −3.492 pp, DocVQA −2.871 pp. This describes results without test-based epoch reselection.','',
        'At 25% retention, native 8B full cold-request TTFT speedup is only about 1.14–1.15× and E2E about 1.08×, below the proposed 2.2–3.1×. Median original LLM visual-token counts are 4,960 for UND and 945 for native 8B, whose native merger already reduces tokens. Model size alone cannot predict larger compression gains. Frontend mean=132.10 ms versus TTFT=220.85 ms gives an ideal zero-backend upper bound of about 1.672× with the frontend unchanged.','',
        '## 4. Audits, Outputs, and Remaining Limitations','',
        'All 330 source-reference signature checks pass. Four runs have 12,500 consecutive steps with matching LR/clipping and actual optimizer state; initial + 4 epochs give 1,120 recomputed development records. All 1,350 native-8B natural-timing outputs match formal outputs, and 1,350 independent memory outputs match timing outputs. The 1,800 bidirectional-fusion natural timing/memory outputs also match formal generation. All trace signatures pass, as do checks preserving the 26,990/34,132 predictions from before parallel resumption.','',
        '- [Final-audit JSON](../artifacts/tessfold_acceleration_20261115/final_audit.json) · [Completion marker](../artifacts/tessfold_acceleration_20261115/all_stages_complete.json)',
        '- [Full scores and initial controls](../artifacts/tessfold_acceleration_20261115/qwen_formal/complete.json)',
        '- [Per-question gold/teacher/student score-drop CSV](../artifacts/tessfold_acceleration_20261115/error_analysis/largest_losses.csv)',
        '- [Full-evaluation / independent-control figure PDF](../artifacts/tessfold_acceleration_20261115/figures/native8b_quality_and_controls.pdf)',
        '- [Per-epoch loss, generation-score, and gradient diagnostics](TESSFOLD_8B_TRAINING_DIAGNOSTICS.md)','',
        'Strict soft latency <0.8 ms, fourfold actual throughput, and 8B cold-request speedups of 2.2–3.1× are not achieved. Reduced visual KV does not imply proportional device-wide peak-memory reduction. Limitations remain: small fixed development set, one training seed, low native8B KLC/DeepForm baseline, and 35/20,000 truncated teacher prefixes. Old results remain; new numerical variants do not inherit old accuracy.','',
        'Rebuild results: `.venv/bin/python -m scripts.tessfold_accel.finish_report --require-complete`; this page: `.venv/bin/python -m scripts.tessfold_accel.final_summary`. Completed stages require no new training submission.']
    path=ROOT/'docs/TESSFOLD_FINAL_RESULTS.md';path.write_text('\n'.join(doc)+'\n')
    save(BASE/'final_summary.json',dict(time=now(),report=str(path.relative_to(ROOT)),final_audit_sha256=sha(BASE/'final_audit.json'),complete_sha256=sha(BASE/'all_stages_complete.json')))
    print(path)

if __name__=='__main__':main()
