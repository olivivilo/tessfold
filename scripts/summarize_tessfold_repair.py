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
    doc=['# TessFold 修复闭环：实测结果解读','',f'生成时间 UTC：{now()}。','',
         '**入口**：`.venv/bin/python -m scripts.summarize_tessfold_repair`。仅消费完成并通过审计的结果，不改变实验或选择模型。','',
         '## 正确性结论','',
         f'448/448 真实 UND 回归通过。原 P、PH、merger 输出逐位相等，严格路径与历史及新清理路径生成 token 相等。前缀 KV 门槛有 {val["prefix_reuse_token_mismatches"]}/48 次 token 不一致，因此本轮未启用前缀 KV 缓存；未把它包装成无损优化。','',
         f'另对照已经完成的独立原生 generate 显存实验，REF、两预算两方法的旧/新路径共 {native_audit["compared"]} 条自然终止/固定64记录全部生成 token 一致，覆盖本轮全部90道计时问题。','',
         f'FP32 分段候选 {val["candidate_mismatches"]["segmented_fp32"]}/448 次特征不逐位相等，BF16 候选 {val["candidate_mismatches"]["segmented_bf16"]}/448 次不相等；FP64 分区 {val["integral_partition_mismatches"]}/448 次改变分区。均没有晋升默认实现。','',
         '## 时延：严格路径相对原路径和 REF','',
         '90 道固定问题，每题每种模式三次。下表为自然终止模式的平均端到端时延比；先对每题三次取均值，再按页面成组 bootstrap 10,000 次。比值大于 1 表示候选更快。中位 TTFT、分阶段时延、固定 64 步和完整配置表见主报告及 JSON。','',
         '|参考|候选|保留率|参考均值 ms|候选均值 ms|平均时延加速比|页面 bootstrap 95% CI|',
         '|---|---|---:|---:|---:|---:|---|']
    for r in paired:
        if r['mode']=='natural_eos' and r['field']=='end_to_end_ms':
            lo,hi=r['document_bootstrap_95_ci']
            doc.append(f'|{r["reference"]}|{r["candidate"]}|{r["ratio"]:.0%}|{r["reference_mean_ms"]:.2f}|{r["candidate_mean_ms"]:.2f}|{r["speedup"]:.3f}×|[{lo:.3f}, {hi:.3f}]|')
    doc+=['','## 显存：全流程峰值与 decoder 峰值分开','',
          '独立的 1 ms 内存采样 pass 共 1,350 次，与 8,100 次时延测量分开。下表取 90 请求中的最大 allocator peak；decoder 峰值取 prefill/generation 两阶段最大值。不是整卡 npu-smi 显存，也不是均值。','',
          '|方法|保留率|全流程最大峰值 GiB|decoder 最大峰值 GiB|decoder 请求峰值中位数 GiB|',
          '|---|---:|---:|---:|---:|']
    for r in memstats:
        if r['method'] in ['REF','C_DENSE','C_STRICT','T_DENSE','T_STRICT']:
            doc.append(f'|{r["method"]}|{r["ratio"]:.0%}|{r["whole_request_peak_max_gib"]:.4f}|{r["decoder_peak_max_gib"]:.4f}|{r["decoder_peak_median_gib"]:.4f}|')
    doc+=['','压缩发生在 encoder 之后，已发生的 encoder 峰值无法消除。减少 decoder 工作集与降低端到端最高峰是两种不同结论。','',
          '## 同页多问题','',
          f'24 张真实页面、每页 10 个不同问题、REF/Column/Tree、三轮。实际启用场景：{load(BASE/"multi/complete.json")["scenarios"]}。缓存与 cold 路径逐题生成 token 不一致次数：{multi["cache_generation_mismatches"]}。','',
          '所有方法均享有相同的特征缓存权限。Column 复用页面压缩结果；Tree 仍逐题重算问题相关压缩。计入首次摄入，K=1/2/5/10 为相同十题序列的前 K 题，不重复问题凑数。实际均摊时延表见主报告。该页面子集仅覆盖 DocVQA/InfoVQA/WTQ/OCRBench，不代表八任务总体准确率。','',
          '## 边界与新版 Qwen','',
          '`encoder`阶段计时的边界是vision tower返回：旧DENSE与S的tower调用包含原生全长merger，新REF/STRICT则在tower内跳过merger、在后续阶段执行。因此`post_encoder_ttft_ms`对全体方法应严格解释为“tower调用结束后的TTFT”，不是所有配置统一的纯ViT物理边界。新REF与STRICT之间的边界一致；旧/新DENSE比较优先使用包含全部工作的端到端时延，不据该阶段差分杜撰独立全长merger耗时。','',
          '本轮没有重训或替换官方 UND；严格路径对照通过后才解释性能。候选方法单列，不把 hard 当作 soft 的无损实现。时延来自有同步与钩子的 batch=1 测量，模型执行顺序固定、节点由 Slurm 共享；queries/s 是串行均摊速率，不是最大并发压测吞吐。','',
          '本轮UND主实验的 LLM 是 Qwen3-1.7B。Qwen2.5-VL-7B 从未接入；新版 Qwen3-VL-8B/Qwen3.5-9B 的官方配置已核对，但无法直接继承 2048 维视觉对齐和问题桥。详见 [新版 Qwen 兼容性检查](TESSFOLD_QWEN_UPGRADE_FEASIBILITY.md)。','',
          (f'另已跑通独立的 [Qwen3-VL-8B原生与短答案56题对照](TESSFOLD_QWEN3VL_8B_PROMPT_COMPARISON.md)，七任务探索性宏平均原问题{extension["macro_score_percent"]:.2f}、统一短答案{short_extension["macro_score_percent"]:.2f}。使用它自己的视觉编码器与对齐，不是UND内部LLM替换，也不代表旧压缩器已迁移。不能与23565题八任务正式成绩直接比较。' if short_extension else '独立Qwen3-VL-8B完整对照尚未完成；本报告不填该分支未执行的分数。'),' ',
          '[完整实测报告](TESSFOLD_SYSTEM_REPAIR_20261115.md) · [原始结果目录](../artifacts/tessfold_system_repair_20261115/) · [机器可读解读](../artifacts/tessfold_system_repair_20261115/measured_interpretation.json)']
    path=ROOT/'docs/TESSFOLD_REPAIR_MEASURED_RESULTS.md'
    path.write_text('\n'.join(doc)+'\n')
    primary=ROOT/'docs/TESSFOLD_SYSTEM_REPAIR_20261115.md'
    content=primary.read_text()
    marker='\n## 完成后的独立审计与阶段边界说明\n'
    content=content.split(marker)[0]
    content+=marker+'\n完整实验与历史原生generate的逐token核对、页面bootstrap和内存解读见 [实测结果解读](TESSFOLD_REPAIR_MEASURED_RESULTS.md)。\n\n'+audit['phase_boundary']+'\n'
    primary.write_text(content)
    print(json.dumps(dict(report=str(path),comparisons=len(paired),memory_configs=len(memstats))))


if __name__=='__main__':
    main()
