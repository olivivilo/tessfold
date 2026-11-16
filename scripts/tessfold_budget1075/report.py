"""Report actual committed rows only; certify full runs with fail-closed audit."""
import collections
import json
import math
from pathlib import Path
import numpy as np
from scripts.tessfold_budget1075.common import (
    ROOT, BASE, PARENT, REGISTRY, REPORT, MODELS, model_id, full_label, completed,
    load, lines, save, sha, now, verify, refresh,
)
from scripts.tessfold_accel.live_io import live_lines


def strict(directory, pattern='rows.rank*.jsonl'):
    rows = []
    for path in sorted(directory.glob(pattern)):
        data = path.read_bytes()
        assert not data or data.endswith(b'\n'), f'Incomplete tail: {path}'
        rows.extend(json.loads(s) for s in data.splitlines() if s.strip())
    return rows


def training_rows():
    result = []
    for kind, ratio in MODELS:
        identity = model_id(kind, ratio)
        run = BASE / full_label(kind, ratio) / identity
        refresh(run / 'manifest.json')
        manifest = load(run / 'manifest.json')
        initial = load(run / 'initial.summary.json')
        logs = live_lines(run / 'training.jsonl')
        epochs = []
        for epoch in range(1, 5):
            rr = [r for r in logs if r['epoch'] == epoch]
            dev = load(run / f'epoch{epoch}.summary.json')
            if rr:
                epochs.append(dict(epoch=epoch, steps=len(rr), train_kl=float(np.mean([r['kl'] for r in rr])),
                                   dev_kl=dev.get('kl'), dev_macro=dev.get('macro'), complete=bool(dev),
                                   norm_p50=float(np.median([r['gradient_norm'] for r in rr])),
                                   norm_p99=float(np.quantile([r['gradient_norm'] for r in rr], .99)),
                                   clip_fraction=float(np.mean([r['clip_multiplier'] < 1 for r in rr]))))
        result.append(dict(id=identity, directory=str(run.relative_to(ROOT)), manifest=manifest,
                           initial=initial, steps=len(logs), epochs=epochs,
                           complete=load(run / 'complete.json')))
    return result


def graphs(training):
    signature = [(r['id'], [(e['epoch'], e['steps'], e['dev_kl']) for e in r['epochs']]) for r in training]
    # Regenerate only at a new completed epoch, never on every live step.
    stable = [(r['id'], [e for e in r['epochs'] if e['complete']]) for r in training]
    identity = json.dumps(stable, sort_keys=True)
    if not any(es for _, es in stable) or load(BASE / 'figures/signature.json').get('value') == identity:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = BASE / 'figures'
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, run in zip(axes.flat, training):
        epochs = [e for e in run['epochs'] if e['complete']]
        if not epochs:
            ax.set_title(run['id'] + ' (pending)')
            continue
        xs = [e['epoch'] for e in epochs]
        ax.plot(xs, [e['train_kl'] for e in epochs], 'o-', label='Training KL')
        initial = run['initial']
        ax.plot([0, *xs], [initial['kl'], *[e['dev_kl'] for e in epochs]], 's-', label='Dev KL')
        ax.set(title=run['id'], xlabel='Epoch', ylabel='KL')
        ax.grid(alpha=.2)
        other = ax.twinx()
        other.plot([0, *xs], [initial['macro'], *[e['dev_macro'] for e in epochs]], '^-', color='green', label='Dev macro')
        other.set_ylabel('Dev native macro (%)')
        ax.legend(loc='upper left', fontsize=8)
        other.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'training_curves.pdf')
    fig.savefig(directory / 'training_curves.png', dpi=180)
    plt.close(fig)
    save(directory / 'signature.json', dict(value=identity, observed_signature=signature))


def memory_summary():
    rows = strict(BASE / 'qwen_memory')
    timing = [r for r in strict(BASE / 'qwen_timing') if r['mode'] == 'natural_eos']
    index = {(r['id'], r['model'], r['repeat']): r for r in timing}
    assert len(rows) == len(index) == 1350
    by_model = collections.defaultdict(list)
    for r in rows:
        assert r['generated_ids'] == index[(r['id'], r['model'], r['repeat'])]['generated_ids']
        assert sha(ROOT / r['trace']) == r['trace_sha256']
        by_model[r['model']].append(r)
    stats = {}
    for identity, rr in sorted(by_model.items()):
        stats[identity] = dict(full_max_gib=max(r['peak_allocated_bytes'] for r in rr) / 2**30,
                              front_max_gib=max(r['stages']['front_processor_vision_mergers']['peak_allocated_bytes'] for r in rr) / 2**30,
                              downstream_max_gib=max(r['downstream_peak_allocated_bytes'] for r in rr) / 2**30,
                              prefill_mean_gib=float(np.mean([r['stages']['qwen_prefill']['time_average_allocated_bytes'] for r in rr])) / 2**30,
                              visual_kv_median_mib=float(np.median([r['decoder_state']['visual_kv_bytes'] for r in rr])) / 2**20)
    result = dict(time=now(), results=stats, all_timing_tokens_equal=True, traces_sha_verified=True)
    save(BASE / 'qwen_memory/summary.json', result)
    return result


def audit():
    registration = verify()
    assert all(completed(stage) for stage in ['qwen_formal', 'qwen_timing', 'qwen_memory'])
    npu = load(BASE / 'full_training_audit.json')
    assert npu['all_passed'] and len(npu['runs']) == 4
    assert npu['source_sha256'] == sha(ROOT / 'scripts/tessfold_accel/training_audit_npu.py')
    reg = load(REGISTRY)
    assert len(reg['runs']) == 4 and reg['formal_used_for_selection'] is False
    assert {(r['kind'], r['ratio']) for r in reg['runs']} == set(MODELS)
    training = []
    for meta in reg['runs']:
        out = ROOT / meta['full_run']
        m, result, initial = load(out / 'manifest.json'), load(out / 'complete.json'), load(out / 'initial.summary.json')
        assert m['epochs'] == 4 and m['train_questions'] == 20000
        assert m['backbone_trainable'] == result['backbone_trainable'] == 0
        assert m['seed'] == 20261115 and m['clip'] == 5 and m['weight_decay'] == 0
        for path, digest in m['source_sha256'].items():
            assert sha(ROOT / path) == digest
        pilot = ROOT / m['gate_root'] / model_id(m['kind'], m['ratio']) / 'complete.json'
        assert sha(pilot) == m['gate_sha256'] and load(pilot)['pilot_gate_passed']
        rr = strict(out, 'training.jsonl')
        total = math.ceil(20000 / m['batch']) * 4
        assert [r['step'] for r in rr] == list(range(1, total + 1))
        warm = max(1, round(total * .1))
        for r in rr:
            step = r['step'] - 1
            factor = (step + 1) / warm if step < warm else .1 + .9 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm - 1))) / 2
            assert math.isfinite(r['kl']) and r['kl'] >= 0 and math.isfinite(r['gradient_norm'])
            assert abs(r['lr'] - m['lr'] * factor) < 1e-12
            assert abs(r['clip_multiplier'] - min(1., 5. / (r['gradient_norm'] + 1e-6))) < 1e-9
        candidates = [load(out / f'epoch{i}.summary.json') for i in range(1, 5)]
        eligible = [r for r in candidates if r['kl'] <= initial['kl'] * .95 and
                    (r['macro'] >= initial['macro'] + .25 or
                     (initial['macro'] >= initial['teacher']['macro'] - .25 and r['macro'] >= initial['teacher']['macro'] - .25))]
        best = max(eligible, key=lambda r: (r['macro'], -r['kl'])) if eligible else None
        assert best == result['best']
        assert meta['stage'] == (best['stage'] if best else 'initial')
        assert sha(ROOT / meta['checkpoint']) == meta['checkpoint_sha256']
        training.append(dict(model=meta['id'], steps=total, selected=meta['stage'], lr=m['lr'], batch=m['batch']))
    formal = strict(BASE / 'qwen_formal')
    qa = lines(ROOT / 'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl')
    ids = {r['id'] for r in qa}
    from scripts.tessfold_accel.qwen_control_stats import control_rows, paired_results, scores
    from vqa.benchmark_metrics import native_score
    controls, _ = control_rows()
    control_ids = {r['id'] for r in controls}
    expected = {(i, identity) for i in ids for identity in ['REF', *[r['id'] for r in reg['runs']]]}
    expected |= {(i, r['id'] + '_initial') for i in control_ids for r in reg['runs']}
    index = {(r['id'], r['model']): r for r in formal}
    assert len(formal) == len(index) == len(expected) == 127441
    assert set(index) == expected
    result = completed('qwen_formal')
    for r in formal:
        assert abs(r['score'] - native_score(r['prediction'], r)) < 1e-10
        assert not r['benchmark_used_for_selection']
        assert r['visual_tokens'] == (r['original_visual_tokens'] if r['model'] == 'REF' else max(1, round(r['original_visual_tokens'] * r['ratio'])))
    for identity in ['REF', *[r['id'] for r in reg['runs']]]:
        metric = scores([r for r in formal if r['model'] == identity])
        assert abs(metric['native_macro'] - result['native_metric_results'][identity]['macro']) < 1e-9
    assert paired_results(formal, reg['runs']) == result['initial_control_results']
    historical = [r for r in strict(PARENT / 'qwen_formal') if r['model'] == 'REF']
    assert len(historical) == 23565
    assert all(r['generated_ids'] == index[(r['id'], 'REF')]['generated_ids'] for r in historical), 'Teacher differs from previous budget experiment'
    checked = 0
    for stage in ['qwen_formal', 'qwen_timing', 'qwen_memory']:
        for path in (BASE / stage).glob('manifest.rank*.json'):
            stored = load(path)
            assert stored['run_registry_sha256'] == sha(REGISTRY)
            for source, digest in stored.get('source_sha256', stored.get('sources', {})).items():
                assert sha(ROOT / source) == digest
                checked += 1
    from scripts.tessfold_accel import qwen_timing_stats
    previous = qwen_timing_stats.BASE
    try:
        qwen_timing_stats.BASE = BASE
        timing = qwen_timing_stats.compute()
    finally:
        qwen_timing_stats.BASE = previous
    assert timing['formal_natural_token_matches'] == 1350
    memory_summary()
    audit_result = dict(time=now(), all_checks_passed=True, training=training,
                        formal_records=len(formal), timing_records=2700, memory_records=1350,
                        teacher_identical_to_previous_questions=23565, initial_control_questions=2404,
                        checked_source_references=checked, registration_sha256=sha(BASE / 'registration.json'),
                        registry_sha256=sha(REGISTRY), training_optimizer_audit_sha256=sha(BASE / 'full_training_audit.json'),
                        source_sha256=sha(__file__), all_base_parameters_frozen=True,
                        single_training_seed=True, no_formal_checkpoint_selection=True)
    save(BASE / 'final_audit.json', audit_result)
    save(BASE / 'all_stages_complete.json', dict(time=now(), final_audit_sha256=sha(BASE / 'final_audit.json'), all_complete=True))
    return audit_result


def write_report():
    reg = load(BASE / 'registration.json')
    if not reg:
        return
    state = load(BASE / 'controller_state.json')
    selected = {r['id']: r for r in load(REGISTRY).get('runs', [])}
    train = training_rows()
    save(BASE / 'training_diagnostics.json', dict(time=now(), runs=train))
    graphs(train)
    final = load(BASE / 'final_audit.json')
    doc = ['# 原生8B：10%与75%保留率完整训练和评测', '',
           f'更新时间（UTC）：{now()}。状态：**' + ('全部训练、正式评测及审计完成' if final.get('all_checks_passed') else '进行中，未完成项不填最终分数') + '**。', '',
           '本轮响应补齐10%/75%预算的请求，对标上一轮25%/50%协议。原模型是完整原生Qwen3-VL-8B-Instruct，使用其自己的视觉塔、merger、DeepStack及MRoPE；不是MonkeyOCRv2-Und换LLM。所有基模参数冻结，仅训练新增空间压缩器。', '',
           '## 入口及产物', '',
           '- 状态：`.venv/bin/python -m scripts.tessfold_budget1075.controller status`',
           '- 继续控制器：`.venv/bin/python -m scripts.tessfold_budget1075.controller resume`',
           '- 重建本报告：`.venv/bin/python -m scripts.tessfold_budget1075.controller report`',
           '- 最终审计：`.venv/bin/python -m scripts.tessfold_budget1075.controller audit`',
           '- [实验登记](../artifacts/tessfold_acceleration_20261115/budget1075/registration.json) · [控制器状态](../artifacts/tessfold_acceleration_20261115/budget1075/controller_state.json)',
           '- [上一轮最终结果](TESSFOLD_FINAL_RESULTS.md) · [原生8B结构](TESSFOLD_NATIVE8B_METHOD.md)', '',
           '## 统一协议与选择规则', '',
           '20,000训练QA/16,731图，4 epochs；固定56QA dev，正式23,565QA/6,494图，文档互斥。种子20261115；完整词表teacher greedy公共回答前缀KL，不加entropy、不以gold训练。P位于原生merger后，对最终4096维特征及3条DeepStack分支同步凸组合，保留代表位置原MRoPE。', '',
           'AdamW、weight_decay=0、clip=5、10%线性warmup后cosine到0.1LR。先用原固定512题训练子集跑3epoch先导；按预登记顺序先LR3e-4，再1e-4、1e-3，只在先导失败时尝试下一项。Column若三项均失败，再试与上一轮StableColumn50相同的固定detail划分变体（LR1e-3、batch16），显式标记结构差异。标准配置batch32。第一次通过先导门槛后，从相同种子的初始化重新训练20k×4epochs，不从先导权重接续。', '',
           '先导及full候选门槛沿用前轮：dev KL比initial至少下降5%，生成宏分至少提高0.25pp；若initial已接近teacher，则沿用teacher饱和例外。full在合格epoch中按dev宏分最高、再按KL最低选模型；没有合格epoch则正式评测initial，并保留完整训练负结果。正式测试不用于选模型、调学习率或追选epoch。', '',
           '固定512文档/2404QA初始P对照沿用上一轮登记。正式生成greedy、短答案、自然EOS、cap256；训练teacher前缀cap64（原缓存35/20000条截断的限制不变）。DeepForm/KLC用DUE corpus F1，其他任务用原生指标，八任务等权；另存共享fieldEM口径。', '',
           '## 实际先导与完整训练进度', '',
           '|模型|先导选择|LR / batch|实际full更新数|full完成|dev选中epoch|',
           '|---|---|---|---:|---|---|']
    for r in train:
        choice = state.get('pilot_choices', {}).get(r['id'], {})
        m = r['manifest']
        label = ('StableColumn' if choice.get('stable') else r['id']) if choice else '先导中'
        hyper = f'{m["lr"]:g} / {m["batch"]}' if m else '待确定'
        doc.append(f'|{r["id"]}|{label} {choice.get("name", "")}|{hyper}|{r["steps"]}|{bool(r["complete"])}|{selected.get(r["id"], {}).get("stage", "待选")} |')
    doc += ['', '### 所有先导尝试（保留失败结果）', '',
            '|模型|先导目录|通过门槛|initial KL / macro|best KL / macro|', '|---|---|---|---|---|']
    for directory in sorted(BASE.glob('pilot*')):
        for path in sorted(directory.glob('*_r*/complete.json')):
            rr = load(path)
            initial, best = rr['initial'], rr.get('best')
            fmt = lambda x: f'{x["kl"]:.4f} / {x["macro"]:.4f}' if x else '无合格epoch'
            doc.append(f'|{path.parent.name}|{directory.name}|{rr["pilot_gate_passed"]}|{fmt(initial)}|{fmt(best)}|')
    doc += ['', '## 每epoch loss、生成分数和梯度', '',
            '|模型|epoch|dev完成|train KL|dev KL|dev宏分|norm p50|norm p99|裁剪频率|',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in train:
        for e in r['epochs']:
            fmt = lambda v: '待完成' if v is None else f'{v:.4f}'
            doc.append(f'|{r["id"]}|{e["epoch"]}|{e["complete"]}|{e["train_kl"]:.4f}|{fmt(e["dev_kl"])}|{fmt(e["dev_macro"])}|{e["norm_p50"]:.3f}|{e["norm_p99"]:.3f}|{e["clip_fraction"]:.1%}|')
    if (BASE / 'figures/training_curves.pdf').exists():
        doc += ['', '[逐epoch训练/开发集曲线PDF](../artifacts/tessfold_acceleration_20261115/budget1075/figures/training_curves.pdf)',
                '', '![训练曲线](../artifacts/tessfold_acceleration_20261115/budget1075/figures/training_curves.png)']
    doc += ['', '## 正式全量任务分数：与25%/50%并列', '',
            '旧预算数据直接引用上一轮已审计的固定模型，新增预算独立训练；StableColumn50为固定detail划分，与标准Column不同。每行23,565QA。', '',
            '|模型|预算|DocVQA|InfoVQA|ChartQA|OCRBench|DeepForm F1|KLC F1|WTQ|DT-VQA|原生macro|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    previous = load(PARENT / 'qwen_formal/complete.json').get('native_metric_results', {})
    new_formal = completed('qwen_formal')
    current = new_formal.get('native_metric_results', {})
    task_order = ['DocVQA', 'InfoVQA', 'ChartQA', 'OCRBench', 'DeepForm', 'KLC', 'WTQ', 'DT-VQA']
    for identity, ratio in [('REF', 1.), *[(model_id(k, r), r) for r in [.1, .25, .5, .75] for k in ['Column', 'Tree']]]:
        scores = (current if ratio in [.1, .75] else previous).get(identity)
        if not scores:
            doc.append(f'|{identity}|{ratio:.0%}|' + '|'.join(['待完成'] * 9) + '|')
            continue
        stable_variant = identity == 'Column_r0.5' or bool(selected.get(identity, {}).get('allocation'))
        label = identity.replace('Column_', 'StableColumn_') if stable_variant else identity
        values = [scores['by_task'][t] for t in task_order] + [scores['macro']]
        doc.append(f'|{label}|{ratio:.0%}|' + '|'.join(f'{v:.4f}' for v in values) + '|')
    if new_formal:
        doc += ['', '[新增预算完整分数及初始对照JSON](../artifacts/tessfold_acceleration_20261115/budget1075/qwen_formal/complete.json)', '',
                '### 相同2404题：初始P与训练后P独立对照', '',
                '下表只对应固定子集，不是全量测试宏分；95%CI仅用于共享fieldEM宏分。', '',
                '|模型|初始原生macro|选中原生macro|改善pp|共享fieldEM改善95%CI|', '|---|---:|---:|---:|---|']
        for identity, v in new_formal['initial_control_results']['comparisons'].items():
            ci = v['shared_fieldEM_task_stratified_document_bootstrap_95_ci']
            doc.append(f'|{identity}|{v["initial"]["native_macro"]:.4f}|{v["selected"]["native_macro"]:.4f}|{v["native_macro_delta"]:+.4f}|[{ci[0]:.4f}, {ci[1]:.4f}]|')
    doc += ['', '## 完整部署速度与独立显存', '',
            '沿用90题、每模型每模式3次。natural EOS及fixed64分开；同题重复先平均、以时延总和比计算加速，页面bootstrap10000次。内存另做1ms采样，不把采样耗时当作速度。所有计时包含processor、完整视觉塔及merger；P不节省前端。', '']
    stats = load(BASE / 'qwen_timing/statistics.json')
    if stats:
        doc += ['|模型|模式|E2E加速|95%页面CI|TTFT加速|95%页面CI|', '|---|---|---:|---|---:|---|']
        for v in stats['comparisons']:
            if v['stratum'] != 'all':
                continue
            ci, ti = v['e2e_bootstrap_95_ci'], v['ttft_bootstrap_95_ci']
            doc.append(f'|{v["model"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
    else:
        doc.append('时延尚未完成最终审计。')
    mem = load(BASE / 'qwen_memory/summary.json')
    if mem:
        doc += ['', '|模型|全流程最大GiB|前端最大GiB|下游最大GiB|prefill采样均值GiB|视觉KV中位MiB|', '|---|---:|---:|---:|---:|---:|']
        for identity, v in mem['results'].items():
            doc.append(f'|{identity}|' + '|'.join(f'{v[k]:.4f}' for k in ['full_max_gib', 'front_max_gib', 'downstream_max_gib', 'prefill_mean_gib', 'visual_kv_median_mib']) + '|')
    doc += ['', '## 实际作业与完成标志', '', '|阶段|Job ID|状态|', '|---|---|---|']
    for label, job in state.get('jobs', {}).items():
        doc.append(f'|{label}|{job["job"]}|{job["status"]}|')
    doc += ['', f'最终审计通过：{bool(final.get("all_checks_passed"))}。', '',
            '原始逐题结果：`artifacts/tessfold_acceleration_20261115/budget1075/qwen_formal/rows.rank*.jsonl`；每条含gold、prediction、score、generated_ids、token数和checkpoint。', '',
            '本轮仍为每配置一个训练seed；固定dev仅56题。不同预算若采用StableColumn，必须区分结构差异；不能把预算变化全部解释为保留率的因果效应。未通过先导的候选不伪装成完成full；若没有任何候选通过，控制器会明确记录需处理的科学问题。已完成不等于接近teacher或实现预期加速。']
    if state.get('scientific_attention'):
        doc += ['', '需处理：`' + json.dumps(state['scientific_attention'], ensure_ascii=False) + '`']
    tmp = REPORT.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(doc) + '\n')
    tmp.replace(REPORT)


if __name__ == '__main__':
    write_report()
