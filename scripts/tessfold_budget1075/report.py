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
    doc = ['# Native 8B: Full Training and Evaluation at 10% and 75% Retention', '',
           f'Updated (UTC):{now()}. Status: **' + ('All training, formal evaluation, and audits completed' if final.get('all_checks_passed') else 'In progress; no final scores reported for incomplete stages') + '**. ', '',
           'This round adds 10%/75% budgets under the previous 25%/50% protocol. The backbone is complete native Qwen3-VL-8B-Instruct with its own visual tower, mergers, DeepStack, and MRoPE, not an LLM replacement in MonkeyOCRv2-Und. All backbone parameters are frozen; only the new spatial compressor is trained.', '',
           '## Entry Points and Outputs', '',
           '- Status: `.venv/bin/python -m scripts.tessfold_budget1075.controller status`',
           '- Resume controller: `.venv/bin/python -m scripts.tessfold_budget1075.controller resume`',
           '- Rebuild this report: `.venv/bin/python -m scripts.tessfold_budget1075.controller report`',
           '- Final audit: `.venv/bin/python -m scripts.tessfold_budget1075.controller audit`',
           '- [Experiment registration](../artifacts/tessfold_acceleration_20261115/budget1075/registration.json) · [Controller status](../artifacts/tessfold_acceleration_20261115/budget1075/controller_state.json)',
           '- [Previous-round final results](TESSFOLD_FINAL_RESULTS.md) · [Native 8B architecture](TESSFOLD_NATIVE8B_METHOD.md)', '',
           '## Common Protocol and Selection Rules', '',
           '20,000 training QA / 16,731 images, 4 epochs; 56 fixed development QA; 23,565 formal QA / 6,494 images, with document-disjoint splits. Seed 20261115. Full-vocabulary KL under a shared teacher-greedy response prefix; no entropy term or gold training. P is after native mergers and applies the same convex aggregation to final 4,096-dimensional features and all 3 DeepStack branches, retaining original representative MRoPE positions.', '',
           'AdamW, weight_decay=0, clip=5; 10% linear warmup followed by cosine decay to 0.1 LR. Start with a 3-epoch pilot on the original fixed 512-question training subset. Try LR 3e-4, then 1e-4, then 1e-3 in preregistered order, advancing only after failure. If all three fail for Column, test the fixed-detail StableColumn variant used previously at 50% (LR 1e-3, batch 16), explicitly marking the structural change. Standard batch size is 32. After the first passing pilot, restart 20k × 4-epoch training from the same seeded initialization rather than its pilot weights.', '',
           'Pilot/full eligibility follows the previous round: development KL must fall at least 5% from initial, and generation macro must rise at least 0.25 pp, retaining the teacher-saturation exception when initial is already close to teacher. Full selection maximizes development macro, then minimizes KL among eligible epochs. With no eligible epoch, formal evaluation uses initial and retains the full-training negative result. Tests select neither models, learning rates, nor epochs.', '',
           'Initial-P controls retain the earlier 512-document/2,404-QA registration. Formal generation is greedy, short-answer, natural-EOS, cap=256. Training teacher prefixes use cap=64; the existing 35/20,000 truncated-cache limitation remains. DeepForm/KLC use DUE corpus F1, others native metrics, with eight tasks equally weighted. Shared field-EM statistics are stored separately.', '',
           '## Pilot and Full-Training Progress', '',
           '|Model|Pilot selection|LR / batch|Actual full-training updates|Full training completed|Development-selected epoch|',
           '|---|---|---|---:|---|---|']
    for r in train:
        choice = state.get('pilot_choices', {}).get(r['id'], {})
        m = r['manifest']
        label = ('StableColumn' if choice.get('stable') else r['id']) if choice else 'Pilot running'
        hyper = f'{m["lr"]:g} / {m["batch"]}' if m else 'Undetermined'
        doc.append(f'|{r["id"]}|{label} {choice.get("name", "")}|{hyper}|{r["steps"]}|{bool(r["complete"])}|{selected.get(r["id"], {}).get("stage", "Pending selection")} |')
    doc += ['', '### All Pilot Attempts, Including Failures', '',
            '|Model|Pilot directory|Passed eligibility criterion|initial KL / macro|best KL / macro|', '|---|---|---|---|---|']
    for directory in sorted(BASE.glob('pilot*')):
        for path in sorted(directory.glob('*_r*/complete.json')):
            rr = load(path)
            initial, best = rr['initial'], rr.get('best')
            fmt = lambda x: f'{x["kl"]:.4f} / {x["macro"]:.4f}' if x else 'No eligible epoch'
            doc.append(f'|{path.parent.name}|{directory.name}|{rr["pilot_gate_passed"]}|{fmt(initial)}|{fmt(best)}|')
    doc += ['', '## Per-Epoch Loss, Generation Scores, and Gradients', '',
            '|Model|epoch|Development evaluation complete|train KL|dev KL|Development macro|norm p50|norm p99|Clipping frequency|',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in train:
        for e in r['epochs']:
            fmt = lambda v: 'Pending' if v is None else f'{v:.4f}'
            doc.append(f'|{r["id"]}|{e["epoch"]}|{e["complete"]}|{e["train_kl"]:.4f}|{fmt(e["dev_kl"])}|{fmt(e["dev_macro"])}|{e["norm_p50"]:.3f}|{e["norm_p99"]:.3f}|{e["clip_fraction"]:.1%}|')
    if (BASE / 'figures/training_curves.pdf').exists():
        doc += ['', '[Per-epoch training/development curves PDF](../artifacts/tessfold_acceleration_20261115/budget1075/figures/training_curves.pdf)',
                '', '![Training curves](../artifacts/tessfold_acceleration_20261115/budget1075/figures/training_curves.png)']
    doc += ['', '## Full Formal Task Scores Alongside the 25%/50% Budgets', '',
            'Old budgets quote audited fixed models; new budgets train independently. StableColumn50 uses fixed-detail partitioning unlike standard Column. Each row has 23,565 QA.', '',
            '|Model|Budget|DocVQA|InfoVQA|ChartQA|OCRBench|DeepForm F1|KLC F1|WTQ|DT-VQA|Native macro|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    previous = load(PARENT / 'qwen_formal/complete.json').get('native_metric_results', {})
    new_formal = completed('qwen_formal')
    current = new_formal.get('native_metric_results', {})
    task_order = ['DocVQA', 'InfoVQA', 'ChartQA', 'OCRBench', 'DeepForm', 'KLC', 'WTQ', 'DT-VQA']
    for identity, ratio in [('REF', 1.), *[(model_id(k, r), r) for r in [.1, .25, .5, .75] for k in ['Column', 'Tree']]]:
        scores = (current if ratio in [.1, .75] else previous).get(identity)
        if not scores:
            doc.append(f'|{identity}|{ratio:.0%}|' + '|'.join(['Pending'] * 9) + '|')
            continue
        stable_variant = identity == 'Column_r0.5' or bool(selected.get(identity, {}).get('allocation'))
        label = identity.replace('Column_', 'StableColumn_') if stable_variant else identity
        values = [scores['by_task'][t] for t in task_order] + [scores['macro']]
        doc.append(f'|{label}|{ratio:.0%}|' + '|'.join(f'{v:.4f}' for v in values) + '|')
    if new_formal:
        doc += ['', '[New-budget full scores and initial-control JSON](../artifacts/tessfold_acceleration_20261115/budget1075/qwen_formal/complete.json)', '',
                '### Same 2,404 Questions: Independent Initial-P Versus Trained-P Comparison', '',
                'This table describes a fixed subset, not the full-test macro; 95% CIs apply only to the shared field-EM macro.', '',
                '|Model|Initial native macro|Selected native macro|Improvement (pp)|Shared field-EM improvement: 95% CI|', '|---|---:|---:|---:|---|']
        for identity, v in new_formal['initial_control_results']['comparisons'].items():
            ci = v['shared_fieldEM_task_stratified_document_bootstrap_95_ci']
            doc.append(f'|{identity}|{v["initial"]["native_macro"]:.4f}|{v["selected"]["native_macro"]:.4f}|{v["native_macro_delta"]:+.4f}|[{ci[0]:.4f}, {ci[1]:.4f}]|')
    doc += ['', '## Full-Deployment Speed and Independent Memory Measurements', '',
            'Use the same 90 questions, 3 repeats per model/mode. Natural EOS and fixed64 are separate. Average repeats within each question, then use the ratio of latency sums; bootstrap pages 10,000 times. Memory uses a separate 1 ms sampling pass whose overhead is not included in speed estimates. All timings include processor, full visual tower, and mergers; P does not save frontend computation.', '']
    stats = load(BASE / 'qwen_timing/statistics.json')
    if stats:
        doc += ['|Model|Mode|E2E speedup|95% page-level CI|TTFT speedup|95% page-level CI|', '|---|---|---:|---|---:|---|']
        for v in stats['comparisons']:
            if v['stratum'] != 'all':
                continue
            ci, ti = v['e2e_bootstrap_95_ci'], v['ttft_bootstrap_95_ci']
            doc.append(f'|{v["model"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
    else:
        doc.append('Final timing audit incomplete.')
    mem = load(BASE / 'qwen_memory/summary.json')
    if mem:
        doc += ['', '|Model|Full-pipeline maximum (GiB)|Frontend maximum (GiB)|Downstream maximum (GiB)|Sampled prefill mean (GiB)|Median visual KV (MiB)|', '|---|---:|---:|---:|---:|---:|']
        for identity, v in mem['results'].items():
            doc.append(f'|{identity}|' + '|'.join(f'{v[k]:.4f}' for k in ['full_max_gib', 'front_max_gib', 'downstream_max_gib', 'prefill_mean_gib', 'visual_kv_median_mib']) + '|')
    doc += ['', '## Actual Jobs and Completion Markers', '', '|Stage|Job ID|Status|', '|---|---|---|']
    for label, job in state.get('jobs', {}).items():
        doc.append(f'|{label}|{job["job"]}|{job["status"]}|')
    doc += ['', f'Final audit passed: {bool(final.get("all_checks_passed"))}. ', '',
            'Raw per-question results: `artifacts/tessfold_acceleration_20261115/budget1075/qwen_formal/rows.rank*.jsonl`; each record includes gold, prediction, score, generated_ids, token counts, and checkpoint.', '',
            'There is still one training seed per configuration and only 56 development questions. StableColumn at some budgets is a structural change; differences cannot be attributed entirely to retention. Candidates failing the pilot are not presented as completed full runs; if none passes, the controller records the unresolved scientific issue. Completion does not establish closeness to teacher or achievement of target speedups.']
    if state.get('scientific_attention'):
        doc += ['', 'Requires attention: `' + json.dumps(state['scientific_attention'], ensure_ascii=False) + '`']
    tmp = REPORT.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(doc) + '\n')
    tmp.replace(REPORT)


if __name__ == '__main__':
    write_report()
