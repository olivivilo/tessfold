"""Audit and merge all eight fixed column/tree experiments, including live progress."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path

from scripts.column_tree_post_full_suite import ROOT, BASE, SUITE, REGISTRATION, STATE, load, save, sha, rows, now, verify, benchmark_rows
from vqa.benchmark_metrics import native_score

VARIANTS = ['initial', 'trained', 'mean', 'selection', 'cell_mean', 'cell_selection']
LABEL = {'col_adaptive': 'Adaptive column partitioning', 'quadtree_question': 'Question-conditioned spatial tree'}
REPORT = ROOT / 'docs/COLUMN_TREE_POST_MERGER_FULL_COMPARISON.md'


def jsonlines(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    result = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return result


def training_info(record):
    out = ROOT / record['output']
    rr = jsonlines(out / 'training.jsonl')
    steps = [r['step'] for r in rr]
    assert steps == list(range(1, len(steps) + 1)), f'Duplicate or skipped training updates: {record["name"]}'
    complete = (out / 'complete.json').exists()
    by_epoch = collections.defaultdict(list)
    for r in rr:
        assert all(math.isfinite(r[k]) for k in ['kl', 'total', 'grad_norm', 'update_l2'])
        assert r['update_l2'] > 0 and r['epoch'] == (r['step'] - 1) // 625 + 1
        by_epoch[r['epoch']].append(r['kl'])
    epochs = []
    for e in range(5):
        stage = 'initial' if e == 0 else f'epoch{e}'
        val = load(out / f'validation.{stage}.summary.json')
        audit = load(out / f'train_audit.{stage}.summary.json')
        if val:
            assert val['rows'] == 56
        if audit:
            assert audit['rows'] == 14
        values = by_epoch[e]
        epochs.append(dict(epoch=e, updates=len(values), train_batch_kl=sum(values) / len(values) if values else None,
                           validation_kl=val.get('kl'), audit_kl=audit.get('kl'), validation_score=val.get('student_macro'),
                           teacher_validation_score=val.get('teacher_macro')))
    if complete:
        done = load(out / 'complete.json')
        assert len(steps) == 2500 and done['steps'] == 2500 and done['epochs'] == 4
        assert done['trainable_backbone'] == 0 and done['padded_samples_per_epoch'] == 0
        assert all(len(by_epoch[e]) == 625 for e in range(1, 5))
        assert all(x['validation_kl'] is not None and x['audit_kl'] is not None for x in epochs)
        assert (out / 'epoch4.pt').exists()
    # Count only committed newline records, not an unfinished final JSON line.
    parts = sorted((out / 'benchmark').glob('predictions.rank*.jsonl'))
    count = 0
    for p in parts:
        with p.open('rb') as f:
            count += sum(line.endswith(b'\n') for line in f)
    return dict(logged_updates=len(steps), complete=complete, epochs=epochs, benchmark_records=count)


def audit_benchmark(record, expected, teacher_reference):
    out = ROOT / record['output']
    bench = out / 'benchmark'
    if not (bench / 'complete.json').exists():
        return None
    parts = sorted(bench.glob('predictions.rank*.jsonl'))
    manifest_paths = sorted(bench.glob('manifest.rank*.json'))
    sig = dict(checkpoint_sha256=sha(out / 'epoch4.pt'), initial_sha256=sha(out / 'initial.pt'),
               summary_sha256=sha(bench / 'summary.json'),
               manifest_sha256={p.name: sha(p) for p in manifest_paths},
               prediction_stats={p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in parts})
    cached_path = SUITE / 'audits' / f'{record["name"]}.json'
    cached = load(cached_path)
    if cached.get('signature') == sig and cached.get('passed'):
        return load(bench / 'summary.json')
    cfg = load(ROOT / record['config'])
    summary = load(bench / 'summary.json')
    assert summary['config'] == cfg and summary['questions'] == 23565 and summary['cap'] == 256
    assert summary['benchmark_used_for_selection'] is False
    manifests = [load(p) for p in manifest_paths]
    assert manifests and all(m == manifests[0] for m in manifests)
    assert len(manifests) == manifests[0]['world'] == len(parts)
    assert manifests[0]['checkpoint_sha256'] == sig['checkpoint_sha256']
    assert manifests[0]['initial_sha256'] == sig['initial_sha256']
    for source, digest in manifests[0]['source_sha256'].items():
        assert sha(ROOT / source) == digest
    train_manifest = load(out / 'manifest.json')
    assert train_manifest['config'] == cfg and train_manifest['trainable_backbone'] == 0
    reg = load(REGISTRATION)
    for k, digest in train_manifest['data_sha256'].items():
        assert digest == reg['dataset'][k]['sha256']
    seen = set()
    scores = collections.defaultdict(list)
    teachers = {}
    limit = collections.Counter()
    max_error = 0.
    for p in parts:
        for r in jsonlines(p):
            key = (r['id'], r['variant'])
            assert key not in seen and r['variant'] in VARIANTS
            seen.add(key)
            source = expected[r['id']]
            for field in ['document_id', 'task', 'question', 'answers']:
                assert r[field] == source[field]
            assert r['ratio'] == record['ratio']
            assert r['compressed_tokens'] == max(1, round(record['ratio'] * r['tokens']))
            assert len(r['student_ids']) <= 256 and 0 <= r['row_sum_error'] < 1e-4
            max_error = max(max_error, r['row_sum_error'])
            assert abs(native_score(r['student_prediction'], source) - r['student_score']) < 1e-12
            teacher = (r['teacher_prediction'], r['teacher_score'])
            assert teacher == teacher_reference[r['id']]
            assert abs(native_score(r['teacher_prediction'], source) - r['teacher_score']) < 1e-12
            teachers[r['id']] = teacher[1]
            scores[(r['task'], r['variant'])].append(r['student_score'])
            limit[r['variant']] += bool(r['hit_token_limit'])
    assert seen == {(key, v) for key in expected for v in VARIANTS}
    for task, metrics in summary['tasks'].items():
        for variant in VARIANTS:
            values = scores[(task, variant)]
            assert abs(100 * sum(values) / len(values) - metrics[variant]) < 1e-8
        tt = [teachers[k] for k, r in expected.items() if r['task'] == task]
        assert abs(100 * sum(tt) / len(tt) - metrics['teacher']) < 1e-8
    assert set(summary['tasks']) == {r['task'] for r in expected.values()}
    for variant in ['teacher', *VARIANTS]:
        assert abs(sum(t[variant] for t in summary['tasks'].values()) / 8 - summary['macro'][variant]) < 1e-8
    assert abs(summary['macro']['trained'] - summary['macro']['initial'] - summary['paired_trained_vs_initial']['gain_points']) < 1e-8
    save(cached_path, dict(passed=True, audited_utc=now(), signature=sig, questions=23565, records=len(seen),
                          teacher_matches_legacy10=True, scores_recomputed=True, unique_coverage=True,
                          question_gold_identity=True, budget_and_generation_cap=True, max_row_sum_error=max_error,
                          source_and_checkpoint_hashes=True, token_limit_counts=dict(limit)))
    return summary


def browser(record):
    out = ROOT / record['output'] / 'benchmark'
    if (out / 'review.html').exists():
        return
    from urllib.parse import quote
    from scripts.review_compression_lab_v2_benchmark import TEMPLATE
    source = {r['id']: r for r in benchmark_rows()}
    initial, trained = {}, {}
    for p in sorted(out.glob('predictions.rank*.jsonl')):
        for r in jsonlines(p):
            if r['variant'] == 'initial':
                initial[r['id']] = r
            elif r['variant'] == 'trained':
                trained[r['id']] = r
    assert len(initial) == len(trained) == 23565 and initial.keys() == trained.keys()
    payload = []
    for key, r in trained.items():
        b = initial[key]
        payload.append(dict(id=key, task=r['task'], question=r['question'], answers=r['answers'],
                            teacher=r['teacher_prediction'], initial=b['student_prediction'], trained=r['student_prediction'],
                            teacher_score=r['teacher_score'], initial_score=b['student_score'], trained_score=r['student_score'],
                            gain=r['student_score'] - b['student_score'], tokens=r['tokens'], compressed_tokens=r['compressed_tokens'],
                            hit_token_limit=r['hit_token_limit'], image=quote(os.path.relpath(ROOT / source[key]['image'], out))))
    html = TEMPLATE.replace('BUDGET', f'{record["ratio"]:.0%} ({record["kind"]})').replace('PAYLOAD', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    tmp = out / f'review.{os.getpid()}.tmp'
    tmp.write_text(html)
    tmp.replace(out / 'review.html')


def plot(record, info):
    signature = [{k: v for k, v in e.items() if k != 'train_batch_kl'} | {'complete_train_kl': e['train_batch_kl'] if e['updates'] == 625 else None} for e in info['epochs']]
    out = SUITE / 'plots'
    out.mkdir(exist_ok=True)
    stamp = out / f'{record["name"]}.json'
    if load(stamp, []) == signature:
        return
    if not any(e['validation_kl'] is not None for e in info['epochs']):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    e = info['epochs']
    for key, label in [('audit_kl', 'historical audit KL'), ('validation_kl', 'validation KL')]:
        vv = [x for x in e if x[key] is not None]
        axes[0].plot([x['epoch'] for x in vv], [x[key] for x in vv], 'o-', label=label)
    vv = [x for x in e if x['updates'] == 625]
    axes[0].plot([x['epoch'] for x in vv], [x['train_batch_kl'] for x in vv], 's-', label='full train epoch mean KL')
    vv = [x for x in e if x['validation_score'] is not None]
    axes[1].plot([x['epoch'] for x in vv], [x['validation_score'] for x in vv], 'o-', label='student validation macro')
    if vv:
        axes[1].axhline(vv[0]['teacher_validation_score'], ls=':', label='uncompressed teacher')
    for ax in axes:
        ax.set_xlabel('epoch'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    axes[0].set_ylabel('teacher-prefix KL'); axes[1].set_ylabel('7-task validation macro (%)')
    fig.suptitle(f'{record["kind"]}, post-merger, keep {record["ratio"]:.0%}, seed20261105')
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'{record["name"]}.{ext}', dpi=150)
    plt.close(fig)
    save(stamp, signature)


def smoke_report(state):
    done = load(SUITE / 'smoke/complete.json')
    doc = ['## Startup and Technical Checks', '', f"Current controller stage: `{state.get('suite_stage', 'registered')}`. ", '',
           'Submit this round’s NPU jobs only after all eight pre-merger formal runs/audits complete. In a separate process per configuration, check nonnegative unit-sum P, budget length, exact PX decoder inputs, nonzero finite gradients, frozen backbone, and actual parameter updates. Check-run weights never initialize full runs.', '',
           'Loss/generation checks on two training and two validation questions are technical records, not multi-image generalization, and select no hyperparameters/configurations. Earlier 144-run screening already includes post-merger 10%/25%/50% variants of these architectures.', '']
    if done:
        doc += [f"Eight-configuration technical checks passed; job  {done['job']}. ", '', '|Generator|Retention fraction|Training KL: before → after|Two-question validation KL: before → after|Two-question validation score: before → after|Gradient norm|', '|---|---:|---:|---:|---:|---:|']
        for r in done['results']:
            doc.append(f"|{LABEL[r['kind']]}|{r['ratio']:.0%}|{r['train_before']['kl']:.5f}→{r['train_after']['kl']:.5f}|{r['validation_before']['kl']:.5f}→{r['validation_after']['kl']:.5f}|{r['validation_before']['student_macro']:.2f}→{r['validation_after']['student_macro']:.2f}|{r['gradient_norm']:.5f}|")
        doc.append('')
    else:
        doc += ['Technical checks incomplete; no pass claimed.', '']
    return doc


def position_comparison(reg, results):
    pre = load(BASE / 'column_tree_full_20261108/merged_results.json').get('results', {})
    doc = ['## Paired Comparison Against Pre-Merger Insertion', '',
           'Both insertion points use identical data, seed, training steps, and fixed epoch 4. Before-merger X inputs are zero; after-merger generators may read raw H and projected X. Differences therefore include available input information as well as insertion point, not merely operator order. The official vision constructor passes spatial_merge_size=1 to PatchMerger (the config.json value 2 governs spatial ordering), applying a tokenwise 768→2048 mapping. Token counts match across positions; budget=max(1,round(ratio×N)).', '',
           '|Retention fraction|Generator|Before: eight-task mean|After: eight-task mean|After − before|Before: DocVQA|After: DocVQA|', '|---|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: '—' if x is None else f'{x:.2f}'
    for record in sorted(reg['runs'], key=lambda r: (r['ratio'], r['kind'])):
        a = pre.get(record['name'].replace('_post-merger_', '_pre-merger_'))
        b = results.get(record['name'])
        av = a['macro']['trained'] if a else None
        bv = b['macro']['trained'] if b else None
        doc.append(f"|{record['ratio']:.0%}|{LABEL[record['kind']]}|{fmt(av)}|{fmt(bv)}|{fmt(bv-av if a and b else None)}|{fmt(a['tasks']['DocVQA']['trained'] if a else None)}|{fmt(b['tasks']['DocVQA']['trained'] if b else None)}|")
    return doc + ['']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    reg = verify()
    state = load(STATE)
    expected = {r['id']: r for r in benchmark_rows()}
    assert len(expected) == 23565
    # Historical teacher is the fixed reference, not a newly trained model.
    teacher_reference = {}
    reference = BASE / 'full/full_quadtree_question_pre-merger_r0.1/benchmark'
    for p in reference.glob('predictions.rank*.jsonl'):
        for r in jsonlines(p):
            if r['variant'] == 'trained':
                teacher_reference[r['id']] = (r['teacher_prediction'], r['teacher_score'])
    assert teacher_reference.keys() == expected.keys()
    results, progress = {}, {}
    for record in reg['runs']:
        info = training_info(record)
        progress[record['name']] = info
        summary = audit_benchmark(record, expected, teacher_reference)
        if summary:
            assert info['complete']
            results[record['name']] = summary
            browser(record)
        plot(record, info)
    fmt = lambda x, n=2: '—' if x is None else f'{x:.{n}f}'
    doc = ['# Post-Merger Compression: Full Comparison of Adaptive Column Partitioning and Question-Conditioned Spatial Trees', '',
           f'Updated (UTC): {now()}. Formal results are complete and audited:  {len(results)}/8 runs; all 8 trained afresh this round.', '',
           '**Entry point**: status/report refresh  `.venv/bin/python -m scripts.column_tree_post_full_suite status`; resume after interruption  `.venv/bin/python -m scripts.column_tree_post_full_suite resume`; final verification  `.venv/bin/python -m scripts.report_column_tree_post_full --require-complete`. ', '',
           '## Experimental Design and Corrections', '',
           'Compare col_adaptive and quadtree_question at 10%/25%/50%/75% retention, all post-merger: encoder H → original merger X → PX → Qwen. Generator inputs are normalized H (768 dimensions) and X (2,048), not merger(PH). Column does not use the question to generate P. Tree uses frozen Qwen input embeddings for the current question\'s first 64 tokens. Despite the quadtree identifier, it recursively bisects rectangles rather than forming a standard four-child quadtree.', '',
           'Start only after eight pre-merger evaluations/audits complete. Both architectures/four budgets initialize afresh from the same seed without pre-merger generator weights; earlier pre-merger results remain separate.', '',
           'This requested fixed-position comparison does not select architecture/seed/checkpoint by formal score. Eight real-UND forward/backward checks precede full training, without screening winners or reusing check weights. Formal data were inspected earlier, so this is not a new blind test.', '',
           '## Implemented Generator and Compression Architecture', '',
           'Shared architecture: affine-free LayerNorm on H and X separately, concatenation, a 2816→96 linear layer + GELU, then a 96-channel 3×3 depthwise/pointwise residual convolution. The importance MLP is 96→96→1, with final-weight std=.01. Within-region masked softmax uses τ=.25+softplus(t), initially about .943. P is nonnegative with unit row sums; output values are PX only, not arbitrary learned latent values.', '',
           'Adaptive Column: multiply local high-frequency encoder detail by learned importance to allocate exactly M nonempty disjoint rectangles, preferring vertical bisection and switching at exhausted dimensions. Learn within-region convex weights; P is question-independent.', '',
           'Question-conditioned spatial tree: add a 2048→96 question-word projection and visual word_key to the shared architecture. Maximum normalized visual/word similarity augments importance. Recursively bisect the longer region axis into M rectangles and learn within-region weights. Despite the quadtree identifier, this is a binary rectangular spatial tree. P receives the current question only, not teacher answers, gold, or other QA context.', '',
           'Both methods detach discrete partition decisions; the whole tree is not differentiable. Gradients pass through within-region softmax and importance networks. Respect official block order and sort output rows by their maximum-weight original-token index. See the [second-round architecture document](COMPRESSION_GENERATOR_V2_DESIGNS.md); no new architectures in this round.', '',
           '## Data, Training, and Evaluation Configuration', '',
           '- Original 20,000 QA / 16,731 images; retain files, sample order, and epoch-shuffle rules without resampling or task balancing.',
           '- Fixed validation: 56 QA / 56 images; historical audit: 14 QA / 7 images. Only 2 audit images belong to 20k training; preserve that audit set and separately summarize full-training KL over every batch/epoch.',
           '- Formal evaluation reuses exactly 23,565 QA: DocVQA/InfoVQA from their original validation partitions plus all original-test tasks, excluding unused validation tasks. Training/development/formal document IDs are disjoint.',
           '- Official MonkeyOCRv2-B-Und encoder, original merger, and Qwen3-1.7B are frozen; train only P. Every run initializes fresh with seed 20261105, without a small-run warm start.',
           '- Four epochs/run, 625 updates/epoch, 2,500 total; AdamW LR=3e-4, wd=0, batch=32, clip=1; 250-step warmup, cosine to 10% peak LR; 8 NPUs, 4-question accumulation/rank, no padding.',
           '- KL-only, T=1, teacher||student; teacher-greedy answers are shared prefixes. Training-teacher/small-validation cap=64, formal greedy cap=256. No row-entropy/column-mass regularization; not a decoder-free loss.',
           '- Commit latest and optimizer state every 25 updates; save and evaluate each epoch, with epoch 4 fixed as the final model. Formal per-question outputs retain initial/trained/mean/selection/cell_mean/cell_selection: 141,390 records per run, without averaging over training seeds or compression samples.',
           '- CPU uses .venv and NPU .venv-und. Retain the verified immutable-cache I/O repair without changing generator/loss/inference/scorer computations. Formal evaluation uses 8 independent workers rather than the earlier 14, with identical questions and generation settings.', '',
           'Formal scores are equally weighted eight-task native macros (%). DeepForm/KLC field exact and retained DT-VQA scoring are not official document-F1/unified rankings. One full-training seed; image bootstrap does not measure seed variance.', '',
           '[Immutable experiment registration, configuration hashes, and data audit](../artifacts/compression_lab_v2/column_tree_post_full_20261108/registration.json) · [Live machine-readable status](../artifacts/compression_lab_v2/column_tree_post_full_20261108/state.json) · [Per-run result audits](../artifacts/compression_lab_v2/column_tree_post_full_20261108/audits/). ', '',
           '## Summary of Eight Training Runs and Formal Evaluations', '',
           '|Retention fraction|Generator|Source / stage|Training updates|Formal records|teacher|initial|trained|Teacher-score retention|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for record in sorted(reg['runs'], key=lambda x: (x['ratio'], x['kind'])):
        name = record['name']; info = progress[name]; result = results.get(name); mm = result['macro'] if result else {}
        stage = 'Historical run reused and audited' if record['reuse'] else state.get('runs', {}).get(name, {}).get('stage', 'registered')
        if result and not record['reuse']:
            stage = 'Formal evaluation completed and audited'
        retained = 100 * mm['trained'] / mm['teacher'] if mm else None
        doc.append(f'|{record["ratio"]:.0%}|{LABEL[record["kind"]]}|{stage}|{info["logged_updates"]}/2500|{info["benchmark_records"]}/141390|{fmt(mm.get("teacher"))}|{fmt(mm.get("initial"))}|{fmt(mm.get("trained"))}|{fmt(retained)}|')
    legacy = load(BASE / 'full/full_global_topk_pre-merger_r0.25/benchmark/summary.json')
    doc += ['', f'Historical 25% global_topk reference only: teacher  {legacy["macro"]["teacher"]:.2f}, initial {legacy["macro"]["initial"]:.2f}, trained {legacy["macro"]["trained"]:.2f}. It is excluded from the question-conditioned-tree budget curve.', '',
            '“—” means no completed/audited formal result; do not substitute 56-QA validation or screening scores.', '', '## Formal Per-Task Comparison at Each Budget', '']
    tasks = sorted({r['task'] for r in expected.values()})
    for ratio in [.1, .25, .5, .75]:
        cc = results.get(f'full_col_adaptive_post-merger_r{ratio:g}')
        tt = results.get(f'full_quadtree_question_post-merger_r{ratio:g}')
        doc += [f'### Retained Components{ratio:.0%}', '', '|Task|teacher|Column initial|Column trained|Tree initial|Tree trained|Column trained − Tree trained|', '|---|---:|---:|---:|---:|---:|---:|']
        for task in ['Macro average', *tasks]:
            c = (cc['macro'] if task == 'Macro average' else cc['tasks'][task]) if cc else {}
            t = (tt['macro'] if task == 'Macro average' else tt['tasks'][task]) if tt else {}
            teacher = (c or t).get('teacher')
            delta = c['trained'] - t['trained'] if c and t else None
            doc.append(f'|{task}|{fmt(teacher)}|{fmt(c.get("initial"))}|{fmt(c.get("trained"))}|{fmt(t.get("initial"))}|{fmt(t.get("trained"))}|{fmt(delta)}|')
        doc.append('')
    doc += position_comparison(reg, results)
    doc += smoke_report(state)
    doc += ['## Per-Run Configurations and Per-Epoch Loss/Generation Scores', '']
    for record in reg['runs']:
        name = record['name']; info = progress[name]; cfg = load(ROOT / record['config'])
        doc += [f'### {LABEL[record["kind"]]} / retain {record["ratio"]:.0%}', '',
                f'[Full configuration](../{record["config"]}); seed {cfg["seed"]}, post-merger, fixed epoch 4.', '',
                '|epoch|Recorded updates|Full-training batch-mean KL|Historical audit KL|Development KL|Small-development-set generation macro|', '|---|---:|---:|---:|---:|---:|']
        for e in info['epochs']:
            doc.append(f'|{e["epoch"]}|{e["updates"]}/625|{fmt(e["train_batch_kl"],5)}|{fmt(e["audit_kl"],5)}|{fmt(e["validation_kl"],5)}|{fmt(e["validation_score"])}|')
        doc += ['', 'Epoch 0 is initialization, without updates; means before 625 steps describe partial epochs. Absolute KL differs across sets; cap=64 small validation is not cap=256 formal evaluation.', '']
        curve = SUITE / 'plots' / f'{name}.pdf'
        if curve.exists():
            doc += [f'[Training/audit/validation KL and validation generation curves](../{curve.relative_to(ROOT)}). ', '']
        out = ROOT / record['output']
        if name in results:
            mm = results[name]['macro']; pair = results[name]['paired_trained_vs_initial']; ci = pair['cluster_bootstrap_95ci_points']
            doc += [f'Formal training gain {pair["gain_points"]:+.2f}  points; image-cluster 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]; relative to teacher  {mm["trained"]-mm["teacher"]:+.2f}  points.', '',
                    f'[Per-question CSV](../{record["output"]}/benchmark/answers.csv) · [Image/gold/teacher/initial/trained answers](../{record["output"]}/benchmark/review.html) · [Machine-readable results](../{record["output"]}/benchmark/summary.json) · [Epoch-4 weights](../{record["output"]}/epoch4.pt). ', '']
    doc += ['## Recovery and Completion Criteria', '',
            'The CPU controller runs independently of the interactive session, with at most 2 NPU stage jobs at once. Fixed-order training proceeds to full evaluation, followed by coverage/gold/teacher/score/config/code/weight audits, CSV/HTML/curve generation, and this report.', '',
            'At most 3 infrastructure recoveries/stage retain seed, 8-NPU sharding, and checkpoint without relaxing science. Beyond that retain failures/logs. Formal scores require completion/coverage/audit. All pre-merger state, weights, and answers remain unchanged.', '',
            'Use resume to restart an incomplete controller while retaining existing jobs. Controller/stage logs: artifacts/compression_lab_v2/column_tree_post_full_20261108/.']
    tmp = REPORT.with_name(REPORT.name + f'.{os.getpid()}.tmp')
    tmp.write_text('\n'.join(doc) + '\n')
    tmp.replace(REPORT)
    save(SUITE / 'merged_results.json', dict(updated_utc=now(), completed=len(results), planned=8, results=results, progress=progress))
    if args.require_complete:
        assert len(results) == 8 and all(p['complete'] for p in progress.values()), 'Some registered runs are not yet fully trained/evaluated/audited'
    print(json.dumps(dict(report=str(REPORT), formal_completed=len(results), planned=8,
                          logged_updates={k: v['logged_updates'] for k, v in progress.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
