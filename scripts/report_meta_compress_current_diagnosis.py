"""Synthesize existing multi-image results; no inference, training or selection."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/meta_compress_search'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    state = json.loads((BASE / 'controller.json').read_text())
    search = {r['name']: r for r in json.loads((BASE / 'summary.search.json').read_text())}
    confirm = {r['name']: r for r in json.loads((BASE / 'summary.confirm.json').read_text())}
    name = 'r3_local_kl_adamw50'
    sr = search[name]
    assert sr['training_complete']
    decision = json.loads((BASE / f'confirmation_decision.{name}.json').read_text())
    epoch = decision['candidate']['epoch']
    s0, s1 = sr['stages']['initial/trained'], sr['stages'][f'epoch{epoch}/trained']
    cr = confirm[name]
    c0, c1 = cr['stages']['confirm/initial'], cr['stages']['confirm/selected']
    assert all(s['complete'] for s in [s0, s1, c0, c1])
    cohorts = {}
    for f in (BASE / 'runs' / name / 'confirm').glob('eval.*.rank*.jsonl'):
        for r in read(f):
            byid = cohorts.setdefault(r['variant'], {})
            assert r['id'] not in byid
            byid[r['id']] = r
    a, b = cohorts['initial'], cohorts['selected']
    assert a.keys() == b.keys() and len(a) == 112
    counts = dict(kl_lower=0, kl_lower_score_up=0, kl_lower_score_down=0,
                  kl_lower_score_same=0, kl_higher=0, kl_same=0)
    for i, r in b.items():
        p = a[i]
        if r['prediction'] < p['prediction']:
            counts['kl_lower'] += 1
            delta = r['student_score'] - p['student_score']
            category = 'up' if delta > 1e-10 else 'down' if delta < -1e-10 else 'same'
            counts[f'kl_lower_score_{category}'] += 1
        else:
            counts['kl_higher' if r['prediction'] > p['prediction'] else 'kl_same'] += 1
    official = [r for n, r in search.items() if n.startswith(('r1_', 'r2_', 'r3_'))]
    facts = {
        'controller_phase': state['phase'], 'last_controller_update': state['updated_at'],
        'formal_candidates': len(official),
        'completed_formal_candidates': sum(r['training_complete'] for r in official),
        'early_stopped_formal_candidates': sum(bool(r.get('early_stop')) for r in official),
        'confirmation_attempts': state.get('confirmation_attempts', []),
        'full_retrain_started': bool(state.get('full_config')),
        'candidate': name, 'selected_epoch': epoch,
        'search': {'initial': s0, 'trained': s1},
        'confirm': {'initial': c0, 'trained': c1, 'mean': cr['matched_mean']},
        'paired_confirmation_counts': counts,
        'note': 'Descriptive synthesis only; no new selection or change to existing gates.',
    }
    (BASE / 'current_diagnosis.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2))
    lines = [
        '# MetaCompress: Updated Failure Evidence After Three Search Rounds', '',
        'Detailed gradient/update, matched optimizer/scheduler, and token-budget measurements:  '
        '[Optimizer and budget measurements](METACOMPRESS_OPTIMIZATION_AND_BUDGET.md). ', '',
        'Reads completed outputs without starting training or changing eligibility. Reproduction entry: ', '',
        '```bash',
        '.venv/bin/python -m scripts.report_meta_compress_current_diagnosis',
        '```', '',
        '## Current Status and Corrections to Earlier Updates', '',
        f"Controller last updated: `{state['updated_at']}`, Status `{state['phase']}`. "
        f"Across three rounds,  {len(official)}  formal candidates, "
        f"{facts['completed_formal_candidates']}  runs completed planned training, "
        f"{facts['early_stopped_formal_candidates']}  runs stopped early under recorded criteria; two additional mini local diagnostics. "
        'Both formal independent confirmations failed; no new original-20k retraining started, and no tuning job remains active.', '',
        'An earlier progress update quoted old 10%-retention/20k/2-epoch results and omitted these subsequent tuning runs. '
        'The old run completed, but is not the latest post-search full retraining. '
        'The following uses actual search/confirmation outputs.', '',
        '## 1. Dense Aggregation Compatibility with UND Features Is the Primary Issue', '',
        'Among 64 matrices sampled from the old full run, mean row entropy/log(n)≈.981 and effective entropy support≈3921, '
        'Adjacent row-probability cosine is about 0.903. Many rows mix the whole page, potentially averaging away text, numbers, and layout.'
        '**High entropy alone does not imply failure**: free P and KL-only SGD previously fit one question while retaining high entropy. '
        'More directly, for three teacher-correct questions, even with m=n and unchanged placeholder/text positions, '
        'all fall to zero after initial-P mixing alone. These three cases show mixing itself is sufficient to break their answers, '
        'This does not generalize to every image.', '',
        'On the same 56-image search at 50%, original dense initialization scores 4.799 and fixed local mean 34.001, '
        'Local-bias initialization scores 37.192. Its prior helps at step 0, '
        'This is not a learning gain. More output rows alone do not ensure distinct local evidence. '
        'Queries currently use 1D fractional max pooling; positional tables are supplied by this implementation, '
        'Not verbatim author code; correspondence to native 2D document tokens needs structural ablation.', '',
        '## 2. Original Optimization Was Constrained, but Does Not Explain Every Later Failure', '',
        'Original SGD has no momentum, LR=1e-3, global clip=.01, giving update L2≤1e-5/step; '
        'All 1,112 steps are clipped, with median multiplier about 0.0001765. Local BF16 finite differences also show '
        'Rounding affects forward changes from tiny FP32 parameter movement; insufficient effective updates are directly evidenced.', '',
        'but later runs already test clip=1/10, AdamW, momentum, constant/cosine, and different budgets. '
        'Latest local KL-only AdamW50 epoch-mean training KL is approximately  '
        '0.550→0.482→0.449→0.425; independent-confirmation KL also decreases. '
        'These candidates are not wholly untrainable, nor is the scheduler established as the sole cause.', '',
        '## 3. Gradient Conflict Between Prediction and Regularization Terms', '',
        'Across eight teacher-correct diagnostic images, original-initialization median collapse/KL gradient-norm ratio is about 44.8; '
        'For 4/8 images, the total/KL gradient dot product is negative: local total-loss descent increases KL. '
        'Component-gradient evidence, not comparison of loss magnitudes or a dataset-wide proportion. '
        'Same-image/SGD/LR 2×2 controls: KL-only without strong clipping fits, whereas full loss without clipping still fails.', '',
        'Collapse penalizes column reuse, but a near-uniform global P also has low collapse; '
        'Its gradient can conflict with entropy encouraging each row to focus on few tokens. '
        'Unnormalized total column mass also changes scale with m. '
        'This supports recalibrating regularization/gradient budgets, not dismissing both regularizers universally: local structure plus full loss can fit a question.', '',
        '## 4. Updated Objectives Generalize KL Reduction, but Stable Answer Gains Are Unconfirmed', '',
        f'Representative candidate `{name}`, 672 distinct training images, 4 epochs, 50% retention, local bias=20, '
        'AdamW 1e-4, clip=1, warmup/cosine, KL-only. Local bias is an additional structural prior absent from the paper.', '',
        '| Cohort | Initial score | Trained score | Score change | Initial KL | Trained KL | KL reduction |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for label, initial, final in [('56-image search', s0, s1), ('112-image confirmation', c0, c1)]:
        lines.append(f"| {label} | {initial['score']:.3f} | {final['score']:.3f} | "
                     f"{final['score']-initial['score']:+.3f} | {initial['prediction']:.5f} | "
                     f"{final['prediction']:.5f} | {1-final['prediction']/initial['prediction']:.1%} |")
    ci = c1['paired_vs_initial']['paired_bootstrap_95ci_points']
    lines += [
        '', f'Paired bootstrap 95% CI for confirmation gain over initial: [{ci[0]:.3f}, {ci[1]:.3f}] points. '
        'The interval crosses zero: insufficient evidence for stable gains, not proof of zero benefit. '
        'The search gain interval also crosses zero. Repeated candidate/epoch selection on the same 56 images risks selection bias. '
        'Reusing the same confirmation set across two rounds also introduces adaptive-reuse limitations.', '',
        f"Per-question comparison: {counts['kl_lower']}/112 questions have lower KL, of which  "
        f"{counts['kl_lower_score_up']}  questions improve, {counts['kl_lower_score_down']}  questions have lower scores, "
        f"{counts['kl_lower_score_same']}  questions unchanged. "
        'The main discrepancy is not total validation-KL overfitting, but distribution matching versus free-generation scoring.', '',
        'Mean full-vocabulary KL may decrease on noncritical tokens, probability tails, or later positions under a teacher-provided answer prefix; '
        'It does not guarantee correct argmax for critical digits/characters. Teacher forcing supplies students with a correct teacher prefix, '
        'Free generation follows its own prefixes; one error can change the trajectory. '
        'These are mechanistic explanations whose contributions have not been separated by per-token margins or first/later-token gradient ablations. '
        'Teacher can be wrong; matching teacher is not the same as matching gold.', '',
        'Confirmation tasks differ: DocVQA 46.43→52.00, InfoVQA 42.12→49.86, DeepForm 12.50→25.00; '
        'DT-VQA simultaneously falls 59.24→49.53, ChartQA 37.50→31.25, WTQ 25.00→18.75; KLC is unchanged. '
        'Only 16 images/task; changes are not stable task effects. '
        'Case/whitespace-normalized exact match rises 16.96→22.32, suggesting that format and native partial credit affect trends; '
        'Prespecified auxiliary metric, not a replacement model-selection objective.', '',
        '## 5. Remaining Limits: Aggressive Compression, Training Coverage, and Task Constraints', '',
        'Fixed-mean search scores at 10%/25%/50%/75% are about 4.59/10.51/34.00/54.68, '
        'Teacher scores 58.13. Budget strongly affects document detail, but this is not an upper bound for the original generator. '
        'The 75% local KL-only candidate scores 55.59 before and 55.03 after training, '
        'Again distinguish retaining more tokens from learning gains. '
        'Single-image fitting at the 10% budget also rules out the overstrong claim that every task is impossible at this retention.', '',
        'DT-VQA is nearly half the original 20k; multi-image pilots already use task balancing, '
        'Thus original task proportions do not explain all later failures. Round 3 uses 96 images/task, a small fraction of 20k; '
        'No convergence/repeated-seed evidence establishes sufficient epochs/data coverage. '
        'Currently training uses one question/image and a question-independent generator. '
        'Distillation on one question does not constrain all potential page-level QA evidence; '
        'Multiple questions/image may strengthen coverage, but have not been shown to repair this issue.', '',
        '## 6. Unsupported Attributions and Next Priorities', '',
        'Insufficient evidence to prioritize Qwen3-1.7B capacity or official encoder as the cause. '
        'The same uncompressed model answers normally and remains frozen, so current LLM-unfreezing damage is not a possible path. '
        'Identity, independent-state, causal-shift, and cache-source checks pass. Teacher logits on seven-task anchors are bit-exact to cache. '
        'This reduces these implementation-error explanations, but is not exhaustive over all images/boundaries.', '',
        'Prioritize attributable comparisons next:', '',
        '1. At a fixed 50% budget on image-disjoint splits, compare 2D local selection/aggregation against the current 1D local generator, '
        'Add original-token-selection baselines and inspect support/output distributions instead of merely expanding dense hyperparameter search.',
        '2. Use multiple questions per training image and image-disjoint validation to test page-wide evidence coverage; '
        'Do not restrict training/reporting to teacher-correct images.',
        '3. First audit the first answer token, numeric/entity tokens, EOS, teacher argmax margin, and free-generation divergence positions. '
        'Then compare original KL, critical-token-weighted KL, and student-prefix distillation at matched architecture/optimizer. '
        'These are new decoder-dependent objective/trajectory changes, distinct from the paper’s method.',
        '4. Expand independent validation and repeated seeds, reserving a separate final confirmation set; '
        'Expand to original 20k only after stable gains over same-budget/architecture initialization. '
        'Do not present possible success/failure of an uncompleted full retraining as an observation.', '',
        'Raw evidence: [Search design](METACOMPRESS_TRAINING_SEARCH.md) · '
        '[search](../artifacts/meta_compress_search/RESULTS.search.md) · '
        '[confirm](../artifacts/meta_compress_search/RESULTS.confirm.md) · '
        '[Mechanism interventions](METACOMPRESS_INTERVENTION_RESULTS.md) · '
        '[Machine-readable report statistics](../artifacts/meta_compress_search/current_diagnosis.json). ',
    ]
    dest = ROOT / 'docs/METACOMPRESS_CURRENT_DIAGNOSIS.md'
    dest.write_text('\n'.join(lines) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
