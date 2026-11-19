"""Read recorded updates to explain clipping, optimizers, schedule and budget."""
import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/meta_compress_search'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    search = {r['name']: r for r in json.loads((BASE / 'summary.search.json').read_text())}
    results = {}
    for name, r in search.items():
        if not name.startswith(('r1_', 'r2_', 'r3_')):
            continue
        log = read(BASE / 'runs' / name / 'training.jsonl')
        assert len({x['step'] for x in log}) == len(log)
        cfg = r['config']

        def summarize(rows):
            return {
                'steps': len(rows),
                'pre_clip_gradient_mean': stats.mean(x['grad_norm'] for x in rows),
                'pre_clip_gradient_median': stats.median(x['grad_norm'] for x in rows),
                'nominal_post_clip_gradient_median': stats.median(min(x['grad_norm'], cfg['clip']) for x in rows),
                'clipped_fraction': stats.mean(x['clip_multiplier'] < 1 for x in rows),
                'clip_multiplier_median': stats.median(x['clip_multiplier'] for x in rows),
                'update_l2_mean': stats.mean(x['update_l2'] for x in rows),
                'update_l2_median': stats.median(x['update_l2'] for x in rows),
                'train_KL': stats.mean(x['prediction'] for x in rows),
                'first_lr': rows[0]['lr'], 'last_lr': rows[-1]['lr'],
                'sum_lr': sum(x['lr'] for x in rows),
                'update_path_length': sum(x['update_l2'] for x in rows),
            }
        results[name] = {
            'config': cfg, 'training_complete': r['training_complete'],
            'early_stop': r.get('early_stop'), 'overall': summarize(log),
            'epochs': {str(ep): summarize([x for x in log if x['epoch'] == ep])
                       for ep in sorted({x['epoch'] for x in log})},
            'search_stages': {k: v for k, v in r['stages'].items() if v['complete']},
        }
    (BASE / 'optimization_audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    lines = [
        '# MetaCompress: Measured Gradients, Optimizer, Scheduler, and Token Budgets', '',
        'Reads completed three-round logs only; entry: ', '',
        '```bash', '.venv/bin/python -m scripts.report_meta_compress_optimization', '```', '',
        '## 1. Current Gradient-Norm Behavior', '',
        'Distinguish three quantities:`grad_norm` is generator-parameter L2 gradient norm after DDP averaging and before clipping; '
        '`clip_multiplier=min(1, threshold/(norm+1e-6))` is the clipping multiplier; '
        '`update_l2` instead measures total parameter L2 change before/after optimizer.step. '
        'Only the last quantity directly measures actual per-step parameter movement. '
        'Clipping preserves total-gradient direction and cannot resolve component-gradient conflict.', '',
        'Use current pilot logs throughout rather than mixing original 20k/batch36 and new pilot/batch8 statistics. '
        'Each row summarizes logged updates; 10/25/50 are retention percentages.', '',
        '| Candidate | Settings summary | Median pre-clipping norm | Clipping threshold | Clipping multiplier | Median scaling multiplier | Mean actual update L2 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    selected = [
        ('r1_sgd_original10', 'dense10 / full loss / SGD'),
        ('r1_sgd_cos_clip1_10', 'dense10 / full loss / SGD+cosine'),
        ('r1_adamw_cos10', 'dense10 / full loss / AdamW+cosine'),
        ('r2_dense_kl_sgdm25', 'dense25 / KL-only / momentum SGD'),
        ('r2_local_sgdm50', 'local50 / full loss / momentum SGD'),
        ('r3_local_kl_adamw50', 'local50 / KL-only / AdamW'),
        ('r3_local_kl75', 'local75 / KL-only / AdamW'),
    ]
    for name, label in selected:
        r = results[name]; z = r['overall']
        lines.append(f"| {name} | {label} | {z['pre_clip_gradient_median']:.3f} | {r['config']['clip']:g} | "
                     f"{z['clipped_fraction']:.1%} | {z['clip_multiplier_median']:.6f} | {z['update_l2_mean']:.6g} |")
    old = results['r1_sgd_original10']['overall']
    new = results['r3_local_kl_adamw50']['overall']
    lines += [
        '', 'Original no-momentum SGD update is  `Δθ=-LR*clip(g)`, hence with clip=.01 and LR=.001, '
        'Nominal bound=1e-5/step; actual FP32 differences have minor rounding deviations. '
        f"Original-setting median clipping multiplier {old['clip_multiplier_median']:.7f}, shrinking gradients approximately  "
        f"{1/old['clip_multiplier_median']:.0f} ×. "
        'This is not a nonfinite norm or established exploding gradient; the threshold is tiny relative to current gradient scale.', '',
        f"Latest local50/KL-only/AdamW median norm:  {new['pre_clip_gradient_median']:.3f}, "
        f"Median multiplier {new['clip_multiplier_median']:.3f}, mean actual parameter update is approximately  "
        f"{new['update_l2_mean']/old['update_l2_mean']:.0f} ×."
        '**Frequent clipping persists, but does not imply updates are suppressed as severely as originally.**'
        'Gradient-norm decreases are not solely attributable to optimizer changes: budget, architecture, loss, and trajectory also changed.', '',
        'Original full-20k median multiplier ≈.0001765 differs from this pilot’s .0002030, '
        'are measurements under different data/batch/duration settings and do not overwrite one another. Strong clipping/tiny SGD updates remain findings of the original run.', '',
        'Latest KL-only candidates set entropy/collapse weights to 0, so neither contributes to their update directions; '
        'Original full-loss gradient conflicts cannot automatically be assigned to this objective. '
        'Original component-gradient audits still cover only eight diagnostic images, not all pilot data.', '',
        '## 2. Optimizer Changes and Their Measured Effects', '',
        'Without momentum, SGD movement is proportional to clipped gradient and LR. '
        'Momentum SGD accumulates directions across steps, so one update is no longer bounded by the same LR×clip expression. '
        'AdamW scales each parameter using first-/second-moment gradient history and applies weight decay, '
        'LR×clip does not bound its parameter updates, and LR values across optimizers are not directly comparable.', '',
        'With dense architecture, full loss, and 10% retention fixed:', '',
        '| Configuration | Actual mean update L2 | Initial score | epoch 1 | epoch 2 | epoch 3 | epoch 4 | Final KL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ['r1_sgd_original10', 'r1_sgd_cos_clip1_10', 'r1_adamw_cos10']:
        r = results[name]; st = r['search_stages']
        scores = ' | '.join(f"{st[f'epoch{e}/trained']['score']:.3f}" for e in range(1, 5))
        lines.append(f"| {name} | {r['overall']['update_l2_mean']:.6g} | {st['initial/trained']['score']:.3f} | "
                     f"{scores} | {st['epoch4/trained']['prediction']:.5f} |")
    lines += [
        '', 'These are combined-setting comparisons: relaxing SGD clipping also adds cosine, and AdamW uses another LR, '
        'Not all differences can be attributed to optimizer identity. New settings substantially increase actual parameter movement, '
        'but the original dense architecture still fails to recover task performance. AdamW10’s brief epoch-2 score gain is not sustained.', '',
        'Both momentum SGD and AdamW learn with local structure. Representative combinations:', '',
        '| Candidate | Training images | Target | Search: initial → selected epoch | Confirmation: initial → selected epoch | Confirmation KL reduction |',
        '|---|---:|---|---:|---:|---:|',
        '| r2_local_sgdm50 | 224 | Full loss | 37.192→40.621(epoch 3) | 35.399→33.966 | 19.7% |',
        '| r3_local_kl_adamw50 | 672 | KL-only | 37.192→44.745(epoch 4) | 35.399→35.913 | 20.4% |', '',
        'Training image count, loss, LR, and other settings differ, so this table does not establish AdamW superiority to momentum SGD; '
        'Both optimizers reduce independent-validation KL, but neither passes independent answer-gain criteria.', '',
        '## 3. Scheduler Behavior and Evidence of Improvement', '',
        'Constant uses fixed LR. Warmup/cosine increases linearly to peak LR, '
        'then cosine decay to 10% peak LR. The 112-step pilot has 11 warmup steps; '
        'The 336-step expanded pilot has 34 warmup steps. Latest AdamW50 starts at about 2.94e-6, reaches 1e-4, and ends at 1e-5.', '',
        'Matched scheduler controls: dense25, full loss, same 224 images, AdamW, peak LR=3e-5, '
        'clip=1, identical initialization/order; only scheduler differs:', '',
        '| scheduler | Mean update L2 | Initial score | epoch 1 | epoch 2 | epoch 3 | epoch 4 | Final KL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ['r1_adamw_constant25', 'r1_adamw_cos25']:
        r = results[name]; st = r['search_stages']
        scores = ' | '.join(f"{st[f'epoch{e}/trained']['score']:.3f}" for e in range(1, 5))
        lines.append(f"| {r['config']['scheduler']} | {r['overall']['update_l2_mean']:.6g} | "
                     f"{st['initial/trained']['score']:.3f} | {scores} | {st['epoch4/trained']['prediction']:.5f} |")
    lines += [
        '', '**Both conditions fail; no evidence shows cosine scheduling repaired the original method.**'
        'At the same peak, cosine has smaller cumulative LR (0.0018465 versus constant 0.00336), '
        'This changes early/late steps and cumulative optimization strength, not schedule shape at matched total LR.', '',
        'Latest local AdamW50 per-epoch measurements:', '',
        '| epoch | LR: first → last update | Median gradient norm | Clipping multiplier | Mean update L2 | train KL | search KL |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    r = results['r3_local_kl_adamw50']
    for ep, z in r['epochs'].items():
        v = r['search_stages'][f'epoch{ep}/trained']
        lines.append(f"| {ep} | {z['first_lr']:.3g}→{z['last_lr']:.3g} | {z['pre_clip_gradient_median']:.3f} | "
                     f"{z['clipped_fraction']:.1%} | {z['update_l2_mean']:.6f} | {z['train_KL']:.5f} | {v['prediction']:.5f} |")
    lines += [
        '', 'Actual updates shrink as LR decays while KL falls, showing optimization is active, '
        'This does not establish superiority to constant scheduling at matched architecture/data; no scheduler-only paired control exists here.', '',
        '## 4. Performance Across Token Budgets', '',
        'The following search uses the same 56 distinct images, 8/task over seven tasks; teacher macro is fixed at 58.134. '
        'Ratios denote retention; 10% removes about 90%. Scores are percentages; lower KL is better. '
        'Untrained baseline for budget effects, not a learning gain.', '',
        '| Retention fraction | Fixed local-mean score | mean KL | Initial dense-generator score | Initial local-generator score |',
        '|---|---:|---:|---:|---:|',
    ]
    mapping = {0.1: ('r1_sgd_original10', None),
               0.25: ('r1_adamw_cos25', 'r2_local_balanced25'),
               0.5: ('r1_adamw_cos50', 'r2_local_balanced50'),
               0.75: (None, 'r3_local_kl75')}
    for ratio, (dense, local) in mapping.items():
        rows = [r for f in (BASE / f'baselines/ratio{ratio:g}/search').glob('eval.*.rank*.jsonl') for r in read(f)]
        assert len(rows) == len({r['id'] for r in rows}) == 56
        score = 100 * stats.mean(r['student_score'] for r in rows)
        kl = stats.mean(r['prediction'] for r in rows)
        fmt = lambda n: f"{results[n]['search_stages']['initial/trained']['score']:.3f}" if n else 'Not measured'
        lines.append(f'| {ratio:.0%} | {score:.3f} | {kl:.5f} | {fmt(dense)} | {fmt(local)} |')
    lines += [
        '', 'Only retention changes in fixed mean controls, clearly identifying its benefit there. '
        'Original dense initialization performs poorly at 10/25/50%; more tokens alone do not repair global mixing. '
        'High initial quality of local75% mainly reflects its wider budget/local preservation, not learning gains.', '',
        'Post-training matched 25%/50% controls: same 224 images, AdamW 3e-5, clip=1, cosine, '
        'local bias=20, entropy/collapse weights=.01 each; only retention differs. Report fixed epoch 4:', '',
        '| Retention fraction | Score: initial → epoch 4 | Score increment | KL: initial → epoch 4 |',
        '|---|---:|---:|---:|',
    ]
    for name in ['r2_local_balanced25', 'r2_local_balanced50']:
        r = results[name]; st = r['search_stages']; a, b = st['initial/trained'], st['epoch4/trained']
        lines.append(f"| {r['config']['ratio']:.0%} | {a['score']:.3f}→{b['score']:.3f} | "
                     f"{b['score']-a['score']:+.3f} | {a['prediction']:.5f}→{b['prediction']:.5f} |")
    lines += [
        '', 'At 50%, initial quality and training gains are higher, but this comparison has no separate independent confirmation; '
        'Search gains alone do not establish reliable generalization.', '',
        'Higher-budget r3_local_kl75: 672 images, AdamW 3e-5, KL-only, cosine, clip=1. '
        'Score 55.587→55.030, KL 0.15613→0.10190. Although absolute quality is close to the small-set teacher, '
        'Training does not beat same-budget initialization. LR also differs from r3_local_kl_adamw50, '
        'These are not budget-only controls. No 10% candidate passed multi-image validation.', '',
        '## 5. Scope of Conclusions', '',
        'Strong clipping/tiny SGD updates are directly evidenced; new settings substantially increase movement, '
        'Latest local variants reduce train/search/confirmation KL. '
        'but optimizer/scheduler changes have not delivered independently confirmed stable answer gains; '
        'More budget improves basic preservation, not proof of successful learning. '
        'No original-20k retraining started; these are small multi-image tests, not new full benchmarks.', '',
        '[All 20 optimizer-statistics JSON records](../artifacts/meta_compress_search/optimization_audit.json) · '
        '[Stagewise search](../artifacts/meta_compress_search/RESULTS.search.md) · '
        '[Independent confirmation](../artifacts/meta_compress_search/RESULTS.confirm.md) · '
        '[Combined diagnosis](METACOMPRESS_CURRENT_DIAGNOSIS.md). ',
    ]
    dest = ROOT / 'docs/METACOMPRESS_OPTIMIZATION_AND_BUDGET.md'
    dest.write_text('\n'.join(lines) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
