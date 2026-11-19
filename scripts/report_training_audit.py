"""Audit actual encoder/data and plot recorded training loss plus post-hoc eval endpoints."""
import collections
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/full/loss_audit'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(fig, name):
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / 'models/MonkeyOCRv2-B'
    weight_path = model_dir / 'model.safetensors'
    source = next(r for r in json.loads((model_dir / 'source_manifest.json').read_text()) if r['Name'] == 'model.safetensors')
    weight_hash = digest(weight_path)
    assert weight_hash == source['Sha256']
    with weight_path.open('rb') as handle:
        header = json.loads(handle.read(struct.unpack('<Q', handle.read(8))[0]))
    parameters = sum(math.prod(r['shape']) for name, r in header.items() if name != '__metadata__')
    fingerprints = json.loads((ROOT / 'artifacts/full/corpus_fingerprint.json').read_text())
    assert digest(ROOT / 'data/full/train.jsonl') == fingerprints['train']
    counts, sources = collections.Counter(), collections.Counter()
    for line in (ROOT / 'data/full/train.jsonl').read_text().splitlines():
        row = json.loads(line)
        counts[row['task']] += 1
        sources[row['task'], row['source_repo']] += 1
    total = sum(counts.values())
    # Latest record per step is canonical if a resumed run appended duplicates.
    records = {}
    for line in (ROOT / 'artifacts/full/training.jsonl').read_text().splitlines():
        row = json.loads(line)
        records[row['stage'], row['step']] = row
    epochs = []
    step_rows = []
    for stage in (1, 2):
        rr = [r for (s, _), r in sorted(records.items()) if s == stage]
        assert [r['step'] for r in rr] == list(range(1, 1905))
        assert {r['epoch'] for r in rr} == {1}
        losses = [r['loss'] for r in rr]
        assert all(math.isfinite(v) for v in losses)
        epochs.append({'stage': stage, 'epoch_in_stage': 1, 'optimizer_steps': len(rr),
                       'online_rank0_mean_loss': sum(losses) / len(losses),
                       'first100_mean_loss': sum(losses[:100]) / 100,
                       'last100_mean_loss': sum(losses[-100:]) / 100})
        for row in rr:
            step_rows.append({'stage': stage, 'epoch_in_stage': 1, 'step': row['step'],
                              'fractional_epoch_in_stage': row['step'] / row['total_steps'],
                              'cumulative_training_pass': stage - 1 + row['step'] / row['total_steps'],
                              'online_rank0_loss': row['loss'], 'lr': row['lr']})
    write_csv(OUT / 'training_steps.csv', step_rows)
    write_csv(OUT / 'epoch_summary.csv', epochs)

    endpoints = json.loads((OUT / 'endpoint_summary.json').read_text())
    actual = {}
    for line in (OUT / 'endpoint_records.jsonl').read_text().splitlines():
        r = json.loads(line)
        actual[r['stage'], r['split'], r['id']] = r
    endpoint_rows = []
    for stage in (1, 2):
        for split in ('train', 'val'):
            result = endpoints['stages'][str(stage)][split]
            rr = [r for (s, sp, _), r in actual.items() if s == stage and sp == split]
            assert len(rr) == result['samples'] == 256
            assert abs(sum(r['answer_nll'] for r in rr) / len(rr) - result['sample_mean_nll']) < 1e-8
            for task, summary in [('All four tasks', result), *result['tasks'].items()]:
                endpoint_rows.append({'stage': stage, 'split': split, 'task': task,
                                      **{k: v for k, v in summary.items() if k != 'tasks'}})
    write_csv(OUT / 'checkpoint_loss.csv', endpoint_rows)
    audit = {'encoder': {'repo': 'zenosai/MonkeyOCRv2-B', 'size': 'Base (B)', 'parameters': parameters,
                         'weights_sha256': weight_hash, 'weight_revision': source['Revision'], 'frozen': True},
             'training_records': total, 'tasks': dict(counts),
             'sources': [{'task': task, 'repo': repo, 'count': count} for (task, repo), count in sources.items()],
             'epoch_loss': epochs, 'historical_eval_loss_recorded': False,
             'online_loss_scope': 'Rank-0 local microbatch means at changing parameters; not all-rank or fixed-checkpoint loss',
             'endpoint_scope': endpoints['scope']}
    (OUT / 'training_audit.json').write_text(json.dumps(audit, indent=2))

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), layout='constrained')
    for stage, ax in zip((1, 2), axes):
        rr = [r for (s, _), r in sorted(records.items()) if s == stage]
        x = np.array([r['step'] / r['total_steps'] for r in rr])
        y = np.array([r['loss'] for r in rr])
        ax.plot(x, y, color='#94a3b8', alpha=.4, lw=.6, label='Recorded step loss (rank 0)')
        ax.plot(x[99:], np.convolve(y, np.ones(100) / 100, mode='valid'), color='#2563eb', lw=2, label='Trailing 100-step mean')
        e = epochs[stage - 1]
        ax.set_title(f"Stage {stage}: " + ('MLP only' if stage == 1 else 'MLP + Qwen') + '\n'
                     + f"First/last 100-step mean: {e['first100_mean_loss']:.3f} / {e['last100_mean_loss']:.3f}")
        ax.set_xlabel('Epoch within stage (one epoch total)')
        ax.set_ylabel('Online answer cross-entropy (nats)')
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle('Historical training loss: 1,904 updates per stage; no historical eval-loss series', fontsize=12)
    save(fig, 'training_loss')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for split, label, color in [('train', 'Train (fixed 256 QA)', '#2563eb'), ('val', 'Validation (fixed 256 QA)', '#ea580c')]:
        vals = [endpoints['stages'][str(s)][split]['sample_mean_nll'] for s in (1, 2)]
        axes[0].plot([1, 2], vals, 'o-', color=color, label=label)
        for stage, value in zip((1, 2), vals):
            axes[0].annotate(f'{value:.3f}', (stage, value), xytext=(7, 5 if split == 'val' else -14), textcoords='offset points', color=color)
    axes[0].set_title('Same fixed train / validation samples')
    axes[0].set_ylabel('Sample-mean answer NLL (nats)')
    axes[0].legend(fontsize=9)
    for task in ('DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA'):
        vals = [endpoints['stages'][str(s)]['val']['tasks'][task]['sample_mean_nll'] for s in (1, 2)]
        axes[1].plot([1, 2], vals, 'o-', label=task)
    axes[1].set_title('Validation breakdown (64 QA per task)')
    axes[1].legend(fontsize=9)
    axes[1].set_ylabel('Sample-mean answer NLL (nats)')
    for ax in axes:
        ax.set_xticks([1, 2], ['Stage 1 end\n(one pass)', 'Stage 2 end\n(two passes total)'])
        ax.set_xlim(.85, 2.2)
        ax.grid(alpha=.18)
    fig.suptitle('Post-hoc saved-checkpoint evaluation: only TWO measured endpoints\nLines join endpoints; intermediate eval losses were not recorded', fontsize=12)
    save(fig, 'checkpoint_train_eval_loss')

    lines = ['# Encoder Identity, Training Data, and Loss-Log Audit', '',
             '**Formal training uses the authors’ pretrained MonkeyOCRv2-B (Base); recomputed weight SHA-256 values exactly match the download-source manifest.**', '',
             f'- Source: `zenosai/MonkeyOCRv2-B`; local  `models/MonkeyOCRv2-B/model.safetensors`. ',
             f'- Actual parameter count in the weights: **{parameters:,}** (approximately 113.7M; 113M in the paper); 12 layers, 12 attention heads, embedding width 768, patch size 14.',
             f'- Weight revision: `{source["Revision"]}`. ', f'- SHA256: `{weight_hash}`. ',
             '- The visual encoder is frozen in both stages. Extract frozen features before training MLP/Qwen. The extraction completion marker records the same weight hash. '
             'This confirms weight identity/freezing, not exact author-equivalence of every preprocessing/backend detail.', '',
             '## Data Actually Used for Formal Training', '',
             f'Both formal stages use the same **{total:,}  public VQA training QA**, one pass/stage.'
             '**Formal training includes no local MonkeyDoc v2 OCR/transcription data**. Earlier engineering tests used QA derived from MonkeyDoc v2 synthetic tables, '
             'Experiment directory:  `artifacts/alignment/`, and  `artifacts/full/` training is reported separately for this round.', '',
             '| Dataset | Actual training QA | Fraction | Source |', '|---|---:|---:|---|']
    for task, count in counts.most_common():
        repo = next(repo for t, repo in sources if t == task)
        lines.append(f'| {task} | {count:,} | {100 * count / total:.2f}% | {repo} |')
    lines.extend(['', '**`echo840/Monkey_Data` is an original Monkey VQA source, distinct from MonkeyDoc v2 pretraining/OCR data.**'
                  'OCRBench is test-only. Public DocVQA/InfoVQA validation is held out for evaluation.', '',
                  '## Historical Training Loss: Available Records', '',
                  'Both stages use **1 epoch / 1,904 optimizer steps**, 3,808 updates total. '
                  'Thus no within-stage multi-epoch convergence curve exists; 100-step intervals cannot be relabeled epochs.', '',
                  '![Historical stepwise training loss](../artifacts/full/loss_audit/training_loss.png)', '',
                  '| Stage | Epochs | Online mean loss across the entire stage | Mean of first 100 steps | Mean of last 100 steps |',
                  '|---|---:|---:|---:|---:|'])
    for e in epochs:
        lines.append(f"| Stage {e['stage']} | 1 | {e['online_rank0_mean_loss']:.4f} | {e['first100_mean_loss']:.4f} | {e['last100_mean_loss']:.4f} |")
    lines.extend(['', 'Original source `artifacts/full/training.jsonl`. Logged loss is the **mean of rank 0’s local microbatches**, '
                  'not aggregated across four ranks; parameters/samples change each step. Epoch means are arithmetic means of these logs, '
                  'not a fixed-endpoint sweep of the entire training set. '
                  'Gray: raw steps; blue: trailing 100-step mean. Decreases fluctuate and are not necessarily monotonic.', '',
                  '## Train/Eval Loss: Post-Hoc Measurements at Two Stage Endpoints', '',
                  '**Historical validation recorded generation metrics such as ANLS/accuracy, but did not save validation loss.**'
                  'Not all intermediate checkpoints remain; historical 100-step or multi-epoch evaluation-loss trajectories cannot be recovered.', '',
                  'Load saved Stage 1/2 models and measure answer NLL on fixed **256 train + 256 validation** questions. '
                  'Each condition uses 64 distinct documents/task from DocVQA, InfoVQA, ChartQA, and DT-VQA, one question/image, '
                  'Deterministic SHA-256 ordering. Equal task weights aid comparison but differ from weighting the complete seven-task training mixture. '
                  'No training updates or test use.', '',
                  '![Fixed training and validation endpoint losses](../artifacts/full/loss_audit/checkpoint_train_eval_loss.png)', '',
                  '| Saved model | Train NLL (256 questions) | Validation NLL (256 questions) |', '|---|---:|---:|'])
    for stage in ('1', '2'):
        s = endpoints['stages'][stage]
        lines.append(f"| Stage {stage} Final | {s['train']['sample_mean_nll']:.4f} | {s['val']['sample_mean_nll']:.4f} |")
    lines.extend(['', 'NLL is causal cross-entropy on first gold answer+EOS, averaged first over answer tokens then questions; lower is better; '
                  'Also export token-weighted loss, matching the training objective. Teacher-forced loss is not free-generation accuracy.'
                  '**Only two endpoints were measured; the connecting line does not imply intermediate measurements or monotonic loss reduction.**', '',
                  '| Validation tasks (64 questions each) | Stage 1 NLL | Stage 2 NLL |', '|---|---:|---:|'])
    for task in ('DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA'):
        values = [endpoints['stages'][str(s)]['val']['tasks'][task]['sample_mean_nll'] for s in (1, 2)]
        lines.append(f'| {task} | {values[0]:.4f} | {values[1]:.4f} |')
    lines.extend(['', 'Logs show overall training-loss decrease and permit comparing validation loss at two saved endpoints; '
                  'A falling loss alone does not establish visual text reading or generalization. '
                  'Next training should log evaluation NLL and generation together on a fixed validation subset, retaining enough checkpoints; '
                  'For multi-epoch curves, actually run multiple epochs and retain stage boundaries.', '',
                  '## Entry Points and Exported Files', '', '```bash',
                  '# Measure fixed training/validation endpoint losses on saved models; NPU, with venv/CANN set by the script.',
                  'sbatch scripts/slurm_loss_endpoints.sh',
                  '# After measurement, rebuild weight/data audits, curves, and this report.',
                  '.venv/bin/python -m pip install -r requirements-report.txt',
                  '.venv/bin/python -m scripts.report_training_audit', '```', '',
                  '- [Historical training-curve PNG](../artifacts/full/loss_audit/training_loss.png), '
                  '[SVG](../artifacts/full/loss_audit/training_loss.svg), [PDF](../artifacts/full/loss_audit/training_loss.pdf). ',
                  '- [Train / eval loss PNG](../artifacts/full/loss_audit/checkpoint_train_eval_loss.png), '
                  '[SVG](../artifacts/full/loss_audit/checkpoint_train_eval_loss.svg), [PDF](../artifacts/full/loss_audit/checkpoint_train_eval_loss.pdf). ',
                  '- [Stepwise CSV](../artifacts/full/loss_audit/training_steps.csv), '
                  '[Stagewise epoch-summary CSV](../artifacts/full/loss_audit/epoch_summary.csv), '
                  '[Train / eval loss CSV](../artifacts/full/loss_audit/checkpoint_loss.csv). ',
                  '- [Verification JSON](../artifacts/full/loss_audit/training_audit.json), '
                  '[Per-question endpoint losses](../artifacts/full/loss_audit/endpoint_records.jsonl), '
                  '[Endpoint-measurement samples](../artifacts/full/loss_audit/selection.json), '
                  '[Endpoint-measurement notes and Slurm job IDs](../artifacts/full/loss_audit/manifest.json). ',
                  '- [Full formal evaluation](FULL_TRAINING_RESULTS.md), [Previous alignment diagnosis](ALIGNMENT_DIAGNOSIS.md). '])
    path = ROOT / 'docs/TRAINING_LOSS_AUDIT.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path)


if __name__ == '__main__':
    main()
