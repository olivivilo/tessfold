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

    lines = ['# Encoder、训练数据和 loss 记录核验', '',
             '**正式训练使用作者预训练 MonkeyOCRv2-B（Base）；本次重新计算权重 SHA256，与下载源 manifest 完全一致。**', '',
             f'- 来源：`zenosai/MonkeyOCRv2-B`；本地 `models/MonkeyOCRv2-B/model.safetensors`。',
             f'- 权重内实际参数总数：**{parameters:,}**（约 113.7M，论文表格记为 113M）；12 层、12 个 attention heads、embed dim 768、patch size 14。',
             f'- 权重 revision：`{source["Revision"]}`。', f'- SHA256：`{weight_hash}`。',
             '- 视觉编码器两个阶段均冻结；先提取冻结特征，再训练 MLP / Qwen。特征提取完成标记保存相同权重 hash。'
             '这能确认权重身份与冻结路径；仅凭身份不能证明所有预处理、后端实现都与作者完全等价。', '',
             '## 正式训练究竟用了哪些数据', '',
             f'正式两阶段均使用相同的 **{total:,} 条公开 VQA 训练问答**，每阶段遍历一轮。'
             '**正式训练没有混入本地 MonkeyDoc v2 OCR/转写数据**。早期工程实验使用过 MonkeyDoc v2 合成表格派生 QA，'
             '该实验位于 `artifacts/alignment/`，与 `artifacts/full/` 本轮训练分开。', '',
             '| 数据集 | 实际训练问答 | 占比 | 来源 |', '|---|---:|---:|---|']
    for task, count in counts.most_common():
        repo = next(repo for t, repo in sources if t == task)
        lines.append(f'| {task} | {count:,} | {100 * count / total:.2f}% | {repo} |')
    lines.extend(['', '**`echo840/Monkey_Data` 是原 Monkey 的 VQA 数据来源，不能和 MonkeyDoc v2 预训练/OCR 数据混为一谈。**'
                  'OCRBench 只用于测试，不参与训练。DocVQA/InfoVQA 的公开 validation 留出用于评测。', '',
                  '## 历史 training loss：记录了什么', '',
                  '两个阶段各 **1 epoch、1,904 optimizer steps**，累计 3,808 次更新。'
                  '因此不存在同一训练阶段的多 epoch 收敛曲线，不能把每 100 个 step 改称一个 epoch。', '',
                  '![历史逐 step 训练 loss](../artifacts/full/loss_audit/training_loss.png)', '',
                  '| 阶段 | epoch 数 | 全阶段在线 loss 均值 | 前 100 step 均值 | 后 100 step 均值 |',
                  '|---|---:|---:|---:|---:|'])
    for e in epochs:
        lines.append(f"| Stage {e['stage']} | 1 | {e['online_rank0_mean_loss']:.4f} | {e['first100_mean_loss']:.4f} | {e['last100_mean_loss']:.4f} |")
    lines.extend(['', '原始来源 `artifacts/full/training.jsonl`。日志中的 loss 是 **rank 0 本地微批的均值**，'
                  '没有跨四个 rank 汇总；参数和样本随 step 改变。上述 epoch 均值是日志的算术均值，'
                  '并不是在 epoch 末固定模型上重新遍历全训练集得到的 loss。'
                  '图中灰色为原始 step，蓝色为尾随 100 step 均值；下降有波动，并非保证单调。', '',
                  '## Train / eval loss：补测的两个阶段终点', '',
                  '**当时的 validation 记录的是 ANLS / accuracy 等生成分数，没有保存 validation loss。**'
                  '中途检查点没有全部保留，无法补回真实的逐 100 step 或多 epoch eval-loss 历史。', '',
                  '本次加载已保存的 Stage 1 和 Stage 2 模型，在固定 **256 条 train + 256 条 val** 上实测答案 NLL。'
                  '两边均从 DocVQA、InfoVQA、ChartQA、DT-VQA 各取 64 张不同文档、每图一题，'
                  '以 SHA256 排序确定性抽样。相同任务权重有利于比较；不代表完整七任务训练混合的加权均值。'
                  '没有训练更新，没有使用 test。', '',
                  '![固定 train 与 validation 终点 loss](../artifacts/full/loss_audit/checkpoint_train_eval_loss.png)', '',
                  '| 保存模型 | Train NLL（256 题） | Validation NLL（256 题） |', '|---|---:|---:|'])
    for stage in ('1', '2'):
        s = endpoints['stages'][stage]
        lines.append(f"| Stage {stage} 末 | {s['train']['sample_mean_nll']:.4f} | {s['val']['sample_mean_nll']:.4f} |")
    lines.extend(['', 'NLL 是第一 gold answer＋EOS 的因果交叉熵，先按每题答案 token 取均值，再按题取均值，越低越好；'
                  '同时导出 token 加权结果。它与训练 objective 一致，但 teacher forcing loss 不等于自由生成准确率。'
                  '**图中只有两个实测终点，连线不代表中间测过，也不能保证中间持续下降。**', '',
                  '| Validation 任务（各 64 题） | Stage 1 NLL | Stage 2 NLL |', '|---|---:|---:|'])
    for task in ('DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA'):
        values = [endpoints['stages'][str(s)]['val']['tasks'][task]['sample_mean_nll'] for s in (1, 2)]
        lines.append(f'| {task} | {values[0]:.4f} | {values[1]:.4f} |')
    lines.extend(['', '这些记录说明训练 loss 总体下降，可核对两个保存模型的验证 loss 差异；'
                  '不能仅凭 loss 下降认定视觉文字读取与泛化已经学好。'
                  '接下来的训练应在固定验证子集上同时记录 eval NLL 与生成分数，保留足够检查点；'
                  '需要多 epoch 收敛曲线时，应实际运行多个 epoch，并保留阶段边界。', '',
                  '## 明确入口与导出文件', '', '```bash',
                  '# 补测保存模型的固定 train/val loss：NPU，venv/CANN 由脚本配置',
                  'sbatch scripts/slurm_loss_endpoints.sh',
                  '# 补测完成后重建权重/数据审计、曲线与本文档',
                  '.venv/bin/python -m pip install -r requirements-report.txt',
                  '.venv/bin/python -m scripts.report_training_audit', '```', '',
                  '- [历史训练曲线 PNG](../artifacts/full/loss_audit/training_loss.png)、'
                  '[SVG](../artifacts/full/loss_audit/training_loss.svg)、[PDF](../artifacts/full/loss_audit/training_loss.pdf)。',
                  '- [Train / eval loss PNG](../artifacts/full/loss_audit/checkpoint_train_eval_loss.png)、'
                  '[SVG](../artifacts/full/loss_audit/checkpoint_train_eval_loss.svg)、[PDF](../artifacts/full/loss_audit/checkpoint_train_eval_loss.pdf)。',
                  '- [逐 step CSV](../artifacts/full/loss_audit/training_steps.csv)、'
                  '[逐阶段 epoch 汇总 CSV](../artifacts/full/loss_audit/epoch_summary.csv)、'
                  '[Train / eval loss CSV](../artifacts/full/loss_audit/checkpoint_loss.csv)。',
                  '- [核验 JSON](../artifacts/full/loss_audit/training_audit.json)、'
                  '[补测逐题 loss](../artifacts/full/loss_audit/endpoint_records.jsonl)、'
                  '[补测样本](../artifacts/full/loss_audit/selection.json)、'
                  '[补测说明与 Slurm job ID](../artifacts/full/loss_audit/manifest.json)。',
                  '- [正式完整评测](FULL_TRAINING_RESULTS.md)、[先前对齐诊断](ALIGNMENT_DIAGNOSIS.md)。'])
    path = ROOT / 'docs/TRAINING_LOSS_AUDIT.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path)


if __name__ == '__main__':
    main()
