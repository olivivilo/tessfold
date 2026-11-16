"""Summarize completed alignment diagnostics without changing formal eval results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/full/alignment_diagnosis'


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    overfit = json.loads((OUT / 'overfit/summary.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    # Predictions are append-only; use the most recent record per measured case.
    records = {}
    for line in (OUT / 'predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        records[row['stage'], row['split'], row['id']] = row
    for stage in (1, 2):
        for split in ('train', 'val'):
            rows = [records[stage, split, row['id']] for row in selection[split]]
            actual = summary['stages'][str(stage)][split]
            assert len(rows) == actual['samples'] == 32
            assert abs(100 * sum(row['correct_image']['anls'] for row in rows) / 32 - actual['correct_image_anls_percent']) < 1e-8
            assert abs(sum(row['wrong_minus_correct_nll'] for row in rows) / 32 - actual['mean_wrong_minus_correct_nll']) < 1e-8
    for branch, result in overfit.items():
        final = result['final']
        assert final == json.loads((OUT / f"overfit/{branch}_step{final['step']}.json").read_text())
        assert final['exact_lowercase'] == sum(any(row['prediction'].lower() == a.lower() for a in row['gold_answers']) for row in final['records'])

    lines = [
        '# 对齐失败诊断与下一步', '',
        '**优先排查视觉对齐与训练方案；当前证据不支持先换大 LLM，也不支持简单归咎于解冻 Qwen。**'
        '本轮完整 DocVQA validation ANLS 仍为 11.52；以下新增结果均为诊断，不替代正式 benchmark。', '',
        '论文第 4.7 节、表 8 使用同一个 Qwen3-1.7B，第二阶段明确联合训练 MLP 与 LLM。'
        '因此 1.7B 并不是论文目标分数的已知容量障碍；原文方法也并非全程冻结 LLM。'
        '但本项目尚未还原作者未公开的准确训练混合和训练日程，不能仅凭相同骨干、LR 和 batch 宣称完整协议一致。', '',
        '## 同题比较 Stage 1 与 Stage 2', '',
        '从实际 DocVQA train 和 val 各确定性选择 32 张不同文档、每图一题。'
        '同一批问题分别运行 Stage 1（原始 Qwen＋对齐后的 MLP）和 Stage 2（联合训练后的 Qwen＋MLP）。'
        '本实验没有更新权重，没有使用 test。', '',
        '| 实际模型 | 训练题 ANLS (%) | 验证题 ANLS (%) | 训练题答案 NLL | 验证题答案 NLL |',
        '|---|---:|---:|---:|---:|',
    ]
    for stage, label in [('1', 'Stage 1：仅 MLP'), ('2', 'Stage 2：MLP＋Qwen')]:
        train, val = (summary['stages'][stage][split] for split in ('train', 'val'))
        lines.append(f"| {label} | {train['correct_image_anls_percent']:.2f} | {val['correct_image_anls_percent']:.2f} | {train['mean_correct_image_answer_nll']:.3f} | {val['mean_correct_image_answer_nll']:.3f} |")
    lines.extend([
        '', '训练题本身也答得较差，不能将现象概括为“训练集已学好，只是验证集过拟合”。'
        'Stage 2 的生成分数和正确答案概率在这组 train/val 中均有改善。'
        '原先固定每任务 64 题的 validation 中，DocVQA 也由 11.77 升至 15.98，其他三个任务均提高。'
        '这些小样本结果不能证明所有解冻设置都有效，也不构成冻结与解冻在相同训练预算下的严格消融。', '',
        '## 正确图像是否让 gold 更可信', '',
        '每题保持问题与 gold 不变，换用同 split 中另一张真实文档，优先匹配视觉 token 数。'
        'NLL 只计算第一 gold 与 EOS，越低越好；token accuracy 使用 teacher forcing，不能当作自由生成准确率。', '',
        '| 模型 / 样本 | 正常图 ANLS (%) | 换错图 ANLS (%) | 错图 NLL − 正常图 NLL | 正常图 NLL 更低的题数 |',
        '|---|---:|---:|---:|---:|',
    ])
    for stage in ('1', '2'):
        for split, name in [('train', '训练'), ('val', '验证')]:
            r = summary['stages'][stage][split]
            lines.append(f"| Stage {stage} / {name} 32 题 | {r['correct_image_anls_percent']:.2f} | {r['wrong_image_anls_percent']:.2f} | {r['mean_wrong_minus_correct_nll']:.3f} | {r['correct_image_lower_nll_samples']}/32 |")
    lines.extend([
        '', '模型确实利用了一部分图像信息，联合训练后平均 gold 概率的图像依赖增强，但正常图的作答仍很差。'
        '此结果不能证明细粒度文字读取已学会，也不能单独定位 encoder、MLP 或优化器中的某一环。'
        'train 有 28/32、val 有 29/32 对照的视觉 token 数完全相同；其余长度差异保留在逐条记录中。', '',
        '## 8 图同问题训练检查', '',
        '仅从 DocVQA **训练集**选择 8 张图，8 个不同页码；问题统一为 `What is the page number?`。'
        '因为问题完全相同，模型必须区分图像才能分别答对。两条独立分支均从 Stage 1 MLP＋原始 Qwen 开始，'
        'encoder 特征始终冻结。每分支最多 80 个 optimizer step，effective batch 8，每 20 步生成评测，8/8 正确则提前停止。', '',
        '| 训练分支 | LR | 更新步 | 初始正确 | 最终正确 | 最终答案 NLL | 换图后答 donor gold |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for branch, label in [('mlp_only', '仅 MLP'), ('mlp_and_llm', 'MLP＋Qwen')]:
        r = overfit[branch]
        initial, final = r['initial'], r['final']
        lines.append(f"| {label} | {r['lr']:g} | {r['steps']} | {initial['exact_lowercase']}/8 | {final['exact_lowercase']}/8 | {final['mean_answer_nll']:.4g} | {final['wrong_image_matches_donor_gold']}/8 |")
    mlp_pass = overfit['mlp_only']['final']['exact_lowercase'] == 8
    joint_pass = overfit['mlp_and_llm']['final']['exact_lowercase'] == 8
    if mlp_pass or joint_pass:
        passed_label = '两个分支都' if mlp_pass and joint_pass else '至少一个分支'
        lines.extend(['', passed_label + '能记住这 8 个图像—答案对应，说明当前特征→MLP→Qwen 的路径有可训练的图像区分能力。'
        '这排除了“该路径完全无法学习”的强说法，**不证明通用 OCR、定位或未见文档泛化能力**。'
        '模型也可能记忆版面、特征模式或长度等捷径，而不是真正读出页码。'])
    else:
        lines.extend(['', '本次有限步数与两组 LR 下仍未达到 8/8，首先需要继续检查小样本训练收敛、梯度与数值路径。'
                      '有限的训练预算不能证明架构不可训练或 LLM 容量不足。'])
    lines.extend([
        '', '两分支 LR 不同，且这是极小训练集的记忆实验；不能据此给冻结/解冻做泛化优劣排名。'
        '正式模型和正式 eval 文件没有被修改，本实验模型不保存为候选正式结果。', '',
        '| 题目 ID | Gold（第一项） | 仅 MLP 最终作答 | MLP＋Qwen 最终作答 |',
        '|---|---|---|---|',
    ])
    joint = {r['id']: r for r in overfit['mlp_and_llm']['final']['records']}
    def cell(text):
        return str(text).replace('|', '\\|').replace('\n', '<br>')
    for row in overfit['mlp_only']['final']['records']:
        lines.append(f"| {row['id']} | {cell(row['gold_answers'][0])} | {cell(row['prediction'])} | {cell(joint[row['id']]['prediction'])} |")
    lines.extend([
        '', '## 现在怎么推进', '',
        '1. 保留作者 encoder 和 Qwen3-1.7B，先在独立输出目录做 DocVQA 单任务对照。固定训练子集、'
        '独立的 validation 文档、训练步数和生成方式，比较继续仅训 MLP、MLP＋小 LR 全量 Qwen、MLP＋LoRA；'
        '按 validation 的 ANLS 与正确/错图差异选择方案，不按训练 loss 或 test 分数选模型。',
        '2. 如果小样本可以记忆但未见文档仍差，优先检查监督是否足以教会读字和定位。'
        '当前混合中 DT-VQA 为 120,444/243,652（约 49.4%），DocVQA 为 39,457（约 16.2%）；'
        '比例值得控制实验，但尚未证明是原因。可将带转写监督的 OCR 对齐 warmup 作为独立实验，'
        '再加入 VQA，并检查分辨率、特征尺度、模板和答案终止标记；这些是待验证方案，不是原文已披露配方。',
        '3. 若冻结对照在相同步数下的 validation 持续优于解冻，才调整解冻时机、LR 或 LoRA。'
        '当前结果不支持直接取消第二阶段。只有可靠的图像读取与对齐基线建立后，再用更大 LLM 做容量对照。', '',
        '本轮确定了排查优先级，尚未把论文差距归因于单一原因，也尚未达到论文成绩。'
        '缺失的作者训练数据/日程，以及数值后端、预处理与原生实现的完整等价性，仍需后续受控验证。', '',
        '## 复现入口与证据', '',
        '```bash',
        '# NPU 计算节点，venv/CANN 由脚本配置；不要同时重复提交同一个诊断',
        'sbatch scripts/slurm_alignment_diagnosis.sh',
        'sbatch scripts/slurm_overfit_probe.sh',
        '# 两项均成功完成后重建本文档，并核对逐条指标',
        '.venv/bin/python -m scripts.report_alignment_diagnosis',
        '```', '',
        '- 训练/验证对照：Slurm `147042`，实际 128 个样本模型组合、256 次生成。',
        '- 8 图训练检查：Slurm `147044`。首次 `147043` 在更新前因诊断脚本使用 inference-mode 张量缓存失败；'
        '改成 no-grad 评测后重跑。正式训练代码原本即使用 no-grad，不涉及这一诊断脚本问题。',
        '- [selection.json](../artifacts/full/alignment_diagnosis/selection.json)：固定 train/val 样本。'
        '首次自动创建，重跑会核对原始数据中的确定性选择是否相同。',
        '- [summary.json](../artifacts/full/alignment_diagnosis/summary.json)、'
        '[predictions.jsonl](../artifacts/full/alignment_diagnosis/predictions.jsonl)：逐题正确/错图作答、gold、NLL、token 准确率。',
        '- [8 图汇总](../artifacts/full/alignment_diagnosis/overfit/summary.json)、'
        '[8 图样本](../artifacts/full/alignment_diagnosis/overfit/selection.json)、'
        '[逐步训练记录](../artifacts/full/alignment_diagnosis/overfit/training.jsonl)：分支初始化、学习率、训练步和评测明细。',
        '- [跨问答上下文调查](CONTEXT_AUDIT.md)：已完成的 226 题输入隔离证据。',
        '- [正式评测结果](FULL_TRAINING_RESULTS.md)、[DocVQA 全部作答对照](DOCVQA_REVIEW.md)。',
    ])
    path = ROOT / 'docs/ALIGNMENT_DIAGNOSIS.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path)


if __name__ == '__main__':
    main()
