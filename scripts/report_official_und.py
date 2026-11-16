"""Score native official-model predictions and maintain a provenance-aware report."""
import argparse
import collections
import json
from pathlib import Path

from vqa.benchmark_metrics import due_field_f1, dtvqa_scores, native_score
from scripts.eval_official_und import OUT, ROOT, rows_for

PAPER = {'DocVQA': 79.3, 'InfoVQA': 46.3, 'DeepForm': 65.8, 'KLC': 38.2,
         'WTQ': 43.2, 'ChartQA': 62.0, 'DT-VQA': 64.3, 'OCRBench': 58.1}


def score(args):
    expected = rows_for(args.mode, args.limit_per_task)
    files = sorted(OUT.glob(f'{args.label}.rank*.jsonl'))
    assert files, 'No predictions'
    reference = {r['id']: r for r in expected}
    assert len(reference) == len(expected), 'Duplicate IDs in reference dataset'
    records = {}
    for path in files:
        assert path.with_suffix('.complete.json').exists(), f'Rank incomplete: {path}'
        completion = json.loads(path.with_suffix('.complete.json').read_text())
        assert completion['world'] == len(files), 'Missing or extra rank shards'
        rank = int(path.stem.rsplit('rank', 1)[1])
        shard_ids = set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['id'] not in records, f'Duplicate prediction {row["id"]}'
            gold = reference[row['id']]
            assert all(row[k] == gold[k] for k in ('question', 'answers', 'task', 'image', 'document_id'))
            source = str(args.adapter) if args.adapter else 'Author full MonkeyOCRv2-B-Und'
            assert row['source'] == source, f'Model source mismatch: {row["source"]} vs {source}'
            assert abs(native_score(row['prediction'], row) - row['score']) < 1e-8
            records[row['id']] = row
            shard_ids.add(row['id'])
        assert len(shard_ids) == completion['samples']
        assert shard_ids == {r['id'] for r in expected[rank::len(files)]}, 'Rank partition differs from reference'
    assert set(records) == {r['id'] for r in expected}, f'Coverage {len(records)}/{len(expected)}'
    groups = collections.defaultdict(list)
    for row in records.values():
        groups[row['task']].append(row)
    result = {'label': args.label, 'mode': args.mode, 'limit_per_task': args.limit_per_task,
              'samples': len(records), 'adapter': args.adapter, 'benchmarks': {},
              'pipeline': 'Author AutoProcessor + process_vision_info + full remote-code model.generate',
              'hardware_choice': 'Ascend BF16, author eager vision attention, Qwen SDPA, greedy, independent questions',
              'hit_token_limit': sum(r['runtime']['hit_token_limit'] for r in records.values())}
    for task, rows in groups.items():
        item = {'samples': len(rows), 'score_percent': 100 * sum(r['score'] for r in rows) / len(rows),
                'metric': 'ANLS' if task in ('DocVQA', 'InfoVQA') else 'AccANLS' if task == 'DT-VQA' else
                'relaxed_accuracy' if task == 'ChartQA' else 'WTQ denotation accuracy' if task == 'WTQ' else 'OCRBench_accuracy'}
        if task in ('DeepForm', 'KLC'):
            item.update(due_field_f1(rows))
        if task == 'DT-VQA':
            scores = [dtvqa_scores(r['prediction'], r['answers']) for r in rows]
            item['component_scores'] = {k: 100 * sum(r[k] for r in scores) / len(scores) for k in ('ANLS', 'accuracy', 'accanls')}
        if task in ('ChartQA', 'OCRBench'):
            by_type = collections.defaultdict(list)
            for r in rows:
                by_type[r.get('question_type', 'unspecified')].append(r)
            item['by_type'] = {k: {'samples': len(v), 'score_percent': 100 * sum(r['score'] for r in v) / len(v)} for k, v in by_type.items()}
        result['benchmarks'][task] = item
    result['task_macro_mean_percent'] = sum(r['score_percent'] for r in result['benchmarks'].values()) / len(groups)
    (OUT / f'{args.label}_metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))
    return result


def report():
    def get(label):
        path = OUT / f'{label}_metrics.json'
        return json.loads(path.read_text()) if path.exists() else None
    baseline, tuned = get('baseline_full'), get('finetuned_full')
    val0, val1 = get('baseline_validation'), get('finetuned_validation')
    lines = ['# 官方 MonkeyOCRv2-B-Und 基线与微调', '',
             '**本实验从作者已对齐的完整 Und 权重开始；官方基线、在此基础上的 LoRA 微调、此前自行对齐模型的成绩分别报告。**', '',
             '主入口：`scripts/run_official_und.sh`；官方源代码：`vendor/MonkeyOCRv2/understanding/`；'
             '官方训练框架：`vendor/MonkeyOCRv2/parsing/train/ms-swift/`。', '',
             '原生推理保留作者 AutoProcessor、process_vision_info、ChatML、图像 token 替换和完整模型 generate。'
             'Ascend 选择作者已有 eager 视觉 attention、Qwen SDPA；BF16，max pixels 1,003,520，'
             'greedy、每题独立生成、最多 256 个新 token，明确启用新题独立 KV cache。', '',
             '| 任务 | 论文表 8 | 官方 Und 基线 | 官方 Und＋本次微调 |', '|---|---:|---:|---:|']
    if baseline:
        lines[4:4] = [f"**完整官方基线：{baseline['samples']:,} 题，八任务均分 {baseline['task_macro_mean_percent']:.4f}；论文报告 57.2。**"
                      '这是官方成品权重复测，非从零重新对齐的成绩；评测协议差异见下文。', '']
    if baseline and tuned:
        lines[6:6] = [f"**微调完成：30,000 条训练问答、1 epoch、235 步；完整八任务均分 {tuned['task_macro_mean_percent']:.4f}，"
                      f"较基线变化 {tuned['task_macro_mean_percent'] - baseline['task_macro_mean_percent']:+.4f}。**"
                      '完整评测上有小幅提升；固定小验证集未改善，模型选择仍保留 baseline，两个结果同时保留。', '']
    for task, paper in PAPER.items():
        values = [f"{r['benchmarks'][task]['score_percent']:.2f}" if r and task in r['benchmarks'] else '尚未完成' for r in (baseline, tuned)]
        lines.append(f'| {task} | {paper:.1f} | {values[0]} | {values[1]} |')
    means = [f"{r['task_macro_mean_percent']:.2f}" if r else '尚未完成' for r in (baseline, tuned)]
    lines.append(f'| 八任务均分 | 57.2 | {means[0]} | {means[1]} |')
    if baseline:
        cap = str(tuned['hit_token_limit']) if tuned else '待完成'
        lines.extend(['', f"生成触及 256-token 上限的题数：官方基线 **{baseline['hit_token_limit']} / {baseline['samples']}**；"
                      f'微调 **{cap}**。原始截断输出保留，未人工补答或改写。'])
    lines.extend(['', 'DocVQA/InfoVQA 使用公开 validation；其他六项使用本地既有 test。'
                  'DeepForm/KLC 保留原 Monkey 单页 QA 到 DUE 的转换协议。'
                  '因此与论文隐藏 test / 文档级细节可能不同，不能宣称完全同协议。'
                  '作者原始训练集是否与这些公开 split 重叠未能完整审计；本次新增训练严格只取公开 train。', '',
                  '## 微调方案与选择依据', '',
                  '作者发布仓库内的 ms-swift、`monkeyocrv2_und` 模板和官方 LoRA 方案；'
                  'rank 8、alpha 32、all-linear，encoder 冻结，LLM 与 aligner 允许 LoRA；'
                  'LR 1e-5、batch 128、1 epoch、5% warmup＋cosine。'
                  '关闭 CUDA flash-only padding_free 和 DeepSpeed，使用 Ascend DDP 和梯度累积；这些硬件适配单独记录。', '',
                  '固定 30,000 条 train：DocVQA 8,000，InfoVQA/ChartQA/WTQ/KLC/DT-VQA 各 4,000，DeepForm 2,000。'
                  '固定 validation 256 条：DocVQA/InfoVQA/ChartQA/DT-VQA 各 64 张不同文档一题。'
                  '每 50 optimizer steps 记录 eval loss 并保存检查点，按 validation loss 选候选模型。'
                  '再以相同 256 题的生成指标比较原始权重与候选，测试集不参与选择。', '',
                  '| 同一固定验证集 | 原始 Und | 微调候选 |', '|---|---:|---:|'])
    tasks = ('DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA')
    for task in tasks:
        values = [f"{r['benchmarks'][task]['score_percent']:.2f}" if r else '尚未完成' for r in (val0, val1)]
        lines.append(f'| {task} | {values[0]} | {values[1]} |')
    means = [f"{r['task_macro_mean_percent']:.2f}" if r else '尚未完成' for r in (val0, val1)]
    lines.append(f'| 四任务均分 | {means[0]} | {means[1]} |')
    selection = OUT / 'model_selection.json'
    if selection.exists():
        s = json.loads(selection.read_text())
        lines.extend(['', f"根据固定验证生成分数选择：**{s['selected']}**；{s['reason']}。"])
        if s['selected'] == 'baseline':
            lines.extend(['', '本次 LoRA 的 validation NLL 改善，但固定验证集生成均分下降，因此未选择该 adapter。'
                          '这只能说明本次数据混合／超参数下的继续微调未取得整体验证收益；'
                          '不能归纳为所有 LLM 微调都无效，也不能归因于 encoder 容量。'])
            if tuned:
                lines.extend(['', '完整评测与固定小验证集的变化方向不同，ChartQA 和 DT-VQA 尤其明显。'
                              '每项仅 64 题的小验证集存在抽样不确定性；本轮保留其既定选择结果，'
                              '不根据测试分数追选模型。完整微调模型和 adapter 仍保留，便于后续独立验证。'])
    lines.extend(['', '## 新发现：官方模型附带的训练线索', '',
                  '`models/MonkeyOCRv2-B-Und/args.json` 包含公式、表格、文本转写、块识别与多个 QA 数据路径，'
                  '并记录从 checkpoint-4688 继续训练、LR 2e-5 等信息。'
                  '这些是发布包内的训练历史线索；文件名不能还原数据内容、准确配比或完整两阶段，'
                  '也不能证明该 args 文件逐项对应论文表 8。它更正了此前“没有公开训练数据线索”的过强表述。', '',
                  '## 复现与证据', '', '```bash',
                  'bash scripts/run_official_und.sh',
                  '# 单图原生推理见 docs/OFFICIAL_UND.md', '```', '',
                  '- 权重：`models/MonkeyOCRv2-B-Und/source_manifest.json` 和 `VERIFIED`，逐文件固定 revision 与 SHA256。',
                  '- 数据：`data/official_und/manifest.json`，训练和验证 JSONL 为官方 messages/images 格式。',
                  '- 原始作答：`artifacts/official_und/{label}.rank*.jsonl`，包含 gold、预测、分数、token 数和模型来源。',
                  '- 微调日志：`artifacts/official_und/sft/`；训练 train/eval loss 与 epoch 在 trainer_state / logging.jsonl 中保留。',
                  '- [真实 train/eval loss 曲线与 DocVQA gold/作答对比](OFFICIAL_UND_TRAINING.md)。',
                  '- 官方版本与硬件适配说明：[OFFICIAL_UND.md](OFFICIAL_UND.md)。',
                  '- 旧自行对齐结果：[FULL_TRAINING_RESULTS.md](FULL_TRAINING_RESULTS.md)，不会被覆盖。'])
    (ROOT / 'docs/OFFICIAL_UND_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label')
    parser.add_argument('--mode', default='benchmark')
    parser.add_argument('--limit-per-task', type=int, default=0)
    parser.add_argument('--adapter')
    args, _ = parser.parse_known_args()
    if args.label:
        score(args)
    report()


if __name__ == '__main__':
    main()
