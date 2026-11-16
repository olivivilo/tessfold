"""Create a report exclusively from measured run artifacts."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/alignment.json')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    out = ROOT / cfg['output_dir']
    results = json.loads((out / 'results.json').read_text())
    logs = [json.loads(x) for x in (out / 'train.jsonl').read_text().splitlines()]
    environment = json.loads((out / 'environment.json').read_text())
    lines = ['# 实际运行结果', '',
             '这些结果来自本地 MonkeyDocv2 英文 nonsemantic 表格构造的问答子集。'
             '不是论文八个标准 benchmark 的复现分数；小样本实验用于验证基础对齐、训练、生成和保存恢复闭环。', '',
             '| 阶段/集合 | 问题数 | EM (%) | ANLS (%) | Relaxed accuracy (%) | Answer NLL |',
             '|---|---:|---:|---:|---:|---:|']
    for key in ('baseline_test', 'stage1_val', 'final_test'):
        m = results[key]
        lines.append(f"| {key} | {m['samples']} | {m['exact_match']:.2f} | {m['anls']:.2f} | {m['relaxed_accuracy']:.2f} | {m['answer_nll']:.4f} |")
    lines.extend(['', '最终测试集按任务分解：', '',
                  '| 任务 | 问题数 | EM (%) | ANLS (%) | Relaxed accuracy (%) |',
                  '|---|---:|---:|---:|---:|'])
    for task, m in results['final_test']['by_task'].items():
        lines.append(f"| {task} | {m['samples']} | {m['exact_match']:.2f} | {m['anls']:.2f} | {m['relaxed_accuracy']:.2f} |")
    lines.extend(['', f"运行环境：`{environment['device']}` / `{environment['dtype']}`，{cfg['threads']} CPU threads。",
                  f"视觉编码器冻结；MLP 参数 {environment['projector_parameters']:,}；语言模型参数 {environment['language_parameters']:,}。",
                  f"输入上限 {cfg['max_pixels']} pixels；greedy decoding；禁用 Qwen3 thinking；最多生成 {cfg['max_new_tokens']} tokens。", '',
                  '训练阶段记录：', ''])
    if environment.get('slurm_job_id'):
        lines.insert(-2, f"Slurm job `{environment['slurm_job_id']}`，节点 `{environment['hostname']}`，申请 16 CPU / 64 GB RAM。")
    for stage in (1,2):
        rows = [x for x in logs if x['stage']==stage]
        if rows:
            lines.append(f"- Stage {stage}: {len(rows)} optimizer steps，LR {rows[0]['lr']}，"
                         f"首/末训练样本 loss {rows[0]['loss']:.4f}/{rows[-1]['loss']:.4f}，"
                         f"用时 {rows[-1]['elapsed_seconds']:.1f}s。不同 step 对应不同样本，不能据此直接计算收敛幅度。")
    baseline, final = results['baseline_test'], results['final_test']
    lines.extend(['', f"相同测试集的 token 加权 Answer NLL：{baseline['answer_nll']:.4f} → {final['answer_nll']:.4f}。",
                  'Answer NLL 使用 teacher forcing；生成 EM/ANLS 使用独立 greedy generation，不向模型提供参考答案。', '',
                  f"Checkpoint 保存/恢复一致性：{results['checkpoint_roundtrip']['passed']}（一个留出样本的 greedy 输出一致）。", '',
                  '原始证据（相对于仓库根目录）：', '',
                  f"- `{cfg['output_dir']}/results.json`：全部实际指标。",
                  f"- `{cfg['output_dir']}/final_test_predictions.jsonl`：逐条预测、参考答案、指标。",
                  f"- `{cfg['output_dir']}/train.jsonl`：全部优化步的 loss/梯度范数。",
                  f"- `artifacts/{Path(args.config).stem}_pipeline.log`：完整运行输出（smoke 初次日志为 `artifacts/pipeline.log`）。", '',
                  '论文表 8 的 MonkeyOCRv2-B 指标为 Overall 57.2、DocVQA 79.3、InfoVQA 46.3、DF 65.8、'
                  'KLC 38.2、WTQ 43.2、ChartQA 62.0、DT-VQA 64.3、OCRBench 58.1。这些是论文引用值，'
                  '未在本次运行中测量，不能与本子集直接比较。', '',
                  '低分或者零分如实保留。32 张表格、低分辨率、极少优化步不能证明文档理解能力已达到论文水平。'])
    predictions = [json.loads(line)['prediction'] for line in (out / 'final_test_predictions.jsonl').read_text().splitlines()]
    if len(set(predictions)) == 1:
        lines.extend(['', f"本次所有测试问题都生成 `{predictions[0]}`，出现固定回答退化。"
                      'NLL 降低只说明优化改变了答案分布，不能证明视觉与语言已经有效对齐。'
                      '因此，本次完成的是可运行的两阶段工程闭环，模型尚未获得可用的 VQA 能力。'])
    verification = out / 'verification.json'
    if verification.exists():
        v = json.loads(verification.read_text())
        lines.extend(['', f"独立检查：指标从逐条预测重新计算通过 `{v['metrics_recalculated']}`；"
                      f"单图 `infer` 入口重新读取原图、提取视觉特征和加载 stage-2 模型通过 `{v['public_infer_entry']}`。",
                      f"证据：`{cfg['output_dir']}/verification.json`。"])
        if 'llm_weight_check' in v:
            w = v['llm_weight_check']
            lines.extend(['', f"联合微调实际权重更新核对：`{w['tensor']}` 的平均绝对更新量为 {w['mean_absolute_update']:.8f}，"
                          f"最大绝对更新量为 {w['max_absolute_update']:.8f}；stage-2 MLP 也与 stage-1 不同。"])
    (ROOT / 'docs/EVAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(ROOT / 'docs/EVAL_RESULTS.md')

if __name__ == '__main__':
    main()
