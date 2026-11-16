"""Report only results actually produced by the formal run; never substitute paper scores."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/full'
PAPER={'DocVQA':79.3,'InfoVQA':46.3,'DeepForm':65.8,'KLC':38.2,'WTQ':43.2,'ChartQA':62.0,'DT-VQA':64.3,'OCRBench':58.1}

def main():
    cfg=json.loads((ROOT/'configs/full.json').read_text())
    manifest=json.loads((ROOT/'data/full/manifest.json').read_text()) if (ROOT/'data/full/manifest.json').exists() else {}
    status=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {'state':'waiting_for_resources_or_preparation'}
    if not (OUT/'status.json').exists() and (ROOT/'data/full/ALL_TASKS_READY').exists() and (ROOT/'artifacts/lm_backend_verification.json').exists():
        status={'state':'waiting_for_accelerator_resources'}
    lines=['# 正式训练与标准评测', '',f"当前状态：`{status['state']}`。", '',
           '配置：MonkeyOCRv2-B 原始预训练骨干，逐 token 两层 MLP，原始 Qwen3-1.7B；视觉骨干冻结。'
           '第一阶段 LR 1e-3，仅训练 MLP；第二阶段 LR 1e-5，全参数更新 MLP 和 LLM；effective batch 128，max pixels 1003520。', '',
           '第一阶段和第二阶段各遍历完整公开训练混合数据一轮；采用 5% warmup + cosine。'
           '论文没有披露准确训练混合比例和两阶段的训练轮数，因此这些选择不代表作者未公开的原始设置。', '',
           '训练入口：`sbatch scripts/slurm_full_npu.sh`。自动执行分布式特征提取、两阶段训练、完整 heldout 评测和本报告生成。'
           '重跑会恢复 `artifacts/full/resume/` 的模型、优化器、scheduler 和 optimizer step。', '',
           '| Benchmark | 本次 split | 实测分数 (%) | 指标 | 论文表 8 引用值 (%) |',
           '|---|---|---:|---|---:|']
    measured={}
    for split in ['val','test']:
        p=OUT/f'final_{split}_metrics.json'
        if p.exists():
            for task,result in json.loads(p.read_text())['benchmarks'].items():
                if task not in measured or split=='test':measured[task]=(split,result)
    for task,paper in PAPER.items():
        if task in measured:
            split,result=measured[task];lines.append(f"| {task} | {split} | {result['score_percent']:.2f} | {result['metric']} | {paper} |")
        else:lines.append(f'| {task} | 尚未完成 | 待实际评测 | — | {paper} |')
    if len(measured)==8:
        average=sum(r['score_percent'] for _,r in measured.values())/8
        lines.extend(['',f'上述八个实际分数的算术平均为 {average:.2f}%。'
                      'DocVQA/InfoVQA 使用公开 validation，DeepForm/KLC 使用原 Monkey 的单页 QA 转换，'
                      '与论文测试集、准确处理细节可能不同，不能称为论文表 8 完全同协议复现。'])
        if average<PAPER.get('Overall',57.2)-10:
            lines.extend(['','**本轮训练和工程闭环已完成，但实际表现远低于论文；尚未实现接近论文的有效对齐目标。**'
                          '闭环核验通过仅说明训练、保存、重载和评分一致，不能证明对齐效果达标。'])
    if (ROOT/'docs/DOCVQA_REVIEW.md').exists():
        lines.extend(['','## DocVQA 作答与 Gold 对照','',
                      '入口：[对照说明与前 60 条样例](DOCVQA_REVIEW.md)。'
                      '全部 5,349 条问答：[可筛选看图页面](../artifacts/full/docvqa_review.html)、'
                      '[完整 TSV](../artifacts/full/docvqa_comparison.tsv)、'
                      '[完整 JSONL](../artifacts/full/docvqa_comparison.jsonl)。'
                      '保留原文 prediction、全部 gold answers、原始图像路径与预测文件行号。'])
    if (ROOT/'docs/CONTEXT_AUDIT.md').exists():
        lines.extend(['','[固定人名作答与跨问答上下文调查](CONTEXT_AUDIT.md)：'
                      '完整 226 题换序/清空特征缓存、KV 初始长度、独立进程、关闭 KV cache 及换图对照证据。'])
    if (ROOT/'docs/ALIGNMENT_DIAGNOSIS.md').exists():
        lines.extend(['','[对齐失败诊断与下一步](ALIGNMENT_DIAGNOSIS.md)：'
                      'Stage 1/2 在相同训练题和验证题上的比较、正确/错图对照，以及 8 图同问题的实际训练检查。'])
    if (ROOT/'docs/TRAINING_LOSS_AUDIT.md').exists():
        lines.extend(['','[Encoder、训练数据和 loss 曲线核验](TRAINING_LOSS_AUDIT.md)：'
                      '作者 Base 预训练权重身份、实际 VQA 数据配比、历史训练曲线与事后补测的固定 train/validation loss。'])
    if (ROOT/'docs/PAPER_REPRODUCTION_GAPS.md').exists():
        lines.extend(['','[论文原方案与本轮训练的边界](PAPER_REPRODUCTION_GAPS.md)：'
                      '正文规定、未公开训练配方，以及官方 Und 继续微调示例与从原始组件对齐的区别。'])
    lines.extend(['', '所有论文分数只用于引用对照；“待实际评测”不会填入估计值或成品模型分数。', '',
                  '完整数据划分与数量：', '', '```json',json.dumps(manifest,indent=2,ensure_ascii=False),'```', '',
                  '证据文件：', '', '- `data/full/download_manifest.json`：下载版本、文件大小、LFS SHA256。',
                  '- `data/full/manifest.json`：完整数据数量及训练/评测图像去重策略。',
                  '- `artifacts/full/training.jsonl`：每个 optimizer step 的 loss、LR、梯度范数。',
                  '- `artifacts/full/final_{val,test}.rank*.jsonl`：逐条原始生成与参考答案。',
                  '- `artifacts/full/final_{val,test}_metrics.json`：完整标准任务指标。',
                  '- `artifacts/full/stage{1,2}/`：训练检查点。',
                  '- `artifacts/full/queue.json`：本次 Slurm 提交记录。', '',
                  '评测：DocVQA/InfoVQA 使用原始字符 ANLS；DT-VQA 使用原论文推荐 AccANLS，并同时保留 ANLS 和 accuracy；ChartQA 使用 5% relaxed accuracy 并分人类/合成问题；'
                  'WTQ 使用 DUE 官方 denotation scorer；DeepForm/KLC 使用 DUE 官方 corpus set F1，'
                  '将原 Monkey per-field QA 转为 DU-schema、None 视为缺失字段；OCRBench 使用原始 substring scorer 并按任务分解。'])
    diagnostics={}
    for stage in (1,2):
        p=OUT/f'stage{stage}_alignment_val_metrics.json'
        if p.exists():diagnostics[stage]=json.loads(p.read_text())['benchmarks']
    if diagnostics:
        lines.extend(['','## MLP-only 与联合对齐的实际对照','',
                      '两阶段使用同一组固定 validation 问题（每任务最多 64 条）；此表是对齐诊断，不替代上面的完整 benchmark。','',
                      '| 任务 | Stage 1：仅 MLP (%) | Stage 2：MLP + Qwen (%) |','|---|---:|---:|'])
        for task in sorted({t for results in diagnostics.values() for t in results}):
            values=[f"{diagnostics[s][task]['score_percent']:.2f}" if task in diagnostics.get(s,{}) else '待评测' for s in (1,2)]
            lines.append(f'| {task} | {values[0]} | {values[1]} |')
    verification=OUT/'verification.json'
    if verification.exists():
        result=json.loads(verification.read_text())
        lines.extend(['','## 完整闭环核验','',
                      f"原始 encoder 权重校验、完整预测覆盖、逐条评分与汇总重算：`{result['passed']}`。",
                      f"保存模型的单图入口核验：`{result.get('public_infer_entry',{}).get('passed',False)}`。",
                      '详细证据：`artifacts/full/verification.json`。'])
    (ROOT/'docs/FULL_TRAINING_RESULTS.md').write_text('\n'.join(lines)+'\n')
    readme=ROOT/'README.md'
    if readme.exists():
        text=readme.read_text()
        start,end='<!-- FULL_STATUS_START -->','<!-- FULL_STATUS_END -->'
        if start in text and end in text:
            if status['state']=='complete':
                banner='正式两阶段训练与完整评测已完成。实测指标见 [docs/FULL_TRAINING_RESULTS.md](docs/FULL_TRAINING_RESULTS.md)。'
                if len(measured)==8:
                    banner+=f'本轮八项平均 {average:.2f}%，论文引用值 57.2%；尚未达到接近论文结果的目标。'
            elif status['state'] in ('failed','interrupted'):
                banner=f"正式任务状态：`{status['state']}`。检查点可恢复；错误与进度见 `artifacts/full/status.json` 及 Slurm 日志。"
            elif status['state'] in ('training','caching','evaluating','verifying'):
                banner=f"正式任务状态：`{status['state']}`。完整指标尚未全部生成；实际进度见 `artifacts/full/status.json`。"
            elif status['state'] in ('waiting_for_accelerator_resources','cached_waiting_for_training'):
                banner='全量数据与数值/恢复/保存模型推理验证已通过，正式任务等待 NPU 训练资源；尚未产生正式 eval 分数。任务记录：`artifacts/full/queue.json`。'
            else:
                banner='全量数据已准备，正式任务已提交并等待 NPU 验证和资源；尚未产生正式 eval 分数。任务记录：`artifacts/full/queue.json`。'
            before=text.split(start,1)[0];after=text.split(end,1)[1]
            readme.write_text(before+start+'\n'+banner+'\n'+end+after)
    print(ROOT/'docs/FULL_TRAINING_RESULTS.md')

if __name__=='__main__':main()
