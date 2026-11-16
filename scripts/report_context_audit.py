"""Render the context audit report from measured diagnostics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/full/context_audit'


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    runtime = load('runtime_summary.json')
    group = load('full_group_summary.json')
    prior = load('training_prior.json')
    sequence = load('sequence_statistics.json')
    features = load('shank_feature_audit.json')
    all_rows = [json.loads(line) for line in (ROOT / 'artifacts/full/docvqa_comparison.jsonl').read_text().splitlines()]
    lookup = {r['id']:r for r in all_rows}
    n = group['samples']
    target = 'Dr. Robert E. Shank'
    baseline_targets = [k for k,v in runtime['per_sample'].items() if v['baseline'] == target]
    stayed = sum(runtime['per_sample'][k]['wrong_image'] == target for k in baseline_targets)
    lines = ['# DocVQA 固定人名作答：跨问答上下文串扰调查', '',
             '**结论：本次代码检查与实际模型实验未发现跨问答的历史上下文或 KV cache 被带入下一题。'
             '全部 226 道目标题在重新加载的最终模型中复现，倒序并逐题清空视觉缓存后保持一致。'
             '目前证据更支持视觉内容利用不足与模型参数中的答案偏好；尚未确定造成对齐失败的具体训练或视觉表示原因。**', '',
             '使用正式 `artifacts/full/stage2/` 模型、原始冻结视觉特征、同样的 BF16 autocast、greedy 生成和 256 token 上限。'
             '没有改正式模型权重、训练数据或 benchmark 结果。实际实验任务：`147017`（多条件及独立进程）、`147021`（完整重复回答组）。', '',
             '## 最关键的实际对照', '',
             '| 对照 | 实测结果 | 含义 |', '|---|---|---|',
             f'| 全部目标题重新加载模型后独立生成 | {group["forward_matches_historical"]}/{n} 与历史作答一致 | 不依赖原评测进程或它之前处理过的题目 |',
             f'| 全部目标题倒序，且每题清空视觉特征内存缓存 | {group["reverse_and_cache_clear_matches_forward"]}/{n} 答案不变 | 未观察到顺序或视觉缓存串扰 |',
             f'| 倒序前后模型输入 embedding 的 SHA256 | {group["prefill_hash_invariant_to_order_and_feature_cache"]}/{n} 相同 | 同题的实际输入不受之前题目影响 |',
             '| 每次 generate 首次 forward 的 KV 长度 / cache_position 起点 | 全部为 0 / 0 | 新题从空 KV cache 与位置 0 开始 |',
             '| 首次 forward 输入与单独构造的当前图像＋当前问题 | 全部长度及 SHA256 一致 | 没有附加上一题问题或答案 embedding |',
             f'| 6 道目标题＋4 道对照题反序重答 | {runtime["order_invariant"]}/10 答案不变 | 包含不同类型答案的换序验证 |',
             f'| 完全禁用生成 KV cache | {runtime["kv_cache_disabled_matches"]}/{runtime["kv_cache_disabled_samples"]} 仍答相同人名 | 错误不依赖 KV cache 路径 |',
             f'| 两道题分别在全新 Python 进程中第一题生成 | {runtime["fresh_process_matches"]}/{runtime["fresh_process_samples"]} 相同 | 进程内不存在任何先前问答 |',
             f'| 将当前记录 gold 字段替换成审计哨兵文本 | {runtime["gold_metadata_unused"]}：输入 hash 和作答不变 | 推理不读取 gold 字段 |', '',
             '逐条输入摘要、实际首轮 KV 长度、SHA256 和作答见 '
             '[runtime_predictions.jsonl](../artifacts/full/context_audit/runtime_predictions.jsonl)。'
             '完整组汇总见 [full_group_summary.json](../artifacts/full/context_audit/full_group_summary.json)。', '',
             '## 换图及零特征：确有内容利用问题，但不能说完全没有视觉依赖', '',
             f'在 6 道原本回答目标人名的题目中，保持问题、替换为另一张真实图像，{stayed}/6 仍回答目标人名，'
             '其余 2 道改为其他人名；4 道非目标回答对照也会随换图改变答案。'
             '这说明模型对图像存在一定依赖，但目标人名在不同文档上容易成为默认猜测。', '',
             '将图像特征置零（保留 token 数和问题）后，10/10 答案改变，6 道目标题都不再回答目标人名，'
             '输出转为带 `<think>` 的泛化推理/询问背景。零向量是分布外输入，'
             '只能说明视觉块会影响生成行为，不能据此认定模型能正确读取图像内容。', '',
             '| 样本 | Gold（全部） | 正常图像 | 换为其他图像后 |', '|---|---|---|---|']
    def cell(text):
        return str(text).replace('|','&#124;').replace('\n','<br>')
    for key,value in runtime['per_sample'].items():
        lines.append(f'| {key} | {cell(json.dumps(lookup[key]["answers"],ensure_ascii=False))} | {cell(value["baseline"])} | {cell(value["wrong_image"])} |')
    lines += ['', '个别零特征作答中，模型会自行写出“Let me check the previous conversation”。'
              '**这是生成的文字，不是模型拿到了历史对话的证据**：该题实际首轮 KV 为 0、输入只有当前问题及零视觉块。', '',
              '## 代码路径检查', '',
              '- [VLM.prompt](../vqa/model.py#L74) 每次只构造当前 user 消息与 assistant 起始标记，没有 conversation/history 容器。',
              '- [VLM.embeds](../vqa/model.py#L86) 每次重新拼接当前图像经过 MLP 的 embedding 与当前问题；推理 answer=None，直接返回 prefix。',
              '- [VLM.predict](../vqa/model.py#L109) 每题单独调用 generate，没有传入上一题的 past_key_values。'
              '本机 Transformers 4.57.1 在默认路径中为每次 generate 新建 DynamicCache；运行时 hook 确认它开始为空。',
              '- [视觉缓存](../vqa/model.py#L59) 保存的是以图像 SHA256 为键的冻结 encoder 特征，不保存问题、答案、projector 输出或 LLM hidden states。',
              '- [训练 forward](../vqa/full_train.py#L170) 按 batch 的不同样本维度独立前向、use_cache=False；本轮 microbatch=1、没有 packing。'
              '梯度累积会合并参数梯度，这是正常训练，不会拼接各题上下文。',
              '- [评测循环](../vqa/full_train.py#L223) 按行调用 predict；仅在生成结束后把 prediction 和 gold 写入输出记录。', '',
              '实际输入示例（DocVQA_validation_10；视觉块为本题图像的 4900 个视觉 token，总长度 4925）：', '',
              '```text', '<|im_start|>user', '<|vision_start|>[当前图像的视觉 embedding]<|vision_end|>Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?<|im_end|>',
              '<|im_start|>assistant', '```', '',
              '## 数据与特征检查', '',
              f'- 涉及的 {features["documents"]} 张原图 SHA256 均与 document_id 一致；缓存 tensor 有 {features["distinct_feature_hashes"]} 个不同 SHA256，全部为有限值。'
              '排除了这些图像全部误用同一份缓存 tensor 的情况；本检查并未重新提取所有图像，不能单凭 hash 证明 encoder 特征质量。',
              f'- DocVQA 训练共 {prior["train"]["questions"]:,} 条；实际训练使用的第一 gold 中，目标完整人名出现 {prior["train"]["exact_target_first_answer_count"]} 次；'
              f'任意 gold 包含 Shank 的问答共 {prior["train"]["shank_any_gold_questions"]} 条。该姓名在训练监督中确实存在。',
              f'- Validation 仅 {prior["val"]["exact_target_any_gold_count"]} 道题有目标完整姓名 gold，却有 {n} 道模型预测该姓名；'
              '不能由正确答案的真实频率解释。训练出现频率只证明可能来源，未证明具体因果机制。',
              f'- 226 道重复回答中，{sequence["target_questions_with_who"]} 道问题含 `who`（也包括 whose/whom）；'
              '提示模型容易在问人名时猜测这一姓名。',
              f'- 完整 DocVQA validation 问题中，含 ChatML/vision 控制标记的条数为 {sequence["question_includes_chat_special_tokens"]}，未发现问题文本注入额外聊天轮次。', '',
              '## 复现入口与证据', '', '```bash',
              '# 数据统计；--features 会重查受影响的 171 张图像及特征文件',
              '.venv/bin/python -m scripts.audit_vqa_data --features',
              '# 实际正式模型对照（NPU 计算节点；不需要重新训练）',
              'sbatch scripts/slurm_context_audit.sh',
              '# 全部 226 道题的顺序/缓存对照',
              'sbatch scripts/slurm_context_audit.sh --full-group',
              '# 实验结果完成后生成本文档',
              '.venv/bin/python -m scripts.report_context_audit', '```', '',
              '同一输出目录中的实验应顺序执行。Slurm 脚本配置 venv/CANN 和独立 Python bytecode 缓存；'
              '逐条日志为 append-only，多次执行会保留全部诊断记录，汇总 JSON 对应最近一次完成的实验。', '',
              '- `artifacts/full/context_audit/runtime_summary.json`：10 道题的多条件与独立进程结果。',
              '- `artifacts/full/context_audit/full_group_summary.json`：226 道题的完整换序及缓存结果。',
              '- `artifacts/full/context_audit/shank_feature_audit.json`：171 张图像和特征的 hash/shape。',
              '- `artifacts/full/context_audit/training_prior.json`、`train_shank_gold.jsonl`、`val_shank_gold.jsonl`：训练/验证监督中的姓名分布。',
              '- `artifacts/full/context_audit/sequence_statistics.json`：原评测各 rank 的首次出现和前序问答。',
              '- `artifacts/full/context_audit/source_fingerprints.json`：所检查的模型实现、Transformers 实现和 projector 文件 SHA256。', '',
              '本调查覆盖完整重复回答组的输入隔离，以及小样本视觉干预。它排查了用户提出的跨问答上下文问题，'
              '尚未把论文性能差距归因到某个特定训练设置。后续应优先检查训练集的小样本过拟合能力、'
              '同题多图的答案概率变化、视觉特征空间与 MLP 输出，而非盲目关闭生成 cache 或直接重复全量训练。']
    (ROOT / 'docs/CONTEXT_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(ROOT / 'docs/CONTEXT_AUDIT.md')


if __name__ == '__main__':
    main()
