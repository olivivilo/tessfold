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
        '# MetaCompress：三轮调参后的失败原因与证据更新', '',
        '梯度范数、实际参数更新、optimizer/scheduler 配对对照与 token 预算的详细数值见 '
        '[优化与预算实测说明](METACOMPRESS_OPTIMIZATION_AND_BUDGET.md)。', '',
        '本文件读取已完成产物，不启动训练，不修改既有选择门槛。可复现入口：', '',
        '```bash',
        '.venv/bin/python -m scripts.report_meta_compress_current_diagnosis',
        '```', '',
        '## 当前实验状态与此前回复更正', '',
        f"controller 最后更新：`{state['updated_at']}`，状态 `{state['phase']}`。"
        f"三轮共有 {len(official)} 组正式候选，"
        f"{facts['completed_formal_candidates']} 组完成原定训练，"
        f"{facts['early_stopped_formal_candidates']} 组按已记录规则提前停止；另有两组 mini 局部诊断。"
        '两次正式独立确认均未通过，新的原 20k 完整重训未启动，当前也没有继续运行的调参作业。', '',
        '此前进度回复引用了旧的 10% token、20k/2 epochs 实验，并漏报这些后续调参结果。'
        '旧实验确实完成，但不能把它写成最新调参后的完整重训。'
        '以下以真实的 search/confirm 产物为准。', '',
        '## 1. 原 dense 聚合结构与 UND 特征的适配是首要问题', '',
        '旧完整训练抽查的 64 个矩阵中，行熵/log(n) 均值约 0.981，熵有效支持数约 3921，'
        '相邻行概率向量 cosine 约 0.903。大量输出行广泛混合整页 token，文字、数字和布局证据可能被平均掉。'
        '**高熵本身不是失败的充分条件**：此前自由 P 和 KL-only SGD 均可在保持高熵时拟合单题。'
        '更直接的证据是：三道 teacher 正确的诊断题，即使 m=n、不缩短占位块和文本位置，'
        '仅经过原初始化 P 混合也全部变为零分。这个 3 题对照说明混合本身足以导致这些题失败，'
        '不能外推为所有图像都如此。', '',
        '同一 56 图 search、50% 预算下，原 dense 初始化为 4.799 分，固定局部 mean 为 34.001，'
        '加局部 bias 的生成器在训练前为 37.192。局部先验的改善发生在第 0 步，'
        '不能归功于训练。单纯增加输出行数不能保证输出行包含不同的局部证据。'
        '当前 query 是一维序列 fractional max pooling，位置表为本项目补全，'
        '并非公开作者代码的逐项复用；其与原生二维文档 token 的对应是否充分，仍待结构消融。', '',
        '## 2. 原始优化确实受限，但不是后续所有失败的共同解释', '',
        '旧训练使用无 momentum SGD，LR=1e-3、全局 norm clip=.01，每步参数更新 L2 至多 1e-5；'
        '全部 1112 步均裁剪，裁剪乘数中位数约 0.0001765。BF16 的局部有限差分也显示，'
        '很小 FP32 参数移动的前向效果受舍入影响。原设置的有效更新不足有直接证据。', '',
        '但后续已尝试 clip=1/10、AdamW、momentum、constant/cosine、不同预算。'
        '最新局部 KL-only AdamW50 的 train KL 各 epoch 均值约为 '
        '0.550→0.482→0.449→0.425，独立 confirm KL 也下降。'
        '因此不能再用“完全没学动”解释这些新候选；也没有证据证明 scheduler 是唯一主因。', '',
        '## 3. 完整 loss 的正则与预测目标存在实际梯度冲突', '',
        '八张经过 teacher 正确条件筛选的诊断图像上，原初始化 collapse/KL 梯度 norm 比中位数约 44.8；'
        '4/8 图像的 total 梯度与 KL 梯度点积为负，即 total 的局部下降方向反而增加 KL。'
        '这是分项求导证据，不是比较三个 loss 的绝对数值；也不代表整个训练集的比例。'
        '同图、同 SGD、同 LR 的 2×2 对照中，KL-only 去除强裁剪能拟合，完整 loss 仅取消裁剪仍失败。', '',
        'collapse 惩罚重复使用同一列，但近均匀的全局 P 也有较小 collapse；'
        '它可以与希望每行聚焦少量源 token 的 entropy 项争夺梯度。'
        '未归一化的列质量总和还随 m 改变尺度。'
        '这支持重新校准正则与梯度预算，但不能断言两项正则普遍无效：局部结构配完整 loss 曾可拟合单题。', '',
        '## 4. 更新目标已能泛化降低 KL，却没有证实答案收益稳定泛化', '',
        f'代表候选 `{name}`，672 张不同训练图、4 epochs、50% token、局部 bias=20、'
        'AdamW 1e-4、clip=1、warmup cosine、KL-only。局部 bias 是论文没有的额外结构先验。', '',
        '| 集合 | 初始分数 | 训练后分数 | 分数变化 | 初始 KL | 训练后 KL | KL 降幅 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for label, initial, final in [('search 56 图', s0, s1), ('confirm 112 图', c0, c1)]:
        lines.append(f"| {label} | {initial['score']:.3f} | {final['score']:.3f} | "
                     f"{final['score']-initial['score']:+.3f} | {initial['prediction']:.5f} | "
                     f"{final['prediction']:.5f} | {1-final['prediction']/initial['prediction']:.1%} |")
    ci = c1['paired_vs_initial']['paired_bootstrap_95ci_points']
    lines += [
        '', f'确认集相对初始分数变化的配对 bootstrap 95% CI 为 [{ci[0]:.3f}, {ci[1]:.3f}] 点。'
        '该区间跨零，表示证据不足以确认稳定收益，不表示已经证明收益必为零。'
        'search 自己的收益区间也跨零；同一 56 图上反复选择候选/epoch，有选择偏差风险。'
        '两轮使用同一 confirm 还存在适应性复用限制。', '',
        f"逐题比较：{counts['kl_lower']}/112 题的 KL 降低，其中 "
        f"{counts['kl_lower_score_up']} 题得分提高、{counts['kl_lower_score_down']} 题得分下降、"
        f"{counts['kl_lower_score_same']} 题得分不变。"
        '因此这里的主要脱节不是“验证 KL 完全过拟合”，而是分布改善与自由生成评分并非同一个目标。', '',
        '全词表平均 KL 的降低可以发生在非关键 token、概率尾部或 teacher 给定的答案后续位置；'
        '它不保证关键数字/字符的 argmax 排名正确。teacher forcing 时学生拥有正确 teacher 前缀，'
        '自由生成时则必须处理自己的前缀，一处错误可改变后续轨迹。'
        '这些是机制解释，尚未用逐 token margin、首 token/后续 token 梯度消融来分离各自贡献。'
        'teacher 也会答错，因此贴近 teacher 不等同于贴近 gold。', '',
        '确认集的任务变化也不一致：DocVQA 46.43→52.00、InfoVQA 42.12→49.86、DeepForm 12.50→25.00；'
        '同时 DT-VQA 59.24→49.53、ChartQA 37.50→31.25、WTQ 25.00→18.75，KLC 不变。'
        '每任务仅 16 图，不能将这些变化视为稳定的任务效应。'
        '严格大小写/空白归一化 exact 由 16.96→22.32，也提示答案格式与原生部分分会影响趋势；'
        '这是预先标明的辅助指标，不替代主指标选模型。', '',
        '## 5. 高压缩率、训练覆盖与任务约束仍有限制', '',
        '固定 mean 的 search 分数随保留比例 10%/25%/50%/75% 约为 4.59/10.51/34.00/54.68，'
        'teacher 为 58.13；预算对文档细节影响明显，但这不是原生成器的可达上限。'
        '75% 的局部 KL-only 候选训练前 55.59、训练后 55.03，'
        '再次说明“保留更多 token”与“训练提升”必须分开。'
        '10% 预算的单图可拟合证据也排除了“所有任务在该预算绝无可能”的过强结论。', '',
        '原 20k 中 DT-VQA 接近半数；多图 pilot 则已经改为任务均衡，'
        '故原数据配比不能解释全部后续失败。第三轮每任务 96 张训练图，只覆盖原 20k 的一小部分；'
        '训练轮数和数据覆盖是否足够尚无收敛及重复 seed 证据。'
        '另外当前每图只取一题训练，而生成器不接收问题。'
        '一个问题的蒸馏信号不能约束整页所有潜在问答证据；'
        '同图多题有望加强这种约束，但目前并未实测证明它能修复。', '',
        '## 6. 目前不支持的归因与下一步优先级', '',
        '没有足够证据优先归咎于 Qwen3-1.7B 容量或官方 encoder。'
        '未压缩的同一模型可正常答题，当前实验主干全部冻结，也没有“本次解冻 LLM 训坏”的路径。'
        '已有 identity、独立状态、causal shift、缓存来源检查通过；七任务锚点 teacher logits 与缓存逐位一致。'
        '这降低了这些实现错误的可能性，但不是对所有图像与所有边界条件的穷尽证明。', '',
        '下一轮应以可归因的实验为主：', '',
        '1. 固定 50% 预算和图像隔离集合，对比二维局部选择/聚合与当前一维局部生成器，'
        '加入原 token 选择基线，检查结构和输出 token 分布，而不只扩大 dense 超参数搜索。',
        '2. 同一训练图加入多个不同问题，验证也按图像隔离，明确测整页证据覆盖；'
        '不要只在 teacher 正确图像上训练和报告。',
        '3. 先审计首个答案 token、数字/实体 token、EOS、teacher argmax margin 与自由生成分歧位置。'
        '再在同结构、同 optimizer 下比较原 KL、关键 token 加权 KL 或学生前缀上的蒸馏。'
        '这些属于 decoder-dependent 的新 loss/轨迹修改，必须与原论文方案分开。',
        '4. 扩大独立验证与重复 seed，单独保留最终确认集合；'
        '只有同预算、同结构初始化的训练收益稳定，再增加到原 20k。'
        '目前不应把任意完整重训已经可能改善或已经失败写成观测结果。', '',
        '原始证据：[调参设计](METACOMPRESS_TRAINING_SEARCH.md) · '
        '[search](../artifacts/meta_compress_search/RESULTS.search.md) · '
        '[confirm](../artifacts/meta_compress_search/RESULTS.confirm.md) · '
        '[机制干预](METACOMPRESS_INTERVENTION_RESULTS.md) · '
        '[本报告机器可读统计](../artifacts/meta_compress_search/current_diagnosis.json)。',
    ]
    dest = ROOT / 'docs/METACOMPRESS_CURRENT_DIAGNOSIS.md'
    dest.write_text('\n'.join(lines) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
