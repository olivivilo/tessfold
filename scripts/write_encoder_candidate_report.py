"""Render the completed encoder-only experiment, without selecting a winning loss."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def main():
    a=json.loads((OUT/'analysis.json').read_text());assert a['has_full_token_OT'] and a['has_line_probe']
    m=a['metrics'];p=json.loads((OUT/'probe_metrics.json').read_text());l=json.loads((OUT/'line_probe_metrics.json').read_text());d=a['full_token_transport_diagnostics'];sp=json.loads((OUT/'subspace_audit.json').read_text())['summary']
    def f(x):return '—' if x is None else f'{x:.4f}'
    def ci(x,percent=False):return '['+', '.join(f'{v*100:.2f}%' if percent else f'{v:.4f}' for v in x)+']'
    names=[('问题加权 coverage + 位置','coverage_pos_question'),('完整 token Sinkhorn / uniform','sinkhorn_full_uniform'),('完整 token Sinkhorn / 问题增量','sinkhorn_full_question_excess'),('文字＋数值＋粗邻接证据','evidence_combined_question'),('文字＋数值＋OCR 行关系','evidence_combined_line_question'),('问题条件子空间重建','subspace_question')]
    mainrows='\n'.join(f"| {label} | {f(m[n]['pooled_spearman'])} | {f(m[n]['within_question_ratio_family'])} {ci(m[n]['within_question_ratio_family_ci95'])} | {m[n]['paired_seed_ordering']['accuracy']*100:.2f}% {ci(m[n]['paired_seed_ordering']['image_bootstrap_ci95'],True)} |" for label,n in names)
    taskrows=[]
    for t,v in a['per_task'].items():
        taskrows.append(f"| {t} | {v['images']} | {24*v['images']:,} | {v['teacher_mean']*100:.2f} | "+' | '.join(f(m[n]['per_task'][t]['pooled_spearman']) for n in ['coverage_pos_question','sinkhorn_full_question_excess','evidence_combined_line_question','subspace_question'])+' |')
    conditionrows='\n'.join('| '+t+' | '+' | '.join(f(m[n]['per_task'][t]['within_question_config']) for n in ['coverage_pos_question','sinkhorn_full_question_excess','evidence_combined_line_question','subspace_question'])+' |' for t in a['per_task'])
    allrows='\n'.join(f"| `{n}` | {f(v['pooled_spearman'])} | {f(v['within_question'])} | {f(v['within_question_ratio_family'])} | {v['paired_seed_ordering']['accuracy']*100:.2f}% |" for n,v in m.items())
    subsetrows='\n'.join(f"| {label} | {f(m[n]['teacher_positive_spearman'])} | {f(m[n]['teacher_perfect_spearman'])} | {f(m[n]['high_retention']['pooled_spearman'])} | {f(m[n]['high_retention']['within_question_config'])} |" for label,n in names)
    configrows='\n'.join('| '+key+' | '+' | '.join(f(m[n]['per_ratio_family'][key]['pooled_spearman']) for n in ['coverage_pos_question','sinkhorn_full_question_excess','evidence_combined_line_question','subspace_question'])+' |' for key in m['coverage_pos_question']['per_ratio_family'])
    apr='\n'.join(f"| {key} | {value:.4f} |" for key,value in p['type_AP'].items())
    pair=m['coverage_pos_question']['paired_seed_ordering'];zero=a['student_zero_scores']
    counter=json.loads((OUT/'counterexamples.json').read_text())
    def escape(value):return str(value).replace('|','\\|').replace('\n',' ')[:180]
    examples=[]
    for name,pairs in counter.items():
        if not pairs:continue
        ex=pairs[0];lo=ex['low_loss'];hi=ex['high_loss']
        examples.append(f"**`{name}`**：{escape(lo['sample_id'])}，{lo['family']}，保留率 {lo['ratio']}。问题：{escape(lo['question'])}。Gold：{escape(lo['answers'])}；teacher：{escape(lo['teacher_prediction'])}（{lo['teacher_score']:.4f}）。\n\n| 试次 | loss | student 作答（截取至多 180 字符） | student 分数 | 分数下降 |\n|---|---:|---|---:|---:|\n"+'\n'.join(f"| {'低 loss' if r is lo else '高 loss'} / seed {r['replicate']} | {r['loss'][name]:.7g} | {escape(r['student_prediction'])} | {r['student_score']:.4f} | {r['score_drop']:.4f} |" for r in [lo,hi]))
    ot=m['sinkhorn_full_question_excess'];ln=m['evidence_combined_line_question']
    sensitivity=json.loads((OUT/'statistical_sensitivity.json').read_text())
    sens=sensitivity['both_trial_max_marginal_error_le_1e_6'];sd=sensitivity['all']
    text=rf'''# 严格 encoder-only 的文档 VQA 压缩 loss 验证

**状态：已完成。** 官方 MonkeyOCRv2-B-Und；八任务共 **3,601 张不同图像、3,601 道题、86,424 次随机压缩、7,202 次均值池化对照**。每次压缩的原始作答、gold、teacher/student 分数和所有新 loss 均单独保留。

**主结论：本轮没有验证出能稳定判断“同题、同压缩预算下哪个随机矩阵更好”的新 loss。** 合并不同题目/压缩配置可见正相关，但应同时看条件相关和随机种子配对。完整 token 的问题加权 Sinkhorn 增量：pooled Spearman **{ot['pooled_spearman']:.4f}**，同题同率同族条件相关 **{ot['within_question_ratio_family']:.4f}**，非平局排序正确率 **{ot['paired_seed_ordering']['accuracy']*100:.2f}%**。文字＋数值＋OCR 行关系组合对应为 **{ln['pooled_spearman']:.4f} / {ln['within_question_ratio_family']:.4f} / {ln['paired_seed_ordering']['accuracy']*100:.2f}%**。置信区间见主表；不因某项 pooled 相关稍高而选定新训练目标。

**完整问题 OT 有很弱的正向信号，不能说所有新候选都完全随机。** 它的未校正 95% 排序 CI 略高于 50%，但优势仅约 1.63 个百分点，且没有显示优于旧 reconstruction baseline；本轮没有建立强相关或实用的保分保证。所有区间均未作多重比较校正，24 个固定字段也并非相互独立的检验。

**明确入口：** [`bash scripts/run_encoder_candidates.sh status`](../scripts/run_encoder_candidates.sh)；[逐试次 CSV](../artifacts/token_proxy/encoder_candidates_v1/trials.csv)、[完整统计 JSON](../artifacts/token_proxy/encoder_candidates_v1/analysis.json)、[反向排序作答实例](../artifacts/token_proxy/encoder_candidates_v1/counterexamples.json)、[原始逐次作答可筛选页面](../artifacts/token_proxy/random_v1_20x/review.html)。

## 1. 本轮做了什么，哪些方案放弃了

| 文献建议 | 本轮实际实现 | 对严格约束的处理 |
|---|---|---|
| 问题证据加权 coverage / Sinkhorn（A / A′） | 独立 OCR 字符/二元组头桥接问题；raw encoder 空间的采样 coverage、完整 token Sinkhorn divergence | 保留 A；放弃官方 projector 空间和 Qwen embedding 的 A′ 变体 |
| 非生成式文字/数值/关系证据保留（A） | 字符存在、hashed bigram、数字/格式类型分布 JS；粗文字–数字邻接；补充训练 same-OCR-line probe | 全部读取 raw X 或 PX；没有精确字符串、算术或表头–数值语义关系识别器 |
| 问题条件子空间重建（A′ baseline） | 改成 raw encoder 全部 768 维的 ridge 子空间重建；用独立词法桥给 teacher token 加权 | 修改后不使用 projector 或 Qwen embedding，正常验证；不是原接口空间版本的逐式复现 |

明确放弃：① 使用官方 merger / alignment MLP 输出定义距离；② 使用 Qwen token embedding 构造问题 query；③ 用 LLM attention、logits、NLL 或 VQA 作答分数训练 evidence head。它们依赖用户明确排除的 decoder/interface，不符合本轮要求。**没有因为候选在论文中被称为 training-free，就把它当作 decoder-agnostic。**

本轮计算图：

```text
官方冻结 Und encoder → X[N,768] ──────→ 官方 merger/LLM → teacher 作答（已有缓存）
                         ├─ P → C[M,768] → 官方 merger/LLM → student 作答（已有缓存）
                         └─ X、C、U、PU、问题 → 独立 OCR/行探针 + encoder-only 数学距离 → loss
```

loss 分支没有读取右侧的官方 merger/LLM 张量。为辅助探针提取训练特征时，加载前将官方 `vision_tower.merger` 替换为 Identity，只加载其余 vision weights；未加载 Qwen 权重。[实现](../scripts/cache_encoder_probe.py)。独立的 768→192→120 evidence MLP 和 578→64→1 行关系 MLP **是用 OCR 标注训练的小型探针，不是官方 encoder–LLM 对齐层**；其参数及输出没有进入下游 LLM。

计算上满足 decoder-free、接口权重独立的要求；但只在一个 Und decoder 上验证，**尚不能宣称已证明跨 decoder 的统计预测能力**。训练一个真正能提升 VQA 的 compressor 也不在本轮范围内。

## 2. 数据、随机矩阵与逐次结果的对应

沿用 [20 倍规模实验](TOKEN_PROXY_20X_RESULTS.md) 的相同图像、问题、预处理、encoder 特征及矩阵文件。**下游 teacher/student 答案及分数来自旧实验缓存；本轮重新计算候选 loss，没有重新运行 93,626 次 LLM 生成。** 这是同一压缩干预的配对重评，避免生成差异混入比较。

- 模型：`models/MonkeyOCRv2-B-Und`，官方成品权重，未挂 LoRA。原始 encoder 输出 BF16、768 维，max_pixels=1,003,520。
- 每图一题；保留率 $r=M/N\in\{{0.125,0.25,0.375,0.5,0.625,0.875\}}$。local/global 两族，各两个 seed：每图 24 个随机试次。另有每图 0.25/0.5 两个 mean-pooling controls，不混入主相关性。
- 未改动 [`vqa/token_proxy.py`](../vqa/token_proxy.py)。每个 token 只属于一个平衡 support，support 大小差至多 1；行内 uniform[-2,2] logits 做 softmax，再混合 30% 均匀权重；加入 $10^{{-6}}$ 全局 floor。所有元素严格正、逐行和为 1，$C=PX$ 是凸组合；等价地 $P=\operatorname{{softmax}}(\log P)$。不是完全无约束的高斯稠密随机矩阵。
- 每个 trial 独立保留 `matrix_seed / matrix_file / teacher_prediction / student_prediction / answers / teacher_score / student_score / score_drop`。**没有先平均 seed 的 ANLS 等分数再算相关性。**
- 新 worker 验证全部矩阵 SHA256、全部 raw X SHA256，以及每次 $PX$ 转为 BF16 后的 SHA256；与原来真正进入下游的压缩张量一致。每图还独立重生成首个随机 P，验证 group/weight 完全一致。loss 使用 float32 $PX$；下游使用其 BF16 版本。
- 原接口审计中 native/cached/identity 的 logits 和生成一致、teacher 与既有官方 baseline 3,601/3,601 一致。[既有审计](../artifacts/token_proxy/random_v1_20x/audit.json)。原有评分器不变：DeepForm/KLC 用逐题 normalized exact match，不能当作文档集合 F1；各任务的 score 都在 [0,1]，跨任务 pooled 结果仅为诊断。

| 任务 | 不同图像/题目 | 随机试次 | 本子集 teacher 分数×100 | coverage ρ | 完整 OT 增量 ρ | evidence＋行 ρ | subspace ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(taskrows)}

这里的 teacher 数值仅说明所用子集基线，不是论文完整 benchmark 的重新评测。

## 3. 实际 loss 结构

### 3.1 不依赖 decoder 的问题桥与 feature 坐标

以 InfoVQA **训练集** OCR WORD 框和文字（confidence≥70）训练独立非自回归探针：720 图训练、240 图按图像隔离验证；与本项目 VQA val/test 的图像 hash 交集为 0。没有用 VQA gold、作答分数或官方 LLM 表征训练。训练固定 20 epochs、采用最后 epoch，不按下游相关性挑 checkpoint。

$H(X)$ 输出多标签概率：小写字符存在、64 个 hashed bigrams、8 个类型（text/digit/decimal/percent/currency/negative/date/alphabet）。问题去停用词后，映射为同一显式字符/bigram 坐标；与 $H(X_i)$ 的 cosine maxsim 得 $s_i(q)$，固定温度 0.1：

$$a_i(q)=0.3/N+0.7\operatorname{{softmax}}_i(s_i(q)/0.1).$$

这是**词法桥**，没有完整的语义问题理解；VQA 的真正答案证据可能不与问题共享文字。240 图 OCR 验证只能检验基础探针，不证明 VQA evidence grounding 正确。

距离中使用同一个固定随机正交投影 $R\in\mathbb{{R}}^{{768\times64}}$（seed=20261022），$h(x)=xR/(s\sqrt{{64}})$，$s={p['rms']:.6f}$ 是训练集 scalar RMS。teacher/student 使用同一尺度，**没有对 student 独立归一化**。该投影与官方对齐层完全无关。坐标 $U$ 为原 patch 的归一化二维位置，student 坐标取 $PU$。

### 3.2 问题加权 coverage

每图固定均匀抽至多 256 个 teacher anchors；对全部 student token 做 nearest-neighbor：

$$L_{{cov}}=\sum_{{i\in A}}\tilde a_i\min_j\big[\|h(X_i)-h(C_j)\|^2+0.25\|U_i-(PU)_j\|^2\big].$$

保留 uniform / question、是否加位置四种固定变体。$\tilde a$ 在所抽 anchors 内重新归一化。**这是采样 coverage，不是全 teacher 精确 coverage**；问题权重很集中时会有漏采证据的局限，不能把它当作原建议的无限精度实现。

### 3.3 完整 token Sinkhorn

完整 $N\times M$ token 间的 cost 与上式相同，target mass 固定 $b_j=1/M$。熵正则 $\epsilon=0.05$，目标为 $\langle\pi,D\rangle+\epsilon\mathrm{{KL}}(\pi\|ab^T)$；用 debiased Sinkhorn divergence：

$$S_\epsilon(\mu,\nu)=OT_\epsilon(\mu,\nu)-\tfrac12 OT_\epsilon(\mu,\mu)-\tfrac12 OT_\epsilon(\nu,\nu).$$

两种 source mass：uniform 或 $a(q)$。**问题加权 source 与均匀 target 的边际不同，即使 $P=I$，raw divergence 也可能非零。** 因而完整保留 raw 值，并单独报告：

$$L_{{OT,q,excess}}=S_\epsilon((X,a(q)),(C,\mathrm{{unif}}))-S_\epsilon((X,a(q)),(X,\mathrm{{unif}})).$$

这是相对未压缩参考的增量，可为负，**不是保证非负的严格距离**。同题内部，减去常量不会改变随机矩阵排序；跨题 pooled 可能改变。

最终主结果采用**全 token**传输，不合并空间质心；64 维 feature 投影仍保留。512 空间质心近似另外保留作计算消融，不能替代最终 OT 结论。NPU float32 使用稳定 scaling，必要时 fallback log-domain；最少 150、最多 1,500 次迭代，目标 marginal tolerance $10^{{-6}}$，实际误差见第 6 节。

### 3.4 非生成式文字、数值与关系证据保留

在固定 16×8 空间查询点，各从 X/U 和 C/PU 独立读取最近 8 个 token，Gaussian bandwidth=0.04 加权；距最近 token 很远时回退 toward $H(0)$。比较两侧探针输出的 Bernoulli JS divergence：

- `evidence_text_*`：字符＋bigram 概率差异。
- `evidence_number_*`：digit、decimal、percent、currency、negative、date 六类概率差异。
- `evidence_relation_question`：相邻查询区域的文字/数字四种绑定状态差异；这是**粗邻接**，没有训练表格语义关系。
- `evidence_combined_question`：上述三项固定等权求和。

补充结构诊断：在同一 720/240 图划分上，用 WORD→LINE 标注训练 37,121 参数的 same-line head。训练输入是两个 OCR word ROI 的 raw X 经冻结 evidence head 第一层所得 192 维特征，加绝对差和二维位置差。验证时，对同一固定空间网格预测相邻区域的 same-line 概率，比较 teacher/student 的 JS；得到 `evidence_line_uniform/question`。`evidence_combined_line_question` 为 text＋number＋line 三项等权和。**这是补充实验，不伪装为最初所有设计均已预注册；也没有按 VQA 分数挑权重或最终候选。**

限制：字符存在/bigram hashing **不能恢复字符顺序和精确数值字符串**；本轮未实现表头–值语义关系、算术关系或非自回归精确 OCR head。对这一组 loss 的负结果，只能否定**上述实际探针实例**在当前分布上的预测能力，不能说所有“精确文字/数值/关系保留”方案已被验证无效。探针以 OCR word/token 标签训练，压缩混合 token 与固定网格读取还存在分布变化。

### 3.5 问题条件子空间重建

改成 encoder 原生 **768 维**空间；不把 feature 降到 64 维后做 subspace。令 $\lambda=0.001s^2$，用 Cholesky 解 ridge 投影残差：

$$E_i=\lambda X_i(C^TC/M+\lambda I)^{{-1}},\qquad L_{{sub,q}}=\frac{{\sum_{{i\in A}}\tilde a_i\|E_i\|^2}}{{\sum_{{i\in A}}\tilde a_i\|X_i\|^2}}.$$

风险评估仍使用同一至多 256 个 teacher anchors；协方差使用全部 C。保留 uniform、question 和减去未压缩 ridge reference 的 excess 三项。该 baseline 可不运行 LLM，但计算 768×768 Cholesky 仍有成本，不是零成本。

它衡量“student span 能否重建 teacher”，并不要求恢复系数等于 P，也不限制为凸重建。混合 C 可能覆盖 feature span、但破坏局部阅读和 decoder 可用性；因此不能将低 residual 直接解释为 VQA 信息充足。

对 hash 选取的八任务各 1 图、64 次压缩另做 [float64 谱/系数审计](../artifacts/token_proxy/encoder_candidates_v1/subspace_audit.json)：相对 1e-4 阈值的秩范围 {sp['rank_at_relative_1e_4']['min']:.0f}–{sp['rank_at_relative_1e_4']['max']:.0f}；ridge covariance 条件数范围 {sp['ridge_cov_condition']['min']:.3g}–{sp['ridge_cov_condition']['max']:.3g}。重建系数中负值占比 median={sp['coefficient_negative_fraction']['median']*100:.2f}%；逐 trial 的系数绝对值行和第 95 分位数 median={sp['coefficient_abs_row_sum_p95']['median']:.2f}、max={sp['coefficient_abs_row_sum_p95']['max']:.2f}。这与压缩 P 本身的非负凸组合约束是两回事。float32 主 loss 对比 float64 参考的 uniform residual 最大相对误差为 {sp['float32_uniform_relative_error']['max']*100:.4f}%，在这 64 个审计点上未显示明显数值失真；不能将此抽查当作全部 93,626 次的精度证明。

## 4. 独立辅助头是否真的学到了东西

| 独立验证指标 | 数值 |
|---|---:|
| text AP | {p['text_AP']:.4f} |
| digit AP | {p['digit_AP']:.4f} |
| 字符 macro AP（text token） | {p['character_macro_AP_text_tokens']:.4f} |
| bigram macro AP（text token） | {p['bigram_macro_AP_text_tokens']:.4f} |
| OCR 词 query Top-10 命中 | {p['OCR_word_query_localization']['top10_hit_rate']*100:.2f}% |
| 同查询 uniform 近似 Top-10 命中 | {p['OCR_word_query_localization']['uniform_approx_top10_hit_rate']*100:.2f}% |
| same-line AP / accuracy | {l['same_line_AP']:.4f} / {l['same_line_accuracy']:.4f} |
| **geometry-only same-line AP** | **{l['geometry_only_AP']:.4f}** |

OCR head：170,808 参数、540,622 train tokens / 181,207 holdout tokens；正负 token 有抽样，AP 对应该抽样分布，不代表全图自然频率。OCR query 检查为 80 张 heldout 图中的 950 个词查询，不是 VQA query 的证据定位准确率。行 head：167,820 train pairs / 56,620 holdout pairs。几何规则的行关系 AP **优于**所训练 head，不应声称 head 学会了强语义关系。

| OCR 类型 | AP |
|---|---:|
{apr}

[OCR 逐 epoch 训练/验证指标](../artifacts/token_proxy/encoder_candidates_v1/probe_metrics.json)、[训练曲线 PDF](../artifacts/token_proxy/encoder_candidates_v1/probe_training_loss.pdf)、[行 head 逐 epoch 指标](../artifacts/token_proxy/encoder_candidates_v1/line_probe_metrics.json)、[OCR 权重](../artifacts/token_proxy/encoder_candidates_v1/evidence_head.pt)、[行权重](../artifacts/token_proxy/encoder_candidates_v1/line_head.pt)。所有辅助训练仅为了构造 loss；没有训练 compressor、官方 projector 或 LLM。

## 5. 相关性验证结果

定义 $\Delta=score_{{teacher}}-score_{{student}}$，不截断负值。期望 loss 越大、$\Delta$ 越大。pooled Spearman 使用 **86,424 个独立保留的 trial**；同题同率同族指标在每个双 seed 组内取 rank、中心化后汇总相关。后者衡量同一压缩预算下的排序能力，不能与跨所有配置 pooled 数值混为一谈。

所有置信区间用 **1,000 次 image-cluster bootstrap**，将一张图的所有 trial 一起重采样；pooled Spearman 每次重新排名。同 seed 配对从不把两个下游分数取平均，只比较高/低 loss 与高/低下降是否同向。条件相关基于组内秩，仅两个 seed 时重在检查方向；不是精细的条件函数估计。

| 实际方案 | pooled Spearman | 同题＋同率＋同族相关 [95% CI] | 非平局 seed 排序正确率 [95% CI] |
|---|---:|---|---|
{mainrows}

总共 **{pair['pairs']:,}** 个双 seed 配对组，其中 **{pair['outcome_ties']:,}** 组下游分数相同、仅 **{pair['pairs']-pair['outcome_ties']:,}** 组非平局。随机 trial 中 student 得 0 分为 **{zero:,}/{a['random_trials']:,}（{zero/a['random_trials']*100:.2f}%）**。因此当前随机族存在严重地板效应；大量试次无法用作答分数辨别强弱。CI 含 50% 时，不能认定排序优于随机。

**弱信号与数值误差敏感性：**只保留两个 trial 的最大 OT 边际误差均≤1e-6 的配对后，剩余 {sens['nontie_pairs']:,} 个非平局对，问题 OT 排序正确率为 {sens['OT_order_accuracy']*100:.2f}%、95% CI {ci(sens['OT_accuracy_image_bootstrap_ci95'],True)}。全量同一批配对中，OT 相对旧 `V1_rec` 的正确率差为 {sd['OT_minus_old_rec_accuracy']*100:.3f} 个百分点，配对 image-bootstrap 95% CI 为 [{sd['paired_image_bootstrap_difference_ci95'][0]*100:.3f}, {sd['paired_image_bootstrap_difference_ci95'][1]*100:.3f}] 个百分点。因此没有验证出相对旧重建项的改善。[完整补充统计](../artifacts/token_proxy/encoder_candidates_v1/statistical_sensitivity.json)。这是敏感性检查，不替换全量主结果，也不用于选择候选。

### 各任务的同题同率同族相关

| 任务 | coverage | 完整 OT 增量 | evidence＋行 | subspace |
|---|---:|---:|---:|---:|
{conditionrows}

### 排除 teacher 本来不会作答，以及较高保留率

| 方案 | teacher>0 pooled ρ | teacher=1 pooled ρ | r≥0.625 pooled ρ | r≥0.625 条件相关 |
|---|---:|---:|---:|---:|
{subsetrows}

teacher>0 共 **{a['teacher_positive_images']:,}** 张图。过滤可以减少“teacher 已经错了导致下降无法体现”的干扰，但不能自动解决同题同预算的随机矩阵排序问题。

### 直接作答反例

以下按反向下降幅度挑出示例，仅用于说明失败机制，**不是这些事件的无偏频率估计**。每一对的图像、问题、保留率和矩阵族都相同；完整答案、矩阵 seed 和其余示例见 counterexamples.json。

{chr(10).join(chr(10)+ex+chr(10) for ex in examples)}

### 固定保留率和矩阵类型后的 pooled Spearman

这里仍混合不同问题，不能代替主表的同题条件检验。

| r / family | coverage | 完整 OT 增量 | evidence＋行 | subspace |
|---|---:|---:|---:|---:|
{configrows}

### 全部固定变体与旧基线

| loss 字段 | pooled ρ | 同题跨全部配置相关 | 同题同率同族相关 | seed 非平局排序 |
|---|---:|---:|---:|---:|
{allrows}

`V1`、`V1_rec` 是此前实际 loss 的同批对照；旧 reconstruction 项可能有弱正排序信号，不能把“新候选未证实有效”说成“所有基线精确随机”。`sinkhorn_*` 无 `full` 前缀是 512 空间质心近似，仅作消融。所有 mean-pooling control 的相关性、双向 rank 残差、每任务 seed 排序数及每项 loss 的分位数见 [analysis.json](../artifacts/token_proxy/encoder_candidates_v1/analysis.json)；没有用对照挑选新组合。

![逐试次 loss 与作答下降散点](../artifacts/token_proxy/encoder_candidates_v1/correlation_scatter.png)

[散点 PDF](../artifacts/token_proxy/encoder_candidates_v1/correlation_scatter.pdf)。图中为固定随机抽取的至多 16,000 个 trial，仅用于可视化；所有统计使用全量随机 trial。横轴范围不同，不可通过斜率大小比较 loss 好坏。

## 6. 数值、接口与精度审计

- [数学检查](../artifacts/token_proxy/encoder_candidates_v1/math_checks.json)：tiny entropic OT 与独立 SciPy constrained optimizer 的目标值误差 $7.76\times10^{{-9}}$；对称性、identity 和 C/PU 联合行置换检查通过。question-raw OT 的 identity 非零是边际定义所致，excess identity 为零。
- 在 8 任务各 hash 选 1 图、64 个固定 trial 上比较 512 / 2048 空间质心与 full-token OT：[分辨率审计](../artifacts/token_proxy/encoder_candidates_v1/transport_resolution_audit.json)。512→full 的 uniform/问题增量排序相关约 0.851/0.861，但相对 MAE 约 52%；2048 仍有约 36%–39% 相对 MAE。因而**追加了全部 93,626 个 trial 的 full-token OT**，主表不使用粗空间近似。
- [CPU float64 / NPU float32 交叉验证](../artifacts/token_proxy/encoder_candidates_v1/npu_transport_check.json)：同 64 个 full-token trial 的最大 loss 差 $8.94\times10^{{-8}}$。
- 全量 full-token OT 的 per-trial 最大边际误差：median={d['max_marginal_error']['median']:.3g}、max={d['max_marginal_error']['max']:.3g}；**{d['trials_above_1e_6']:,}/{d['trial_count']:,}** 个 trial 的最大误差高于 $10^{{-6}}$（每 trial 包括复用的 teacher-self 求解）。因此不声称所有求解都精确满足目标阈值。最大迭代数 {d['max_iterations']['max']:.0f}。完整诊断见 JSON。
- 合成满秩混合例中 $X\approx XC^+C$ 的相对 residual 约 $1.23\times10^{{-15}}$，说明 span 重建的结构盲区；这不是对全部真实 C 的满秩证明。记录的 `student_m_ge_d` 仅为行数条件，不能解释为实测秩。
- set/span/固定空间读取 loss 对 C 和 PU 的联合行置换不变，而 causal decoder 不一定置换不变；因此这些指标不可能自动提供对任意 decoder 的保真保证。
- generator、samples、head、line head 与关键源码 SHA256 保存在 protocol/analysis 中，完整输出另由 [final_audit.json](../artifacts/token_proxy/encoder_candidates_v1/final_audit.json) 逐图逐 trial 检查。源码与原始作答缓存不以结果好坏进行替换。

## 7. 如何解释和继续

本轮支持的结论是：**这组严格 encoder-only 的具体实例，在现有 balanced random convex-mixing 分布上，尚不足以充当可信的逐次下游质量 proxy。** 不支持推断“encoder 有问题”“所有 decoder-free 指标都不可能”或“按这些 loss 训练一定无效”。这不是上述参考论文原算法的完整复现，更不是对其方法有效性的反证。

问题权重依赖词法重合；数字 head 判断的是类型而非精确值；行结构不是语义关系；subspace 允许 LLM 不会实现的任意线性重建；随机 global mixing 会把空间证据大量聚拢。这些都是“loss 可变而下游已几乎全错”或“loss 小却仍丢失答案”的具体机制。另有 80% 左右零分和高平局率，令当前实验对较细差异的辨别力有限。

所有 VQA 图像来自已经分析过的旧评测批次，因此是**探索性重放，不是未见过的确认性 holdout**。公式及组合未用 VQA 分数拟合，补充 line probe 与 OT 精度改进已明确标注；没有从结果中追选“最好方案”作为经验证的结论。

下一步更值得做的是：先用独立 OCR/表格训练标注构建能区分“12.5 / 125 / 12.6”以及表头–值绑定的非生成式探针，并单独验证其准确率；再在下游不过度归零、仍有相同预算内随机差异的压缩分布上检验。若改变压缩族，应作为新的实验协议，不能替换本轮相同生成器的结果。若要声称跨 decoder 泛化，应冻结全部 loss 参数后迁移另一个接口/LLM 验证。

## 8. 运行与文件入口

```bash
cd /path/to/tessfold
# 已有 .venv（CPU 统计/探针）和 .venv-und（Ascend encoder/OT），均为 venv。
bash scripts/run_encoder_candidates.sh status
# 完整恢复/重跑：复用已有 checkpoint 和逐图缓存，提交有依赖关系的 Slurm jobs。
bash scripts/run_encoder_candidates.sh submit
# 已有所有 loss 时，仅重做统计、数值检查与 Markdown：
bash scripts/run_encoder_candidates.sh check
bash scripts/run_encoder_candidates.sh report
```

`submit` 会串联：准备 OCR protocol → encoder-only OCR feature cache → evidence head（已有则复用）→ CPU candidates / line probe / full-token NPU OT → 数学检查与统计报告。**它要求原 `random_v1_20x` 作答/特征/矩阵缓存存在**；若从零生成下游缓存，先按 [20 倍规模运行文档](TOKEN_PROXY_20X.md) 操作。若修改 head/公式，必须使用新的实验目录或完整清理所有相关派生 loss，不能只替换权重后混用旧缓存。当前入口默认复用，不自动破坏既有实验。

| 文件 | 作用 |
|---|---|
| [protocol.json](../artifacts/token_proxy/encoder_candidates_v1/protocol.json) | 固定输入、超参数、排除变体、原矩阵/样本 hash |
| [encoder_evidence.py](../vqa/encoder_evidence.py) / [encoder_candidate_losses.py](../vqa/encoder_candidate_losses.py) | 词法桥、采样 coverage、证据 JS、ridge、空间近似 OT |
| [exact_encoder_transport_npu.py](../scripts/exact_encoder_transport_npu.py) | 完整 token OT，无空间合并 |
| [train_encoder_evidence.py](../scripts/train_encoder_evidence.py) / [train_encoder_relation.py](../scripts/train_encoder_relation.py) | 独立 OCR / same-line head 训练 |
| [report_encoder_candidates.py](../scripts/report_encoder_candidates.py) | 不平均 trial 的统计与完整 CSV |
| [trials.csv](../artifacts/token_proxy/encoder_candidates_v1/trials.csv) | 所有 93,626 个 trial，gold/双作答/双分数/全部新 loss |
| [counterexamples.json](../artifacts/token_proxy/encoder_candidates_v1/counterexamples.json) | 同题同率同族内低 loss 反而下降更大的具体作答对 |
| [analysis.json](../artifacts/token_proxy/encoder_candidates_v1/analysis.json) | 全量 pooled、分层、bootstrap CI、control、诊断及源码 hash |

CSV 的 `ratio` 是**保留率 M/N**；`score_drop` 越大越差；loss 越大按设计也应越差。`loss_sinkhorn_full_question_excess`、`loss_evidence_combined_line_question` 等为本报告字段；按 `sample_id / family / ratio` 筛选两个 seed，便可查看不经平均的直接对照。
'''
    text=text.replace(chr(92)*2,chr(92))
    (ROOT/'docs/ENCODER_ONLY_LOSS_VALIDATION.md').write_text(text)
    print('Wrote docs/ENCODER_ONLY_LOSS_VALIDATION.md')
if __name__=='__main__':main()
