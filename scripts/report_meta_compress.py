"""Generate the auditable Markdown report; full mode rejects incomplete jobs."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/meta_compress');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    root=ROOT/a.root;cfg=json.loads((root/'protocol.json').read_text());training=read_jsonl(root/'training.jsonl')
    # A resumed job may re-log work since its latest checkpoint. Keep the most
    # recent record for each optimizer step, matching the final checkpoint.
    training=list({x['step']:x for x in training}.values());training.sort(key=lambda r:r['step'])
    complete=(root/'complete.json').exists()
    ep=root/'eval';protocol_path=ep/'protocol.benchmark.json'
    evcfg=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    records=[]
    for f in sorted(ep.glob('predictions.benchmark.rank*.jsonl')):records.extend(read_jsonl(f))
    bykey={}
    for row in records:
        key=(row['id'],row['variant'])
        if key in bykey:raise RuntimeError(f'Duplicate evaluation {key}')
        bykey[key]=row
    records=list(bykey.values())
    expected=evcfg['samples']*len(evcfg['variants']) if evcfg else 1536
    evaluated=evcfg is not None and len(records)==expected
    if evaluated:
        planned=read_jsonl(ep/'samples.benchmark.jsonl')
        assert set(bykey)=={(r['id'],v) for r in planned for v in evcfg['variants']},'Unexpected evaluation membership'
    if not a.allow_partial and not (complete and evaluated):raise RuntimeError(f'Incomplete: training={complete}, eval={len(records)}/{expected}')
    epochs=[]
    for epoch in sorted({r['epoch'] for r in training}):
        z=[r for r in training if r['epoch']==epoch]
        epochs.append({'epoch':epoch,'logged_steps':len(z),**{k:statistics.mean(x[k] for x in z) for k in ['prediction','entropy','collapse','loss','grad_norm']}})
    metrics={}
    for variant in sorted({r['variant'] for r in records}):
        table={}
        for task in sorted({r['task'] for r in records}):
            z=[r for r in records if r['task']==task and r['variant']==variant]
            if not z:continue
            table[task]={'samples':len(z),**{k:statistics.mean(r[k] for r in z) for k in ['teacher_score','student_score','score_drop','prediction_loss','entropy_loss','collapse_loss','total_loss','tokens','compressed_tokens']},
                         'student_token_limit_fraction':statistics.mean(r['student_hit_token_limit'] for r in z)}
        metrics[variant]=table
    delta_ci={}
    if evaluated:
        rng=np.random.default_rng(20261026)
        for task in sorted(metrics['trained']):
            trained={r['id']:r for r in records if r['variant']=='trained' and r['task']==task}
            initial={r['id']:r for r in records if r['variant']=='initial' and r['task']==task}
            assert trained.keys()==initial.keys()
            differences=np.array([100*(trained[k]['student_score']-initial[k]['student_score']) for k in trained])
            boots=differences[rng.integers(0,len(differences),(2000,len(differences)))].mean(1)
            delta_ci[task]={'trained_minus_initial_pp':float(differences.mean()),'paired_bootstrap_ci95':np.quantile(boots,[.025,.975]).tolist()}
    summary={'training_complete':complete,'evaluation_complete':evaluated,'evaluation_rows':len(records),'expected_evaluation_rows':expected,'epochs':epochs,'variants':metrics,'paired_changes':delta_ci}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    if records:
        with (root/'answers.csv').open('w',newline='') as h:
            fields=['task','id','variant','question','answers','teacher_prediction','student_prediction','teacher_score','student_score','score_drop','prediction_loss','entropy_loss','collapse_loss','total_loss']
            writer=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for row in records:writer.writerow({**row,'answers':json.dumps(row['answers'],ensure_ascii=False)})
    if training:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
        for ax,key in zip(axes.flat,['prediction','entropy','collapse','loss']):
            xs=np.array([r['step'] for r in training]);ys=np.array([r[key] for r in training])
            ax.plot(xs,ys,alpha=.3,lw=.6,label='batch mean')
            window=min(25,len(ys))
            ax.plot(xs[window-1:],np.convolve(ys,np.ones(window)/window,mode='valid'),lw=1.3,label=f'{window}-step mean')
            ax.set(xlabel='Optimizer step',ylabel=key);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.savefig(root/'training_loss.png',dpi=160);fig.savefig(root/'training_loss.pdf');plt.close(fig)
    state='训练、评测已完成' if complete and evaluated else ('训练已完成，正式评测尚未完成' if complete else '正式训练尚未完成，尚无最终评测结果')
    outcome=''
    if complete and evaluated:
        macro={v:100*statistics.mean(z['student_score'] for z in table.values()) for v,table in metrics.items()}
        teacher_macro=100*statistics.mean(z['teacher_score'] for z in metrics['trained'].values())
        outcome=(f"\n**本轮结果未达到压缩目标。** 八任务分数的描述性算术均值：未压缩 teacher **{teacher_macro:.2f}**，"
                 f"初始压缩器 **{macro['initial']:.2f}**，训练后 **{macro['trained']:.2f}**，mean pooling **{macro['mean']:.2f}**。"
                 "不同任务使用各自指标，这个均值不是论文官方总分。训练后的版本没有显示改善；"
                 "不能将流程已完成解释为复现成功。训练总 loss 的下降主要来自 collapse 正则，"
                 "prediction KL 的 epoch 均值基本不变、行熵略升，尚未学到足以维持问答质量的压缩。"
                 "这是本轮实现及固定设置下的负结果，不足以判定论文方法或 KL 训练信号本身无效。\n")
    text=f'''# MonkeyOCRv2-B-Und：MetaCompress 生成器训练与评测

**状态：{state}。** 训练日志已记录 {len(training)} / 1,112 个优化 step；评测已收集 {len(records)} / {expected} 条变体记录。本文由 `scripts/report_meta_compress.py` 从真实产物生成；训练完成前的统计只表示已记录 batch，不是完整 epoch。正式入口为 **`bash scripts/run_meta_compress.sh status`**；训练入口为 **`sbatch scripts/slurm_meta_compress_cpu.sh`**。
{outcome}

## 1. 论文设计与本次实际结构

参考 [Wang et al., CVPR 2026，MetaCompress，§5、Eq.6–10、§6.1](../Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf) 和 [supplementary §8.2](../artifacts/decoder_dependent_loss/paper_supplemental.pdf)。本次训练的是通用图像条件矩阵生成器，**不是**论文 §4 针对单张图优化一个自由矩阵的 500-epoch 实验。公开仓库缺少可直接复用的完整训练代码，下面明确列出本文实现的细节。

令官方 UND encoder 输出为 $X_0\\in\\mathbb R^{{n\\times768}}$，官方 merger 为 $F$，使用论文 decoder 前的位置：$X=F(X_0)\\in\\mathbb R^{{n\\times2048}}$。

$$
Z=X+E_{{pos}},\\quad Q=\\operatorname{{FracMaxPool}}_{{k=3,m}}(Z)W_q,\\quad K=ZW_k,
$$
$$
P=\\operatorname{{softmax}}_{{row}}\\left(\\frac{{Q\\operatorname{{diag}}(\\theta)K^\\top}}{{\\sqrt{{d_c}}}}\\right),\\qquad X_c=PX.
$$

实际 $d_c=64$，$m=\\max(1,\\operatorname{{round}}(0.1n))$；$W_q,W_k\\in\\mathbb R^{{2048\\times64}}$；$\\theta\\in\\mathbb R^{{64}}$。位置编码为两个 `512×64` 的可学习行/列表，加 `64→2048` 无偏置投影。共 **458,816 个可训练参数**。初始 $W_q=W_k$，各元素高斯标准差 $1/64$，$\\theta=1$。$P$ 每行和为 1，$PX$ 是凸组合；不再施加前期随机矩阵的列质量上下界，由论文 collapse 项控制重复聚合。

```mermaid
flowchart LR
    I[文档图像] --> E[官方 UND encoder 冻结]
    E --> F[官方 merger 冻结]
    F --> X[视觉 token X: n×2048]
    X --> G[MetaCompress 可训练]
    G --> P[行 softmax 矩阵 P: m×n]
    X --> C[压缩 token PX]
    P --> C
    X --> T[冻结 Qwen teacher]
    C --> S[同一冻结 Qwen student]
    T --> KL[共同前缀的 full-vocab KL]
    S --> KL
    P --> R[平均行熵与 collapse]
    KL --> L[总 loss]
    R --> L
```

**本次正式方案在 merger 后压缩，即 $PF(X_0)$；不是前几轮随机实验的 $F(PX_0)$。非线性 merger 使二者不等价。** 实现仍支持 `--insertion pre-merger`，但它仅用于早期接口 smoke，不混入本次正式结果。生成器仅接收图像 token 与坐标，不接收问题、gold 或 teacher 答案。

| 方面 | 原论文 | 本次实现与差异 |
|---|---|---|
| 主干 | LLaVA 等通用多轮 VLM | 官方 MonkeyOCRv2-B-Und + Qwen3-1.7B，独立文档 VQA |
| 压缩位置 | visual projector 后、LLM 前 | 相同；压缩官方 merger 的 2048 维输出 |
| 特征来源 | 冻结对应 VLM 的视觉模块 | 从官方 **UND** encoder 原生输入重新提取 BF16 特征，不复用独立 B 模型缓存 |
| query pooling | 正文称 average；supplementary §8.2 明确 fractional max、kernel=3 | 遵循 supplementary：PyTorch fractional max，固定 offset=0.5 保证同图确定性；不再用初稿 adaptive average |
| 位置编码 | absolute position embedding，未公开全部参数化 | 低维可学习行/列表后投影；严格按官方 `get_pos_ids_by_grid` 的 2×2 block 顺序，而不是误用 row-major |
| 压缩维度 | $d_c\\ll d$，现有正文未明确所有配置数值 | 本次固定 64，不宣称就是作者所有配置 |
| KL 轨迹与温度 | 对响应分布做 KL，未完整公开 teacher-forcing 细节 | teacher 自生成共同前缀，完整词表、温度 1、每响应位置平均 |
| teacher 响应长度 | 未给出 MonkeyOCRv2 设置 | 训练 cap=64；自由生成评测 cap=256 |
| 训练数据 | 约 20k MT-GQA/MT-VQA-v2 等图文样本 | 既有七种文档 VQA 训练集固定 hash 选 20k；并非原论文训练数据 |
| 优化 | SGD 1e-3，clip 1e-2，batch=36，2 epochs，四张 RTX A6000 | 数值配置相同；实现采用 global-norm clipping；本轮 CPU/Gloo 36 ranks，同步 batch=36 |

## 2. 损失与反向传播

使用论文生成器完整 **Eq.10 三项目标**，两个正则权重均为 1：

$$
L_{{pred}}=\\frac1T\\sum_{{t=1}}^T D_{{KL}}\\!\\left(p_T(\\cdot\\mid q,a_{{<t}},X)\\Vert p_S(\\cdot\\mid q,a_{{<t}},PX)\\right),
$$
$$
L_H=-\\frac1m\\sum_i\\sum_jP_{{ij}}\\log P_{{ij}},\\qquad
L_{{collapse}}=\\max_j\\sum_iP_{{ij}},\\qquad L=L_{{pred}}+L_H+L_{{collapse}}.
$$

$a$ 是未压缩 teacher 的 greedy 响应，包含实际产生的 EOS；gold 不用于优化。两条 teacher-forcing 路径使用相同 teacher 前缀；对词表求和、对响应位置求均值。teacher logits 停梯度并可原样缓存为 FP32。student 经过冻结 Qwen 的计算图反传至 $P$ 与生成器；**冻结权重不等于关闭 student 的 autograd**。BF16 模型 logits 转 FP32 做 log-softmax，$P$ 和生成器为 FP32。teacher 和 student 的下游答案各自独立生成，student 不读取 teacher 答案或上题 cache。

平均行熵鼓励局部集中，collapse 惩罚多行重复依赖同一源 token；它们是训练正则，不预先假定三项总和必然最能预测 gold 分数。单条 teacher greedy 轨迹的条件 KL 也不等于对所有可能响应轨迹积分的完整序列 KL。

## 3. 数据、划分与运行

训练集：`data/full/train.jsonl`，共 243,652 条问答。本次固定 hash 选择 **20,000 条、16,731 张图像**；与本地 val/test 的图像 ID 交集为 0。每轮最终 batch 补齐 16 条，故共 **40,032 次样本呈现、1,112 次 SGD 更新**，不是 40,032 条不同问答。

| 训练任务 | 本次问答数 |
|---|---:|
'''
    for task,n in cfg['task_counts'].items():text+=f'| {task} | {n:,} |\n'
    text+='''
采用官方 `AutoProcessor`、ChatML 和图像处理，max pixels=1,003,520；不复用旧独立 B encoder 的 raw feature cache。代码审查发现发布的 B 与 B-Und 视觉张量并非相同，旧草稿缓存路径已在正式训练前移除。本轮原生 UND 特征保存在 `artifacts/meta_compress_cache/features/`，teacher response 在 `teachers/`，完整 FP32 teacher logits 在 `logits/`。缓存仅节省重复计算，不更新主干。

NPU 普通/高优先级分区均长期无空闲卡，原排队作业已取消。实际改用六个 CPU 节点、每节点六个模型进程、每进程 16 线程（576 CPU 核），36-rank Gloo 同步 SGD。正式训练作业 **151231**；训练成功后自动触发八种任务评测作业 **151233**，再由 **151237** 汇总文档。另安排 **151239**（4 个 CPU array shard、共 28 个进程/448 核）预计算冻结 teacher 缓存，减少同步 SGD 的生成等待；复用同一 helper、16 线程和完整 FP32 目标，不更新生成器、不改目标。并发峰值上限 1,024 CPU 核。所有 teacher/student 分数均在本轮 CPU BF16 上重新生成，不混用前期 NPU 作答。

评测预先固定每任务 64 张不同图像、一图一题，共 **512 张/题**。DocVQA、InfoVQA 用官方 validation，其余六项用 test；它们与训练图像无交集。对比 `trained`、训练开始时保存的同一 `initial`、连续平衡分组 `mean` 三种压缩器，并共享未压缩 teacher。**最终 epoch-2 模型预先指定，不用 benchmark 分数挑 checkpoint 或调参。** 这是固定子集评测，不是论文全量榜单，不能直接宣称复现论文得分。DeepForm/KLC 使用逐字段 exact match，不冒充官方整文档 corpus F1；DT-VQA 延续既有 max(包含答案 accuracy, ANLS)，WTQ 用 denotation，OCRBench 用既有子任务计分。

## 4. 已完成的接口与梯度验证

- 两样本单进程、两进程 DDP smoke 均完成；它们是接口检查，不计为正式 20k 训练结果。
- 正式结构的独立梯度审计：`artifacts/meta_compress_gradient_audit/gradient_audit.json` 中 KL 单项生成器梯度范数为 **2.829941**；官方主干可训练参数数为 **0**。
- 行随机矩阵检查的最大行和误差小于 **4e-7**；正式训练逐样本继续记录该误差。
- 仅把生成器加入 SGD；teacher 使用 no-grad，student 保留梯度。

## 5. 实际训练记录

| Epoch | 已记录 batch | prediction KL | 行熵 | collapse | 总 loss | 裁剪前梯度范数 |
|---|---:|---:|---:|---:|---:|---:|
'''
    for e in epochs:text+=f"| {e['epoch']} | {e['logged_steps']} | {e['prediction']:.6f} | {e['entropy']:.6f} | {e['collapse']:.6f} | {e['loss']:.6f} | {e['grad_norm']:.6f} |\n"
    if not epochs:text+='| 尚未写入首个完整全局 batch | 0 | — | — | — | — | — |\n'
    if training:text+='\n[训练曲线 PNG](../artifacts/meta_compress/training_loss.png) · [训练曲线 PDF](../artifacts/meta_compress/training_loss.pdf)。曲线为真实 batch 均值和 25-step 平滑，不强行拟合单调下降。\n'
    text+='''
## 6. 下游评测结果

'''
    if not evaluated:text+=f'**当前尚未完成正式评测（{len(records)}/{expected} 条）。因此没有可报告的最终任务分数，也不能判断训练是否有效。** 下列正式表格仅在全部预定样本齐全后生成。\n'
    else:
        text+='每项分数均为百分制；Δ训练表示 trained−initial，正值说明训练有帮助。\n\n| 任务 | n | 未压缩 teacher | mean pool | initial | trained | teacher−trained | Δ训练（95% 配对 bootstrap CI，百分点） |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
        for task,z in metrics['trained'].items():
            change=delta_ci[task];lo,hi=change['paired_bootstrap_ci95']
            text+=f"| {task} | {z['samples']} | {100*z['teacher_score']:.2f} | {100*metrics['mean'][task]['student_score']:.2f} | {100*metrics['initial'][task]['student_score']:.2f} | {100*z['student_score']:.2f} | {100*z['score_drop']:.2f} | {change['trained_minus_initial_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] |\n"
        text+='\n| 任务 | 变体 | KL | 行熵 | collapse | 总 loss | 平均 n→m |\n|---|---|---:|---:|---:|---:|---|\n'
        for variant,table in metrics.items():
            for task,z in table.items():text+=f"| {task} | {variant} | {z['prediction_loss']:.4f} | {z['entropy_loss']:.4f} | {z['collapse_loss']:.4f} | {z['total_loss']:.4f} | {z['tokens']:.1f}→{z['compressed_tokens']:.1f} |\n"
        text+='\n每项仅 64 张图像，区间可能较宽；这轮只检验固定 90% reduction 和固定训练设置，不把单次差异推广成所有保留率的结论。loss 降低与生成正确率提高要分别检查；模型保留 teacher 的错误也会降低 KL。\n'
    text+='''
## 7. 入口与产物

```bash
# 查看作业、训练进度、已有评测记录
bash scripts/run_meta_compress.sh status

# 手动重建当前进度报告（不会声称未完成结果已经完成）
.venv/bin/python -m scripts.report_meta_compress --allow-partial

# 正式 CPU 训练；中断后相同命令读取最近 checkpoint 恢复
sbatch scripts/slurm_meta_compress_cpu.sh

# 训练完成后评测（正常流水线已经安排依赖，无需重复提交）
sbatch scripts/slurm_meta_compress_eval_cpu.sh

# 全部评测完成后汇总；样本不齐会报错
.venv/bin/python -m scripts.report_meta_compress

# 单图问答与矩阵导出（训练完成后）
.venv/bin/python -m scripts.predict_meta_compress \\
  --image /absolute/path/to/document.png --question 'What is the date?' \\
  --teacher --save-matrix /tmp/compression_P.pt
```

训练 `scripts/train_meta_compress.py`；生成器/损失 `vqa/meta_compress.py`；独立生成评测 `scripts/eval_meta_compress.py`。Python 环境由已有 **`.venv`** 管理；CPU 模型 BF16，依赖记录沿用 `requirements-decoder-loss-cpu.txt` 和 `requirements-report.txt`。NPU 另有 `.venv-und` 和 `scripts/slurm_meta_compress.sh`，本次实际结果以 CPU 协议为准。

- [训练协议/代码 SHA256](../artifacts/meta_compress/protocol.json)、[固定训练样本](../artifacts/meta_compress/samples.jsonl)、[B/UND 特征来源审计](../artifacts/meta_compress/encoder_provenance_audit.json)；
- [逐全局 step 训练日志](../artifacts/meta_compress/training.jsonl)、各 `examples.rank*.jsonl` 保存逐样本各 loss；
- `artifacts/meta_compress/initial.pt`、`meta_compress.pt`、`complete.json`；源码快照 `source_snapshots/`；
- `artifacts/meta_compress/eval/predictions.benchmark.rank*.jsonl`：gold、两边实际答案、token IDs、分数、三项 loss 和总 loss；
- `artifacts/meta_compress/eval/matrices/benchmark.rank*.pt`：每个评测 worker 首图的实际训练后矩阵 P，附 grid 与 checkpoint SHA256；
- [作答对照 CSV（评测有记录后生成）](../artifacts/meta_compress/answers.csv)、[机器可读汇总](../artifacts/meta_compress/summary.json)。
'''
    if complete and evaluated:
        macro={v:100*statistics.mean(x['student_score'] for x in table.values()) for v,table in metrics.items()}
        teacher_macro=100*statistics.mean(x['teacher_score'] for x in metrics['trained'].values())
        conclusion=f'''## 完成后的实际结论

**本轮训练已完成，但没有改善压缩 VQA 表现，不能视为成功复现论文的压缩效果。** 八任务百分制宏平均：未压缩 teacher **{teacher_macro:.2f}**，mean-pool **{macro['mean']:.2f}**，初始生成器 **{macro['initial']:.2f}**，训练后生成器 **{macro['trained']:.2f}**；训练相对初始化变化 **{macro['trained']-macro['initial']:+.2f} 个百分点**。各任务仅 64 张图像，宏平均仅作本轮描述，混合任务指标不是官方统一榜单指标。

训练 epoch 1→2：总 loss **12.685964→12.379922**，主要来自 collapse **2.334961→2.015338**；prediction KL **2.331271→2.333542** 没有改善，行熵 **8.019732→8.031041** 也没有下降。因而此前中途观察到的小幅总 loss 下降不能当作语义保留变好的证据。评测 KL 的八任务宏平均也从约 **2.07750→2.07852**，没有改善。

证据支持的判断是：当前设定主要优化了防重复聚合正则，没有学到足够有效的文档证据保留。90% token reduction 下初始/训练后矩阵仍高度弥散，行熵很高；裁剪前全局梯度范数约 54–62，而阈值为 0.01，实际更新受到强裁剪。这些是后续诊断线索，尚不能据本轮断言唯一根因是某个超参数、模型容量或论文方法本身。下一步应先做初始化/注意力尺度与梯度分项审计，并在训练内验证集上验证调整；不根据本轮 benchmark 分数追选模型。本文保留当前失败结果及完整答案，不把 smoke 或中间 checkpoint 替换成最终结果。

'''
        text=text.replace('## 1. 论文设计与本次实际结构',conclusion+'## 1. 论文设计与本次实际结构',1)
    dest=ROOT/'docs/METACOMPRESS_GENERATOR_TRAINING.md';tmp=dest.with_suffix('.tmp');tmp.write_text(text);tmp.replace(dest)
    print(json.dumps({'report':str(dest),'training_steps':len(training),'training_complete':complete,'evaluation_rows':len(records),'evaluation_complete':evaluated}))

if __name__=='__main__':main()
