"""Strict report for the completed isolated diagnostics; refuses partial tables."""
import csv
import json
import math
import statistics as st
from pathlib import Path
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from scripts.diagnose_meta_compress import ROOT,OUT,VARIANTS
from scripts.diagnose_meta_compress_factorial import VARIANTS as FACTORIAL
from scripts.train_meta_compress import sha,atomic_json


def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def table(headers,rows):
    def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in rows])
def f(v):return f'{v:.5f}'


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'samples.jsonl')==protocol['samples_sha256']
    assert sha(ROOT/'artifacts/meta_compress/initial.pt')==protocol['original_initial_sha256']
    assert sha(ROOT/'artifacts/meta_compress/meta_compress.pt')==protocol['original_final_sha256']
    samples=read(OUT/'samples.jsonl');ids={r['id'] for r in samples};fit=next(r for r in samples if r['role']=='fit')
    results=[];records=[];traces={};grad_rows=[]
    variants=VARIANTS+FACTORIAL
    for cfg in variants:
        out=(OUT if cfg in VARIANTS else OUT/'factorial')/cfg['name'];complete=json.loads((out/'complete.json').read_text())
        assert complete['steps']==50 and complete['base_trainable_parameters']==0
        ev=read(out/'evaluations.jsonl');trace=read(out/'training.jsonl');traces[cfg['name']]=trace
        assert [r['step'] for r in trace]==list(range(1,51))
        expected=ids if cfg['kind']!='free' else {fit['id']}
        assert {(r['id'],r['stage']) for r in ev}=={(i,s) for i in expected for s in ('initial','final')}
        assert len(ev)==len(expected)*2
        for r in ev:r['variant']=cfg['name'];r['ratio']=cfg['ratio'];records.append(r)
        a=next(r for r in ev if r['role']=='fit' and r['stage']=='initial')
        b=next(r for r in ev if r['role']=='fit' and r['stage']=='final')
        held={s:[r for r in ev if r['role']=='holdout' and r['stage']==s] for s in ['initial','final']}
        result={'variant':cfg['name'],'config':cfg,'fit_initial':a,'fit_final':b,
            'holdout':{s:{'n':len(v),'score_percent':100*st.mean(r['student_score'] for r in v) if v else None,
                         'kl':st.mean(r['prediction'] for r in v) if v else None} for s,v in held.items()},
            'update_path_length':sum(r['update_norm'] for r in trace),
            'median_clip_multiplier':st.median(r['clip_multiplier'] for r in trace),
            'parameter_changes':json.loads((out/'parameter_changes.json').read_text())}
        results.append(result)
        ga=json.loads((out/'gradients.initial.json').read_text());gb=json.loads((out/'gradients.final.json').read_text())
        for stage,g in [('initial',ga),('final',gb)]:
            grad_rows.append([cfg['name'],stage,*[f(g['norms'][k]) for k in ['prediction','entropy','collapse']],
                *[f(g['cosines'][k]) for k in ['prediction__entropy','prediction__collapse','entropy__collapse']]])
    controls=read(OUT/'controls/evaluations.jsonl');cc=json.loads((OUT/'controls/complete.json').read_text())
    assert cc['evaluations']==48 and len(controls)==48
    assert {(r['id'],r['method'],r['ratio']) for r in controls}=={(i,m,q) for i in ids for m in ('mean','sample') for q in (.1,.25,.5)}
    for r in controls:r['variant']='fixed_'+r['method'];r['stage']='fixed';records.append(r)
    full_budget=read(OUT/'full_budget/records.jsonl')
    assert json.loads((OUT/'full_budget/complete.json').read_text())['records']==3 and len(full_budget)==3
    for r in full_budget:
        assert r['tokens']==r['compressed_tokens'] and r['prompt_and_position_length_unchanged']
        r['variant']='dense_full_budget';r['stage']='fixed';r['ratio']=1.;records.append(r)
    fieldnames=list(dict.fromkeys(k for r in records for k in r))
    with (OUT/'answers.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fieldnames);w.writeheader();w.writerows(records)
    interface=json.loads((OUT/'interface_audit.json').read_text())
    precision=json.loads((OUT/'controls/feature_precision_audit.json').read_text())
    directions=json.loads((OUT/'precision/complete.json').read_text())
    matrix_manifest=json.loads((OUT/'matrices/complete.json').read_text())
    assert len(matrix_manifest['matrices'])==5
    assert all(r['code_sha256']==sha(ROOT/'scripts/visualize_meta_compress_diagnosis.py') for r in matrix_manifest['matrices'])
    task_grads=read(OUT/'task_gradients/records.jsonl')
    assert json.loads((OUT/'task_gradients/complete.json').read_text())['records']==16 and len(task_grads)==16
    assert {(r['id'],r['state']) for r in task_grads}=={(i,s) for i in ids for s in ['original_initial','original_20k_final']}
    extension=json.loads((OUT/'factorial/protocol.json').read_text())
    summary={'protocol':protocol,'extension_protocol':extension,'interface':interface,'results':results,'fixed_controls':controls,'full_budget':full_budget,'precision':precision,'finite_differences':directions,'task_gradients':task_grads}
    atomic_json(OUT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for color_index,cfg in enumerate(variants):
        tr=traces[cfg['name']]
        for ax,key in zip(axes.flat,['prediction','entropy','collapse','update_norm']):
            ax.plot([r['step'] for r in tr],[r[key] for r in tr],label=cfg['name'],lw=1.25,
                color=plt.get_cmap('tab20')(color_index))
    for ax,label in zip(axes.flat,['Teacher-prefix KL (nats)','Row entropy (nats)','Collapse (max column mass)','Actual update L2 norm']):
        ax.set_ylabel(label);ax.set_xlabel('Single-image update');ax.grid(alpha=.2)
    axes[0,0].set_yscale('log');axes[1,0].set_yscale('log');axes[1,1].set_yscale('log')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.14,1,1));fig.savefig(OUT/'ablation_curves.png',dpi=160);fig.savefig(OUT/'ablation_curves.pdf');plt.close(fig)
    main_rows=[]
    for r in results:
        a,b=r['fit_initial'],r['fit_final'];h=r['holdout']
        hp='N/A' if h['final']['n']==0 else f"{h['initial']['score_percent']:.2f} → {h['final']['score_percent']:.2f}"
        hk='N/A' if h['final']['n']==0 else f"{h['initial']['kl']:.4f} → {h['final']['kl']:.4f}"
        main_rows.append([r['variant'],r['config']['ratio'],f(a['prediction'])+' → '+f(b['prediction']),
            f"{100*a['student_score']:.2f} → {100*b['student_score']:.2f}",hp,hk])
    fixed_rows=[]
    for ratio in [.1,.25,.5]:
        for method in ['mean','sample']:
            subset=[r for r in controls if r['method']==method and r['ratio']==ratio]
            train=next(r for r in subset if r['role']=='fit');hold=[r for r in subset if r['role']=='holdout']
            fixed_rows.append([method,ratio,f(train['prediction']),f"{100*train['student_score']:.2f}",
                f(st.mean(r['prediction'] for r in hold)),f"{100*st.mean(r['student_score'] for r in hold):.2f}"])
    sample_rows=[[r['role'],r['task'],r['id'],r['diagnostic_tokens'],r['diagnostic_teacher_prediction']] for r in samples]
    configs=[[v['name'],v['kind'],v['objective'],v['optimizer'],v['lr'],v['clip'],v['ratio']] for v in variants]
    update_rows=[[r['variant'],f"{r['median_clip_multiplier']:.7g}",f"{r['update_path_length']:.7g}",
        f(r['fit_final']['entropy_fraction']),f(r['fit_final']['effective_support']),f(r['fit_final']['collapse'])] for r in results]
    pred_rows=[[r['variant'],r['fit_initial']['student_prediction'],r['fit_final']['student_prediction'],fit['diagnostic_teacher_prediction']] for r in results]
    text='''# MetaCompress 失败排查：输入审计与十二组干预实验

本报告记录实际完成的隔离诊断，原 20k/2 epochs 训练与 checkpoint 均保留。十二组各完成 50 次单图更新，另完成 48 次固定压缩对照、3 次完整预算对照及 BF16/FP32 方向有限差分；不是新的正式 benchmark，也不是根据测试集重新选模型。

## 入口与原始数据

- 实验脚本：[`scripts/diagnose_meta_compress.py`](../scripts/diagnose_meta_compress.py)。
- Slurm 入口：[`scripts/slurm_meta_compress_diagnosis.sh`](../scripts/slurm_meta_compress_diagnosis.sh)。
- 固定对照：[`scripts/diagnose_meta_compress_controls.py`](../scripts/diagnose_meta_compress_controls.py)。
- 补全 KL-only 裁剪对照与局部 full-loss 对照：[`scripts/diagnose_meta_compress_factorial.py`](../scripts/diagnose_meta_compress_factorial.py)，先 `--prepare`，再分别运行 `--variant 0` / `--variant 1`。未修改第一批冻结的代码或 protocol。
- 精度审计：[`scripts/diagnose_meta_compress_precision.py`](../scripts/diagnose_meta_compress_precision.py)。
- 跨任务梯度审计：[`scripts/diagnose_meta_compress_task_gradients.py`](../scripts/diagnose_meta_compress_task_gradients.py)。
- 完整 token 预算对照：[`scripts/diagnose_meta_compress_full_budget.py`](../scripts/diagnose_meta_compress_full_budget.py)。
- 矩阵导出 / 图示：[`scripts/visualize_meta_compress_diagnosis.py`](../scripts/visualize_meta_compress_diagnosis.py)，`.pt` 中包含 P、grid、配置和 checkpoint SHA256。
- 状态 / 报告统一入口：`bash scripts/run_meta_compress_diagnosis.sh status|report`。
- 报告入口：[`scripts/report_meta_compress_diagnosis.py`](../scripts/report_meta_compress_diagnosis.py)。
- 每题原始作答：[`answers.csv`](../artifacts/meta_compress_diagnosis/answers.csv)；完整数值：[`summary.json`](../artifacts/meta_compress_diagnosis/summary.json)。每组目录另有 50 步训练记录、首末梯度分项、参数变化、最终 checkpoint。

```bash
# 首次准备固定 cohort；已有 protocol 时会拒绝覆盖
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.diagnose_meta_compress --prepare
sbatch scripts/slurm_meta_compress_diagnosis.sh
# 固定对照也应在 compute node 执行
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 .venv/bin/python -m scripts.diagnose_meta_compress_controls
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.report_meta_compress_diagnosis
```

## 1. 固定协议与结论边界

模型为官方 MonkeyOCRv2-B-Und，vision encoder、官方 merger 和 Qwen3-1.7B 全冻结。沿用原训练的 post-merger 插入位置、BF16 主干、FP32 生成器。训练目标是 teacher 自由生成答案的共享前缀、全词表正向 KL，含 EOS；gold 不参与 loss。

这里的官方 merger 属于 **MonkeyOCRv2**，将 encoder 的 768 维 token 对齐到 2048 维 Qwen 输入；它不是 MetaCompress，当前实例不减少 token 数。实验链路如下，两次 Qwen 前向共享权重、独立状态：

```mermaid
flowchart LR
    V[冻结官方 encoder] --> M[冻结官方 merger]
    M --> X[原始 token X]
    X --> T[冻结 Qwen teacher]
    X --> G[压缩矩阵生成器]
    G --> P[逐行 softmax 得到 P]
    P --> C[压缩 token PX]
    X --> C
    C --> S[冻结 Qwen student]
    T --> K[共同 teacher 前缀的 KL]
    S --> K
    P --> R[Entropy 与 Collapse]
    K --> L[损失消融]
    R --> L
    L -. 只更新生成器 .-> G
```

从原 20k **训练记录**中按固定 hash 选取 8 个不同图像：1 个 DocVQA 图像用于单图 overfit，7 个独立图像用于训练内诊断 holdout，覆盖全部 7 个有训练数据的任务。OCRBench 无训练样本，本轮没有加入。与官方 val/test 图像无交集。为检验能否保住正确证据，筛选 teacher 得分为 1、非空/非 None、答案不超过 12 token 的图像；一般图像不超过 4096 token，KLC/DeepForm 放宽到 5200。**因此本表 teacher 均为 100%，这是筛选条件，不是新的模型成绩；每任务只有 1 个 holdout，不能估计总体泛化或显著性。**

同一初始化用于所有生成器变体。同一图像与问题重复 50 次、batch=1；这与原 batch=36 的梯度统计不同，不能把此处的步数或超参数效果直接外推到原规模。自由 P 的参数数目和优化器不同，只用于可优化性 oracle，不能归因为单独的“生成器结构效应”。最终固定第 50 步，不按 gold 或 holdout 选 checkpoint。

'''+table(['角色','任务','问题 ID','原 token 数','teacher / gold 正确答案'],sample_rows)+'''

## 2. 输入、缓存和因果位移审计

对 fit 图像，原生官方输入路径与缓存路径生成完全相同；fresh native encoder 特征与缓存特征的最大绝对差为 '''+f(interface['native_cached_feature_max_abs'])+'''，raw 差为 '''+f(interface['native_cached_raw_max_abs'])+'''。缓存 teacher logits 与重新前向 logits 最大差 '''+f(interface['cached_logits_max_abs'])+'''；显式 P=I 经相同乘法/cast/占位路径后 KL='''+f(interface['identity_KL'])+'''。第一个和最后一个响应位置单独重算，与一次性 teacher forcing 对应 logits 一致。主干可训练参数为 0。

这些检查支持当前数据路径与 causal shift 正确，不支持“缓存错 encoder”“没有把图像送入 decoder”“teacher/student 有共享 KV cache”等解释；逐次生成均新建独立状态。审计只覆盖一张 fit 图像的数值一致性，不能据此声称穷尽所有数据问题。

## 3. 十二组实际设置

'''+table(['名称','结构','优化目标','优化器','lr','全局 norm clip','保留率'],configs)+'''

`generator` 是原 MetaCompress 生成器。`free` 将原生成器在 fit 图像上的 P 取 log 作为独立可训练 logits，再逐行 softmax；不是论文 §4 的另一套初始化/SGD 配置，也不能用于独立图像推理。`local` 是明确的修改：在原 logits 上加固定局部 bias，同一均衡连续 token 分组内为 0，组外为 log(1e-6)，再 softmax。仍是凸组合，但它是 native 2×2 block **序列**的连续分组，不等于完整二维邻域，也不是原文方法。Adam 为额外诊断；paper 主实验仍是 SGD。全 loss 保留 KL + H + max-column-mass、权重均为 1。

自由 P 的 log/softmax 往返在浮点中不保证逐位等于生成器 P，BF16 前向可能放大微小差异，因此各变体的初始 KL 均独立重算，没有人为把 free 的首值替换为 generator 的首值。

## 4. 训练图像与独立图像的首末结果

下表“分数”为单题现有 scorer×100：DocVQA/InfoVQA 使用 ANLS，ChartQA 使用 relaxed accuracy，DT-VQA 使用 max(containment accuracy, ANLS)，WTQ 使用 denotation，DeepForm/KLC 使用单字段 exact match，**不等于官方 DUE corpus F1**。holdout 为七题混合任务指标的描述性均值，不能称为官方多任务成绩。KL 均在相同 teacher 响应上计算。

'''+table(['实验','保留率','fit KL：初→末','fit 分数：初→末','holdout 分数：初→末','holdout KL：初→末'],main_rows)+'''

训练问题：**'''+fit['question']+'''**；gold / teacher：**'''+fit['diagnostic_teacher_prediction']+'''**。

'''+table(['实验','初始生成','第 50 步生成','teacher'],pred_rows)+'''

## 5. 梯度强度、冲突与实际更新

以下是同一 fit 问题分别对 KL、H、collapse 求导，报告对该变体全部可训练参数拼接后的 L2 norm 与方向 cosine。数值大小不等于全数据梯度结论，cosine<0 代表该点的一阶下降方向有冲突。

'''+table(['实验','阶段','||g_KL||','||g_H||','||g_C||','cos(KL,H)','cos(KL,C)','cos(H,C)'],grad_rows)+'''

'''+table(['实验','裁剪乘数中位数','50 步实际更新路径长度','末行熵/log(n)','末 exp(H)','末 collapse'],update_rows)+'''

自由 P 与生成器的参数空间不同，更新路径长度不可直接比较。未裁剪采用 clip=inf 以测 norm；没有有限阈值的缩放。BF16 可以让 FP32 小更新的部分前向变化被舍入；当前原正式 checkpoint 相对 initial 的 BF16 压缩特征元素变化比例逐图见 `controls/feature_precision_audit.json`，不应把它简化为“所有梯度均失效”。

![训练轨迹](../artifacts/meta_compress_diagnosis/ablation_curves.png)

训练记录每步的 loss 是该步更新**前**的值；首末表是更新前初始状态与第 50 次更新**后**独立重算的值。

## 6. 无训练的保留率 / 局部压缩对照

固定 `mean`：按原生 token 顺序均衡分成 m 组，各组做均值；`sample`：各组取中间一个原 token，P 为 one-hot 行随机矩阵。不使用问题、答案或 decoder attention 选 token。它们用于区分“token 数减少”和“稠密全局混合”的影响；不能把更高保留率的收益算成同预算方法收益。

'''+table(['固定方法','保留率','fit KL','fit 分数','holdout KL','holdout 分数'],fixed_rows)+'''

## 7. 可复核性与局限

全部 12 组完成检查、每组 50 步顺序检查、每图每阶段唯一性检查和 48 个固定对照完整性检查通过后才生成本报告。原 initial / final checkpoint SHA256 与 protocol 中记录一致。全部问题、gold、teacher/student 作答均保留在 answers.csv。原 benchmark 未参与此次 checkpoint 选择。

本轮是单 seed、单图 overfit 与极小 holdout 的机制诊断，不能替代多图任务均衡训练、重复 seed、完整独立评测。局部 bias、Adam 和较高保留率均为与原实验不同的干预，必须分别解释。
'''
    drows=[]
    for r in directions['records']:
        if r['step_norm']==0:
            drows.append([r['precision'],0,f(r['kl']),f(r['repeat_delta']),'N/A','N/A'])
        else:
            drows.append([r['precision'],r['step_norm'],f(r['kl']),f(r['actual_delta']),
                f(r['first_order_predicted_delta']),f(r['compressed_elements_changed_fraction'])])
    text+='''
## 8. 数值精度与小步更新

从原生成器初始化出发，沿该精度下 KL 的单位负梯度方向移动指定的**全部参数 L2 距离**，逐次复位，不是连续训练。固定同一 BF16 cache source 与 BF16 teacher 目标。FP32 模式仅把同一 BF16 checkpoint 的权重值转为 FP32 后计算 student，不是加载另一份高精度预训练权重。各模式从各自基线计算 delta，不能比较绝对 KL 来判断模型质量。

'''+table(['Student 精度','参数步长 L2','KL','实际 delta（0步为重复差）','一阶预测 delta','压缩特征元素变化比例'],drows)+'''

BF16 的 cast 会使参数到前向 loss 的映射分段、非光滑；有限差分与反传的一阶预测在非常小步长可能不一致。这个审计只回答当前图像与局部步长的数值问题，不证明 FP32 是完整训练的最佳配置。

精度控制保留了原生 BF16 模型的 FP32 RoPE buffers。早期版本直接对全模型 `.to(bfloat16)`，会额外改变这些 buffers，因此该版本的数值不用于结论；记录保留在 `precision_invalid_buffer_cast/`。这只是精度审计脚本中已修正的问题，原正式训练没有做该额外 dtype cast。
'''
    grows=[]
    for r in task_grads:
        g=r['norms'];c=r['cosines']
        dot=g['prediction']**2+g['prediction']*g['entropy']*c['prediction__entropy']+g['prediction']*g['collapse']*c['prediction__collapse']
        grows.append([r['task'],r['id'],r['state'],f(g['prediction']),f(g['entropy']),f(g['collapse']),
            f(g['collapse']/max(g['prediction'],1e-30)),f(c['prediction__entropy']),f(c['prediction__collapse']),f(dot)])
    text+='''
## 9. 跨任务梯度审计

不能把一张 fit 图像的梯度方向推广到全部数据，因此另对同一 cohort 的 8 张图像（7 个任务）分别检查原生成器 initial 与原 20k final 两个状态。不做训练，不对各图像的梯度向量先平均；报告的是每图 norm 与 cosine，不能解释成原 batch=36 的确切全局梯度。

'''+table(['任务','问题 ID','状态','||g_KL||','||g_H||','||g_C||','C/KL norm 比','cos(KL,H)','cos(KL,C)','dot(KL,total)'],grows)+'''

末列由三个 norm 与 cosine 还原：dot(g_KL,g_total)=||g_KL||²+||g_KL||||g_H||cos(KL,H)+||g_KL||||g_C||cos(KL,C)。负值表示沿负 total 梯度的一阶小步**增加** KL；正的全局裁剪系数不会改变这个符号。这里是一阶梯度诊断，不能保证 BF16 的实际离散小步严格符合近似。
'''
    byname={r['variant']:r for r in results}
    factorial_rows=[]
    for objective,clip_name,free_name in [('完整 loss','original_all_sgd_clip001','all_sgd_no_clip'),('KL-only','kl_sgd_clip001','kl_sgd_no_clip')]:
        a=byname[clip_name]['fit_final'];b=byname[free_name]['fit_final']
        factorial_rows.append([objective,f(a['prediction']),a['student_prediction'],f(b['prediction']),b['student_prediction']])
    k=byname['kl_sgd_no_clip'];free=byname['free_dense_kl_adam'];local=byname['local_bias_all_adam'];dense=byname['kl_adam_keep025']
    initial_grads=[r for r in task_grads if r['state']=='original_initial']
    ratios=[r['norms']['collapse']/max(r['norms']['prediction'],1e-30) for r in initial_grads]
    conflicts=sum(r['cosines']['prediction__collapse']<0 for r in initial_grads)
    def total_dot(r):
        g=r['norms'];c=r['cosines']
        return g['prediction']**2+g['prediction']*g['entropy']*c['prediction__entropy']+g['prediction']*g['collapse']*c['prediction__collapse']
    total_conflicts=sum(total_dot(r)<0 for r in initial_grads)
    text+='''
## 10. 证据更新：哪些解释得到支持，哪些不能成立

**当前问题首先出在压缩器的优化设置及结构约束组合，不能优先归咎于 encoder 或 Qwen3-1.7B 容量。单纯取消裁剪、单纯改 Adam、单纯增加 token 预算，都不是这组实验的充分修复。**

### 10.1 同一 SGD、同一初始值下的 2×2 对照

'''+table(['优化目标','clip=.01：末 KL','clip=.01：生成','无裁剪：末 KL','无裁剪：生成'],factorial_rows)+'''

四组均 lr=.001、同一训练图像、50 步、保留率 10%。KL-only 无裁剪的 KL 从 '''+f(k['fit_initial']['prediction'])+' 降至 '+f(k['fit_final']['prediction'])+'''，独立生成答对；完整 loss 无裁剪仍未答对。该对照支持“更新幅度”和“损失组合”共同影响当前失败；不是靠增大 SGD 学习率数值就能充分解决。原强裁剪对全部 1,112 步的约束及参数变化仍以原 failure_audit 为准。

### 10.2 正则的方向与初始化相互影响

在 8 张诊断图像的原初始化上，collapse/KL 参数梯度范数比的中位数为 **'''+f(st.median(ratios))+'**，范围 **'+f(min(ratios))+'～'+f(max(ratios))+'**；其中 **'+str(conflicts)+'/8** 张图像的 collapse 与 KL 梯度 cosine 为负，**'+str(total_conflicts)+'''/8** 张图像的完整 total 梯度与 KL 梯度点积为负，即 total 的一阶下降方向会增大预测 KL。这不是看 loss 数值大小得到的判断，而是直接的分项梯度测量；只对这 8 张条件筛选图像成立，不能冒充整个训练集的统计。

局部 bias + 完整 loss 的末 KL 为 '''+f(local['fit_final']['prediction'])+'，collapse='+f(local['fit_final']['collapse'])+'''，能拟合当前问题。这说明不能把 entropy/collapse 全盘判为无效；它们与 P 的初始支持结构、特征尺度、优化器和梯度裁剪存在相互作用。当前稠密初始化下完整 loss 优先找到的低 collapse 解，未必保留文档证据。

### 10.3 高熵不是失败的充分条件，低熵也不是成功的保证

自由 P 的末 KL 为 '''+f(free['fit_final']['prediction'])+'、行熵 '+f(free['fit_final']['entropy'])+'''，也能答对单题；KL-only SGD 的末行熵仍为 '''+f(k['fit_final']['entropy'])+'''。因此应修正“行熵高就必然答不对”的过强解释。另一方面，dense KL-only Adam 在 25% 预算下末行熵仅 '''+f(dense['fit_final']['entropy'])+'，最大列质量却为 '+f(dense['fit_final']['collapse'])+' / '+str(dense['fit_final']['compressed_tokens'])+'''，出现大量输出行集中同一源 token 的退化，仍未答对。仅看 entropy 或单题 KL 都不能确认通用压缩质量。

自由 P 的自由度与优化器不同，成功只能证明该图像的预算和 decoder 路径允许找到可用解，不能单独证明原生成器存在表达能力缺陷。KL-only SGD 也能拟合，更不支持“原生成器完全学不动”的结论。

### 10.4 10% 预算困难，但不能把失败全归因于预算

固定取样/均值池化的保留率对照见第 6 节。高保留率下局部保留原 token 能改善一些图像；稠密全局混合即使增加行数，也不自动恢复局部文字。10% 下的单图成功排除了“该图预算绝对不可能”的解释，不能证明任意文档问题在该预算下都能恢复。

### 10.5 尚未解决的是多图泛化

单图学会输出 38 可能只保住了回答该问题所需的局部信号，甚至利用了针对该 teacher 输出的捷径；它不是整页文字和所有潜在问题均被保留的证据。七题 holdout 的首末作答与 KL 必须与 fit 分开看，不能以训练题满分宣称方案已修复。当前不据此选择或替换原正式模型。

25% 和 50% 的 dense KL-only Adam 确实各有 holdout 均分 **14.29→28.57** 的上涨：两组均来自同一道 WTQ 题，由回答 `1` 变为正确答案 `10`，其余六题得分不变。这一改善保留在结果中，不应说“所有设置都没有提升”。但其 holdout 平均 KL 分别由 **1.5030→1.9077**、**1.5189→1.8846** 变差，且矩阵发生单列集中、训练题仍错。七题中多答对一题不足以推断稳定泛化，也不能据此忽略矩阵退化。

建议下一轮先做训练集内图像互斥、任务均衡的小规模多图实验，以同图多问题约束避免只记单个答案：对比 dense KL-only SGD 与 local-bias + 完整 loss；分别监控生成分数、KL、collapse/m、梯度范数和 BF16 前向变化。先在 50%/25%/10% 分别验证，不混报预算；使用独立验证生成结果选择模型，最后再做完整评测。现有证据不要求先解冻 LLM，也不支持直接重复原 20k 原设置。
'''
    full_failures=sum(r['student_score']==0 for r in full_budget)
    lead='''
主要发现：输入/缓存/causal shift 审计通过；**不减少 token、不缩短 prompt，仅经过原始稠密 P 混合，'''+str(full_failures)+'''/3 道 teacher 正确的题变为零分**。同一 SGD 下，KL-only 去除强裁剪可单图拟合，而完整 loss 仅取消裁剪仍失败；局部 bias 配合完整 loss 也能拟合。梯度方向冲突、稠密矩阵退化与小步 BF16 舍入都获得直接诊断证据。**单图成功不等于独立图像性能恢复，详见第 10～11 节。**
'''
    text=text.replace('\n## 入口与原始数据',lead+'\n## 入口与原始数据',1)
    fbrows=[[r['task'],r['id'],r['tokens'],r['teacher_prediction'],r['student_prediction'],f(r['prediction']),f"{100*r['student_score']:.2f}",f(r['entropy_fraction'])] for r in full_budget]
    text+='''
## 11. 完整预算对照：没有减少 token，混合本身会怎样

对 cohort 的两张 DocVQA 和一张 ChartQA 图像，将原始生成器的保留率设为 1；m=n，图像占位块、问题 prompt 及连续 position 长度与 teacher 完全相同。原生成器此时 pool kernel=1，但 softmax(QKᵀ/sqrt(dc)) **不等于单位阵**。比较同预算的原始 X（teacher）与生成器混合后的 PX；生成器未训练。

'''+table(['任务','问题 ID','n=m','原始 teacher','混合后生成','KL','混合后分数','行熵/log(n)'],fbrows)+'''

这个对照可以检验“序列缩短是唯一问题”的解释，但不能证明缩短位置和预算没有额外影响，也不能将 keep=1 的 kernel=1 与 keep=.1 的 fractional max kernel=3 当成完全相同条件。
'''
    text+='''
三题的原始 teacher 均正确，混合后 '''+str(full_failures)+''' 题为零分。就这些图像而言，初始化产生的混合本身已足以破坏作答，不能把退化全归因于 90% reduction 或缩短文本位置。原论文共享 Wq=Wk 的理论初始化并不保证在实际 UND 特征统计和当前位置/pool 参数化下近似保留原 token；这是一项迁移差异的实证，不能推断论文原模型也存在相同问题。
'''
    text+='''
## 12. 同一图像的矩阵对比

![压缩矩阵](../artifacts/meta_compress_diagnosis/matrices/matrix_comparison.png)

颜色为 log10(P)，统一范围 [-6,0]；为避免单列峰值在缩图时消失，显示时每 3 个源 token 取最大权重，**仅显示降采样，不改变保存的 P 或任何 loss**。横轴为原生 2×2 block 序列位置，纵轴为输出 token。各面板分别标注 m，25% 保留率的输出行数与其他面板不同。原始矩阵见 [`matrices/`](../artifacts/meta_compress_diagnosis/matrices/complete.json)。可直接 `torch.load(path, weights_only=True)['P']` 读取，不需要加载 decoder。
'''
    (ROOT/'docs/METACOMPRESS_INTERVENTION_RESULTS.md').write_text(text)
    print(json.dumps({'variants':len(results),'updates':600,'records':len(records),'report':'docs/METACOMPRESS_INTERVENTION_RESULTS.md'},ensure_ascii=False))


if __name__=='__main__':main()
