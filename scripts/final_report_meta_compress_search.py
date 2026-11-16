"""Final Markdown and CSV; refuses incomplete full training/benchmark artifacts."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import statistics
from scripts.report_meta_compress_search import ROOT,BASE,read,collect,paired,average


def main():
    state=json.loads((BASE/'controller.json').read_text());cfgpath=ROOT/state['full_config'];cfg=json.loads(cfgpath.read_text())
    run=ROOT/cfg['output'];bench=run/'benchmark'
    assert (run/'complete.json').exists(),'Training is incomplete'
    completion=json.loads((run/'complete.json').read_text());assert completion['rows']==20000 and completion['steps']==1112
    for rank in range(32):
        assert (bench/f'complete.rank{rank}.json').exists()
        assert (bench/f'teacher_check.rank{rank}.json').exists()
    records=[r for f in bench.glob('predictions.rank*.jsonl') for r in read(f)]
    expected={r['id'] for r in read(bench/'samples.jsonl')}
    keys=[(r['id'],r['variant']) for r in records]
    assert len(keys)==len(set(keys)) and set(keys)=={(i,v) for i in expected for v in ['initial','trained','mean']}
    assert len(expected)==512
    fresh_teacher={}
    for r in records:
        value=(r['teacher_prediction'],r['teacher_generated_ids'],r['teacher_score'],r['original_teacher_ids_equal'])
        if r['id'] in fresh_teacher:assert fresh_teacher[r['id']]==value
        fresh_teacher[r['id']]=value
    teacher_ids_equal=sum(v[3] for v in fresh_teacher.values())
    variants={v:[r for r in records if r['variant']==v] for v in ['initial','trained','mean']}
    tasks=sorted({r['task'] for r in records});table={}
    for t in tasks:
        table[t]={}
        for v,rr in variants.items():
            rows=[r for r in rr if r['task']==t]
            table[t][v]={'n':len(rows),'score':100*average([r['student_score'] for r in rows]),
                'teacher':100*average([r['teacher_score'] for r in rows]),
                'KL':average([r['prediction'] for r in rows]),'C_fraction':average([r['column_mass_fraction'] for r in rows])}
    macro={v:average([table[t][v]['score'] for t in tasks]) for v in variants}
    teacher=average([table[t]['trained']['teacher'] for t in tasks])
    changes={v:paired(variants[v],variants['trained']) for v in ['initial','mean']}
    old_records=[r for f in (ROOT/'artifacts/meta_compress/eval').glob('predictions.benchmark.rank*.jsonl') for r in read(f) if r['variant']=='trained']
    old_macro=average([100*average([r['student_score'] for r in old_records if r['task']==t]) for t in tasks])
    pilot=state['selection']['candidate'];confirmation=state['selection']['confirmation'];confmean=state['selection']['matched_mean']
    selected=collect(BASE/f"runs/{pilot['name']}");search=selected['stages'][f"epoch{pilot['epoch']}/trained"]
    pilot_training_rows=read(ROOT/selected['config']['train_rows'])
    pilot_training_images=len({r['document_id'] for r in pilot_training_rows})
    search_images=len({r['document_id'] for r in read(BASE/'search.jsonl')})
    confirm_images=len({r['document_id'] for r in read(BASE/'confirm.jsonl')})
    train=read(run/'training.jsonl');assert len(train)==1112 and {r['step'] for r in train}==set(range(1,1113))
    epochs=[]
    for ep in [1,2]:
        rows=[r for r in train if r['epoch']==ep]
        epochs.append({'epoch':ep,'steps':len(rows),**{k:average([r[k] for r in rows]) for k in ['prediction','entropy','collapse','objective','grad_norm','clip_multiplier','update_l2']},
            'clipped_fraction':average([float(r['clip_multiplier']<1) for r in rows])})
    summary={'complete':True,'full_config':str(cfgpath.relative_to(ROOT)),'pilot':pilot,'confirmation':confirmation,
        'benchmark_samples':512,'benchmark_records':len(records),'teacher_macro':teacher,'student_macros':macro,
        'old_10pct_trained_macro':old_macro,'matched_paired_changes':changes,'tasks':table,'epochs':epochs,
        'teacher_retention':macro['trained']/teacher,'fresh_teacher_questions':len(fresh_teacher),'teacher_old_exact_ids_equal':teacher_ids_equal}
    (BASE/'FINAL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    fields=['task','id','document_id','variant','question','answers','teacher_prediction','student_prediction','teacher_score','student_score','prediction','entropy','collapse','objective','tokens','compressed_tokens']
    with (BASE/'final_answers.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in records:writer.writerow({**row,'answers':json.dumps(row['answers'],ensure_ascii=False)})
    structure=('保持原 dense 生成器结构。' if not cfg.get('local_bias') else
        f"**增加了论文原生成器没有的固定局部 logit bias（显著结构差异）**：令 `g(j)=floor(j*m/n)`，在原 attention logits 中，对 `g(j) != i` 的位置减去 {cfg['local_bias']:g}，再做行 softmax。该约束按官方 token 序列的连续组建立，每行主要聚合约 n/m 个 token，仍对所有源 token 保留非零权重。它增加固定先验，没有新增可训练参数；不能把其收益归因于单纯 optimizer/clipping 改动。")
    weightnote=('三项权重仍为原文的 1。' if cfg['entropy_weight']==cfg['collapse_weight']==1 else
        f"**loss 权重偏离原文**：entropy={cfg['entropy_weight']:g}，collapse={cfg['collapse_weight']:g}；原文均为 1。")
    text=f'''# MetaCompress：多图调参、独立确认与原 20k 完整重训结果

**已完成真实 pilot、独立确认、原 20,000 条训练样本的 2 epochs 重训，以及固定 512 题八任务评测。** 最终保留 token 比例为 **{cfg['ratio']:.0%}**；评测 teacher / 同结构初始化 / mean / 训练后宏平均为 **{teacher:.2f} / {macro['initial']:.2f} / {macro['mean']:.2f} / {macro['trained']:.2f}**。训练后相对同预算 mean 的配对变化为 **{changes['mean']['gain_points']:+.2f}** 个百分点，95% bootstrap CI 为 **{changes['mean']['paired_bootstrap_95ci_points']}**。这些是实际观测，不代表达到论文压缩成绩。

入口：[实验设计](METACOMPRESS_TRAINING_SEARCH.md)；[所有候选逐阶段结果](../artifacts/meta_compress_search/RESULTS.search.md)；[独立确认结果](../artifacts/meta_compress_search/RESULTS.confirm.md)；[逐题 teacher/student/gold CSV](../artifacts/meta_compress_search/final_answers.csv)；[机器可读最终结果](../artifacts/meta_compress_search/FINAL_SUMMARY.json)。

## 最终方案与论文差异

使用官方 `MonkeyOCRv2-B-Und`，冻结 encoder、官方 merger 和 Qwen3-1.7B。在 merger 后的 2048 维视觉 token 上压缩。原生成器为：可学习位置编码 → fractional max query pooling（kernel=3）→ 64 维 Q/K 与通道权重 → 行 softmax → `P @ X`；共 458,816 个可训练参数。原论文主文的 average pooling 与补充材料的 fractional max pooling 存在差异，本实现采用后者。位置编码采用可学习行/列表，未宣称与未公开的作者代码逐项相同。

{structure}

实际训练目标是 `KL(teacher || student) + {cfg['entropy_weight']:g} * mean_row_entropy(P) + {cfg['collapse_weight']:g} * max_column_mass(P)`。{weightnote} KL 用完整词表、温度 1、teacher 自生成共同前缀，gold 不进入训练。每题独立生成 student 答案，生成得分用于验证。压缩矩阵各行和为 1，token 为凸组合。

最终 optimizer=`{cfg['optimizer']}`，peak LR=`{cfg['lr']}`，clipping norm=`{cfg['clip']}`，scheduler=`{cfg['scheduler']}`，warmup fraction=`{cfg.get('warmup_fraction')}`，最小 LR 比例=`{cfg.get('min_lr_fraction')}`，momentum=`{cfg.get('momentum',0)}`。AdamW 时 betas=(0.9,0.999)、weight decay={cfg.get('weight_decay',.01)}。完整配置：[config](../{state['full_config']})。

## 调参与独立确认

胜出候选的 pilot 使用 {pilot_training_images} 张训练图 / {len(pilot_training_rows)} 道题，search 为 {search_images} 张不同图，confirm 为 {confirm_images} 张不同图，包含七项任务。第一、二轮 pilot 是每任务 32 图、共 224 图；若进入第三轮，则从原 20k 扩展至每任务最多 96 图，实际计数见对应 protocol。search/confirm 的全部图像均不在原 20k 中，且排除正式 val/test 图像；不筛选 teacher 正确样本。候选按 search 选择，最终 benchmark 不参与模型选择。完整轮次失败结果保留，不删除不利结果。确认集用于过门槛，未用于反向传播；实际进入确认的候选数为 {len(state.get('confirmation_attempts',[]))}，多次使用同一确认集存在适应性选择的限制。

胜出 pilot：`{pilot['name']}`，epoch {pilot['epoch']}。

| 集合 | 训练后分数 | 相对同结构初始变化 | 相对 mean 变化 | KL 相对初始降低 | 配对 95% CI（相对初始） |
|---|---:|---:|---:|---:|---|
| search 56 图 | {search['score']:.3f} | {search['paired_vs_initial']['gain_points']:+.3f} | {search['score']-selected['matched_mean']['score']:+.3f} | {search['kl_relative_reduction']:.1%} | {search['paired_vs_initial']['paired_bootstrap_95ci_points']} |
| confirm 112 图 | {confirmation['score']:.3f} | {confirmation['paired_vs_initial']['gain_points']:+.3f} | {confirmation['score']-confmean['score']:+.3f} | {confirmation['kl_relative_reduction']:.1%} | {confirmation['paired_vs_initial']['paired_bootstrap_95ci_points']} |

原始筛选要求：相对初始分数至少 +2 点、KL 降低至少 5%、平均最大列质量/m 不超过 0.25；正式更新前另登记超过同预算 mean 至少 2 点。第一批 epoch-1 结果出现后、尚无候选过门槛且尚未查看 confirm 时，另登记 teacher 得分保留率至少 50% 的实用性要求，详见 `suitability_addendum.json`。这不是最初协议的一部分；所有原始观测保留，原 `screen_pass` 与最终 `practical_pass` 分开报告。这些门槛不等同于显著性检验。

## 完整训练过程与曲线

重新从**原初始生成器**启动，optimizer 状态也重置；没有从 pilot checkpoint 继续。原 20k 样本和顺序未改变，global batch=36，36-rank CPU/Gloo，实际每 rank threads={json.loads((run/'manifest.json').read_text())['threads']}，2 epochs 共 1,112 updates。每 epoch 最后一个不足 batch 按该 epoch 已打乱的序列补齐，实际每 epoch 20,016 次样本呈现。冻结主干为 BF16，保留原有 FP32 RoPE buffer，生成器和 optimizer 为 FP32。初始、每 epoch、latest checkpoint 与源代码快照都保存于 [run](../{cfg['output']})。

| epoch | updates | train KL | entropy | collapse | weighted objective | grad norm | clipping 发生比例 | 平均 clip multiplier | 参数 update L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for ep in epochs:text+=f"| {ep['epoch']} | {ep['steps']} | {ep['prediction']:.5f} | {ep['entropy']:.4f} | {ep['collapse']:.4f} | {ep['objective']:.5f} | {ep['grad_norm']:.4f} | {ep['clipped_fraction']:.1%} | {ep['clip_multiplier']:.5f} | {ep['update_l2']:.7f} |\n"
    text+=f'''
[训练与验证曲线 PDF](../{cfg['output']}/curves.pdf)。图中 KL 是 teacher-prefix 分布蒸馏 loss，**不是 gold answer 的交叉熵**；验证还报告自由生成任务分数，避免仅以 NLL/KL 下降判断有效。

![完整训练与验证曲线](../{cfg['output']}/curves.png)

完整训练固定使用 epoch 2 做最终评测，不根据 benchmark 回选 checkpoint。

## 固定八任务评测

与原失败训练使用完全相同的 512 张图/512 题，各任务 64 题；DocVQA、InfoVQA 用原公开 validation，其余任务沿用原 test。max_new_tokens=256，teacher 不压缩；新模型、同结构初始化、固定 mean 使用相同 token 预算。**512 题全部重新执行独立的未压缩 teacher 生成**，以当前 CPU BF16 / 8 threads 的 teacher 分数作为各压缩变体的共同对照；与旧评测的完整 token ID 精确一致为 {teacher_ids_equal}/512。旧 teacher 数据只用于一致性审计，不替代新 teacher 作答。所有压缩路径独立生成、重新计算相同 teacher 前缀的 KL。

| 任务 | n | teacher | 同结构初始化 | matched mean | 训练后 | 训练后−mean |
|---|---:|---:|---:|---:|---:|---:|
'''
    for t in tasks:
        d=table[t];text+=f"| {t} | {d['trained']['n']} | {d['trained']['teacher']:.2f} | {d['initial']['score']:.2f} | {d['mean']['score']:.2f} | {d['trained']['score']:.2f} | {d['trained']['score']-d['mean']['score']:+.2f} |\n"
    text+=f"| 八任务原生得分宏平均 | 512 | {teacher:.2f} | {macro['initial']:.2f} | {macro['mean']:.2f} | {macro['trained']:.2f} | {macro['trained']-macro['mean']:+.2f} |\n"
    text+=f'''
最终 teacher 分数保留率为 {macro['trained']/teacher:.1%}。相对初始化的配对变化 {changes['initial']['gain_points']:+.3f} 点，95% CI={changes['initial']['paired_bootstrap_95ci_points']}；相对 mean 的变化 {changes['mean']['gain_points']:+.3f} 点，95% CI={changes['mean']['paired_bootstrap_95ci_points']}。bootstrap 按任务分层、以图像为重采样单位，共 10,000 次；这些区间只衡量本固定样本的抽样不确定性。

旧训练结果为 {old_macro:.2f}，预算 10%；新预算为 {cfg['ratio']:.0%}。预算不同带来的提升不能归因于训练设置，以上 matched-budget 初始化和 mean 才是本轮主要对照。本评测是固定 512 题，不是全部 23,565 题。DeepForm/KLC 仍为逐字段 exact，不能与论文 corpus F1 直接等同；正式论文任务成绩不可仅凭此小评测宣称复现。

所有逐题答案、gold、loss、保留 token 数、截断标记及全部失败候选保留。最终结果不论是否优于小验证集趋势，均按预先选定 epoch 原样报告。
'''
    destination=ROOT/'docs/METACOMPRESS_RETRAIN_RESULTS.md';destination.write_text(text)
    readme=ROOT/'README.md';s=readme.read_text()
    start='<!-- META_RETRAIN_FINAL_START -->';end='<!-- META_RETRAIN_FINAL_END -->'
    block=(f'{start}\n**MetaCompress 本轮调参与 20k 重训已完成：token 保留 {cfg["ratio"]:.0%}，'
           f'固定 512 题 teacher / 训练后宏均分 {teacher:.2f} / {macro["trained"]:.2f}；'
           f'[最终实测报告](docs/METACOMPRESS_RETRAIN_RESULTS.md)、[逐题作答](artifacts/meta_compress_search/final_answers.csv)。**\n{end}')
    if start in s:s=s[:s.index(start)]+block+s[s.index(end)+len(end):]
    else:
        first,rest=s.split('\n',1);s=first+'\n\n'+block+'\n'+rest
    readme.write_text(s)
    print(destination)


if __name__=='__main__':main()
