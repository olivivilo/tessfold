"""Report the explicitly authorized 50% extension without rewriting its failed gate."""
import collections
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/compression_lab_v2'


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    state=load(BASE/'stages.json')['ratios']['0.5']
    if not state.get('user_authorized_extension'):return
    cfg=load(ROOT/state['full_config']);out=ROOT/cfg['output'];list(out.iterdir())
    training=[]
    if (out/'training.jsonl').exists():
        training=[json.loads(s) for s in (out/'training.jsonl').read_text().splitlines() if s.strip()]
    doc=['# 50% token保留率：用户授权的完整训练与正式全量评测','',
         f'更新UTC：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。阶段：`{state["stage"]}`。',
         '', '**入口**：状态 `bash scripts/run_compression_lab_v2.sh status`；中断恢复 `bash scripts/run_compression_lab_v2.sh resume`；本页更新 `.venv/bin/python -m scripts.report_compression_lab_v2_keep50`；严格全量审计 `.venv/bin/python -m scripts.report_compression_lab_v2_final --require-complete`。',
         '', '## 实验登记与结构','',
         '用户要求对50%保留率下表现较好的方案实施与10%/25%相同的全量实验。本次选用已经完成双seed扩展训练的问题条件四叉树（quadtree_question，pre-merger）。原独立确认的两个seed得分为50.27和51.12，teacher为58.82；第一个seed的增益95%CI为[-0.26,10.41]，原双seed门槛仍为失败。此次是后续明确授权的扩展，不改变原确认结果。',
         '', '沿用官方MonkeyOCRv2-B-Und，冻结encoder、官方merger和Qwen3-1.7B。生成器读取768维encoder输出H、图像网格和问题embedding，产生行随机矩阵P；每行是原token的凸组合，学生视觉输入为merger(PH)。保留token数为max(1,round(0.5×N))。教师为未压缩官方模型。具体生成器结构见[结构文档](COMPRESSION_GENERATOR_V2_DESIGNS.md)。',
         '', '## 固定实验流程','',
         '与10%/25%使用相同原20,000QA / 16,731图像。从头初始化生成器，seed20261105，不复用小规模权重、不改选确认分数较好的seed。训练4epochs、2,500全局更新、batch32；8NPU每rank累积4题，无样本填充。AdamW LR3e-4、weight_decay=0、clip=1；前250步warmup，随后cosine下降至峰值LR的10%。仅优化共同teacher前缀的全词表KL，不增加行熵或其他正则。',
         '', '每25步保存latest.pt和optimizer状态，每epoch固定保存与验证。小筛选集为56QA、生成上限64token，只观察训练；最终固定epoch4，不按小验证或正式测试分数改选。正式集合与此前相同：23,565QA、8任务、greedy、生成上限256token，逐题保留initial、trained、mean、selection、cell_mean、cell_selection六种作答，共141,390条。teacher沿用同一未压缩官方模型缓存。各问题独立，不跨题复用生成上下文。',
         '', '本次仅改变50%预算的全量实验登记，10%/25%的权重及正式作答保留。正式评测使用14个独立工作进程；并行度只改变题目分配。缓存、断点和控制器使用已验证的修复版本。实际核对的14个初始权重张量与10%模型逐一相同，见[初始化审计](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/initialization_audit.json)。',
         '', '## 实际训练进度','',
         f'训练作业：`{state.get("full_job","尚未提交")}`；评测作业：`{state.get("benchmark_job","尚未提交")}`。已记录全局更新：{max((r["step"] for r in training),default=0)}/2500；训练完成标志：{(out/"complete.json").exists()}。记录步数不等于已提交checkpoint步数。',
         '', '|epoch|已记录batch / 625|全训练batch平均KL|固定训练审计KL|验证KL|小筛选集生成分|',
         '|---|---:|---:|---:|---:|---:|']
    by_epoch=collections.defaultdict(list)
    for r in training:by_epoch[r['epoch']].append(r['kl'])
    fmt=lambda v:'—' if v is None else f'{v:.5f}'
    for e in range(1,5):
        rr=by_epoch[e];audit=load(out/f'train_audit.epoch{e}.summary.json');val=load(out/f'validation.epoch{e}.summary.json')
        doc.append(f'|{e}|{len(rr)}/625|{fmt(sum(rr)/len(rr) if rr else None)}|{fmt(audit.get("kl"))}|{fmt(val.get("kl"))}|{fmt(val.get("student_macro"))}|')
    doc+=['', '全训练均值覆盖该epoch实际完成batch，未满625表示尚未完成；固定审计/验证为不同小集合。小筛选集cap64与正式集合cap256不同，不能将它们直接当作训练前后对照。',
          '', '## 正式评测结果','']
    bench=out/'benchmark'
    try:list(bench.iterdir())
    except FileNotFoundError:pass
    audited=load(BASE/'infrastructure/recovery_20261106/final_artifact_audit.json').get('budgets',{}).get('0.5')
    if (bench/'complete.json').exists() and audited:
        summary=load(bench/'summary.json');m=summary['macro'];p=summary['paired_trained_vs_initial'];ci=p['cluster_bootstrap_95ci_points']
        variants=['teacher','initial','trained','mean','selection','cell_mean','cell_selection']
        doc+=['已完成23,565QA / 141,390条变体记录，并通过逐题覆盖、gold对应、teacher一致性、checkpoint/代码哈希和作答分数复算。',
              '', '|任务|'+ '|'.join(variants)+'|','|---|'+ '|'.join(['---:']*len(variants))+'|',
              '|八任务宏平均|'+ '|'.join(f'{m[v]:.2f}' for v in variants)+'|']
        for task,values in sorted(summary['tasks'].items()):
            doc.append('|'+task+'|'+ '|'.join(f'{values[v]:.2f}' for v in variants)+'|')
        doc+=['',f'训练后相对初始化为{p["gain_points"]:+.2f}点，按图像聚类bootstrap95%CI [{ci[0]:.2f},{ci[1]:.2f}]；相对teacher为{m["trained"]-m["teacher"]:+.2f}点，teacher分数保留率{100*m["trained"]/m["teacher"]:.2f}%。',
              '', '这是各任务原生分数的等权宏平均，DeepForm/KLC为字段exact，DT-VQA沿用既有scorer，不能直接视为论文的官方统一成绩。10%与50%为相同生成器不同预算；25%为不同生成器，跨预算差异不能全部归因于token数量。']
        doc+=['', '与此前全量结果使用相同集合、评分及生成设置的对照：',
              '', '|保留率|生成器|teacher|initial|trained|', '|---|---|---:|---:|---:|']
        for ratio,name in [('10%','full_quadtree_question_pre-merger_r0.1'),('25%','full_global_topk_pre-merger_r0.25'),('50%','full_quadtree_question_pre-merger_r0.5')]:
            result=load(BASE/'full'/name/'benchmark/summary.json')
            values=result['macro']
            doc.append(f'|{ratio}|{result["config"]["kind"]}|{values["teacher"]:.2f}|{values["initial"]:.2f}|{values["trained"]:.2f}|')
        doc+=['', '50%训练后八项任务均高于自身初始化；仍有teacher差距，尤其OCRBench。第4轮小验证生成分从第2/3轮的68.60回落至65.03，虽然验证KL进一步下降，因此KL下降并不保证生成分单调提高；正式结果使用预先固定epoch4，未追选较早权重。本轮为一个全量训练seed，图像bootstrap的区间衡量题目/图像抽样不确定性，不衡量跨训练seed的不确定性。']
    else:doc+=['正式全量结果尚未完成或尚未通过最终审计，不用小规模确认成绩代替。']
    rel='../'+cfg['output']
    doc+=['',f'[逐题CSV]({rel}/benchmark/answers.csv) · [图像/gold/teacher/训练前后作答]({rel}/benchmark/review.html) · [正式机器结果]({rel}/benchmark/summary.json) · [固定epoch4权重]({rel}/epoch4.pt)。',
          '', '[逐epoch训练/验证KL曲线](../artifacts/compression_lab_v2/plots/loss_v2_full_20k_pre-merger_keep0.5.pdf) · [逐epoch生成分数曲线](../artifacts/compression_lab_v2/plots/score_v2_full_20k_pre-merger_keep0.5.pdf) · [三个预算正式结果](COMPRESSION_GENERATOR_V2_FINAL_RESULTS.md) · [扩展登记](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/registration.json) · [训练完整性审计](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/training_audit.json)。']
    target=ROOT/'docs/COMPRESSION_GENERATOR_V2_KEEP50_FULL.md';tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
    print(json.dumps({'stage':state['stage'],'logged_updates':len(training),'formal_audited':bool(audited)}))


if __name__=='__main__':main()
