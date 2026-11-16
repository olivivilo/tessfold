"""Feedback supplementary protocol, live progress, audited results and tables."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from scripts.supplement_common import ROOT,BASE,REG,load,save,lines,sha,now,verify,benchmark_rows,summary,legacy,teacher_records
from vqa.benchmark_metrics import native_score


def audited(record,expected,teacher):
    out=ROOT/record['output']/'benchmark'
    for parent in [out.parent,out]:
        try:list(parent.iterdir())
        except FileNotFoundError:pass
    if not (out/'complete.json').exists():return None
    signature=dict(summary_sha256=sha(out/'summary.json'),manifests={p.name:sha(p) for p in out.glob('manifest.rank*.json')},
                   predictions={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in out.glob('predictions.rank*.jsonl')})
    cache=out/'supplement_audit.json';old=load(cache)
    if old.get('signature')==signature and old.get('passed'):return load(out/'summary.json')
    cfg=load(ROOT/record['config']);wanted=lines(ROOT/cfg['eval_rows']) if cfg.get('eval_rows') else list(expected.values())
    variants=cfg.get('variants',['trained']);items=[r for p in sorted(out.glob('predictions.rank*.jsonl')) for r in lines(p)]
    assert len(items)==len(wanted)*len(variants)
    assert {(r['id'],r['variant']) for r in items}=={(r['id'],v) for r in wanted for v in variants}
    manifests=[load(p) for p in out.glob('manifest.rank*.json')]
    assert manifests and all(m==manifests[0] for m in manifests)
    assert len(manifests)==manifests[0]['world']
    if cfg.get('checkpoint'):assert manifests[0]['checkpoint_sha256']==sha(ROOT/cfg['checkpoint'])
    for r in items:
        source=expected[r['id']]
        for k in ['task','document_id','question','answers']:assert r[k]==source[k]
        assert r['teacher_prediction']==teacher[r['id']]['teacher_prediction']
        assert abs(r['teacher_score']-teacher[r['id']]['teacher_score'])<1e-12
        assert abs(native_score(r['student_prediction'],source)-r['student_score'])<1e-12
        assert len(r['student_ids'])<=256
        if cfg['method']!='R':assert r['compressed_tokens']==max(1,round(cfg['ratio']*r['tokens']))
        if cfg['method']=='S':assert r['first_layer_visual_tokens']==r['tokens']
        else:assert r['first_layer_visual_tokens']==r['compressed_tokens']
    ss=load(out/'summary.json')
    for v in variants:
        direct=summary([r for r in items if r['variant']==v]);assert direct==ss['variants'][v]
    if cfg['method']=='QSWAP':
        for r in items:
            if r['variant']=='QQ':
                assert r['student_ids']==teacher[r['id']]['student_ids'],'QQ no longer reproduces original production answer'
    if cfg['method']=='R':
        amendment=load(BASE/'resize_recovery/registration.json')
        recovery=[load(p) for p in out.glob('resize_recovery.rank*.json')]
        assert len(recovery)==manifests[0]['world'] and amendment
        assert all(r['amendment_sha256']==sha(BASE/'resize_recovery/registration.json') for r in recovery)
    save(cache,dict(passed=True,audited_utc=now(),signature=signature,questions=len(wanted),records=len(items),scores_recomputed=True,teacher_gold_identity=True,unique_coverage=True,qq_production_regression=cfg['method']=='QSWAP'))
    return ss


def training(record):
    out=ROOT/record['output']
    try:list(out.iterdir())
    except FileNotFoundError:pass
    rr=lines(out/'training.jsonl')
    if not rr:return dict(updates=0)
    assert [r['step'] for r in rr]==list(range(1,len(rr)+1))
    assert all(math.isfinite(r[k]) for r in rr for k in ['kl','total','grad_norm','update_l2'])
    cfg=load(out/'effective_config.json') or load(ROOT/record['config'])
    if (out/'complete.json').exists():
        done=load(out/'complete.json');assert done['trainable_backbone']==0
        assert done['steps']==cfg.get('max_steps',2500)
    return dict(updates=len(rr),latest=rr[-1],config=cfg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    reg=verify();state=load(BASE/'state.json');expected={r['id']:r for r in benchmark_rows()};teacher=teacher_records()
    results={};progress={}
    for record in reg['runs']:
        if record['phase']=='eval':
            result=audited(record,expected,teacher)
            if result:results[record['name']]=result
        if record['phase']=='train':progress[record['name']]=training(record)
    fmt=lambda v:'—' if v is None else f'{v:.2f}'
    complete=load(BASE/'complete.json');smoke=load(BASE/'smoke/complete.json');selection=load(BASE/'development_selection.json')
    doc=['# 文档视觉压缩：反馈补充实验设置、过程与结果','',f'更新UTC：{now()}。新增正式评测已审计{len(results)}组；完整流水线完成标志：{bool(complete)}。','',
         '**入口**：`.venv/bin/python -m scripts.compression_supplement_suite status`查看；`resume`恢复独立后台控制器；最终`.venv/bin/python -m scripts.report_compression_supplement --require-complete`核验。','',
         '## 实验设计与冻结协议','',
         '依据[反馈解读](COMPRESSION_SCHEME_FEEDBACK_ACTIONS.md)、feedback/experiment_runbook.docx和paper_experiment_tables.xlsx。主预算25%，次预算50%；固定20,000QA/16,731图训练、56QA验证、23,565QA/6,494图正式评测，正式集已被查看，本轮是固定基准扩展，不称盲测。官方encoder、merger、Qwen3-1.7B全冻结。','',
         '- 八组完整训练：M25/M50；V25/J25；T25 seed20261106/07；M25 seed20261106/07。全局batch32、8NPU、每epoch625更新、4epochs/2500更新、固定epoch4，不按正式分数选模。',
         '- T/V/J为KL-only；T保留原AdamW3e-4、clip1、250步warmup及cosine到10%。M保留原全局query/key矩阵及KL+行熵+最大列质量，UND移植细节沿用vqa/meta_compress.py并披露，而非声称作者官方代码。',
         '- 新增六个优化开发试跑：M与T各三候选，224QA来自当前full_train，固定56QA验证，每候选100更新；原SGD1e-3/clip1e-2/constant、AdamW3e-4/clip1/cosine、AdamW3e-5/clip1/cosine。M按endpoint验证生成宏分、KL、登记顺序选一次优化设置，所有预算/seed共用；T主实验保持历史固定设置，试跑均报告。此额外600更新不计入八组正式20,000更新。',
         '- M原作者设置若未被选中，保留完整试跑与曲线，不将其优化失败冒充强基线胜利。M正式manifest/effective_config记录实际选择。开发小样本的选择不保证正式集最优，不再据正式分数调优。',
         '- V仅将C切轴改长轴；J仅将T能量中位切点改floor几何中点，仍按能量选叶。两者新初始化，其他训练参数不改。',
         '- H/A复用T25的真实叶子和生产排序，分别取最高权重原token/均值，是推理干预，不是另训hard/mean方法；G*用既有T分数，4x4合法固定格、rho均值softmax配额、精确容量约束取整和格内s topk，非原GridPrune。',
         '- QQ/QW/WQ/WW只更换压缩器问题，Qwen始终本题；每一既定分区用本题代表序固定，保存叶ID/叶序及边界变化，同页错题映射不依赖答案/分数。QQ必须逐题回归历史生产作答。',
         f'- 同页去重后锁定{reg["cohorts"]["same_page_questions"]}QA/{reg["cohorts"]["same_page_images"]}图；只在同一子集和同一有样本任务集比较，不与全集宏分相减。',
         '- S使用已固定commit的官方S²Prune allocation.py；首Qwen block全长，ERC=L2(block输出−输入)，再剪视觉hidden/mask/原1D RoPE位置及第0层KV，其他文本保留；后27块K。UND不使用Qwen2.5-VL原预处理及M-RoPE，故明确标UND移植。',
         '- S粗格边长为min(8在25%或9在50%, h,w,isqrt(K))，原图全局Laplacian用于复杂度，官方非空矩形分配与局部代表选择；2x2 block顺序正确置换，原代表恢复序列顺序。',
         '- R保持原宽高比近似的合法processor尺度搜索，按实际K′最接近目标K选择，重新编码小图；记录K′误差，不把目标r写成实得率。',
         '- 正式各方法独立greedy cap256；训练teacher/验证cap64、T=1、全词表逐题平均KL，teacher由原模型生成而非gold。DeepForm/KLC字段EM；八任务等权宏平均不能冒充官方统一榜单。',
         '- 效率：固定90页题、按原N三层各30题，batch1冷视觉缓存、权重常驻；方法逐个驻留以避免多生成器增加内存不公平。各方法10请求预热，3轮交错，分别自然EOS与固定64生成；包含processor、视觉编码/原merger、压缩/CPU与生成，真实首token回调计TTFT、p50/p95和allocated/reserved峰值。无跨题KV复用。','',
         '[完整登记](../artifacts/compression_supplement_20261111/registration.json) · [机器状态](../artifacts/compression_supplement_20261111/state.json) · [原文/官方代码](../artifacts/compression_supplement_20261111/sources/)。','',
         '## 技术检查与开发集选择','',f'真实UND技术检查通过：{bool(smoke)}。选择完成：{bool(selection)}。未完成的阶段不报告为成功。','']
    if selection:
        doc += [f'选定M优化配置：`{selection["settings"]}`；依据固定开发验证集，不使用正式集。','', '|方法|候选|endpoint验证KL|endpoint验证宏分|','|---|---|---:|---:|']
        for r in selection['all_trials']:doc.append(f'|{r["method"]}|{r["candidate"]}|{r["validation"]["kl"]:.5f}|{r["validation"]["student_macro"]:.2f}|')
    recovery=load(BASE/'resize_recovery/registration.json')
    if recovery:
        doc += ['', '### 缩图基线实现错误及版本化恢复', '',
                '原尺寸搜索在部分细长页的候选target上产生高或宽为0的smart_resize结果，排序时除零；25%/50%各三次尝试失败。修复只过滤非正尺寸候选，保留原排序/目标预算/processor/生成设置，不修改签名原评测源码。已检查完整图像/预算候选，并核对所有已提交旧作答的选择尺寸不变；续跑保留这些作答，不改变训练或其他方法。', '',
                '[恢复登记、原失败记录及全图候选核对](../artifacts/compression_supplement_20261111/resize_recovery/)。恢复版本与原manifest同时保存，未完成时不报告R成绩。', '']
    doc+=['','## 全部运行实际进度','', '|运行|阶段|状态|更新|已写正式记录|','|---|---|---|---:|---:|']
    for r in reg['runs']:
        st=state['runs'].get(r['name'],{});count=0
        if r['phase']=='eval':
            for f in (ROOT/r['output']/'benchmark').glob('predictions.rank*.jsonl'):
                with f.open('rb') as h:count+=sum(line.endswith(b'\n') for line in h)
        doc.append(f'|{r["name"]}|{r["phase"]}|{st.get("stage","registered")}|{progress.get(r["name"],{}).get("updates",0)}|{count}|')
    doc+=['','## T1：同主干正式结果','', '|预算|方法|来源|宏分|DocVQA|ChartQA|DeepForm|OCRBench|','|---|---|---|---:|---:|---:|---:|---:|']
    for ratio in [.25,.5]:
        for label,kind,pos in [('T','quadtree_question','pre-merger'),('C','col_adaptive','pre-merger'),('TP','quadtree_question','post-merger')]:
            s=load(ROOT/legacy(kind,ratio,pos)/'benchmark/summary.json')
            doc.append(f'|{ratio:.0%}|{label}|历史实测|{s["macro"]["trained"]:.2f}|{s["tasks"]["DocVQA"]["trained"]:.2f}|{s["tasks"]["ChartQA"]["trained"]:.2f}|{s["tasks"]["DeepForm"]["trained"]:.2f}|{s["tasks"]["OCRBench"]["trained"]:.2f}|')
        for name,ss in results.items():
            c=ss['config']
            if c['ratio']!=ratio or c.get('seed') not in [None,20261105] or c['method'] in ['HA','QSWAP']:continue
            m=ss['variants']['trained'];t=m['by_task']
            doc.append(f'|{ratio:.0%}|{c["method"]}|新增正式，已审计|{m["macro"]:.2f}|{t["DocVQA"]:.2f}|{t["ChartQA"]:.2f}|{t["DeepForm"]:.2f}|{t["OCRBench"]:.2f}|')
    doc += ['', '空白方法仍在运行或未通过审计，不能用小验证替代。完整八任务分数见各运行的benchmark/summary.json及answers.csv。REF宏分57.00409130135，DocVQA76.9342。','', '## T3/T4：机制与同页换题','']
    for name in ['HA_r0.25','QSWAP_r0.25']:
        if name not in results:doc+=[f'`{name}`：尚未完成并审计。',''];continue
        ss=results[name];doc+=['|变体|QA|图像|有效任务数|子集宏分|','|---|---:|---:|---:|---:|']
        for v,s in ss['variants'].items():doc.append(f'|{v}|{s["rows"]}|{s["images"]}|{s["tasks"]}|{s["macro"]:.2f}|')
        doc.append('')
    doc += ['## 训练与验证过程','', '|运行|epoch|训练KL|训练total|验证KL|验证生成宏分|','|---|---:|---:|---:|---:|---:|']
    for r in reg['runs']:
        if r['category']!='full_training':continue
        rr=lines(ROOT/r['output']/'training.jsonl')
        for epoch in range(5):
            val=load(ROOT/r['output']/f'validation.{"initial" if epoch==0 else "epoch"+str(epoch)}.summary.json');batch=[x for x in rr if x['epoch']==epoch]
            doc.append(f'|{r["name"]}|{epoch}|{fmt(sum(x["kl"] for x in batch)/len(batch) if batch else None)}|{fmt(sum(x["total"] for x in batch)/len(batch) if batch else None)}|{fmt(val.get("kl"))}|{fmt(val.get("student_macro"))}|')
    doc+=['','## 统计、效率与最终结论','', '正式主比较为三个共同seed的T_pre−M_post；先对每题同seed差平均，再按图像簇bootstrap10000次；报告各seed差的样本标准差，不能以trained−initial区间替代跨方法区间。主方法未完整覆盖前不作优越性结论。','',
          'T5实际时延与内存、T6配对区间及Excel回填由最终统计阶段在全部相关产物齐全后生成；pending不写成0或成功。若结果不支持问题条件/软加权/实际加速，保留负结果并收缩结论。', '',
          '本轮增加开发集小试以避免已知的优化不适配；八组正式实验预算、已看benchmark和历史数据限制均如实记录。方法实现、每次任务日志、checkpoint、训练/验证记录、逐题作答、区域诊断与计时保存在独立目录，不覆盖原16组。']
    target=ROOT/'docs/COMPRESSION_SUPPLEMENT_RESULTS.md';tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
    save(BASE/'merged_results.json',dict(updated_utc=now(),results=results,progress=progress,planned=len(reg['runs'])))
    if args.require_complete:
        assert len(results)==sum(r['phase']=='eval' for r in reg['runs'])
        assert (BASE/'efficiency/complete.json').exists()
        import subprocess
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.analyze_compression_supplement'],cwd=ROOT,check=True)
        final=load(BASE/'analysis_complete.json')
        assert final['three_seed_complete'] and final['timing_complete'] and final['xlsx']
        target.write_text(target.read_text().replace('完整流水线完成标志：False。','完整流水线完成标志：True。'))
        with target.open('a') as handle:
            handle.write('\n## 最终配对统计、效率与回填工作簿\n\n全部正式结果已覆盖并审计，三个seed和效率记录齐全。详见[完整统计与结果](COMPRESSION_SUPPLEMENT_STATISTICS.md)及[完成的Excel工作簿](../artifacts/compression_supplement_20261111/paper_experiment_tables_completed.xlsx)。科学结论以实际配对区间、任务分数和实测时延为准，不预设本方法获胜。\n')
    print(json.dumps(dict(formal_audited=len(results),registered=len(reg['runs']),report=str(target))),flush=True)


if __name__=='__main__':main()
