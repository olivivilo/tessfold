"""Read-only experiment report and image-clustered uncertainty; no model selection on test."""
import csv
import json
import math
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/meta_compress_strict'

def read(path):
    if not path.exists():return []
    result=[];lines=path.read_text().splitlines()
    for i,s in enumerate(lines):
        try: result.append(json.loads(s))
        except json.JSONDecodeError:
            if i<len(lines)-1:raise
    return result

def save(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.report.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def mean(v):return sum(v)/len(v)

def summarize(rows,expected):
    ids=[r['id'] for r in rows]
    assert len(ids)==len(set(ids)),'Duplicate evaluation question'
    if set(ids)!=set(expected):return {'complete':False,'n':len(ids),'expected':len(expected)}
    tasks=defaultdict(list)
    for r in rows:tasks[r['task']].append(r)
    fields=['prediction','entropy','collapse','total','student_score','teacher_score']
    bytask={task:{k:mean([r[k] for r in rr]) for k in fields} for task,rr in tasks.items()}
    return {'complete':True,'n':len(rows),'images':len({r['document_id'] for r in rows}),
            **{k:mean([r[k] for r in bytask.values()]) for k in fields},
            'score':100*mean([r['student_score'] for r in bytask.values()]),
            'teacher':100*mean([r['teacher_score'] for r in bytask.values()]),'tasks':bytask,
            'student_truncations':sum(r['hit_token_limit'] for r in rows),
            'teacher_truncations':sum(r['teacher_hit_token_limit'] for r in rows)}

def paired(initial,final):
    aa={r['id']:r for r in initial};bb={r['id']:r for r in final};assert aa.keys()==bb.keys()
    groups=defaultdict(lambda:defaultdict(list))
    for key,a in aa.items():groups[a['task']][a['document_id']].append(100*(bb[key]['student_score']-a['student_score']))
    # Each sampled image retains all its QA; the estimator is the same task
    # QA-average as the report, not an unclustered independent-question sample.
    rng=np.random.default_rng(20261101);samples=np.zeros(10000);point=0.
    for task,docs in groups.items():
        sums=np.array([sum(v) for v in docs.values()]);counts=np.array([len(v) for v in docs.values()])
        idx=rng.integers(len(sums),size=(10000,len(sums)))
        samples+=(sums[idx].sum(1)/counts[idx].sum(1))/len(groups)
        point+=sums.sum()/counts.sum()/len(groups)
    return {'gain_points':float(point),'cluster_bootstrap_95ci_points':np.quantile(samples,[.025,.975]).tolist()}

def stable(stages,epochs,kl_tail_tolerance=0.):
    reasons=[];practical=[]
    for cohort in ['train_audit','validation']:
        initial=stages.get(f'{cohort}/initial',{})
        last=[stages.get(f'{cohort}/epoch{e}',{}) for e in range(epochs-2,epochs+1)]
        if not initial.get('complete') or not all(v.get('complete') for v in last):
            reasons.append(f'{cohort}: missing complete initial/last3');continue
        if mean([v['total'] for v in last])>initial['total']*.97:reasons.append(f'{cohort}: total drop <3%')
        if mean([v['prediction'] for v in last])>initial['prediction']*.95:reasons.append(f'{cohort}: KL drop <5%')
        if any(b['total']>a['total']*1.02 for a,b in zip(last,last[1:])):reasons.append(f'{cohort}: total rebound >2%')
        if last[-1]['total']>last[0]['total']:reasons.append(f'{cohort}: last3 total ending above start')
        if last[-1]['prediction']>last[0]['prediction']*(1+kl_tail_tolerance):
            reasons.append(f'{cohort}: last3 KL ending above start' if not kl_tail_tolerance
                           else f'{cohort}: last3 KL rebound >{100*kl_tail_tolerance:g}%')
        if min(v['score'] for v in last)<initial['score']+1:reasons.append(f'{cohort}: last3 score gains not all >=1pt')
        if mean([v['score'] for v in last])<initial['score']+2:reasons.append(f'{cohort}: mean score gain <2pt')
        if last[-1]['score']<last[0]['score']:reasons.append(f'{cohort}: last3 score ending below start')
        if mean([v['score'] for v in last])<.7*initial['teacher']:practical.append(f'{cohort}: retains <70% teacher score')
    return {'pass':not reasons and not practical,'trend_pass':not reasons,
            'kl_tail_relative_tolerance':kl_tail_tolerance,
            'practical_pass':not practical,'reasons':reasons+practical,
            'trend_reasons':reasons,'practical_reasons':practical}

def collect(run):
    cfg=json.loads((run/'manifest.json').read_text())['config'];grouped=defaultdict(list)
    for f in sorted(run.glob('eval.*.rank*.jsonl')):
        _,cohort,stage,_=f.name.split('.',3);grouped[f'{cohort}/{stage}']+=read(f)
    expected={'validation':{r['id'] for r in read(ROOT/cfg['validation_rows'])},
              'train_audit':{r['id'] for r in read(ROOT/cfg['audit_rows'])},
              'confirm':{r['id'] for r in read(BASE/'confirm.jsonl')}}
    stages={k:summarize(v,expected[k.split('/')[0]]) for k,v in grouped.items()}
    for k,s in stages.items():
        cohort,stage=k.split('/');initial=stages.get(f'{cohort}/initial',{})
        if s.get('complete') and initial.get('complete') and stage!='initial':
            s['gain_points']=s['score']-initial['score']
            s['kl_drop_fraction']=1-s['prediction']/initial['prediction']
            s['total_drop_fraction']=1-s['total']/initial['total']
            if cohort=='confirm':s['paired']=paired(grouped[f'{cohort}/initial'],grouped[k])
    log=read(run/'training.jsonl')
    result={'name':run.name,'config':cfg,'complete':(run/'complete.json').exists(),
            'updates':max([r['step'] for r in log],default=0),
            'planned_updates':math.ceil(len(read(ROOT/cfg['train_rows']))/cfg['batch_size'])*cfg['epochs'],
            'stages':stages,'stability':stable(stages,cfg['epochs']),
            'requested_stability':stable(stages,cfg['epochs'],kl_tail_tolerance=.02)}
    if (run/'early_stop.json').exists():result['early_stop']=json.loads((run/'early_stop.json').read_text())
    return result,grouped

def expanded_overview(results):
    selected=[r for r in results if r['name'].startswith('early_expanded') or r['name']=='full_verified_family']
    if not selected:return []
    lines=['## 扩大训练与重复 seed：最新完整 epoch','',
           '每行使用同一 epoch 的完整训练审计和验证记录；正在评测的 epoch 不计入分数。初始→最新表示同预算、同位置、同 seed 的训练变化。',
           '训练审计为固定训练子集，不是所有训练 batch 的均值。只有完成预定 epochs、重复 seed 和独立确认通过，才认定本轮稳定泛化。当前选择以总 loss/VQA 为主，保留原分项 KL 严格判定作诊断；变更属于看过调参结果后的自适应方案，独立确认未据此放宽。','',
           '| 位置 / 保留率 / seed | global batch / LR | 完整 epoch | 更新步 | 训练分数 初始→最新 | 验证分数 初始→最新 | 训练 total 初始→最新 | 验证 total 初始→最新 | 验证 KL 初始→最新 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        stages=r['stages'];cfg=r['config']
        epochs=[e for e in range(1,cfg['epochs']+1) if all(stages.get(f'{c}/epoch{e}',{}).get('complete')
                for c in ['train_audit','validation'])]
        epoch=max(epochs,default=0);stage=f'epoch{epoch}' if epoch else 'initial'
        def transition(cohort,field,digits):
            a=stages.get(cohort+'/initial',{});b=stages.get(cohort+'/'+stage,{})
            return f'{a[field]:.{digits}f}→{b[field]:.{digits}f}' if a.get('complete') and b.get('complete') else '—'
        lines.append(f"| {cfg['insertion']} / {cfg['ratio']:.0%} / {cfg['seed']} | {cfg['batch_size']} / {cfg['lr']:.3g} | {epoch}/{cfg['epochs']} | {r['updates']}/{r['planned_updates']} | "
                     +' | '.join(transition(c,f,d) for c,f,d in [('train_audit','score',2),('validation','score',2),
                         ('train_audit','total',4),('validation','total',4),('validation','prediction',4)])+' |')
    lines+=['','验证集未压缩 teacher 为 '+', '.join(f"{v:.2f}" for v in sorted({r['stages']['validation/initial']['teacher']
                 for r in selected if r['stages'].get('validation/initial',{}).get('complete')}))+' 分；混合任务宏平均不是官方统一榜单指标。','']
    for r in selected:
        rel=f'../artifacts/meta_compress_strict/runs/{r["name"]}'
        lines.append(f'- `{r["name"]}`：[曲线 PDF]({rel}/curves.pdf) · [逐步日志]({rel}/training.jsonl)')
    lines+=['','以下保留所有初始化筛查、负结果、分项 loss 和同预算固定对照。','']
    return lines

def confirmed_report(results, state):
    """Keep confirmed evidence separate from the subsequent coverage experiment."""
    indexed={r['name']:r for r in results}
    records=[json.loads(p.read_text()) for p in sorted(BASE.glob('confirmation.*.json'))]
    if not records:return []
    paired_answers=[]
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        directory=BASE/'runs'/r['name']
        before={v['id']:v for f in directory.glob('eval.confirm.initial.rank*.jsonl') for v in read(f)}
        after={v['id']:v for f in directory.glob('eval.confirm.selected.rank*.jsonl') for v in read(f)}
        for key in sorted(before.keys()&after.keys()):
            a=before[key];b=after[key]
            paired_answers.append({'seed':r['config']['seed'],'id':key,'document_id':a['document_id'],
                'task':a['task'],'question':a['question'],'gold_answers':json.dumps(a['answers'],ensure_ascii=False),
                'initial_answer':a['student_prediction'],'trained_answer':b['student_prediction'],
                'teacher_answer':a['teacher_prediction'],'initial_score':a['student_score'],
                'trained_score':b['student_score'],'teacher_score':a['teacher_score'],
                'score_gain':b['student_score']-a['student_score'],
                'initial_total_loss':a['total'],'trained_total_loss':b['total'],
                'initial_KL':a['prediction'],'trained_KL':b['prediction']})
    if paired_answers:
        with (BASE/'confirmation_answers.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired_answers[0]));writer.writeheader();writer.writerows(paired_answers)
    lines=['# MetaCompress：双 seed 独立确认与扩大训练进度','',
        '本页由真实逐题记录和确认结果生成。确认使用未参与选择的 224 张图、420 道题；固定 epoch8，不根据确认分数重选 checkpoint。',
        '七任务原生分数按任务宏平均，百分制；DeepForm/KLC 为字段 exact，并非官方整文档 F1。','',
        '## 已完成的独立确认','',
        '| seed | QA | 初始 VQA | 训练后 VQA | teacher | 增益与图像聚类 95% CI | total 初始→训练后 | KL 初始→训练后 | 确认判据 |',
        '|---|---:|---:|---:|---:|---|---|---|---|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        a=r['stages'].get('confirm/initial',{});b=record['confirmation']
        if not (a.get('complete') and b.get('complete')):continue
        ci=b['paired']['cluster_bootstrap_95ci_points']
        lines.append(f"| {r['config']['seed']} | {b['n']} | {a['score']:.2f} | {b['score']:.2f} | {b['teacher']:.2f} | {b['gain_points']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {a['total']:.4f}→{b['total']:.4f} | {a['prediction']:.4f}→{b['prediction']:.4f} | {'通过' if record['pass'] else '未通过'} |")
    lines+=['','上述两组来自同一训练方案：官方 MonkeyOCRv2-B-Und 全部冻结；原 MetaCompress 生成器与 KL + 行熵 + 最大列质量三项原始总 loss，权重均为 1；post-merger，保留 75% token；dense-position 参数初始化 strength=512；AdamW LR=3e-6、global batch=32、8 epochs、8 NPU ranks。',
        '训练 672 图/1,801 QA；固定训练审计 28 图/78 QA；调参验证 112 图/213 QA；确认 224 图/420 QA，按图像互斥。确认的 teacher/student 各自自由生成，cap=64；gold 只用于评分。',
        '', '## 训练与调参验证','',
        '| seed | 集合 | 初始 VQA→epoch8 | 初始 total→epoch8 | 总 loss/VQA 主判据 | 原 KL 严格诊断 |',
        '|---|---|---|---|---|---|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        for cohort in ['train_audit','validation']:
            a=r['stages'].get(cohort+'/initial',{});b=r['stages'].get(cohort+'/epoch8',{})
            if not (a.get('complete') and b.get('complete')):continue
            lines.append(f"| {r['config']['seed']} | {cohort} | {a['score']:.2f}→{b['score']:.2f} | {a['total']:.4f}→{b['total']:.4f} | {'通过' if r['requested_stability']['pass'] else '未通过'} | {'通过' if r['stability']['pass'] else '未通过'} |")
    lines+=['','主判据属于查看调参结果后的自适应修改：末三轮 KL 允许最多 2% 末端反弹；原判据要求无反弹。两种判据并列保留，不能宣称原严格协议通过。总 loss/VQA 的条件和独立确认的条件没有放宽。详见 [修订原始记录](../artifacts/meta_compress_strict/primary_endpoint_addendum.json)。',
        '这里的“稳定改善”是已记录的末三轮判据及独立确认改善，不表示每一个 epoch、每个任务或每个问题均单调变好。训练审计是固定训练子集，不能代替全部训练数据的生成分数。',
        '', '同预算 post-merger mean pooling 在调参验证集为 **50.374**；两 seed 的 epoch8 为 **46.61 / 49.97**。当前未证明优于 mean pooling，也未达到无压缩 teacher。batch32 同时改变梯度平均和累计更新量；匹配累计 LR 的 batch8 对照被跳过，不能唯一归因于梯度噪声降低。',
        '', '## 各任务独立确认','',
        '| seed | 任务 | 初始 | epoch8 | teacher | 增益 |','|---|---|---:|---:|---:|---:|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        a=r['stages'].get('confirm/initial',{});b=record['confirmation']
        if not (a.get('complete') and b.get('complete')):continue
        for task,v in sorted(b['tasks'].items()):
            initial=100*a['tasks'][task]['student_score'];final=100*v['student_score']
            lines.append(f"| {r['config']['seed']} | {task} | {initial:.2f} | {final:.2f} | {100*v['teacher_score']:.2f} | {final-initial:+.2f} |")
    if (BASE/'confirmation_per_task_uncertainty.json').exists():
        lines+=['','[按任务分层、按图像聚类的各任务增益区间](../artifacts/meta_compress_strict/confirmation_per_task_uncertainty.json)：ChartQA、DeepForm、KLC 两 seed 的名义 95% 区间下界均大于 0；DocVQA、InfoVQA、DT-VQA 两 seed 区间均跨 0。当前独立证据支持宏平均改善，尚不足以确认后面三个任务分别改善。各任务区间为单项区间，没有进行多重比较校正。']
    lines+=['','## 全部训练 batch 的逐 epoch loss','',
        '下面的 training total 是该 epoch 全部 global batch 的均值，包含末 batch 既定循环补齐，按训练样本权重；validation total 是固定验证集按任务宏平均。两者聚合权重不同，分别观察各自时间趋势。两 seed 的 training total 均逐 epoch 下降；验证 loss 总体下降后进入平台，VQA 有中途波动，不能描述为八轮严格单调上升。','',
        '| seed | epoch | training total | validation total | validation VQA |',
        '|---|---:|---:|---:|---:|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        grouped=defaultdict(list)
        for row in read(BASE/'runs'/r['name']/'training.jsonl'):grouped[row['epoch']].append(row['total'])
        expected=math.ceil(len(read(ROOT/r['config']['train_rows']))/r['config']['batch_size'])
        for epoch,values in sorted(grouped.items()):
            v=r['stages'].get(f'validation/epoch{epoch}',{})
            if len(values)==expected and v.get('complete'):
                lines.append(f"| {r['config']['seed']} | {epoch} | {mean(values):.5f} | {v['total']:.5f} | {v['score']:.3f} |")
    lines+=['','## 后续覆盖扩大实验的真实状态','',f"控制器阶段：`{state.get('phase')}`，记录时间 `{state.get('updated_at')}`。",
        '扩大数据为 12,661 图/19,999 QA、8 个完整 epoch。初始化重训，LR 按调度 LR 总量缩放为约 2.7408e-7；这与双 seed 确认训练的覆盖和优化过程不同，不能沿用其成功结论。']
    full=indexed.get('full_verified_family')
    if full:
        epochs=[e for e in range(1,9) if all(full['stages'].get(f'{c}/epoch{e}',{}).get('complete') for c in ['train_audit','validation'])]
        lines+=['',f"实际更新 {full['updates']}/{full['planned_updates']}；完整生成评测 {max(epochs,default=0)}/8 epochs；训练完成标志：{full['complete']}。"]
        if full['complete']:
            lines+=['',f"扩大训练总 loss/VQA 主判据：{'通过' if full['requested_stability']['pass'] else '未通过：'+'；'.join(full['requested_stability']['reasons'])}。"]
    else:lines+=['','尚无扩大训练 manifest，不能视为已经训练完成。']
    final=ROOT/'docs/METACOMPRESS_STRICT_FINAL_RESULTS.md'
    if final.exists():lines+=['','[实际完成的全量 benchmark 结果](METACOMPRESS_STRICT_FINAL_RESULTS.md)。']
    else:lines+=['','23,565 QA 的 benchmark 尚无最终报告，不能报告全量分数。']
    lines+=['','## 明确入口与产物','',
        '```bash','bash scripts/run_meta_compress_strict.sh status','bash scripts/run_meta_compress_strict.sh report',
        '# 新训练（先检查 status，避免与控制器重复提交）',
        'bash scripts/run_meta_compress_strict.sh train artifacts/meta_compress_strict/configs/early_expanded3_screen37_batch32.json',
        '# 独立图像问答；将 IMAGE 和 QUESTION 替换为实际输入',
        'bash scripts/run_meta_compress_strict.sh predict \\',
        '  --checkpoint artifacts/meta_compress_strict/runs/early_expanded3_screen37_batch32/epoch8.pt \\',
        '  --image IMAGE --question QUESTION --teacher --save-json artifacts/meta_compress_strict/prediction.json','```','',
        '[完整方案](METACOMPRESS_STRICT_OPTIMIZATION.md) · [实时逐轮结果](METACOMPRESS_STRICT_RESULTS.md) · [确认集初始/训练后/teacher/gold 逐题对比](../artifacts/meta_compress_strict/confirmation_answers.csv) · [全部逐题记录](../artifacts/meta_compress_strict/answers.csv)',
        '', '确认对比 CSV 中逐题 score 为 0～1；正文宏平均为百分制。每行保留一次确定性生成结果，没有按随机压缩重复平均。']
    for record in records:
        name=record['candidate'];rel=f'../artifacts/meta_compress_strict/runs/{name}'
        lines+=['',f'`{name}`：[曲线 PDF]({rel}/curves.pdf) · [训练日志]({rel}/training.jsonl) · [配置](../artifacts/meta_compress_strict/configs/{name}.json) · [独立确认机器记录](../artifacts/meta_compress_strict/confirmation.{name}.json)。']
    (ROOT/'docs/METACOMPRESS_STRICT_CONFIRMED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return ['## 独立确认','', '[双 seed 独立确认、限制和后续训练状态](METACOMPRESS_STRICT_CONFIRMED_RESULTS.md)。','']

def main():
    results=[];allrows=[]
    for run in sorted((BASE/'runs').glob('*')):
        if not (run/'manifest.json').exists():continue
        result,grouped=collect(run);results.append(result)
        for rows in grouped.values():allrows.extend({'run':run.name,**r} for r in rows)
    save(BASE/'summary.json',results)
    lines=['# 原结构、完整三项 loss：训练与验证实测','',
        '只将完整评测阶段计为结果；分数为七任务各自原生指标的百分制宏平均。DeepForm/KLC 为字段 exact。',
        '训练审计集固定不变，独立生成答案；训练 batch loss 和固定审计 loss 分开记录。','',
        '主判据与原附加 KL 严格诊断分开列出，均非独立确认结果。主判据允许最后三轮 KL 末端≤起点的 102%，其他总 loss、VQA、KL 整体降幅与质量门槛全部保留。该修订发生在查看调参结果后、独立确认前。','',
        '| run | 完成 | updates | 总 loss/VQA 主判据 | 原单 seed 趋势诊断 | 原趋势+质量保留诊断 |','|---|---|---:|---|---|---|']
    for r in results:lines.append(f"| {r['name']} | {'提前停止，未完成原计划' if r.get('early_stop') else r['complete']} | {r['updates']}/{r['planned_updates']} | {r['requested_stability']['pass']} | {r['stability']['trend_pass']} | {r['stability']['pass']} |")
    lines+=['','| run | 集合/阶段 | QA | score | teacher | KL | H | collapse | total | Δscore |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for key,s in sorted(r['stages'].items()):
            if not s['complete']:
                lines.append(f"| {r['name']} | {key}（未完成） | {s['n']}/{s['expected']} | — | — | — | — | — | — | — |")
            else:lines.append(f"| {r['name']} | {key} | {s['n']} | {s['score']:.3f} | {s['teacher']:.3f} | {s['prediction']:.5f} | {s['entropy']:.3f} | {s['collapse']:.3f} | {s['total']:.4f} | {s.get('gain_points',0):+.3f} |")
    lines+=['','## 同预算固定对照（非生成器候选）','',
            '| 位置/对照 | QA | score | teacher | KL | total |','|---|---:|---:|---:|---:|---:|']
    expected={r['id'] for r in read(BASE/'validation.jsonl')}
    for directory in sorted((BASE/'baselines').glob('*')):
        rows=[r for f in directory.glob('eval.validation.fixed.rank*.jsonl') for r in read(f)]
        allrows.extend({'run':'control:'+directory.name,**r} for r in rows)
        s=summarize(rows,expected)
        if s['complete']:lines.append(f"| {directory.name} | {s['n']} | {s['score']:.3f} | {s['teacher']:.3f} | {s['prediction']:.5f} | {s['total']:.4f} |")
        else:lines.append(f"| {directory.name}（未完成） | {s['n']}/{s['expected']} | — | — | — | — |")
    text='\n'.join(lines)+'\n'
    (BASE/'RESULTS.md').write_text(text)
    state=json.loads((BASE/'controller.json').read_text()) if (BASE/'controller.json').exists() else {}
    doc=['# 严格 MetaCompress 优化：实时结果','',
         f'控制器最近记录阶段：`{state.get("phase","initializing")}`；更新：`{state.get("updated_at","pending")}`。',
         '本文件读取真实产物自动生成；未完成阶段不作为最终结果，门槛失败不等于训练仍在运行。',
         '[方案和明确入口](METACOMPRESS_STRICT_OPTIMIZATION.md) · [调度状态](../artifacts/meta_compress_strict/RUNNING_STATUS.md) · [逐题作答](../artifacts/meta_compress_strict/answers.csv)','',*confirmed_report(results,state),*expanded_overview(results),text,
         '## 曲线与未达标原因','']
    for r in results:
        rel=f'../artifacts/meta_compress_strict/runs/{r["name"]}'
        doc+=['',f'### {r["name"]}','',f'[训练/验证曲线 PDF]({rel}/curves.pdf) · [PNG]({rel}/curves.png)','',
              '总 loss/VQA 主判据：'+('通过（仍需重复 seed 与独立确认）' if r['requested_stability']['pass'] else '尚未通过：'+'；'.join(r['requested_stability']['reasons'])),
              '', '原额外 KL 严格诊断：'+('通过' if r['stability']['pass'] else '未通过：'+'；'.join(r['stability']['reasons']))]
        cfg=r['config']
        doc+=['',f"位置 `{cfg['insertion']}`；token 保留率 {cfg['ratio']:.0%}；初值 `{cfg['initialization']}`；optimizer `{cfg['optimizer']}`；LR `{cfg['lr']}`；clip `{cfg['clip']}`；seed `{cfg['seed']}`。"]
        epochs=[(int(k.split('epoch')[1]),v) for k,v in r['stages'].items()
                if k.startswith('validation/epoch') and v.get('complete')]
        initial=r['stages'].get('validation/initial',{})
        if epochs and initial.get('complete'):
            epoch,last=max(epochs,key=lambda v:v[0])
            doc+=['',f'最新完整验证阶段 epoch {epoch}，不是追选的最高分 epoch：','',
                  '| 任务 | 初始 | 最新 | teacher | Δscore | KL 初始→最新 |',
                  '|---|---:|---:|---:|---:|---:|']
            for task,v in sorted(last['tasks'].items()):
                a=initial['tasks'][task]
                doc.append(f"| {task} | {100*a['student_score']:.2f} | {100*v['student_score']:.2f} | {100*v['teacher_score']:.2f} | {100*(v['student_score']-a['student_score']):+.2f} | {a['prediction']:.4f}→{v['prediction']:.4f} |")
    (ROOT/'docs/METACOMPRESS_STRICT_RESULTS.md').write_text('\n'.join(doc)+'\n')
    if allrows:
        keys=['run','cohort','stage','id','document_id','task','question','answers','teacher_prediction','student_prediction',
              'teacher_score','student_score','prediction','entropy','collapse','total','tokens','compressed_tokens','hit_token_limit']
        with (BASE/'answers.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(allrows)
    plot(results)
    print(json.dumps([{'name':r['name'],'updates':r['updates'],'complete':r['complete'],
        'primary_stable':r['requested_stability']['pass'],'legacy_strict_stable':r['stability']['pass']} for r in results]))

def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    for r in results:
        run=BASE/'runs'/r['name']
        log=read(run/'training.jsonl');byepoch=defaultdict(list)
        for row in log:byepoch[row['epoch']].append(row)
        per_epoch=math.ceil(len(read(ROOT/r['config']['train_rows']))/r['config']['batch_size'])
        online=[(e,{k:mean([v[k] for v in rr]) for k in ['total','prediction']})
                for e,rr in sorted(byepoch.items()) if len(rr)==per_epoch]
        signature=json.dumps({'stages':r['stages'],'online':online},sort_keys=True)
        sig=run/'plot_signature.json'
        if sig.exists() and sig.read_text()==signature:continue
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,field in zip(axes[:2],['total','prediction']):
            ax.plot([v[0] for v in online],[v[1][field] for v in online],marker='x',linestyle=':',color='tab:blue',label='train batches (epoch mean)')
        for cohort in ['train_audit','validation']:
            series=[]
            for k,s in r['stages'].items():
                c,stage=k.split('/')
                if c!=cohort or not s['complete'] or (stage!='initial' and not stage.startswith('epoch')):continue
                series.append((0 if stage=='initial' else int(stage[5:]),s))
            series.sort(key=lambda v:v[0])
            for ax,field in zip(axes,['total','prediction','score']):
                ax.plot([v[0] for v in series],[v[1][field] for v in series],marker='o',
                        color={'train_audit':'tab:orange','validation':'tab:green'}[cohort],label=cohort)
                ax.set_xlabel('epoch');ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_title('prediction KL' if field=='prediction' else field);ax.grid(alpha=.25);ax.legend()
        fig.suptitle(r['name']);fig.tight_layout()
        fig.savefig(run/'curves.png',dpi=140);fig.savefig(run/'curves.pdf');plt.close(fig);sig.write_text(signature)

if __name__=='__main__':main()
