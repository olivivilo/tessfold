"""Audit and report per-intervention decoder KL / entropy correlations."""
import argparse,collections,csv,hashlib,json,re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.report_encoder_candidates import analyze, rho
from vqa.benchmark_metrics import native_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/token_proxy/random_v1_20x'
OUT=ROOT/'artifacts/decoder_dependent_loss'
NAMES=['prediction_kl','entropy','total','first_token_kl','entropy_normalized']


def fmt(v):return '不可估计（常量）' if v is None else f'{v:.4f}'
def ci(v):return f'[{v[0]:.4f}, {v[1]:.4f}]'


def paired_metric_comparison(x_kl,x_total,y,draws=1000):
    """Paired image bootstrap for adding entropy; preserve outcome ties."""
    a=np.diff(x_kl.reshape(len(y),12,2),axis=2)[:,:,0]
    b=np.diff(x_total.reshape(len(y),12,2),axis=2)[:,:,0]
    d=np.diff(y.reshape(len(y),12,2),axis=2)[:,:,0]
    eligible=(a!=0)&(b!=0)&(d!=0)
    ac=((a*d>0)&eligible).sum(1);bc=((b*d>0)&eligible).sum(1);n=eligible.sum(1)
    rng=np.random.default_rng(20261025);ix=rng.integers(len(y),size=(draws,len(y)))
    boot=(bc[ix].sum(1)-ac[ix].sum(1))/np.maximum(1,n[ix].sum(1))
    observed=float((bc.sum()-ac.sum())/max(1,n.sum()))
    # For pooled rho use the same image draws for both metrics and re-rank each.
    dr=[]
    for idx in ix:
        left=rho(x_kl[idx].ravel(),y[idx].ravel());right=rho(x_total[idx].ravel(),y[idx].ravel())
        if left is not None and right is not None:dr.append(right-left)
    return {'seed_order_total_minus_kl':observed,'seed_order_delta_ci95':list(map(float,np.quantile(boot,[.025,.975]))),
            'seed_joint_eligible_pairs':int(n.sum()),'pooled_rho_total_minus_kl':rho(x_total.ravel(),y.ravel())-rho(x_kl.ravel(),y.ravel()),
            'pooled_rho_delta_ci95':list(map(float,np.quantile(dr,[.025,.975])))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=5);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();out=a.output
    samples=[json.loads(l) for l in (BASE/'samples.jsonl').open()]
    samples_by_id={r['id']:r for r in samples}
    complete=[r for r in samples if (out/'losses'/f"{r['document_id']}.json").exists()]
    if not a.allow_partial:assert len(complete)==len(samples),'Missing image losses; results are not complete'
    wanted={r['id'] for r in complete};losses={};audits=[];audit_by_id={};manifests=set();seconds=[]
    for r in complete:
        d=json.loads((out/'losses'/f"{r['document_id']}.json").read_text())
        assert len(d['trials'])==26 and d['sample_id']==r['id']
        audits.append(d['audit']);seconds.append(d['elapsed_seconds']);manifests.add(json.dumps(d['code_sha256'],sort_keys=True))
        audit_by_id[d['sample_id']]=d['audit']
        for trial in d['trials']:
            assert trial['id'] not in losses
            losses[trial['id']]=trial
    manifest_list=[json.loads(m) for m in sorted(manifests)]
    # Scheduling runner was upgraded from fixed shards to atomic image claims.
    # Accept that change only when the numerical loss/model/generator hashes match.
    for m in manifest_list:
        assert all(m[f]==manifest_list[0][f] for f in m if f!='scripts/eval_decoder_dependent.py'), 'Numerical implementation changed within cohort'
    rows=[]
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for line in path.open():
            r=json.loads(line)
            if r['sample_id'] not in wanted:continue
            loss=losses[r['id']]
            assert loss['matrix_sha256']==r['matrix_file_sha256'] and loss['compressed_feature_sha256']==r['compressed_feature_sha256']
            for name in ['teacher_score','student_score','score_drop']:assert loss[name]==r[name]
            # Original trial JSON omits OCRBench dataset_name. Recover sample
            # metadata so HME100k uses its whitespace-insensitive native metric.
            score_row={**samples_by_id[r['sample_id']],**r}
            assert native_score(r['teacher_prediction'],score_row)==r['teacher_score']
            assert native_score(r['student_prediction'],score_row)==r['student_score']
            assert abs(loss['prediction_kl']+loss['entropy']-loss['total'])<1e-9
            r['new_loss']={k:loss[k] for k in NAMES};rows.append(r)
    assert len(rows)==26*len(complete)
    # Reuse saved CPU generation audits only when EOS proves the response is
    # complete. A 16-token truncated prefix is never scored as a full answer.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/MonkeyOCRv2-B-Und'),local_files_only=True,trust_remote_code=True)
    lowest={r['sample_id']:r for r in rows if r['kind']=='random' and r['ratio']==.125 and r['family']=='local' and r['replicate']==0}
    generation_checks=[]
    for sid,ad in audit_by_id.items():
        ti,si=ad.get('teacher_regenerated_ids',[]),ad.get('student_regenerated_ids',[])
        if 151645 not in ti or 151645 not in si:continue
        r=lowest[sid];score_row={**samples_by_id[sid],**r}
        tp=tokenizer.decode(ti,skip_special_tokens=True);sp=tokenizer.decode(si,skip_special_tokens=True)
        ts,ss=native_score(tp,score_row),native_score(sp,score_row)
        generation_checks.append({'sample_id':sid,'task':r['task'],'question':r['question'],'answers':r['answers'],
            'teacher_cpu_prediction':tp,'student_cpu_prediction':sp,'teacher_npu_prediction':r['teacher_prediction'],
            'student_npu_prediction':r['student_prediction'],'cpu_teacher_score':ts,'cpu_student_score':ss,
            'npu_teacher_score':r['teacher_score'],'npu_student_score':r['student_score'],
            'cpu_score_drop':ts-ss,'npu_score_drop':r['score_drop'],**r['new_loss']})
    (out/'cpu_generation_checks.json').write_text(json.dumps(generation_checks,ensure_ascii=False,indent=2)+'\n')
    fields=['id','sample_id','document_id','task','question','image','answers','kind','family','ratio','replicate',
            'matrix_seed','matrix_file','teacher_prediction','teacher_score','student_prediction','student_score','score_drop']
    with (out/'trials.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields+NAMES);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in fields},**r['new_loss']})
    rr=sorted([r for r in rows if r['kind']=='random'],key=lambda r:(r['document_id'],r['ratio'],r['family'],r['replicate']))
    y=np.array([r['score_drop'] for r in rr]).reshape(-1,24)
    teacher=np.array([r['teacher_score'] for r in rr])[::24];tasks=np.array([r['task'] for r in rr])[::24]
    ratios=[r['ratio'] for r in rr[:24]];families=[r['family'] for r in rr[:24]]
    items=[(name,np.array([r['new_loss'][name] for r in rr]).reshape(-1,24),y,teacher,tasks,ratios,families) for name in NAMES]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:metrics=dict(pool.map(analyze,items))
    for name,x,_,_,_,_,_ in items:
        dx=np.diff(x.reshape(len(x),12,2),axis=2).ravel()
        dy=np.diff(y.reshape(len(y),12,2),axis=2).ravel()
        nx=int((dx!=0).sum());ny=int((dy!=0).sum());overlap=int(((dx!=0)&(dy!=0)).sum())
        metrics[name]['within_config_tie_ceiling']=float(overlap/np.sqrt(nx*ny)) if nx and ny else None
        metrics[name]['seed_loss_nontie_pairs']=nx
    for name in NAMES:
        for rate in [.25,.5]:
            cr=[r for r in rows if r['kind']=='mean_pool_control' and r['ratio']==rate]
            metrics[name]['controls'][str(rate)]=rho([r['new_loss'][name] for r in cr],[r['score_drop'] for r in cr])
        for label,condition in [
            ('teacher_first_token_match',lambda d:d['teacher_first_argmax']==d['teacher_response_first_id']),
            ('teacher_full_forced_argmax_match',lambda d:d['teacher_forced_greedy_agreement']==1.0)]:
            subset=[r for r in rr if condition(audit_by_id[r['sample_id']])]
            metrics[name][label]={'images':len(subset)//24,'pooled_spearman':rho([r['new_loss'][name] for r in subset],[r['score_drop'] for r in subset])}
    per_task={}
    for t in sorted(set(tasks)):
        tr=[r for r in rr if r['task']==t];mask=tasks==t
        per_task[t]={'images':int(mask.sum()),'teacher_score':float(teacher[mask].mean()),
                     'student_score':float(np.mean([r['student_score'] for r in tr])),
                     'score_drop':float(np.mean([r['score_drop'] for r in tr])),
                     'student_zero_fraction':float(np.mean([r['student_score']==0 for r in tr]))}
    examples={}
    for name in NAMES[:3]:
        good=[];bad=[]
        for i in range(0,len(rr),2):
            left,right=sorted(rr[i:i+2],key=lambda r:r['new_loss'][name])
            if right['new_loss'][name]==left['new_loss'][name]:continue
            pair={'low_loss':left,'high_loss':right}
            if left['score_drop']<right['score_drop']:good.append(pair)
            elif left['score_drop']>right['score_drop']:bad.append(pair)
        key=lambda p:abs(p['low_loss']['score_drop']-p['high_loss']['score_drop'])
        examples[name]={'concordant':sorted(good,key=key,reverse=True)[:3],
                        'counterexamples':sorted(bad,key=key,reverse=True)[:3]}
    (out/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    audit={'complete':len(complete)==len(samples),'images':len(complete),'expected_images':len(samples),
           'random_trials':len(rr),'controls':len(rows)-len(rr),'unique_matrices':len(losses),
           'all_feature_matrix_compressed_hashes_verified':True,'all_cached_scores_recomputed':True,
           'code_sha256_versions':manifest_list,
           'identity_checks':[r for r in audits if 'identity_KL_max_abs' in r],
           'teacher_forced_first_token_matches':sum(r['teacher_first_argmax']==r['teacher_response_first_id'] for r in audits),
           'teacher_forced_agreement_mean':float(np.mean([r['teacher_forced_greedy_agreement'] for r in audits])),
           'compute_seconds_sum':sum(seconds),'compute_seconds_per_image_median':float(np.median(seconds))}
    regenerated=[r for r in audits if 'teacher_generation_prefix_reproduced' in r]
    audit['regenerated_prefix_checks']={'images':len(regenerated),
        'teacher_match':sum(r['teacher_generation_prefix_reproduced'] for r in regenerated),
        'student_match':sum(r['student_generation_prefix_reproduced'] for r in regenerated),
        'both_match':sum(r['teacher_generation_prefix_reproduced'] and r['student_generation_prefix_reproduced'] for r in regenerated)}
    audit['completed_cpu_generation_checks']={'images':len(generation_checks),
        'teacher_score_equal':sum(r['cpu_teacher_score']==r['npu_teacher_score'] for r in generation_checks),
        'student_score_equal':sum(r['cpu_student_score']==r['npu_student_score'] for r in generation_checks),
        'score_drop_equal':sum(r['cpu_score_drop']==r['npu_score_drop'] for r in generation_checks),
        'cpu_vs_npu_score_drop_spearman':rho([r['cpu_score_drop'] for r in generation_checks],[r['npu_score_drop'] for r in generation_checks]),
        'loss_vs_cpu_drop_spearman':{n:rho([r[n] for r in generation_checks],[r['cpu_score_drop'] for r in generation_checks]) for n in NAMES[:3]},
        'loss_vs_npu_drop_spearman':{n:rho([r[n] for r in generation_checks],[r['npu_score_drop'] for r in generation_checks]) for n in NAMES[:3]}}
    audit['analysis_code_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in [
        'scripts/report_decoder_dependent.py','scripts/report_encoder_candidates.py','vqa/benchmark_metrics.py']}
    audit['source_generation_audit_sha256']=hashlib.sha256((BASE/'audit.json').read_bytes()).hexdigest()
    audit['source_generation_audit']=json.loads((BASE/'audit.json').read_text())
    if (OUT/'model_audit.json').exists():audit['model_audit']=json.loads((OUT/'model_audit.json').read_text())
    summary={'audit':audit,'metrics':metrics,'per_task':per_task,'bootstrap_repeats':1000,
             'student_zero_fraction':float(np.mean([r['student_score']==0 for r in rr])),
             'teacher_positive_images':int((teacher>0).sum()),'teacher_perfect_images':int((teacher==1).sum()),
             'lambda_entropy':1.0,'protocol':json.loads((OUT/'protocol.json').read_text())}
    summary['adding_entropy_paired_bootstrap']=paired_metric_comparison(items[0][1],items[2][1],y)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plot(out,rr,summary)
    sections=[]
    sections.append('# Decoder-dependent 压缩 loss：实际结果\n')
    sections.append(f"本次完成 **{len(complete):,} 张不同图像、{len(rr):,} 次独立随机压缩、{len(rows)-len(rr):,} 次均值池化对照**。"+('全部预定样本已完成。' if audit['complete'] else '**这是部分样本结果，尚未完成预定全规模。**'))
    sections.append('下游作答及得分复用同一个官方基线、同一组原始特征和逐次压缩矩阵的既有独立生成；本轮新增运行 decoder 前向，计算完整词表预测 KL。没有把旧的 encoder-only loss 改名为新 loss。每个种子的得分独立参加统计。')
    sections.append('分数下降定义为 Δ = teacher_score − student_score；正相关表示 loss 越高、压缩后损失越大。主分析仅含随机压缩；均值池化另列。所有分数均在 [0,1] 内。')
    sections.append('## 三个主要指标\n')
    sections.append('| Loss | pooled Spearman ρ（图像 bootstrap 95% CI） | 同题内秩相关 | 固定题目×比例×类型的秩相关（95% CI） | 两种子非平局排序正确率（95% CI） |\n|---|---|---|---|---|')
    for name in NAMES[:3]:
        m=metrics[name];pair=m['paired_seed_ordering']
        sections.append(f"| {name} | {fmt(m['pooled_spearman'])} {ci(m['pooled_image_bootstrap_ci95'])} | {fmt(m['within_question'])} | {fmt(m['within_question_ratio_family'])} {ci(m['within_question_ratio_family_ci95'])} | {100*pair['accuracy']:.2f}% {ci([100*v for v in pair['image_bootstrap_ci95']])} |")
    sections.append('95% CI 使用 1,000 次按图像重采样；pooled Spearman 每次重新排名。同题内相关是组内秩中心化后再计算相关；固定题目×比例×类型每组只有两个种子，因此必须结合排序正确率和平局数解释。它不是另一个 pooled Spearman。')
    sections.append('两种子组内秩相关受大量下游平局限制。固定现有 loss/得分的平局位置，即使所有非平局种子对都排对，其最大值也只有 `非平局交集数 / sqrt(loss 非平局数 × 得分非平局数)`；因此不能机械地用 0.7 要求这个条件指标。相关性和非平局排序正确率应一起看。')
    add=summary['adding_entropy_paired_bootstrap']
    sections.append(f"加入熵的配对图像 bootstrap：总 loss 相对 KL 的 pooled ρ 变化={add['pooled_rho_total_minus_kl']:.4f}，95% CI {ci(add['pooled_rho_delta_ci95'])}；共同非平局种子对上的排序正确率变化={100*add['seed_order_total_minus_kl']:.2f} 个百分点，95% CI {ci([100*v for v in add['seed_order_delta_ci95']])}。这直接比较相同图像/相同矩阵的两种指标，不能把很小的点估计差异自动解释为熵有效。")
    sections.append('| Loss | teacher 得分 > 0 的 pooled ρ | teacher 得分 = 1 的 pooled ρ | 控制题目与配置后的全局秩残差相关 | 同比例同类型种子非平局数 | 下游平局数 | loss 平局但下游不平局数 |\n|---|---|---|---|---|---|---|')
    for name in NAMES[:3]:
        m=metrics[name];pair=m['paired_seed_ordering']
        sections.append(f"| {name} | {fmt(m['teacher_positive_spearman'])} | {fmt(m['teacher_perfect_spearman'])} | {fmt(m['partial_global_rank_question_config'])} | {pair['concordant']+pair['discordant']:,} | {pair['outcome_ties']:,} | {pair['loss_ties_outcome_nontie']:,} |")
    sections.append('## 各任务结果\n')
    sections.append('| 数据集 | 图像 | Teacher 平均分 | Student 平均分 | 平均下降 | KL ρ | 熵 ρ | 总 loss ρ |\n|---|---|---|---|---|---|---|---|')
    for t,d in per_task.items():
        sections.append(f"| {t} | {d['images']} | {100*d['teacher_score']:.2f} | {100*d['student_score']:.2f} | {100*d['score_drop']:.2f} | "+' | '.join(fmt(metrics[n]['per_task'][t]['pooled_spearman']) for n in NAMES[:3])+' |')
    sections.append('表中的平均任务分数只描述整体难度；相关性和排序分析始终使用每一次随机压缩的原始得分，没有先对种子或作答求均值。各任务沿用既有 native_score 指标；跨任务 pooled 相关应同时结合逐任务结果。')
    sections.append('| 数据集 | KL 种子排序正确率 / 非平局对数 | 熵种子排序正确率 / 非平局对数 | 总 loss 种子排序正确率 / 非平局对数 |\n|---|---|---|---|')
    for t in per_task:
        values=[]
        for n in NAMES[:3]:
            d=metrics[n]['per_task'][t]
            values.append(f"{100*d['seed_order_accuracy']:.2f}% / {d['nontie_pairs']:,}" if d['nontie_pairs'] else '无非平局对')
        sections.append('| '+t+' | '+' | '.join(values)+' |')
    sections.append('逐任务种子排序表为点估计；每对均固定题目、比例与类型，只比较随机种子，不把它与跨题目的 pooled 相关混用。')
    sections.append('## 固定比例和矩阵类型\n')
    sections.append('| 保留率/类型 | KL pooled ρ | 熵 pooled ρ | 总 loss pooled ρ |\n|---|---|---|---|')
    for key in metrics['total']['per_ratio_family']:
        sections.append('| '+key+' | '+' | '.join(fmt(metrics[n]['per_ratio_family'][key]['pooled_spearman']) for n in NAMES[:3])+' |')
    sections.append('各比例/类型的 pooled 值仍跨题目；更严格的同题控制和种子排序见主表。')
    sections.append('## 对照和诊断\n')
    for name in NAMES:
        m=metrics[name]
        sections.append(f"- `{name}` 范围/中位数：{m['loss_quantiles'][0]:.6g} / {m['loss_quantiles'][3]:.6g} / {m['loss_quantiles'][-1]:.6g}（min/median/max）；均值池化 25% / 50% 的 ρ：{fmt(m['controls']['0.25'])} / {fmt(m['controls']['0.5'])}。")
        if name in NAMES[:3]:sections.append(f"  固定配置秩相关在现有平局分布下的上限：{fmt(m['within_config_tie_ceiling'])}。")
    sections.append(f"- Student 得分为零的随机试验占 {100*summary['student_zero_fraction']:.2f}%；teacher 得分 > 0 的图像 {summary['teacher_positive_images']:,} 张，teacher 满分图像 {summary['teacher_perfect_images']:,} 张。")
    sections.append(f"- 共享 teacher 前缀的第一个位置 argmax 与历史 teacher 首 token 一致：{audit['teacher_forced_first_token_matches']:,}/{len(complete):,}。完整前缀平均 greedy 一致率：{100*audit['teacher_forced_agreement_mean']:.4f}%。不同设备/并行归约与逐 token KV-cache 可引入 BF16 数值差异，这些核对值保留在 audit.json。")
    rg=audit['regenerated_prefix_checks']
    sections.append(f"- 跨 CPU/NPU 的自由生成前缀复核：{rg['images']} 张图，teacher 一致 {rg['teacher_match']}，student 一致 {rg['student_match']}，两者均一致 {rg['both_match']}。前缀最长 16 token；不将首个先导样本一致推广为所有样本一致。得分仍为原 NPU 自由生成，loss 为 CPU BF16 共同前缀前向；存在跨设备数值与生成分叉限制。")
    cg=audit['completed_cpu_generation_checks']
    sections.append(f"- 利用已保存的重生记录，在 teacher 和 student 均于 16 token 内生成 EOS 的 **{cg['images']} 张图**上计算完整 CPU 作答分数：teacher 得分一致 {cg['teacher_score_equal']}，student 得分一致 {cg['student_score_equal']}，得分下降一致 {cg['score_drop_equal']}；CPU/NPU 得分下降的 Spearman={fmt(cg['cpu_vs_npu_score_drop_spearman'])}。这是按完成 EOS 筛选的短作答诊断子集，只检查 12.5% local seed 0，不代表所有 93,626 次干预。完整记录见 `cpu_generation_checks.json`。")
    sections.append('| 完整 EOS 子集上的 loss | 对 CPU 得分下降的 ρ | 对同一子集 NPU 得分下降的 ρ |\n|---|---|---|')
    for n in NAMES[:3]:sections.append(f"| {n} | {fmt(cg['loss_vs_cpu_drop_spearman'][n])} | {fmt(cg['loss_vs_npu_drop_spearman'][n])} |")
    sections.append('首 token KL 是没有答案前缀的诊断；归一化熵 H/log N 是长度尺度诊断，均不是替代论文目标的主结果。没有根据评测分数调节 λ 或选择最好的变体。')
    sections.append('| 跨设备一致性诊断子集 | 图像 | KL ρ | 熵 ρ | 总 loss ρ |\n|---|---|---|---|---|')
    for label in ['teacher_first_token_match','teacher_full_forced_argmax_match']:
        sections.append('| '+label+' | '+str(metrics['total'][label]['images'])+' | '+' | '.join(fmt(metrics[n][label]['pooled_spearman']) for n in NAMES[:3])+' |')
    sections.append('## 独立作答实例\n')
    for typ in ['concordant','counterexamples']:
        sections.append('### '+('KL 与下游排序一致' if typ=='concordant' else 'KL 排序的反例')+'\n')
        for pair in examples['prediction_kl'][typ][:2]:
            left,right=pair['low_loss'],pair['high_loss']
            sections.append(f"- `{left['sample_id']}`，{left['task']}，保留率 {left['ratio']}，{left['family']}。问题：{left['question']}；gold：{json.dumps(left['answers'],ensure_ascii=False)}；teacher：{left['teacher_prediction']!r}，分数 {left['teacher_score']:.4f}。")
            for label,r in [('低 KL',left),('高 KL',right)]:
                sections.append(f"  - {label}，seed index {r['replicate']}：KL={r['new_loss']['prediction_kl']:.6f}，熵={r['new_loss']['entropy']:.6f}，总={r['new_loss']['total']:.6f}；student={r['student_prediction']!r}，分数={r['student_score']:.4f}，下降={r['score_drop']:.4f}。")
    sections.append('## 结果文件与审计\n')
    sections.append('- `trials.csv`：所有逐次作答、gold、两项 loss、总 loss、分数与分数下降。\n- `losses/<document_id>.json`：逐响应位置 KL、teacher token IDs、矩阵/特征摘要。\n- `analysis.json`：完整相关性、逐任务与逐配置结果、置信区间。\n- `examples.json`：三项 loss 的一致排序实例及反例。\n- `audit.json`：完整性、分数重算、来源摘要及接口核对。\n- `correlation_scatter.pdf/png`：逐次压缩散点图。')
    markdown='\n\n'.join(sections)+'\n'
    markdown=re.sub(r'(\|)\n\n(?=\|)',r'\1\n',markdown)
    (out/'RESULTS.md').write_text(markdown)
    print(json.dumps({'complete':audit['complete'],'images':len(complete),'random_trials':len(rr),'metrics':{n:metrics[n]['pooled_spearman'] for n in NAMES[:3]}},ensure_ascii=False),flush=True)


def plot(out,rows,summary):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));rng=np.random.default_rng(25)
    chosen=rng.choice(len(rows),size=min(len(rows),18000),replace=False)
    for ax,name in zip(axes,NAMES[:3]):
        for family,color in [('local','#2471a3'),('global','#ca6f1e')]:
            rr=[rows[i] for i in chosen if rows[i]['family']==family]
            ax.scatter([r['new_loss'][name] for r in rr],[r['score_drop'] for r in rr],s=3,alpha=.18,color=color,label=family,rasterized=True)
        m=summary['metrics'][name];ax.set_title(f"{name}\npooled rho={m['pooled_spearman']:.3f}; within config={m['within_question_ratio_family']:.3f}")
        ax.set_xlabel('Loss (nats)');ax.set_ylabel('Teacher score - student score')
    axes[0].legend();fig.tight_layout();fig.savefig(out/'correlation_scatter.pdf');fig.savefig(out/'correlation_scatter.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    import multiprocessing;multiprocessing.set_start_method('spawn');main()
