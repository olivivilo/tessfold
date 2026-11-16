"""Audited24model report, paired3seed statistics, stage memory and raw CSV."""
from __future__ import annotations
import argparse,collections,csv,json,math,statistics
import numpy as np
from scripts.seed_memory_common import *
from vqa.benchmark_metrics import native_score
REPORT=ROOT/'docs/COLUMN_TREE_THREE_SEED_AND_MEMORY.md'

def collect_model(m,expected,teachers):
 out=ROOT/m['output'];train=lines(out/'training.jsonl')
 assert [r['step'] for r in train]==list(range(1,len(train)+1))
 for r in train:assert all(math.isfinite(r[k]) for k in ['kl','grad_norm','update_l2']) and r['update_l2']>0
 progress=dict(steps=len(train),training_complete=(out/'complete.json').exists(),epochs=[])
 for e in range(5):
  stage='initial' if e==0 else f'epoch{e}';vv=load(out/f'validation.{stage}.summary.json');aa=load(out/f'train_audit.{stage}.summary.json');tr=[r['kl'] for r in train if r['epoch']==e]
  progress['epochs'].append(dict(epoch=e,updates=len(tr),training_kl=statistics.mean(tr) if tr else None,validation_kl=vv.get('kl'),validation_score=vv.get('student_macro'),audit_kl=aa.get('kl')))
 if progress['training_complete']:
  done=load(out/'complete.json');assert done['steps']==2500 and done['epochs']==4 and done['trainable_backbone']==0
  assert len(train)==2500 and all(x['updates']==625 for x in progress['epochs'][1:])
  assert all(x['validation_kl'] is not None and x['audit_kl'] is not None for x in progress['epochs'])
 if not (out/'benchmark/complete.json').exists():return progress,None
 assert progress['training_complete']
 files=sorted((out/'benchmark').glob('predictions.rank*.jsonl'))
 signature=dict(checkpoint_sha256=sha(ROOT/m['checkpoint']),summary_sha256=sha(out/'benchmark/summary.json'),
  predictions={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in files})
 cached_path=BASE/'audits'/f'{m["id"]}.json';cached=load(cached_path)
 if cached.get('signature')==signature and cached.get('passed'):return progress,cached['summary']
 rr=model_records(m);assert len(rr)==len(expected)==23565
 seen=set()
 for r in rr:
  assert r['id'] not in seen;seen.add(r['id']);s=expected[r['id']]
  assert all(r[k]==s[k] for k in ['task','question','document_id','answers'])
  assert r['compressed_tokens']==max(1,round(m['ratio']*r['tokens']))
  assert r['row_sum_error']<1e-4 and len(r['student_ids'])<=256
  assert abs(native_score(r['student_prediction'],s)-r['student_score'])<1e-12
  assert (r['teacher_prediction'],r['teacher_score'])==teachers[r['id']]
 summary=summarize(rr);recorded=load(out/'benchmark/summary.json')
 orig=recorded['variants']['trained']['macro'] if 'variants' in recorded else recorded['macro']['trained'] if isinstance(recorded['macro'],dict) else recorded['macro']
 assert abs(summary['macro']-orig)<1e-8
 save(cached_path,dict(passed=True,audited_utc=now(),signature=signature,summary=summary,
  unique_coverage=True,scores_gold_teacher_checked=True,checkpoint_fixedepoch4=True,backbone_frozen=True))
 return progress,summary

def plot_training(m,progress):
 signature=progress['epochs'];stamp=BASE/'plots'/f'{m["id"]}.json'
 if load(stamp)==signature:return
 if not any(x['validation_kl'] is not None for x in signature):return
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,3.5))
 for key,label in [('training_kl','full train KL'),('validation_kl','validation KL'),('audit_kl','historical audit KL')]:
  pp=[x for x in signature if x[key] is not None];ax[0].plot([x['epoch'] for x in pp],[x[key] for x in pp],'o-',label=label)
 pp=[x for x in signature if x['validation_score'] is not None];ax[1].plot([x['epoch'] for x in pp],[x['validation_score'] for x in pp],'o-',label='validation generation')
 for a in ax:a.set_xlabel('epoch');a.grid(alpha=.2);a.legend(fontsize=7)
 ax[0].set_ylabel('KL');ax[1].set_ylabel('score %');fig.suptitle(m['id']);fig.tight_layout()
 stamp.parent.mkdir(exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(stamp.with_suffix('.'+ext),dpi=130)
 plt.close(fig);save(stamp,signature)

def final_statistics(reg,results):
 cached=load(BASE/'score_statistics.json')
 signature={m['id']:sha(ROOT/m['checkpoint']) for m in reg['models']}
 if cached.get('signature')==signature:return cached
 from scripts.analyze_compression_supplement import paired_maps,average_seed_maps
 maps={m['id']:{r['id']:r for r in model_records(m)} for m in reg['models']}
 pairs={};seeds={}
 for ratio in RATIOS:
  values=[]
  for seed in SEEDS:
   a=maps[model_id('quadtree_question',ratio,seed)];b=maps[model_id('col_adaptive',ratio,seed)]
   pairs[f'T-C_r{ratio:g}_seed{seed}']=paired_maps(a,b,10000);values.append((a,b))
  aa,bb=average_seed_maps(values);pairs[f'T-C_r{ratio:g}_three_seed_mean']=paired_maps(aa,bb,10000)
  tt=[results[model_id('quadtree_question',ratio,z)]['macro'] for z in SEEDS];cc=[results[model_id('col_adaptive',ratio,z)]['macro'] for z in SEEDS]
  seeds[str(ratio)]=dict(T_mean=statistics.mean(tt),T_sample_sd=statistics.stdev(tt),C_mean=statistics.mean(cc),C_sample_sd=statistics.stdev(cc),
    paired_seed_gains=[a-b for a,b in zip(tt,cc)],gain_mean=statistics.mean(a-b for a,b in zip(tt,cc)),gain_sample_sd=statistics.stdev(a-b for a,b in zip(tt,cc)))
 result=dict(signature=signature,seeds=seeds,paired=pairs,notes='Image bootstrap conditional on these3trained seeds; seed sample SD distinct,3seeds insufficient for strong population claims; viewed benchmark exploratory.')
 save(BASE/'score_statistics.json',result);return result

def memory_statistics(reg):
 out=BASE/'memory'
 if not (out/'complete.json').exists():return None
 cached=load(out/'statistics.json')
 manifests=sorted(out.glob('manifest.rank*.json'));parts=sorted(out.glob('memory.rank*.jsonl'))
 signature=dict(manifests={p.name:sha(p) for p in manifests},parts={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in parts})
 if cached.get('signature')==signature and cached.get('audited'):return cached
 rr=[r for p in parts for r in lines(p)]
 models=reg['models']+[dict(id='REF',kind='REF',ratio=1.,seed=None)]
 timing={r['id']:r for r in lines(BASE/'timing_rows.jsonl')}
 expected={(m['id'],key,mode,k) for m in models for key in timing for mode in ['natural_eos','fixed64'] for k in range(3)}
 assert len(rr)==13500 and {(r['model'],r['id'],r['mode'],r['repeat']) for r in rr}==expected
 predictions={m['id']:{r['id']:r for r in model_records(m) if r['id'] in timing} for m in reg['models']}
 teacher={}
 for r in next(iter(predictions.values())).values():teacher[r['id']]=r['teacher_prediction']
 groups=collections.defaultdict(list);flat=[]
 for r in rr:
  assert r['tokens']==timing[r['id']]['original_tokens'] and r['compressed_tokens']==max(1,round(r['ratio']*r['tokens']))
  assert sha(ROOT/r['trace'])==r['trace_sha256']
  if r['mode']=='natural_eos':
   if r['kind']=='REF':assert r['prediction']==teacher[r['id']]
   else:assert r['generated_ids']==predictions[r['model']][r['id']]['student_ids'],'Phase instrumentation must preserve native generation'
  else:assert len(r['generated_ids'])==64
  assert r['decoder_state']['prefill_visual_tokens']==r['compressed_tokens']
  for phase,v in r['phases'].items():
   assert v['peak_allocated_bytes']>=v['time_average_allocated_bytes']>=0
   assert v['seconds']>0 and v['samples']>=2
   key=(r['kind'],r['ratio'],r['seed'],r['mode'],phase);groups[key].append(v)
   flat.append({k:r[k] for k in ['model','kind','ratio','seed','mode','id','repeat','tokens','compressed_tokens']}|dict(phase=phase,**v))
 summaries=[]
 for (kind,ratio,seed,mode,phase),vv in sorted(groups.items(),key=lambda x:str(x[0])):
  summaries.append(dict(kind=kind,ratio=ratio,seed=seed,mode=mode,phase=phase,requests=len(vv),
   peak_allocated_max_gib=max(v['peak_allocated_bytes'] for v in vv)/1024**3,
   peak_allocated_mean_gib=statistics.mean(v['peak_allocated_bytes'] for v in vv)/1024**3,
   average_allocated_mean_gib=statistics.mean(v['time_average_allocated_bytes'] for v in vv)/1024**3,
   pooled_time_average_allocated_gib=sum(v['time_average_allocated_bytes']*v['seconds'] for v in vv)/sum(v['seconds'] for v in vv)/1024**3,
   start_allocated_mean_gib=statistics.mean(v['start_allocated_bytes'] for v in vv)/1024**3,
   incremental_peak_mean_gib=statistics.mean(v['incremental_peak_allocated_bytes'] for v in vv)/1024**3,
   peak_reserved_max_gib=max(v['peak_reserved_bytes'] for v in vv)/1024**3,
   average_reserved_mean_gib=statistics.mean(v['time_average_reserved_bytes'] for v in vv)/1024**3,
   max_sample_gap_seconds=max(v['max_sample_gap_seconds'] for v in vv)))
 with (out/'stage_memory_per_request.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 with (out/'stage_memory_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 result=dict(signature=signature,audited=True,records=len(rr),summaries=summaries,native_generation_regression_passed=True,traces_sha256_verified=True)
 save(out/'statistics.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();reg=verify();state=load(BASE/'state.json')
 expected={r['id']:r for r in lines(BASE/'formal_rows.jsonl')};teacherpath=BASE/'audits/teacher_reference.json';teachers=load(teacherpath)
 if not teachers:
  first=next(m for m in reg['models'] if m['reuse']);rr=model_records(first);teachers={r['id']:[r['teacher_prediction'],r['teacher_score']] for r in rr};save(teacherpath,teachers)
 teachers={k:tuple(v) for k,v in teachers.items()}
 results={};progress={}
 for m in reg['models']:
  pp,res=collect_model(m,expected,teachers);progress[m['id']]=pp
  if res:results[m['id']]=res
  plot_training(m,pp)
 stats=final_statistics(reg,results) if len(results)==24 else None
 memory=memory_statistics(reg)
 fmt=lambda x,n=2:'—' if x is None else f'{x:.{n}f}'
 doc=['# 列划分与问题条件空间树：四预算共同三seed及分阶段设备内存','',f'更新UTC：{now()}；已审计正式模型{len(results)}/24，新完整训练14组，复用已完成10组。','',
 '**入口**：`.venv/bin/python -m scripts.column_tree_seed_memory_suite status`；中断`resume`；固定登记核对`verify`；最终`.venv/bin/python -m scripts.report_seed_memory --require-complete`。','',
 '## 固定设计与参数','',
 '10%/25%/50%/75%均为token保留率（约10×/4×/2×/1.33× token压缩，不是加速比）。统一merger前，官方MonkeyOCRv2-B-Und encoder/merger/Qwen3-1.7B全冻结，只训练P生成器。两种方法均用20261105、20261106、20261107共同种子。','',
 '旧两方法四预算的seed20261105共8组，以及补充实验T25的seed20261106/07共2组复用；其余14组从头训练，不挑seed。原20,000QA/16,731图、验证56QA、历史审计14QA、正式23,565QA/6,494图，固定文件/顺序。AdamW LR3e-4、wd0、clip1、batch32、8NPU每rank累积4题、4epochs/2500更新；250步warmup、cosine到10%峰值；KL-only共同teacher生成前缀T=1，训练/验证cap64、正式greedy cap256、固定epoch4。','',
 '新正式评测仅trained变体23565条/模型，不重复计算旧六种控制；旧initial/control逐题结果保留，不改变比较的trained评分。八任务等权宏平均，DeepForm/KLC为字段exact，非统一官方榜单。正式基准已被查看，本轮是固定重复扩展。','',
 '## 逐seed正式结果','', '|保留率|方法|seed|来源|训练步数|八任务宏分|DocVQA|正式审计|','|---|---|---:|---|---:|---:|---:|---|']
 for m in reg['models']:
  res=results.get(m['id'],{});pp=progress[m['id']]
  doc.append(f'|{m["ratio"]:.0%}|{LABEL[m["kind"]]}|{m["seed"]}|{"复用完整实验" if m["reuse"] else "本轮新训练"}|{pp["steps"]}/2500|{fmt(res.get("macro"))}|{fmt(res.get("tasks",{}).get("DocVQA"))}|{"通过" if res else "待完成"}|')
 doc+=['','## 三seed聚合及配对比较','','均分与样本SD描述训练seed波动；图像聚类bootstrap描述给定这三个已训练模型的评测图像不确定性。每题先在各模型内独立评分，三seed均值仅用于重复训练的统计估计，不是对随机压缩试验跨种子平均后求loss相关性。未校正多重次比较，不能用三seed声称总体方差已精确估计。','']
 if stats:
  doc+=['|保留率|列均分±样本SD|树均分±样本SD|树−列均差|配对seed差SD|三seed均差的95%图像CI|','|---|---:|---:|---:|---:|---|']
  for ratio in RATIOS:
   s=stats['seeds'][str(ratio)];pair=stats['paired'][f'T-C_r{ratio:g}_three_seed_mean'];ci=pair['image_cluster_95ci_points']
   doc.append(f'|{ratio:.0%}|{s["C_mean"]:.4f}±{s["C_sample_sd"]:.4f}|{s["T_mean"]:.4f}±{s["T_sample_sd"]:.4f}|{s["gain_mean"]:+.4f}|{s["gain_sample_sd"]:.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|')
 else:doc+=['尚未收齐24组，未提前用缺失seed填充或发布完整三seed结论。']
 doc+=['','## 分阶段内存测量定义与限制','',
 '独立内存诊断覆盖24个checkpoint及未压缩REF：原固定90QA（按原N三层各30），每组3轮，自然EOS/cap256与固定64两种生成模式；共13,500请求。batch1、权重常驻、不读视觉缓存、不跨题复用KV，一次只驻留一个生成器；10次预热/模型。','',
 '六个阶段顺序：processor；encoder（视觉trunk，原官方merger前结束）；native_merger（原全长merger，单列）；compression（生成P、PH、压缩特征的原官方merger及问题embedding）；Qwen prefill（原generate首次完整prompt forward）；generation（prefill后首token选择及后续解码至结束）。REF的compression近于空操作，仍单列。encoder后原全长X保留，以复现既有流水线，不悄悄加入释放优化。','',
 '每段边界同步并重置allocator峰值。峰值是torch.npu.max_memory_allocated/reserved；平均值是目标1ms CPU观察线程记录memory_allocated/reserved后，按实际时间间隔做梯形加权的采样估计，包括阶段中的CPU间隙，不能冒充精确kernel时间平均。保存原始npz轨迹、样本数/最大间隔，窄脉冲的平均估计有采样误差，但峰值独立来自allocator。','',
 '分别报告阶段起点/终点绝对占用、绝对峰值、相对起点的增量峰值、采样平均；汇总区分跨请求峰值的均值与最大值，以及请求平均值的均值与按阶段时长合并的均值。生成阶段实际KV存储字节单列，token减小不等于常驻权重或encoder峰值减小。设备allocated/reserved不是整卡其他进程或驱动总占用。测量含同步/线程开销，不拿其时延直接宣称加速。','',
 '真实技术检查先回归8种结构/预算的原生生成token，检查prefill实际视觉数与KV、已知64MiB分配平台的采样/峰值、fresh生成器实际UND反向与参数更新；通过后才提交完整训练。完整内存自然EOS还逐题回归对应正式checkpoint答案。','']
 if memory:
  doc+=['内存13,500记录及轨迹SHA已审计，原生生成回归通过。详见[逐请求CSV](../artifacts/column_tree_seed_memory_20261113/memory/stage_memory_per_request.csv)和[各方法/预算/seed/模式/阶段汇总CSV](../artifacts/column_tree_seed_memory_20261113/memory/stage_memory_summary.csv)。','',
   '|保留率|方法|阶段|请求峰值均值GiB|请求平均值均值GiB|请求峰值最大GiB|增量峰值均值GiB|','|---|---|---|---:|---:|---:|---:|']
  for ratio in RATIOS:
   for kind in KINDS:
    for phase in ['encoder','native_merger','compression','qwen_prefill','generation']:
     pp=[x for x in memory['summaries'] if x['kind']==kind and x['ratio']==ratio and x['phase']==phase and x['mode']=='natural_eos'];assert len(pp)==3
     doc.append(f'|{ratio:.0%}|{LABEL[kind]}|{phase}|{statistics.mean(x["peak_allocated_mean_gib"] for x in pp):.4f}|{statistics.mean(x["average_allocated_mean_gib"] for x in pp):.4f}|{max(x["peak_allocated_max_gib"] for x in pp):.4f}|{statistics.mean(x["incremental_peak_mean_gib"] for x in pp):.4f}|')
 else:doc+=['分阶段正式内存尚未完成，不用旧端到端峰值填入阶段表。']
 doc+=['','## 训练/验证loss与原始入口','']
 for m in reg['models']:
  pp=progress[m['id']]
  doc+=[f'### {m["id"]}','',f'[配置](../{m["config"]}) · [权重](../{m["checkpoint"]}) · [曲线](../artifacts/column_tree_seed_memory_20261113/plots/{m["id"]}.pdf) · [逐题结果目录](../{m["output"]}/benchmark/)。','',
   '|epoch|已记录更新|全训练KL|验证KL|验证生成宏分|历史审计KL|','|---|---:|---:|---:|---:|---:|']
  for e in pp['epochs']:doc.append(f'|{e["epoch"]}|{e["updates"]}|{fmt(e["training_kl"],5)}|{fmt(e["validation_kl"],5)}|{fmt(e["validation_score"])}|{fmt(e["audit_kl"],5)}|')
 doc+=['','## 调度与恢复','',
 '独立CPU控制器持有文件锁，最多2个8NPU阶段同时排队/运行；保留原数据/源码/配置签名。每阶段最多3次基础设施恢复，训练沿用25步保存的optimizer/latest；不足完成标志不报告成功。超过恢复次数留attention_required，不更改科学协议。旧实验权重和迁移包不覆盖。','',
 f'当前阶段状态：{dict(collections.Counter(s["status"] for s in state.get("stages",{}).values()))}。',
 '[不可变登记](../artifacts/column_tree_seed_memory_20261113/registration.json) · [实时状态](../artifacts/column_tree_seed_memory_20261113/state.json) · [审计](../artifacts/column_tree_seed_memory_20261113/audits/)。']
 tmp=REPORT.with_suffix('.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(REPORT)
 save(BASE/'merged_results.json',dict(updated=now(),completed_models=len(results),results=results,progress=progress,score_statistics_ready=stats is not None,memory_ready=memory is not None))
 if a.require_complete:assert len(results)==24 and stats is not None and memory is not None
 print(json.dumps(dict(report=str(REPORT),completed_models=len(results),memory=memory is not None)),flush=True)
if __name__=='__main__':main()
