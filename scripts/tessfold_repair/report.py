"""Evidence-only report and paper-ready tables; no invented performance targets."""
import argparse,collections,csv,json,statistics
import numpy as np
from scripts.tessfold_repair.common import *
REPORT=ROOT/'docs/TESSFOLD_SYSTEM_REPAIR_20261115.md'

def speed_rows(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  if r['instrumentation']=='synchronized_wall_and_NPU_events':groups[(r['method'],r['ratio'],r['mode'])].append(r)
 result=[]
 for (method,ratio,mode),rr in sorted(groups.items()):
  phases=sorted(set(p for r in rr for p in r['stages']))
  row=dict(method=method,ratio=ratio,mode=mode,requests=len(rr),ttft_p50_ms=float(np.median([r['ttft_ms'] for r in rr])),ttft_p95_ms=float(np.quantile([r['ttft_ms'] for r in rr],.95)),
    post_encoder_ttft_p50_ms=float(np.median([r['post_encoder_ttft_ms'] for r in rr])),e2e_p50_ms=float(np.median([r['end_to_end_ms'] for r in rr])),
    peak_allocated_max_gib=max(r['peak_allocated_bytes'] for r in rr)/1024**3,
    visual_kv_mib=(float(np.median([r['decoder_state']['visual_kv_bytes'] for r in rr]))/1024**2 if all('visual_kv_bytes' in r['decoder_state'] for r in rr) else None))
  row['phase_wall_medians_ms']={p:float(np.median([r['stages'][p]['wall_ms'] for r in rr if p in r['stages']])) for p in phases}
  row['phase_event_medians_ms']={p:float(np.median([r['stages'][p]['device_event_ms'] for r in rr if p in r['stages']])) for p in phases}
  result.append(row)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();reg=verify();state=load(BASE/'state.json');cpu=load(BASE/'cpu_equivalence.json');val=load(BASE/'validation/complete.json')
 single_done=(BASE/'single/complete.json').exists();multi_done=(BASE/'multi/complete.json').exists()
 single=[r for p in (BASE/'single').glob('results.rank*.jsonl') for r in lines(p)] if single_done else []
 multi=[r for p in (BASE/'multi').glob('results.rank*.jsonl') for r in lines(p)] if multi_done else []
 doc=['# TessFold零重训系统重构、精度核验与效率实测','',f'更新UTC：{now()}。真实验证完成：{bool(val)}；单请求测量完成：{single_done}；多问题测量完成：{multi_done}。','',
 '**入口**：`.venv/bin/python -m scripts.tessfold_repair.suite status|resume|verify`；最终`.venv/bin/python -m scripts.tessfold_repair.report --require-complete`。','',
 '## 本次接口修复','',
 '本次使用独立版本 scripts/tessfold_repair，保留20261114失败实验及官方源码。仅在HF已准备好embeddings首轮输入后恢复真实input_ids；不使用伪造ID、不改变positions/mask/KV或权重。CPU小Qwen已复现原错误，并验证修复后的embeddings生成、原生生成、两题独立prefix及异常恢复。另修复缩图补丁被重复安装的问题。','',
 '## 实际工作范围','',
 '依据tessfold.md执行零重训重构；八个已训练seed20261105/epoch4的列/树checkpoint固定，模型权重全部冻结。历史16组、三seed及原效率数据保留，不覆写签名源码或删除负结果。仅优化推理工作与评测流程，不重训、不更换Qwen、不预设投稿或性能结果。','',
 '## 对参考方案的必要修正','',
 '1. 实数数学等价不等于浮点逐位相等。参考方案把FP32 softmax/GEMM改为BF16乘法累加，改变舍入；FP32 scatter也改变求和顺序。默认strict_dense保留原FP32 partition/softmax/GEMM，只删无用工作。实验性segmented_fp32/BF16分支单独记录，不能偷偷替换后继承旧分数。','',
 '2. 原代码已经有FP32二维积分图和最大堆，主要切点仍用局部NumPy sum/cumsum。提议换FP64积分图加二分搜索改变舍入、质量平局与切点；本轮候选独立比较原实现，不能让reference与candidate共用新分区器。没有证明或承诺JIT必然<0.8ms。','',
 '3. Hard读取是独立方法变体，不与原soft模型逐位等价；不以soft−hard区间跨零声称二者严格等效。现有Hard25%分数38.24只适用于其历史设置，不为新预算自动赋相同成绩。','',
 '4. Post-ViT方法无法节省已经发生的encoder峰值；端到端最高峰保留报告。新部署路径不先运行无用全长merger，压缩后释放raw/P/输入图像，再显式构造相同视觉embeddings传给原generate。独立验证这些生命周期变化没有改变答案。','',
 '5. 25%是视觉KV逻辑子段按实际K/N的比例，不是全KV/整卡显存/吞吐严格四倍。每页K按round取整，文本/答案KV不缩短，常驻权重不变。实际并发上限未做容量压测，不能以缓存比例替代实测吞吐。','',
 '6. 同页前缀缓存必须严格复用原prompt中视觉结束之前的固定因果前缀；不可把上一题或答案KV混入。原Qwen是decoder self-attention，不是独立视觉cross-attention。REF也获得相同的feature/KV复用权限；每题克隆只读cache并计时。前缀拆开执行的浮点内核可能不同，必须单独回归生成。','',
 '7. 本主实验保持官方UND/Qwen3-1.7B，以便与历史答案独立回归。用户20261115另授权考察更新的约7B Qwen；可选扩展的架构适配与环境检查单独报告，不混用主实验成绩。项目没有TeX原稿，提供独立实测表。','',
 '8. 旧固定64基线2532.14ms、T2502.28ms，差约29.86ms；参考文件所谓“仅省4.5ms”“64步250–300ms”与实际单位不符。不能据这些数字精准定位耗时来源。缩图固定64结果和T−M/T−A的CI此前已经存在，应该引用精确原记录，不重复捏造补测。','',
 '## 已完成CPU独立测试','',
 '另行近均匀能量审计已找到FP64分区反例（见fp64_partition_audit.json），所以下列350个普通案例的零差异不代表严格等价。','',f'切分候选案例{cpu.get("partition_cases",0)}，其中FP64候选与原FP32不一致{cpu.get("integral_candidate_mismatches",0)}例；有限案例无反例不构成普遍逐位等价证明。','',
 '|方法|保留率|严格路径逐位一致|FP32分段逐位一致|BF16分段逐位一致|BF16最大绝对差|','|---|---:|---|---|---|---:|']
 for r in cpu.get('feature_tests',[]):doc.append(f'|{r["kind"]}|{r["ratio"]:.0%}|{r["strict_bit_exact"]}|{r["segmented_fp32_equal"]}|{r["segmented_bf16_equal"]}|{r["segmented_bf16_max_error"]:.8f}|')
 doc+=['','测试不是参考md中仅print误差：strict对原生成器P和输出执行torch.equal断言。原group顺序、官方2×2block与逆置换、代表argmax最小序号、稳定排序、round-to-even长度和原energy乘法结合次序均保留。','',
 '## 真实UND独立验证门槛','',
 '固定56题验证×两方法×四预算共448例。逐例原生成器对新strict的P/PH/merger输出逐位相等，并与历史验证student_ids对比；另完整重新执行无多余merger、释放中间张量的clean生成路径，比较生成token。分段候选和FP64切分只作诊断，失败不修改旧权重或参数。每rank另取真实同页两题测试REF/Column/Tree独立prefixcache与原全prompt输出，检查基础cache不被题目或答案污染。','']
 if val:doc += [f'实际完成：{val["cases"]}例，strict通过：{val["strict_passed"]}。分段候选不一致：`{val["candidate_mismatches"]}`；FP64切分不一致{val["integral_partition_mismatches"]}；prefix缓存token不一致{val["prefix_reuse_token_mismatches"]}。','']
 else:doc+=['真实验证尚未完成，不提前声明已通过。','']
 doc+=['## 单请求效率协议','',
 '相同90题/三层各30，主25%次50%，15个方法预算配置（REF及各预算T/C原dense、新strict、TreeHard、R、S），自然EOS/cap256及固定64各三轮；共8100条时延记录。另每配置每题一次独立1ms内存采样，共1350条，不把采样线程引入的耗时混入主时延表。每模型两题cap16预热，JIT若存在须预热不计首次编译。','',
 '实际硬件Ascend，使用torch.npu.Event与同步wall双计时；CPU分区/传输成本看wall，事件包含stream空闲等待，不能把device event当纯kernel或遗漏CPU。记录processor、encoder、压缩+merger、prefill、decode、TTFT、postencoder TTFT、端到端总延迟、KV实际逻辑字节和阶段allocated/reserved。含同步/钩子测量对所有方法一致，不能称完全无扰动服务吞吐。S按原内部剪枝流程给decoder总段，不伪造独立prefill。','']
 summaries=[];regression=[]
 if single_done:
  assert len(single)==9450
  bykey={(r['method'],r['ratio'],r['mode'],r['id'],r['repeat'],r['instrumentation']):r for r in single}
  for r in single:
   if r['method'] not in ['T_STRICT','C_STRICT']:continue
   old=bykey[(r['method'].replace('_STRICT','_DENSE'),r['ratio'],r['mode'],r['id'],r['repeat'],r['instrumentation'])]
   assert r['generated_ids']==old['generated_ids'],'Strict optimization changed benchmark generation'
  summaries=speed_rows(single)
  save(BASE/'single/summary.json',dict(rows=summaries,strict_generation_regression_passed=True))
  with (BASE/'single/latency_summary.csv').open('w') as f:
   fields=[k for k in summaries[0] if not k.startswith('phase_')];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summaries)
  doc+=['|方法|保留率|模式|TTFT p50 ms|Post-encoder TTFT p50 ms|E2E p50 ms|压缩+merger wall p50 ms|峰值GiB|视觉KV MiB|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
  for r in summaries:
   kv='未独立测得' if r['visual_kv_mib'] is None else f'{r["visual_kv_mib"]:.2f}'
   doc.append(f'|{r["method"]}|{r["ratio"]:.0%}|{r["mode"]}|{r["ttft_p50_ms"]:.2f}|{r["post_encoder_ttft_p50_ms"]:.2f}|{r["e2e_p50_ms"]:.2f}|{r["phase_wall_medians_ms"]["compression_and_merger"]:.2f}|{r["peak_allocated_max_gib"]:.4f}|{kv}|')
  # Exactstrata are fixed from originaltiming sample lengths, not chosen by speed.
  timing=lines(BASE/'timing_rows.jsonl');ordered=sorted(timing,key=lambda r:(r['original_tokens'],r['id']));strata={r['id']:['short','medium','long'][i//30] for i,r in enumerate(ordered)}
  bins=[]
  for label in ['short','medium','long']:
   for row in speed_rows([r for r in single if strata[r['id']]==label]):bins.append(dict(stratum=label,**row))
  save(BASE/'single/stratified.json',bins)
  memory=[r for r in single if r['instrumentation']=='memory1ms']
  for r in memory:assert sha(ROOT/r['trace'])==r['trace_sha256']
  save(BASE/'single/audit.json',dict(passed=True,records=len(single),strict_tokens_checked=True,trace_sha_checked=True))
  # Paper tablecontains measured rows only, no asserted submillisecondcaption.
  tex=['% Automatically generated from completed measured results. Historical tables unchanged.',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Keep & TTFT (ms) & Post-encoder TTFT (ms) & E2E (ms) \\',r'\midrule']
  for r in summaries:
   if r['mode']=='natural_eos':tex.append(f"{r['method'].replace('_', chr(92)+'_')} & {100*r['ratio']:.0f}\\% & {r['ttft_p50_ms']:.2f} & {r['post_encoder_ttft_p50_ms']:.2f} & {r['e2e_p50_ms']:.2f} \\\\")
  tex += [r'\bottomrule',r'\end{tabular}'];(BASE/'single/measured_table.tex').write_text('\n'.join(tex)+'\n')
 else:doc+=['单请求正式测量尚未完成，不填预期0.XX毫秒或预设加速倍数。']
 doc+=['','## 同页多问题公平缓存协议','',
 '固定24张至少有10个不同问题的真实页，K=1/2/5/10取同一10题序列前K题；不重复问题凑K。只有DocVQA/InfoVQA/WTQ/OCRBench有足够10问页，此子集存在任务偏移，不称八任务完整泛化。对REF、Column、Tree均给cold、feature_cache、prefix_cache三场景；若prefix试验token不一致，则整组prefix场景弃用并报告，仍测feature_cache。','',
 'cold每题独立图像encoder；featurecache整页缓存H，REF/Column还缓存对齐特征，Tree逐题重算问题相关压缩；prefixcache REF/Column缓存视觉因果前缀，Tree每题重新构建问题相关视觉前缀。前缀克隆和各题processor计时，首次摄入按(摄入+前K题时间)/K摊销；不把跨题答案KV当可复用视觉前缀。所有方法共享同等前端缓存权利。','']
 if multi_done:
  grouped=collections.defaultdict(list)
  for r in multi:
   for k,v in r['amortization'].items():grouped[(r['method'],r['scenario'],int(k))].append(v)
  stats=[dict(method=m,scenario=s,k=k,measurements=len(v),mean_ms_per_query=statistics.mean(v),median_ms_per_query=statistics.median(v),queries_per_second=1000/statistics.mean(v)) for (m,s,k),v in sorted(grouped.items())]
  # Regression uses samepage/method/repeat, compares alternatecache scenarios.
  reference={(r['page'],r['method'],r['repeat']):r for r in multi if r['scenario']=='cold'};mismatch=0
  for r in multi:
   orig=reference[(r['page'],r['method'],r['repeat'])]
   mismatch+=sum(a['generated_ids']!=b['generated_ids'] for a,b in zip(orig['queries'],r['queries']))
  save(BASE/'multi/summary.json',dict(rows=stats,cache_generation_mismatches=mismatch,exact_on_measured_cohort=mismatch==0))
  doc+=['|方法|缓存场景|K|平均ms/题（含首次摄入）|中位ms/题|实测queries/s|','|---|---|---:|---:|---:|---:|']
  for r in stats:doc.append(f'|{r["method"]}|{r["scenario"]}|{r["k"]}|{r["mean_ms_per_query"]:.2f}|{r["median_ms_per_query"]:.2f}|{r["queries_per_second"]:.2f}|')
  doc += ['',f'对cold逐题token回归不一致次数：{mismatch}。非零时不能声称缓存Bit-Exact。','']
  try:
   import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
   fig,ax=plt.subplots(figsize=(6,3.5))
   for m in ['REF','Column','Tree']:
    rr=[r for r in stats if r['method']==m and r['scenario']=='feature_cache'];ax.plot([r['k'] for r in rr],[r['mean_ms_per_query'] for r in rr],'o-',label=m)
   ax.set_xlabel('Questions per real page');ax.set_ylabel('Amortized ms/query incl ingestion');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
   for ext in ['pdf','png']:fig.savefig(BASE/'multi'/f'amortization.{ext}',dpi=170)
   plt.close(fig)
  except ImportError:pass
 else:doc+=['多问题正式测量尚未完成，不能宣称近4倍吞吐。']
 doc+=['','## FLOPs与论文表述边界','',
 'FLOPs只能称解析估计：原merger线性部分约2N(768²+768×2048)，compressed约以实际M/N缩放。Qwen投影/MLP按总序列L=M+文本前缀缩放；注意力因果项近L²，但含文本、层、mask、内核和生成，不是全部计算严格r²。S首层原N，后27层K。当前没有kernel FLOP硬件counter，不把解析值写成实测。','',
 '所有历史负结果保留，新的优化/缓存只在对应部署场景说明。未达到sub-ms或cache优势时如实写明；不删除“旧实现稠密P”或6.414GiB历史事实，而给出旧实现→新实现的版本化对照。完整源清单/固定数据/权重SHA见登记，待完成时不把计划当结果。','',
 f'当前阶段：{state.get("stages",{})}。','[登记](../artifacts/tessfold_system_repair_20261115/registration.json) · [CPU测试](../artifacts/tessfold_system_repair_20261115/cpu_equivalence.json) · [真实验证](../artifacts/tessfold_system_repair_20261115/validation/) · [旧精确统计](COMPRESSION_SUPPLEMENT_STATISTICS.md)。']
 REPORT.write_text('\n'.join(doc)+'\n')
 save(BASE/'summary.json',dict(time=now(),validation=bool(val),single=single_done,multi=multi_done))
 if a.require_complete:assert val and single_done and multi_done
 print(json.dumps(dict(report=str(REPORT),validation=bool(val),single=single_done,multi=multi_done)))
if __name__=='__main__':main()
