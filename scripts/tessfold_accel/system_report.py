"""Complete new paired timing, trace, exact token and memory audit."""
import collections,json
import numpy as np
from scripts.tessfold_accel.system_common import *
from scripts.tessfold_repair.report import speed_rows

def main():
    verify();complete=load(BASE/'single/complete.json');assert complete.get('records')==9450
    rows=[r for f in (BASE/'single').glob('results.rank*.jsonl') for r in lines(f)]
    key=lambda r:(r['id'],r['method'],r['ratio'],r['mode'],r['repeat'],r['instrumentation'])
    index={key(r):r for r in rows};assert len(rows)==len(index)==9450
    reference={key(r):r for f in (ROOT/'artifacts/tessfold_system_repair_20261115/single').glob('results.rank*.jsonl') for r in lines(f)}
    for r in rows:
        if r['method'] in ['T_FAST','C_FAST']:
            k=list(key(r));k[1]=r['method'].replace('_FAST','_DENSE')
            assert r['generated_ids']==index[tuple(k)]['generated_ids'],'Compiled path changed current native generation'
            k[1]=r['method'].replace('_FAST','_STRICT');assert r['generated_ids']==reference[tuple(k)]['generated_ids'],'Compiled path changed historical strict generation'
        if r['method']=='REF':assert r['generated_ids']==reference[key(r)]['generated_ids']
        if r['instrumentation']=='memory1ms':assert sha(ROOT/r['trace'])==r['trace_sha256']
    stats=speed_rows(rows);save(BASE/'single/summary.json',dict(rows=stats,exact_tokens_verified=True,traces_verified=True))
    doc=['# TessFold 编译切分与精确推理：完整效率复测','',f'UTC：{now()}。','',
        '**入口**：`.venv/bin/python -m scripts.tessfold_accel.system_report`；运行：`sbatch scripts/tessfold_accel/system.sh`。','',
        '保持UND、生成器checkpoint、保留率及90道计时问题不变。新FAST路径缓存官方block几何、用C++编译原FP32切分、以可精确表示的FP32代表索引排序避开整数argsort的CPU回退；软聚合仍为原FP32 masked softmax与GEMM。不是O(N)软聚合，不宣称其亚毫秒。','',
        '9450条唯一记录，8100时延+1350独立1ms内存。自然终止与固定64各三次，NPU event和同步wall。所有FAST输出同时对当前DENSE和历史STRICT逐token核对，REF也对历史核对；全部通过后才生成本报告。','',
        '|方法|保留率|模式|TTFT p50 ms|post-ViT TTFT p50 ms|E2E p50 ms|压缩+merger p50 ms|全流程峰值 GiB|视觉KV MiB|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        kv='未独立取得' if r['visual_kv_mib'] is None else f'{r["visual_kv_mib"]:.2f}'
        doc.append(f'|{r["method"]}|{r["ratio"]:.0%}|{r["mode"]}|{r["ttft_p50_ms"]:.2f}|{r["post_encoder_ttft_p50_ms"]:.2f}|{r["e2e_p50_ms"]:.2f}|{r["phase_wall_medians_ms"]["compression_and_merger"]:.2f}|{r["peak_allocated_max_gib"]:.4f}|{kv}|')
    comparisons=[];rng=np.random.default_rng(20261115)
    doc+=['','## 同题配对加速比','',
        '以下使用每题三次平均再计算整体平均时延比，以页面为bootstrap单位10000次。不把中位数比与均值比混用。','',
        '|参考|候选|保留率|模式|E2E平均加速比|95%页面CI|TTFT平均加速比|','|---|---|---:|---|---:|---|---:|']
    for ratio in [.25,.5]:
     for method in ['C_FAST','T_FAST']:
      for ref in ['REF',method.replace('_FAST','_DENSE')]:
       for mode in ['natural_eos','fixed64']:
        pair=[]
        for r in lines(BASE/'timing_rows.jsonl'):
            def average(method,field):
                rr=[index[(r['id'],method,1. if method=='REF' else ratio,mode,k,'synchronized_wall_and_NPU_events')][field] for k in range(3)]
                return float(np.mean(rr))
            pair.append((r['document_id'],average(ref,'end_to_end_ms'),average(method,'end_to_end_ms'),average(ref,'ttft_ms'),average(method,'ttft_ms')))
        docs=collections.defaultdict(lambda:np.zeros(4))
        for page,*values in pair:docs[page]+=values
        a=np.asarray([docs[k] for k in sorted(docs)]);boot=a[rng.integers(len(a),size=(10000,len(a)))].sum(1)
        lo,hi=np.quantile(boot[:,0]/boot[:,1],[.025,.975]);total=a.sum(0)
        result=dict(reference=ref,method=method,ratio=ratio,mode=mode,e2e_speedup=float(total[0]/total[1]),ttft_speedup=float(total[2]/total[3]),e2e_ci=[float(lo),float(hi)])
        comparisons.append(result);doc.append(f'|{ref}|{method}|{ratio:.0%}|{mode}|{result["e2e_speedup"]:.3f}×|[{lo:.3f},{hi:.3f}]|{result["ttft_speedup"]:.3f}×|')
    save(BASE/'single/paired_comparisons.json',comparisons)
    timing=sorted(lines(BASE/'timing_rows.jsonl'),key=lambda r:(r['original_tokens'],r['id']));strata={r['id']:['short','medium','long'][i//30] for i,r in enumerate(timing)}
    save(BASE/'single/stratified.json',[dict(stratum=s,**r) for s in ['short','medium','long'] for r in speed_rows([r for r in rows if strata[r['id']]==s])])
    memory=[]
    for (method,ratio),rr in __import__('itertools').groupby(sorted([r for r in rows if r['instrumentation']=='memory1ms'],key=lambda r:(r['method'],r['ratio'])),key=lambda r:(r['method'],r['ratio'])):
        rr=list(rr);memory.append(dict(method=method,ratio=ratio,requests=len(rr),peak_max_gib=max(r['peak_allocated_bytes'] for r in rr)/2**30,
            decoder_peak_max_gib=max(r['stages'][s]['peak_allocated_bytes'] for r in rr for s in r['stages'] if s in ['qwen_prefill','generation','decoder_total'])/2**30))
    save(BASE/'single/memory_summary.json',memory)
    doc+=['','post-ViT字段在FAST/REF中边界一致；旧DENSE/S的tower阶段包含原全长merger，应解释为post-tower而非统一纯ViT边界。端到端始终计入全部计算。图重放部署另行测量，主表FAST不含提前编译捕获的隐藏摊销。','',
        'Hard是按原score挑代表的独立变体；不继承soft准确率。Resize与S²Prune使用原对照流程。有限固定样本、同步钩子与共享Slurm节点会限制时延推广，不把本表当生产并发吞吐。']
    (ROOT/'docs/TESSFOLD_COMPILED_SYSTEM_RESULTS.md').write_text('\n'.join(doc)+'\n')
    tex=['% Measured only; FAST exact FP32 dense readout, compiled partition.',r'\begin{tabular}{lrrrr}',r'Method & Keep & TTFT(ms) & E2E(ms) & Compress+Merger(ms) \\']
    for r in stats:
        if r['mode']=='natural_eos' and r['method'] in ['REF','C_FAST','T_FAST']:
            tex.append(f'{r["method"].replace("_",chr(92)+"_")} & {r["ratio"]:.0%} & {r["ttft_p50_ms"]:.2f} & {r["e2e_p50_ms"]:.2f} & {r["phase_wall_medians_ms"]["compression_and_merger"]:.2f} '+r'\\')
    tex.append(r'\end{tabular}');(BASE/'single/measured_table.tex').write_text('\n'.join(tex).replace('% &',r'\% &')+'\n')
    save(BASE/'complete.json',dict(time=now(),records=9450,exact_generation_passed=True,traces_verified=True));print(json.dumps(dict(complete=True,records=9450)),flush=True)

if __name__=='__main__':main()
