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
    doc=['# TessFold Compiled Partitioning and Exact Inference: Full Efficiency Remeasurement','',f'UTC: {now()}. ','',
        '**Entry point**: `.venv/bin/python -m scripts.tessfold_accel.system_report`; run: `sbatch scripts/tessfold_accel/system.sh`. ','',
        'UND, generator checkpoints, retention fractions, and 90 timing questions remain fixed. FAST caches official block geometry, compiles the original FP32 partitioner in C++, and sorts exactly representable FP32 representative indices to avoid integer-argsort CPU fallback. Soft aggregation retains the original FP32 masked softmax and GEMM. This is not O(N) soft aggregation, and submillisecond latency is not claimed.','',
        '9,450 unique records: 8,100 timing + 1,350 independent 1 ms memory measurements. Natural/fixed64 each have three repeats, NPU-event and synchronized wall timing. FAST is checked tokenwise against current DENSE and historical STRICT; REF is checked against history. Generate this report only after all checks pass.','',
        '|Method|Retention fraction|Mode|TTFT p50 ms|post-ViT TTFT p50 ms|E2E p50 ms|Compression+merger p50 (ms)|Full-pipeline peak (GiB)|Visual KV (MiB)|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        kv='Not measured separately' if r['visual_kv_mib'] is None else f'{r["visual_kv_mib"]:.2f}'
        doc.append(f'|{r["method"]}|{r["ratio"]:.0%}|{r["mode"]}|{r["ttft_p50_ms"]:.2f}|{r["post_encoder_ttft_p50_ms"]:.2f}|{r["e2e_p50_ms"]:.2f}|{r["phase_wall_medians_ms"]["compression_and_merger"]:.2f}|{r["peak_allocated_max_gib"]:.4f}|{kv}|')
    comparisons=[];rng=np.random.default_rng(20261115)
    doc+=['','## Paired Speedups on Identical Questions','',
        'Average each question’s three repeats, then compare aggregate mean latency; bootstrap pages 10,000 times. Ratios of medians and ratios of means are not interchangeable.','',
        '|Reference|Candidate|Retention fraction|Mode|Mean E2E speedup|95% page-level CI|Mean TTFT speedup|','|---|---|---:|---|---:|---|---:|']
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
    doc+=['','FAST/REF share a post-ViT boundary. Old DENSE/S tower timing includes full-length merger and should be called post-tower, not the same pure-ViT boundary. E2E includes all work. Graph-replay deployment is measured separately; primary FAST tables do not hide capture/compilation amortization.','',
        'Hard selects representatives using original scores and does not inherit soft accuracy. Resize/S²Prune retain original control procedures. Fixed limited samples, synchronization hooks, and shared Slurm nodes limit timing generalization; these are not production-concurrency measurements.']
    (ROOT/'docs/TESSFOLD_COMPILED_SYSTEM_RESULTS.md').write_text('\n'.join(doc)+'\n')
    tex=['% Measured only; FAST exact FP32 dense readout, compiled partition.',r'\begin{tabular}{lrrrr}',r'Method & Keep & TTFT(ms) & E2E(ms) & Compress+Merger(ms) \\']
    for r in stats:
        if r['mode']=='natural_eos' and r['method'] in ['REF','C_FAST','T_FAST']:
            tex.append(f'{r["method"].replace("_",chr(92)+"_")} & {r["ratio"]:.0%} & {r["ttft_p50_ms"]:.2f} & {r["e2e_p50_ms"]:.2f} & {r["phase_wall_medians_ms"]["compression_and_merger"]:.2f} '+r'\\')
    tex.append(r'\end{tabular}');(BASE/'single/measured_table.tex').write_text('\n'.join(tex).replace('% &',r'\% &')+'\n')
    save(BASE/'complete.json',dict(time=now(),records=9450,exact_generation_passed=True,traces_verified=True));print(json.dumps(dict(complete=True,records=9450)),flush=True)

if __name__=='__main__':main()
