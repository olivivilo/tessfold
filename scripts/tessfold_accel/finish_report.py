"""Final paired statistics, plots and paper snippets from completed evidence.

May rebuild completed components while full training runs. --require-complete
requires every planned core experiment and native8B formal/timing artifacts.
"""
import argparse,collections,json,statistics
from pathlib import Path
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
from scripts.tessfold_accel.live_io import live_lines
from vqa.benchmark_metrics import due_field_f1
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def prefix_stats():
    base=BASE/'prefix_formal';assert (base/'complete.json').exists()
    rows=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)]
    assert len(rows)==len({(r['id'],r['method']) for r in rows})==47130
    results={};rng=np.random.default_rng(20261115)
    for method in ['REF','Column']:
        rr=[r for r in rows if r['method']==method];by={};samples=[]
        for task in sorted({r['task'] for r in rr}):
            ss=[r for r in rr if r['task']==task];pages=collections.defaultdict(lambda:np.zeros(2))
            for r in ss:pages[r['document_id']]+=np.array([r['cached_score']-r['reference_score'],1.])
            a=np.asarray(list(pages.values()));boot=[]
            for _ in range(100):
                total=a[rng.integers(len(a),size=(100,len(a)))].sum(1);boot.extend(100*total[:,0]/total[:,1])
            by[task]=dict(questions=len(ss),reference=100*np.mean([r['reference_score'] for r in ss]),cached=100*np.mean([r['cached_score'] for r in ss]),
                delta=100*np.mean([r['cached_score']-r['reference_score'] for r in ss]),document_bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist(),
                token_mismatches=sum(not r['tokens_equal'] for r in ss))
            samples.append(np.asarray(boot))
        macro_delta=sum(r['delta'] for r in by.values())/len(by)
        # Taskwise page-stratified sampling; tasks equally weighted.
        boot=np.asarray(samples).mean(0);lo,hi=np.quantile(boot,[.025,.975])
        f1={}
        for task in ['KLC','DeepForm']:
            ss=[r for r in rr if r['task']==task]
            f1[task]={name:due_field_f1([dict(r,prediction=r[field]) for r in ss]) for name,field in [('reference','reference_prediction'),('cached','cached_prediction')]}
        results[method]=dict(by_task=by,macro_delta=macro_delta,macro_delta_ci=[float(lo),float(hi)],
            observational_within_025pp_lower_margin=bool(lo>-.25),not_bit_exact=any(not r['tokens_equal'] for r in rr),due_corpus_f1=f1,
            criterion='Exploratory reporting margin -0.25pp, not a preregistered noninferiority claim; all task deltas and token changes shown.')
    save(base/'statistics.json',results);return results


def multi_stats():
    base=BASE/'multi_deployment';assert (base/'complete.json').exists()
    rows=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)];assert len(rows)==864
    grouped=collections.defaultdict(list)
    for r in rows:
        for k,v in r['amortization'].items():grouped[(r['method'],r['scenario'],int(k))].append(v)
    stats=[dict(method=m,scenario=s,k=k,mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),serial_queries_per_second=1000/float(np.mean(v)),page_runs=len(v)) for (m,s,k),v in sorted(grouped.items())]
    index={(r['page'],r['method'],r['scenario'],r['repeat']):r for r in rows}
    pages=sorted({r['page'] for r in rows});assert len(pages)==24
    comparisons=[];rng=np.random.default_rng(20261115)
    configs=[('REF','cold','REF','feature_text_only'),('Column','cold','Column','feature_text_only'),
        ('Tree','cold','Tree','feature_text_only'),('REF','feature_text_only','Column','feature_text_only'),
        ('REF','feature_text_only','Tree','feature_text_only'),('Column','feature_text_only','Column','prefix_kv'),
        ('REF','feature_text_only','REF','prefix_kv')]
    for rm,rs,cm,cs in configs:
      for k in [1,2,5,10]:
        arr=np.asarray([[np.mean([index[(page,m,scenario,rep)]['amortization'][str(k)] for rep in range(3)]) for m,scenario in [(rm,rs),(cm,cs)]] for page in pages])
        total=arr.sum(0);boot=arr[rng.integers(len(arr),size=(10000,len(arr)))].sum(1)
        comparisons.append(dict(reference_method=rm,reference_scenario=rs,candidate_method=cm,candidate_scenario=cs,k=k,
            paired_mean_speedup=float(total[0]/total[1]),document_bootstrap_95_ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()))
    save(base/'summary.json',dict(rows=stats,paired_comparisons=comparisons,bootstrap='24pageclusters, eachpage averaged over3repetitions,10000resamples; page-workload weighting, not task-equalmacro'));return stats


def interleaved_stats():
    base=BASE/'interleaved';assert (base/'complete.json').exists()
    rows=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)];assert len(rows)==2520
    index={(r['id'],r['method'],r['ratio'],r['mode'],r['repeat']):r for r in rows};assert len(index)==2520
    comparisons=[];rng=np.random.default_rng(20261115)
    qa=lines(BASE/'system_exact/timing_rows.jsonl')
    for ratio in [.25,.5]:
     for candidate in ['C_FAST','T_FAST']:
      for reference in ['REF']+([candidate.replace('_FAST','_DENSE')] if ratio==.25 else []):
       for mode in ['natural_eos','fixed64']:
        pages=collections.defaultdict(lambda:np.zeros(4))
        for row in qa:
            aa=[index[(row['id'],reference,1. if reference=='REF' else ratio,mode,k)] for k in range(2)]
            bb=[index[(row['id'],candidate,ratio,mode,k)] for k in range(2)]
            pages[row['document_id']]+=np.array([np.mean([r['end_to_end_ms'] for r in aa]),np.mean([r['end_to_end_ms'] for r in bb]),np.mean([r['ttft_ms'] for r in aa]),np.mean([r['ttft_ms'] for r in bb])])
        a=np.asarray(list(pages.values()));tot=a.sum(0);boot=a[rng.integers(len(a),size=(10000,len(a)))].sum(1)
        comparisons.append(dict(reference=reference,candidate=candidate,ratio=ratio,mode=mode,e2e_speedup=float(tot[0]/tot[1]),
            e2e_bootstrap_95_ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist(),ttft_speedup=float(tot[2]/tot[3]),
            ttft_bootstrap_95_ci=np.quantile(boot[:,2]/boot[:,3],[.025,.975]).tolist()))
    save(base/'statistics.json',comparisons);return comparisons


def batch_stats():
    base=BASE/'batched_serving';assert (base/'complete.json').exists()
    rows=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)];assert len(rows)==144
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['stratum'],r['method'],r['batch_size'],r['mode'])].append(r)
    stats=[]
    for (s,m,b,mode),rr in sorted(grouped.items()):
        stats.append(dict(stratum=s,method=m,batch=b,mode=mode,batch_ms_mean=float(np.mean([r['batch_ms'] for r in rr])),
            qps_mean=float(np.mean([r['queries_per_second'] for r in rr])),peak_max_gib=max(r['peak_allocated_bytes'] for r in rr)/2**30,
            mean_qa_score=float(np.mean([q['score'] for r in rr for q in r['queries']])),ids=[q['id'] for q in rr[0]['queries']]))
    save(base/'summary.json',dict(rows=stats,scope='True synchronous batches, not a maximum async concurrency capacity test. Batching may alter floating outputs; retain all task scores.'))
    return stats


def plots():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=BASE/'figures';directory.mkdir(exist_ok=True)
    validation=[r for p in (BASE/'validation1').glob('rows.rank*.jsonl') for r in lines(p)]
    if validation:
        fig,ax=plt.subplots(figsize=(6,3.7))
        for kind,label in [('col_adaptive','Column'),('quadtree_question','Tree')]:
            rr=[r for r in validation if r['kind']==kind and r['ratio']==.25]
            ax.scatter([r['n'] for r in rr],[r['original_compressor_wall_ms'] for r in rr],s=13,alpha=.4,label=label+' Python')
            ax.scatter([r['n'] for r in rr],[r['variants']['dense_exact']['compressor_wall_ms'] for r in rr],s=13,alpha=.7,label=label+' compiled exact')
        ax.set(xlabel='Original visual tokens',ylabel='Pure compressor wall time (ms)');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        for ext in ['pdf','png']:fig.savefig(directory/f'compressor_wall.{ext}',dpi=180)
        plt.close(fig)
    for label in ['qwen_pilot1','qwen_pilot_high','qwen_pilot_stable','qwen_full_pass1','qwen_full_tree25','qwen_full_stable50']:
        runs=[p for p in (BASE/label).glob('*_r*') if (p/'training.jsonl').exists()]
        if not runs:continue
        fig,axs=plt.subplots(1,3,figsize=(11.5,3.4))
        for run in runs:
            manifest=load(run/'manifest.json');logs=live_lines(run/'training.jsonl');initial=load(run/'initial.summary.json')
            if not logs:continue
            epochs=sorted(run.glob('epoch*.summary.json'),key=lambda p:int(p.name.split('.')[0][5:]));ev=[(0,initial)]+[(int(p.name.split('.')[0][5:]),load(p)) for p in epochs]
            ep=collections.defaultdict(list)
            for r in logs:ep[r['epoch']].append(r['kl'])
            axs[0].plot(sorted(ep),[np.mean(ep[k]) for k in sorted(ep)],'o-',label=run.name+' train')
            axs[0].plot([k for k,r in ev if r],[r['kl'] for k,r in ev if r],'s--',label=run.name+' dev')
            axs[1].plot([k for k,r in ev if r],[r['macro'] for k,r in ev if r],'o-',label=run.name)
            axs[2].plot([r['step'] for r in logs],[r['gradient_norm'] for r in logs],linewidth=.7,alpha=.75,label=run.name)
        axs[0].set(xlabel='Epoch',ylabel='KL loss');axs[1].set(xlabel='Epoch',ylabel='Dev generation macro (%)');axs[2].set(xlabel='Optimizer step',ylabel='Gradient norm before clip',yscale='log')
        for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ['pdf','png']:fig.savefig(directory/f'{label}_training.{ext}',dpi=180)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    from scripts.tessfold_accel.progress import main as progress
    progress();sections=[]
    if (BASE/'system_exact/complete.json').exists():sections+=['[Full efficiency table for the compiled strict path](TESSFOLD_COMPILED_SYSTEM_RESULTS.md). [Actual native 8B architecture](TESSFOLD_NATIVE8B_METHOD.md) · [Environment and reproduction entry points](TESSFOLD_RUNTIME_REPRODUCTION.md). ']
    if (BASE/'interleaved/complete.json').exists():
        rr=interleaved_stats();sections+=['','## Interleaved Timing Comparison','',
            'Randomize method order within question: 90 QA, 7 configurations, natural/fixed64, two repeats each; 10,000 page-bootstrap resamples.','',
            '|Reference|Candidate|Retention fraction|Mode|E2E speedup|95%CI|TTFT speedup|','|---|---|---:|---|---:|---|---:|']
        for r in rr:sections.append(f'|{r["reference"]}|{r["candidate"]}|{r["ratio"]:.0%}|{r["mode"]}|{r["e2e_speedup"]:.3f}×|[{r["e2e_bootstrap_95_ci"][0]:.3f},{r["e2e_bootstrap_95_ci"][1]:.3f}]|{r["ttft_speedup"]:.3f}×|')
    if (BASE/'multi_deployment/complete.json').exists():
        rr=multi_stats();sections+=['','## Same-Page Multi-Question Results Including Initial Ingestion','',
            'feature_text_only caches features and reconstructs the original prompt exactly without repeated image processing. prefix_kv is a separate numerical approximation. Neither measures maximum concurrent throughput.','',
            '|Method|Scenario|K|Mean ms/question|Serial queries/s|','|---|---|---:|---:|---:|']
        for r in rr:
            if r['k'] in [1,5,10]:sections.append(f'|{r["method"]}|{r["scenario"]}|{r["k"]}|{r["mean_ms"]:.2f}|{r["serial_queries_per_second"]:.3f}|')
        sections+=['','Same-page K=10 paired mean-latency ratios (3-repeat page means, 24-page bootstrap ×10,000): ','',
            '|Reference|Candidate|Speedup|95% page-level CI|','|---|---|---:|---|']
        for v in load(BASE/'multi_deployment/summary.json')['paired_comparisons']:
            if v['k']==10:
                ci=v['document_bootstrap_95_ci']
                sections.append(f'|{v["reference_method"]}/{v["reference_scenario"]}|{v["candidate_method"]}/{v["candidate_scenario"]}|{v["paired_mean_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|')
        c=load(BASE/'multi_deployment/complete.json');sections+=['',f'Feature-cache tokens match exactly; prefix_kv changes generation on {c["prefix_token_mismatches"]} times; summed per-question task-score differences {c["prefix_task_score_sum_delta"]:.4f}. ']
    if (BASE/'prefix_formal/complete.json').exists():
        rr=prefix_stats();sections+=['','## Full-Task Impact of Visual-Prefix KV Reuse','',
            '23,565 questions/method against independent historical cap=256 generation. Below DeepForm/KLC use shared field EM; JSON also gives DUE corpus F1. Bootstrap task-stratified page clusters 10,000 times. Small macro differences do not imply tokenwise losslessness.','',
            '|Method|Task|Questions|Original complete generation|KV cache|Score difference|95% page-level CI|Token changes|','|---|---|---:|---:|---:|---:|---|---:|']
        for m,result in rr.items():
            for t,r in result['by_task'].items():sections.append(f'|{m}|{t}|{r["questions"]}|{r["reference"]:.4f}|{r["cached"]:.4f}|{r["delta"]:+.4f}|[{r["document_bootstrap_95_ci"][0]:.4f},{r["document_bootstrap_95_ci"][1]:.4f}]|{r["token_mismatches"]}|')
            sections+=['',f'{m}Equally weighted task-macro score difference{result["macro_delta"]:+.4f}, 95%CI[{result["macro_delta_ci"][0]:.4f},{result["macro_delta_ci"][1]:.4f}]. ']
    if (BASE/'batched_serving/complete.json').exists():
        rr=batch_stats();sections+=['','## Real-Batch Throughput and Peak Memory','',
            'Fix 8 real requests in each of three sequence-length strata, with nested subsets for batches 1/2/4/8. Report fixed64 and the mean of three repeats. This is synchronous batch throughput, not asynchronous maximum concurrency. Larger batches change numerical kernels; retain per-question outputs rather than inheriting batch-1 accuracy.','',
            '|Layer|Method|batch|Mean queries/s|Maximum peak (GiB)|Mean QA score|','|---|---|---:|---:|---:|---:|']
        for r in rr:
            if r['mode']=='fixed64':sections.append(f'|{r["stratum"]}|{r["method"]}|{r["batch"]}|{r["qps_mean"]:.3f}|{r["peak_max_gib"]:.4f}|{r["mean_qa_score"]:.4f}|')
    if (BASE/'formal_equivalence/complete.json').exists():
        rr=load(BASE/'formal_equivalence/complete.json');sections+=['','## Full Formal UND Equivalence Regression','',
            f'Completed{rr["questions"]} questions × four configurations, totaling {rr["records"]} records; all P/PH/generated tokens match history. “Lossless” refers only to strict-versus-original compressed implementation, **not equal scores to the uncompressed teacher**.','',
            '|Configuration|Shared eight-task macro|','|---|---:|']
        for k,r in rr['results'].items():sections.append(f'|{k}|{r["macro"]:.4f}|')
    if (BASE/'hard_formal/complete.json').exists():
        r=load(BASE/'hard_formal/complete.json');sections+=['','## Full Evaluation of the Hard Variant Without Dense P','',
            f'Total{r["questions"]} questions; shared eight-task macro={r["macro"]:.4f}. Generated-token changes versus historical H: {r["legacy_h_token_mismatches"]} questions; summed score difference {r["score_sum_delta"]:+.4f}. This separate hard variant does not inherit soft losslessness. Per-task results: `{json.dumps(r["tasks"],ensure_ascii=False)}`. ']
    if (BASE/'tiled_vision_pilot/complete.json').exists():
        r=load(BASE/'tiled_vision_pilot/complete.json');sections+=['','## Conservative Query-Chunked Visual Attention Pilot','',
            'Preserve eager BF16 QK/scaling, FP32 softmax cast to BF16, and AV rounding boundaries. Chunk only queries, retaining the complete key axis. All weights and 56 development questions are fixed. Even matching this set requires broader regression before formal strict use.','',
            '|Query chunk|Feature changes|Generation changes|Median encoder speedup|Original peak (GiB)|Chunk peak (GiB)|Score difference (pp)|','|---|---:|---:|---:|---:|---:|---:|']
        for tile,v in r['results'].items():sections.append(f'|{tile}|{v["feature_mismatches"]}|{v["token_mismatches"]}|{v["median_encoder_speedup"]:.3f}×|{v["max_eager_peak_gib"]:.4f}|{v["max_tiled_peak_gib"]:.4f}|{v["mean_score_delta_pp"]:+.4f}|')
    if (BASE/'bidirectional_vision_pilot/complete.json').exists():
        r=load(BASE/'bidirectional_vision_pilot/complete.json');sections+=['','## Pilot of Explicit Bidirectional Fused Attention','',
            'Independent of the author\'s original call, set pre_tokens=next_tokens=INTMAX so masked PFA permits all future positions within the same image. The interpretation of default next_tokens=0 comes from local interface documentation and must be assessed with the new experiments. Eager/fused-kernel rounding differences remain; bit-exactness is not claimed.','',
            f'Fixed 56 QA: median encoder speedup {r["median_encoder_speedup"]:.3f}×; REF per-question mean change {r["ref_mean_qa_delta_pp"]:+.4f} percentage points; expansion eligibility={r["expansion_gate_passed"]}; questions with generation changes per configuration: `{json.dumps(r["token_changes"],ensure_ascii=False)}`. ']
    if (BASE/'bidirectional_formal/complete.json').exists():
        r=load(BASE/'bidirectional_formal/complete.json');sections+=['','## Independent Formal Evaluation of Explicit Bidirectional Fused Attention','',
            'Same frozen UND/P weights; re-encode and independently regenerate all 23,565 QA × 5 configurations. This numerical candidate uses fresh measurements without inheriting eager scores or training/selecting checkpoints. Separate both DeepForm/KLC metrics.','',
            '|Configuration|Shared field-EM macro|Native DUE-F1 macro|','|---|---:|---:|']
        for m,v in r['shared_fieldEM_results'].items():sections.append(f'|{m}|{v["macro"]:.4f}|{r["native_metric_results"][m]["macro"]:.4f}|')
    if (BASE/'bidirectional_timing/complete.json').exists() or (BASE/'bidirectional_formal/complete.json').exists():
        from scripts.tessfold_accel.bidirectional_stats import compute as bidirectional_stats
        bs=bidirectional_stats()
        if 'quality_vs_original_eager' in bs:
            sections+=['','### Full-Set Paired Quality Differences: Bidirectional Fusion Versus Eager','',
                'Shared field-EM macro difference, task-equal with 10,000 within-task document resamples. This is a numerical implementation change, not a training gain.','',
                '|Configuration|Macro-score difference (pp)|95% stratified page-level CI|Questions with changed decoded text|','|---|---:|---|---:|']
            for m,v in bs['quality_vs_original_eager'].items():
                ci=v['shared_fieldEM_macro_delta_task_stratified_document_bootstrap_95_ci']
                sections.append(f'|{m}|{v["shared_fieldEM_macro_delta"]:+.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|{v["decoded_prediction_changes"]}|')
        if 'timing_summary' in bs:
            sections+=['','### Full Measured Efficiency of the Bidirectional Path','',
                '2,700 independent timing and 450 independent memory measurements on 90 fixed questions / 5 configurations, natural/fixed64 × 3 repeats. REF also uses bidirectional fusion. All frontend work remains; fused-kernel benefits are not attributed to P. Natural and independent-memory outputs must match this variant’s formal outputs.','',
                '|Method|Retention fraction|Mode|E2E p50ms|TTFT p50ms|Compression+merger p50 (ms)|','|---|---:|---|---:|---:|---:|']
            for v in bs['timing_summary']:sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["mode"]}|{v["e2e_p50_ms"]:.2f}|{v["ttft_p50_ms"]:.2f}|{v["phase_wall_medians_ms"]["compression_and_merger"]:.2f}|')
            sections+=['','|Method|Retention fraction|Full-pipeline peak (GiB)|Encoder peak (GiB)|Downstream peak (GiB)|Logical visual KV (MiB)|','|---|---:|---:|---:|---:|---:|']
            for v in bs['memory_summary']:sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["full_max_gib"]:.4f}|{v["encoder_max_gib"]:.4f}|{v["downstream_max_gib"]:.4f}|{v["visual_kv_median_mib"]:.2f}|')
            sections+=['','|Method|Retention fraction|Mode|E2E speedup over this variant\'s REF|95% page-level CI|TTFT speedup|','|---|---:|---|---:|---|---:|']
            for v in bs['paired_speed_vs_same_bidirectional_REF']:
                ci=v['e2e_document_bootstrap_95_ci']
                sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|')
    if (BASE/'full_training_audit.json').exists():
        audit=load(BASE/'full_training_audit.json');assert audit['all_passed']
        sections+=['','## Independent Recalculation Audit of Full Training','',
            'All 12,500 optimizer steps across four runs are consecutive. Actual LR and clipping multipliers match their formulas step by step, and latest includes optimizer state. Native QA/DUE-F1 and KL are recomputed for initial plus 4 epochs × 56 development questions per run: 1,120 records, all matching summaries. Training source signatures remain unchanged; formal test scores are not read for selection. See `artifacts/tessfold_acceleration_20261115/full_training_audit.json`. ']
    if (BASE/'bidir_interleaved/complete.json').exists():
        from scripts.tessfold_accel.bidir_interleaved_stats import compute as bidir_interleaved_stats
        bs=bidir_interleaved_stats();sections+=['','## Interleaved Eager/Bidirectional Comparison on Identical Questions','',
            'To verify compressed-prefill differences independently, each of 90 questions / generation modes / repeats randomly interleaves 6 configurations (two backends × REF/Column25/Tree25), yielding 2,160 records. All timings use the same frozen single_request implementation. Cross-backend comparisons must also report each backend\'s full VQA quality; model weights are unchanged.','',
            '|Reference|Candidate|Mode|Paired E2E speedup|E2E 95% page-level CI|Paired TTFT speedup|TTFT 95% page-level CI|','|---|---|---|---:|---|---:|---|']
        for v in bs['comparisons']:
            ci=v['e2e_document_bootstrap_95_ci'];ti=v['ttft_document_bootstrap_95_ci']
            sections.append(f'|{v["reference_backend"]}/{v["reference_method"]}|{v["candidate_backend"]}/{v["candidate_method"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
        sections+=['','|Backend|Method|Mean encoder time (ms)|Mean prefill time (ms)|Mean TTFT (ms)|Generated-output changes versus this path’s previous measurement|','|---|---|---:|---:|---:|---:|']
        for v in bs['natural_phase_means']:sections.append(f'|{v["backend"]}|{v["method"]}|{v["encoder_mean_ms"]:.2f}|{v["prefill_mean_ms"]:.2f}|{v["ttft_mean_ms"]:.2f}|{v["prior_generation_mismatches"]}|')
    if (BASE/'teacher_prefix_audit.json').exists():
        ta=load(BASE/'teacher_prefix_audit.json');t=ta['splits']['train'];d=ta['splits']['dev']
        sections+=['','## Audit of Actual Teacher-Supervision Prefixes','',
            f'Among 20,000 training QA, {t["truncated"]} responses truncate at 64 tokens without EOS ({100*t["truncated"]/t["questions"]:.3f}%), truncated responses on the fixed 56-question development set: {d["truncated"]} records. Training-answer lengths [min,p50,p90,p99,max]: {t["length_quantiles"]}; original training protocol unchanged, with no formal-score-based relabeling. Report truncation separately from formal greedy cap=256.']
    if (BASE/'qwen_formal/complete.json').exists():
        r=load(BASE/'qwen_formal/complete.json');registry=load(BASE/'qwen_selected_runs.json')
        sections+=['','## Formal Qwen3-VL-8B Compression Results','',
            'Use original 20k training / 56 development / 23,565 formal QA with document-disjoint images, a fixed short-answer prompt, and cap=256. All checkpoints are selected only by development data; native 8B is fully frozen. The same P acts on all 4 visual branches. Primary metrics use DUE F1; the shared field-EM version is saved separately in JSON and is not the same macro score.','',
            '|Model|Formal QA|Native-metric macro|Shared field-EM macro|Development-selected epoch|','|---|---:|---:|---:|---|']
        for model,rr in r['native_metric_results'].items():
            meta=next((x for x in registry['runs'] if x['id']==model),{})
            sections.append(f'|{model}|23565|{rr["macro"]:.4f}|{r["shared_fieldEM_results"][model]["macro"]:.4f}|{meta.get("stage","Native teacher")}|')
        for model,rr in r['native_metric_results'].items():
            sections+=['',f'{model}Per-task results: `{json.dumps(rr["by_task"],ensure_ascii=False)}`. ']
        controls=r.get('initial_control_results')
        if controls:
            sections+=['','### Before/After Comparison on an Independent Fixed Sample','',
                'Select 64 documents/task in advance (512 images, 2,404 questions), including all their questions. Compare each architecture\'s deterministic initial parameters with its dev-selected parameters on identical samples; this formal subset never selects models. It is not the full 23,565-question mean. Native metrics include KLC/DeepForm DUE corpus F1; CIs apply only to the shared field-EM macro.','',
                '|Configuration|Initial native macro|Trained native macro|Native-metric score difference|Shared field-EM score difference|Shared field-EM difference: 95% page CI|','|---|---:|---:|---:|---:|---|']
            for m,c in controls['comparisons'].items():
                ci=c['shared_fieldEM_task_stratified_document_bootstrap_95_ci']
                sections.append(f'|{m}|{c["initial"]["native_macro"]:.4f}|{c["selected"]["native_macro"]:.4f}|{c["native_macro_delta"]:+.4f}|{c["shared_fieldEM_macro_delta"]:+.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|')
    if (BASE/'qwen_formal/complete.json').exists():
        from scripts.tessfold_accel.native_error_analysis import compute as native_error_analysis
        er=native_error_analysis();sections+=['','## Compression Retention on Teacher-Solvable Questions','',
            'Post-hoc formal-result analysis, not selection. Group teacher-positive/perfect questions separately per task. KLC/DeepForm use field EM for conditioning, not corpus F1. Different subset sizes make these conditional analyses, not new unconditional benchmarks.','',
            '|Model|Group|Task|Questions|Teacher mean score|Compressed mean score|Score difference: teacher−student|Questions with lower compressed score|','|---|---|---|---:|---:|---:|---:|---:|']
        for m,groups in er['results'].items():
            for group in ['teacher_gt0','teacher_perfect']:
                for task,v in groups[group]['by_task'].items():sections.append(f'|{m}|{group}|{task}|{v["questions"]}|{v["teacher_mean_percent"]:.3f}|{v["student_mean_percent"]:.3f}|{v["mean_loss_pp"]:+.3f}|{v["lower_questions"]}|')
        sections+=['','The 32 largest score-drop questions per model/task, with fixed-ID tie breaking, are saved as [gold/teacher/student CSV](../artifacts/tessfold_acceleration_20261115/error_analysis/largest_losses.csv), for error analysis only, not selection.']
    if (BASE/'qwen_timing/complete.json').exists():
        rows=[r for f in (BASE/'qwen_timing').glob('rows.rank*.jsonl') for r in lines(f)];grouped=collections.defaultdict(list)
        for r in rows:grouped[(r['model'],r['mode'])].append(r)
        sections+=['','## Full-Deployment Latency for 8B','',
            'Use the same 90 questions, including official processing, native vision/all mergers, P, and the decoder. Post-merger P saves no encoder/merger work. TTFT is recorded after the first lm_head and argmax. Compression+prompt includes P generation, four-branch P@X, representative-position selection, and embedding/scatter; router-only timing is not reported as total compression. prefill_done is saved separately in JSON.','',
            '|Model|Mode|E2E p50ms|TTFT p50ms|P generation p50 (ms)|Compression+prompt p50 (ms)|Peak (GiB)|','|---|---|---:|---:|---:|---:|---:|']
        for (m,mode),rr in sorted(grouped.items()):sections.append(f'|{m}|{mode}|{np.median([r["end_to_end_ms"] for r in rr]):.2f}|{np.median([r["ttft_ms"] for r in rr]):.2f}|{np.median([r["p_generator_ms"] for r in rr]):.2f}|{np.median([r["compressor_ms"] for r in rr]):.2f}|{max(r["peak_allocated_bytes"] for r in rr)/2**30:.4f}|')
    if (BASE/'qwen_timing/complete.json').exists():
        from scripts.tessfold_accel.qwen_timing_stats import compute as compute_timing
        timing_statistics=compute_timing()
        sections+=['','### Paired 8B Speedups and Resolution Strata','',
            'Average three repeats within each question, then divide reference/candidate sums of latency; bootstrap pages 10,000 times. short/medium/long retain the original fixed 90-question strata. Native 8B visual-token counts can differ from UND. Full fixed64 and post-front values are in statistics.json.','',
            '|Model|Stratum|Mode|E2E speedup|E2E 95% page-level CI|TTFT speedup|TTFT 95% page-level CI|','|---|---|---|---:|---|---:|---|']
        for v in timing_statistics['comparisons']:
            if v['mode']=='natural_eos' or v['stratum']=='all':
                ci=v['e2e_bootstrap_95_ci'];ti=v['ttft_bootstrap_95_ci']
                sections.append(f'|{v["model"]}|{v["stratum"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
    if (BASE/'qwen_timing/complete.json').exists():
        from scripts.tessfold_accel.boundary_stats import compute as boundary_stats
        boundary=boundary_stats();sections+=['','## Input-Token Counts and Post-Tower Speedups with Matched Boundaries','',
            'Input visual tokens [min,p50,p90,max] on the same 90 questions: '+json.dumps(boundary['token_scales'],ensure_ascii=False)+'. Native 8B token counts after its merger differ, so LLM parameter count alone cannot predict speed. Compare REF/FAST only: both partition after tower return and before merger, timing merger-to-first-token, not E2E TTFT. S/DENSE with different boundaries are excluded.','',
            '|Method|Retention fraction|Mode|Paired post-tower speedup|95% page-level CI|','|---|---:|---|---:|---|']
        for v in boundary['strict_und_post_tower_pairs']:
            ci=v['document_bootstrap_95_ci'];sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["mode"]}|{v["post_tower_ttft_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|')
    ceiling=load(BASE/'qwen_timing/amdahl_front_ceiling.json')
    if ceiling:
        sections+=['','### Measurement Boundaries for the Missed 8B Speed Targets','',
            f'REF mean frontend time on the same 90 questions (processor + full native vision/mergers): {ceiling["reference_front_mean_ms"]:.2f} ms; mean TTFT {ceiling["reference_ttft_mean_ms"]:.2f} ms. With the frontend unchanged, even a zero-cost backend limits ideal TTFT speedup to {ceiling["ideal_zero_postfront_ttft_upper_bound"]:.3f}×. This is a workload-specific Amdahl bound, not measured speedup or a general bound for larger models. Parameter count alone cannot predict 2.2–3.1×.']
    if (BASE/'qwen_memory/complete.json').exists() and (BASE/'qwen_timing/complete.json').exists():
        memory=[r for p in (BASE/'qwen_memory').glob('rows.rank*.jsonl') for r in lines(p)]
        timing=[r for p in (BASE/'qwen_timing').glob('rows.rank*.jsonl') for r in lines(p) if r['mode']=='natural_eos']
        index={(r['id'],r['model'],r['repeat']):r for r in timing};assert len(index)==len(memory)==1350
        grouped=collections.defaultdict(list)
        for r in memory:
            assert r['generated_ids']==index[(r['id'],r['model'],r['repeat'])]['generated_ids'],'Independent memory pass changed outputs'
            assert sha(ROOT/r['trace'])==r['trace_sha256']
            grouped[r['model']].append(r)
        sections+=['','## Independent Stagewise 8B Memory and Logical Visual KV','',
            '1,350 independent memory measurements: 90 questions × 5 models × 3 repeats, natural EOS with cap=256. Generated tokens must match the unsampled timing pass question by question. Peaks come from the allocator; means use 1 ms time-weighted sampling. Native encoder and mergers form the frontend; downstream includes visual setup, prefill, and generation. All measurements include resident model weights. Logical visual KV size cannot imply a proportional reduction in device-wide peak memory or increase in throughput.','',
            '|Model|Full-pipeline maximum (GiB)|Frontend maximum (GiB)|Downstream maximum (GiB)|Sampled prefill mean (GiB)|Median logical visual KV (MiB)|','|---|---:|---:|---:|---:|---:|']
        stats={}
        for m,rr in sorted(grouped.items()):
            v=dict(full_max_gib=max(r['peak_allocated_bytes'] for r in rr)/2**30,
                front_max_gib=max(r['stages']['front_processor_vision_mergers']['peak_allocated_bytes'] for r in rr)/2**30,
                downstream_max_gib=max(r['downstream_peak_allocated_bytes'] for r in rr)/2**30,
                prefill_mean_gib=float(np.mean([r['stages']['qwen_prefill']['time_average_allocated_bytes'] for r in rr]))/2**30,
                visual_kv_median_mib=float(np.median([r['decoder_state']['visual_kv_bytes'] for r in rr]))/2**20)
            stats[m]=v
            sections.append(f'|{m}|{v["full_max_gib"]:.4f}|{v["front_max_gib"]:.4f}|{v["downstream_max_gib"]:.4f}|{v["prefill_mean_gib"]:.4f}|{v["visual_kv_median_mib"]:.2f}|')
        save(BASE/'qwen_memory/summary.json',dict(results=stats,all_timing_tokens_equal=True,traces_sha_verified=True))
    if (BASE/'attention_semantics.json').exists():
        from scripts.tessfold_accel.attention_semantics_report import main as attention_report
        attention_report()
        sections+=['','[Direct operator evidence on attention-range defaults and bidirectional repair](TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md). ']
    from scripts.tessfold_accel.training_diagnostics import main as training_diagnostics
    training_diagnostics()
    plots();sections+=['','## Curves and References','', '[Per-epoch loss, gradient, and generation diagnostics](TESSFOLD_8B_TRAINING_DIAGNOSTICS.md). ','',
        'Compressor-only latency plot: [PDF](../artifacts/tessfold_acceleration_20261115/figures/compressor_wall.pdf). Per-epoch train/validation KL, generation scores, and stepwise gradient curves for each 8B version are in `artifacts/tessfold_acceleration_20261115/figures/*_training.pdf/png`. Training KL averages actual batches in the epoch; development KL averages all 56 questions at a fixed checkpoint. These are different datasets.','',
        'Compression-matrix and prediction-KL reference: [Wang et al., Rethinking Token Reduction for Large Vision-Language Models, CVPR 2026 (MetaCompress)](../Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf). This round retains prediction KL but does not reproduce its P generator.','',
        'Native8B architecture: [Official Qwen3-VL-8B-Instruct model/configuration](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), fixed revision  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. ','',
        'Original TeX/PDFs, dense-P history, and peak-memory negative results remain. Manuscript wording must distinguish strict dense-soft, approximate sparse, hard, feature-cache, and approximate-KV variants. Analytical FLOPs/cache-byte ratios are not measured throughput.']
    path=ROOT/'docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md';path.write_text(path.read_text().split('\n\n## Measurements for Completed Stages')[0]+'\n\n## Measurements for Completed Stages\n\n'+'\n'.join(sections)+'\n')
    expected=['system_exact','interleaved','formal_equivalence','prefix_formal','multi_deployment','batched_serving','qwen_formal','qwen_timing','qwen_memory','hard_formal','bidirectional_formal','bidirectional_timing','bidir_interleaved']
    state={p:(BASE/p/'complete.json').exists() for p in expected};save(BASE/'final_report_status.json',dict(time=now(),stages=state,all_complete=all(state.values())))
    if a.require_complete:assert all(state.values()),state
    print(json.dumps(dict(report=str(path),stages=state)))

if __name__=='__main__':main()
