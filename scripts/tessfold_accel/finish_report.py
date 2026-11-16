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
    if (BASE/'system_exact/complete.json').exists():sections+=['[编译严格路径完整效率表](TESSFOLD_COMPILED_SYSTEM_RESULTS.md)。[实际原生8B结构](TESSFOLD_NATIVE8B_METHOD.md) · [环境与复现入口](TESSFOLD_RUNTIME_REPRODUCTION.md)。']
    if (BASE/'interleaved/complete.json').exists():
        rr=interleaved_stats();sections+=['','## 方法交错计时对照','',
            '在每题内随机方法顺序，90QA、7配置、自然/固定64、各两轮。页面bootstrap10000次。','',
            '|参考|候选|保留率|模式|E2E加速比|95%CI|TTFT加速比|','|---|---|---:|---|---:|---|---:|']
        for r in rr:sections.append(f'|{r["reference"]}|{r["candidate"]}|{r["ratio"]:.0%}|{r["mode"]}|{r["e2e_speedup"]:.3f}×|[{r["e2e_bootstrap_95_ci"][0]:.3f},{r["e2e_bootstrap_95_ci"][1]:.3f}]|{r["ttft_speedup"]:.3f}×|')
    if (BASE/'multi_deployment/complete.json').exists():
        rr=multi_stats();sections+=['','## 同页多问：首次摄入计入的实际结果','',
            'feature_text_only为纯特征缓存并精确重建原prompt，无重复图像预处理；prefix_kv另列数值近似。不是最大并发吞吐。','',
            '|方法|场景|K|平均ms/题|串行queries/s|','|---|---|---:|---:|---:|']
        for r in rr:
            if r['k'] in [1,5,10]:sections.append(f'|{r["method"]}|{r["scenario"]}|{r["k"]}|{r["mean_ms"]:.2f}|{r["serial_queries_per_second"]:.3f}|')
        sections+=['','同页K=10配对均值时延比（各页面先平均三次，24页bootstrap10000次）：','',
            '|参考|候选|加速比|95%页面CI|','|---|---|---:|---|']
        for v in load(BASE/'multi_deployment/summary.json')['paired_comparisons']:
            if v['k']==10:
                ci=v['document_bootstrap_95_ci']
                sections.append(f'|{v["reference_method"]}/{v["reference_scenario"]}|{v["candidate_method"]}/{v["candidate_scenario"]}|{v["paired_mean_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|')
        c=load(BASE/'multi_deployment/complete.json');sections+=['',f'特征路径逐题token精确一致；prefix_kv生成不同{c["prefix_token_mismatches"]}次，逐题任务分差总和{c["prefix_task_score_sum_delta"]:.4f}。']
    if (BASE/'prefix_formal/complete.json').exists():
        rr=prefix_stats();sections+=['','## 视觉前缀KV全量任务影响','',
            '每方法23565题，与独立历史完整生成cap256对照。下表DeepForm/KLC为共享fieldEM，另在JSON给出DUE corpus F1。页面分组、任务分层bootstrap10000次；不能因小均分差就称逐token无损。','',
            '|方法|任务|题数|原完整生成|KV缓存|分差|95%页面CI|token变化|','|---|---|---:|---:|---:|---:|---|---:|']
        for m,result in rr.items():
            for t,r in result['by_task'].items():sections.append(f'|{m}|{t}|{r["questions"]}|{r["reference"]:.4f}|{r["cached"]:.4f}|{r["delta"]:+.4f}|[{r["document_bootstrap_95_ci"][0]:.4f},{r["document_bootstrap_95_ci"][1]:.4f}]|{r["token_mismatches"]}|')
            sections+=['',f'{m}任务等权宏平均分差{result["macro_delta"]:+.4f}，95%CI[{result["macro_delta_ci"][0]:.4f},{result["macro_delta_ci"][1]:.4f}]。']
    if (BASE/'batched_serving/complete.json').exists():
        rr=batch_stats();sections+=['','## 真实batch吞吐与峰值显存','',
            '三种序列层各固定8个真实请求，batch1/2/4/8使用嵌套子集。以下固定64步，三次均值；这是同步batch吞吐，不是异步服务并发容量上限。较大batch会改变浮点内核，逐题结果保留，不继承batch1准确率。','',
            '|层|方法|batch|queries/s均值|最大峰值GiB|平均QA分数|','|---|---|---:|---:|---:|---:|']
        for r in rr:
            if r['mode']=='fixed64':sections.append(f'|{r["stratum"]}|{r["method"]}|{r["batch"]}|{r["qps_mean"]:.3f}|{r["peak_max_gib"]:.4f}|{r["mean_qa_score"]:.4f}|')
    if (BASE/'formal_equivalence/complete.json').exists():
        rr=load(BASE/'formal_equivalence/complete.json');sections+=['','## UND正式全量无损回归','',
            f'已完成{rr["questions"]}题×四配置，共{rr["records"]}记录；所有P、PH及生成token对历史全部一致。这里的“无损”仅指新严格实现相对原压缩模型，**不是压缩模型与未压缩teacher分数相同**。','',
            '|配置|共享八任务macro|','|---|---:|']
        for k,r in rr['results'].items():sections.append(f'|{k}|{r["macro"]:.4f}|')
    if (BASE/'hard_formal/complete.json').exists():
        r=load(BASE/'hard_formal/complete.json');sections+=['','## 无稠密P的Hard变体完整评测','',
            f'共{r["questions"]}题，共享八任务macro={r["macro"]:.4f}。相对历史H的生成token变化{r["legacy_h_token_mismatches"]}题，逐题分数差之和{r["score_sum_delta"]:+.4f}。这是独立hard变体，不继承soft的无损结论。各任务：`{json.dumps(r["tasks"],ensure_ascii=False)}`。']
    if (BASE/'tiled_vision_pilot/complete.json').exists():
        r=load(BASE/'tiled_vision_pilot/complete.json');sections+=['','## 保守的query分块视觉注意力先导','',
            '保留原eager的BF16 QK矩乘、缩放、FP32 softmax转BF16和AV矩乘舍入边界，仅切query轴，完整key轴不变。冻结全部参数，56题固定dev；即使本组相同，也需更大回归后才能用于正式严格结果。','',
            '|query块|特征变化|生成变化|encoder中位加速|原峰值GiB|分块峰值GiB|分差pp|','|---|---:|---:|---:|---:|---:|---:|']
        for tile,v in r['results'].items():sections.append(f'|{tile}|{v["feature_mismatches"]}|{v["token_mismatches"]}|{v["median_encoder_speedup"]:.3f}×|{v["max_eager_peak_gib"]:.4f}|{v["max_tiled_peak_gib"]:.4f}|{v["mean_score_delta_pp"]:+.4f}|')
    if (BASE/'bidirectional_vision_pilot/complete.json').exists():
        r=load(BASE/'bidirectional_vision_pilot/complete.json');sections+=['','## 显式双向融合注意力先导','',
            '独立于前述作者原调用：显式pre_tokens=next_tokens=INTMAX，使传入mask的PFA允许全部同图未来位置。此前调用默认next_tokens=0的解释来自本地接口文档，须结合新实验判断。仍存在eager与融合内核的舍入差别，不冒充Bit-Exact。','',
            f'固定56QA：encoder中位加速{r["median_encoder_speedup"]:.3f}倍，REF逐题均分变化{r["ref_mean_qa_delta_pp"]:+.4f}个百分点，扩展门槛通过={r["expansion_gate_passed"]}；各配置生成变化题数：`{json.dumps(r["token_changes"],ensure_ascii=False)}`。']
    if (BASE/'bidirectional_formal/complete.json').exists():
        r=load(BASE/'bidirectional_formal/complete.json');sections+=['','## 显式双向融合注意力独立正式评测','',
            '同一冻结UND/P权重，重新编码并完整独立生成23565QA×5配置；这是独立数值候选。每行均为本轮实测，不继承eager成绩，不选择或训练checkpoint。DeepForm/KLC两种指标分开。','',
            '|配置|共享fieldEM macro|原生DUE-F1 macro|','|---|---:|---:|']
        for m,v in r['shared_fieldEM_results'].items():sections.append(f'|{m}|{v["macro"]:.4f}|{r["native_metric_results"][m]["macro"]:.4f}|')
    if (BASE/'bidirectional_timing/complete.json').exists() or (BASE/'bidirectional_formal/complete.json').exists():
        from scripts.tessfold_accel.bidirectional_stats import compute as bidirectional_stats
        bs=bidirectional_stats()
        if 'quality_vs_original_eager' in bs:
            sections+=['','### 双向融合与原eager的全量配对质量差','',
                '共享fieldEM宏分差，任务等权、任务内按文档bootstrap10000次；这是数值实现变化，不是训练收益。','',
                '|配置|宏分差pp|95%分层页面CI|解码文本变化题数|','|---|---:|---|---:|']
            for m,v in bs['quality_vs_original_eager'].items():
                ci=v['shared_fieldEM_macro_delta_task_stratified_document_bootstrap_95_ci']
                sections.append(f'|{m}|{v["shared_fieldEM_macro_delta"]:+.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|{v["decoded_prediction_changes"]}|')
        if 'timing_summary' in bs:
            sections+=['','### 双向融合路径的完整实测效率','',
                '2700次独立时延与450次独立内存，固定90题，5配置，natural/fixed64各三次；REF同样使用双向融合。仍保留所有前端成本；不将融合内核的收益归因于P压缩。所有natural及独立memory输出须与本变体正式评测相同。','',
                '|方法|保留率|模式|E2E p50ms|TTFT p50ms|压缩+merger p50ms|','|---|---:|---|---:|---:|---:|']
            for v in bs['timing_summary']:sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["mode"]}|{v["e2e_p50_ms"]:.2f}|{v["ttft_p50_ms"]:.2f}|{v["phase_wall_medians_ms"]["compression_and_merger"]:.2f}|')
            sections+=['','|方法|保留率|全流程峰值GiB|encoder峰值GiB|下游峰值GiB|逻辑视觉KV MiB|','|---|---:|---:|---:|---:|---:|']
            for v in bs['memory_summary']:sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["full_max_gib"]:.4f}|{v["encoder_max_gib"]:.4f}|{v["downstream_max_gib"]:.4f}|{v["visual_kv_median_mib"]:.2f}|')
            sections+=['','|方法|保留率|模式|相对本变体REF的E2E加速|95%页面CI|TTFT加速|','|---|---:|---|---:|---|---:|']
            for v in bs['paired_speed_vs_same_bidirectional_REF']:
                ci=v['e2e_document_bootstrap_95_ci']
                sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|')
    if (BASE/'full_training_audit.json').exists():
        audit=load(BASE/'full_training_audit.json');assert audit['all_passed']
        sections+=['','## 全量训练的独立重算审计','',
            '四组共12,500个optimizer step全部连续，实际LR和clip multiplier逐步吻合预设公式；latest含optimizer状态。每组初始及4个epoch的56题预测重新计算原生QA/DUE-F1指标和KL，共1120条dev记录，均与摘要吻合。全部训练源签名保持不变；这些审计不读取正式测试分数做选择。详见`artifacts/tessfold_acceleration_20261115/full_training_audit.json`。']
    if (BASE/'bidir_interleaved/complete.json').exists():
        from scripts.tessfold_accel.bidir_interleaved_stats import compute as bidir_interleaved_stats
        bs=bidir_interleaved_stats();sections+=['','## 同题交错的eager/双向融合直接对照','',
            '为核实独立阶段压缩prefill差异，90QA的每个问题/生成模式/重复内随机排列6个配置（两后端×REF/Column25/Tree25），共2160条。所有时延通过同一冻结single_request采集。跨后端比较还须同时看各自全量VQA质量；模型权重无变化。','',
            '|参考|候选|模式|E2E配对加速|E2E 95%页面CI|TTFT配对加速|TTFT 95%页面CI|','|---|---|---|---:|---|---:|---|']
        for v in bs['comparisons']:
            ci=v['e2e_document_bootstrap_95_ci'];ti=v['ttft_document_bootstrap_95_ci']
            sections.append(f'|{v["reference_backend"]}/{v["reference_method"]}|{v["candidate_backend"]}/{v["candidate_method"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
        sections+=['','|后端|方法|encoder均值ms|prefill均值ms|TTFT均值ms|相对该路径先前测量生成变化数|','|---|---|---:|---:|---:|---:|']
        for v in bs['natural_phase_means']:sections.append(f'|{v["backend"]}|{v["method"]}|{v["encoder_mean_ms"]:.2f}|{v["prefill_mean_ms"]:.2f}|{v["ttft_mean_ms"]:.2f}|{v["prior_generation_mismatches"]}|')
    if (BASE/'teacher_prefix_audit.json').exists():
        ta=load(BASE/'teacher_prefix_audit.json');t=ta['splits']['train'];d=ta['splits']['dev']
        sections+=['','## 实际teacher监督前缀审计','',
            f'20,000训练QA中，{t["truncated"]}条在64token上限处未生成EOS而截断（{100*t["truncated"]/t["questions"]:.3f}%），固定56题dev截断{d["truncated"]}条。训练回答长度[min,p50,p90,p99,max]为{t["length_quantiles"]}；原训练协议保持不变，不在观察正式分数后重标训练。该截断限制与正式greedy cap256分开报告。']
    if (BASE/'qwen_formal/complete.json').exists():
        r=load(BASE/'qwen_formal/complete.json');registry=load(BASE/'qwen_selected_runs.json')
        sections+=['','## Qwen3-VL-8B正式压缩结果','',
            '原20k训练/56题dev/23565题正式数据，文档图像互斥。短答案prompt统一固定，cap256；所有checkpoint仅由dev选择，原生8B全冻结。DeepStack所有4条视觉支路使用同一P。主指标给出DUE F1版本，共享fieldEM版本另存JSON，不能混为同一宏分。','',
            '|模型|正式QA|原生指标macro|共享fieldEM macro|dev选择epoch|','|---|---:|---:|---:|---|']
        for model,rr in r['native_metric_results'].items():
            meta=next((x for x in registry['runs'] if x['id']==model),{})
            sections.append(f'|{model}|23565|{rr["macro"]:.4f}|{r["shared_fieldEM_results"][model]["macro"]:.4f}|{meta.get("stage","原生teacher")}|')
        for model,rr in r['native_metric_results'].items():
            sections+=['',f'{model}各任务：`{json.dumps(rr["by_task"],ensure_ascii=False)}`。']
        controls=r.get('initial_control_results')
        if controls:
            sections+=['','### 独立固定样本上的训练前后对照','',
                '预先按任务各取64个文档（共512图、2404题），包含这些文档的全部问题。相同样本比较各架构的确定性初始参数与dev所选参数；该正式子集不参与选模型。此表不是全23565题均分。原生指标包含KLC/DeepForm的DUE corpus F1；CI仅对应共享fieldEM宏分。','',
                '|配置|初始原生macro|训练后原生macro|原生分差|共享fieldEM分差|共享fieldEM分差95%页面CI|','|---|---:|---:|---:|---:|---|']
            for m,c in controls['comparisons'].items():
                ci=c['shared_fieldEM_task_stratified_document_bootstrap_95_ci']
                sections.append(f'|{m}|{c["initial"]["native_macro"]:.4f}|{c["selected"]["native_macro"]:.4f}|{c["native_macro_delta"]:+.4f}|{c["shared_fieldEM_macro_delta"]:+.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|')
    if (BASE/'qwen_formal/complete.json').exists():
        from scripts.tessfold_accel.native_error_analysis import compute as native_error_analysis
        er=native_error_analysis();sections+=['','## teacher可解问题上的压缩保留','',
            '这是正式结果的事后解释，不参与模型选择。按teacher逐题分数>0或满分分组，每个任务分别统计；KLC/DeepForm在此使用逐字段EM进行条件筛选，不把它称为DUE corpus F1。各组题数不同，不是新的无条件benchmark。','',
            '|模型|组别|任务|题数|teacher均分|压缩均分|分差teacher−student|压缩更低题数|','|---|---|---|---:|---:|---:|---:|---:|']
        for m,groups in er['results'].items():
            for group in ['teacher_gt0','teacher_perfect']:
                for task,v in groups[group]['by_task'].items():sections.append(f'|{m}|{group}|{task}|{v["questions"]}|{v["teacher_mean_percent"]:.3f}|{v["student_mean_percent"]:.3f}|{v["mean_loss_pp"]:+.3f}|{v["lower_questions"]}|')
        sections+=['','各模型各任务分数损失最大的32题（固定ID打破平局）保存为[gold/teacher/student对照CSV](../artifacts/tessfold_acceleration_20261115/error_analysis/largest_losses.csv)，用于错误分析，不用于选模型。']
    if (BASE/'qwen_timing/complete.json').exists():
        rows=[r for f in (BASE/'qwen_timing').glob('rows.rank*.jsonl') for r in lines(f)];grouped=collections.defaultdict(list)
        for r in rows:grouped[(r['model'],r['mode'])].append(r)
        sections+=['','## 8B完整部署时延','',
            '同90题，包含官方processor、原生vision与全部merger、P和decoder；新P位于merger后，没有节省视觉encoder/merger。TTFT在首个lm_head和argmax完成后计时。压缩+prompt包括P生成、四支路P@X、代表位置选择和embedding/scatter，不把router-only时间冒充总压缩。prefill_done另存JSON。','',
            '|模型|模式|E2E p50ms|TTFT p50ms|生成P p50ms|压缩+prompt p50ms|峰值GiB|','|---|---|---:|---:|---:|---:|---:|']
        for (m,mode),rr in sorted(grouped.items()):sections.append(f'|{m}|{mode}|{np.median([r["end_to_end_ms"] for r in rr]):.2f}|{np.median([r["ttft_ms"] for r in rr]):.2f}|{np.median([r["p_generator_ms"] for r in rr]):.2f}|{np.median([r["compressor_ms"] for r in rr]):.2f}|{max(r["peak_allocated_bytes"] for r in rr)/2**30:.4f}|')
    if (BASE/'qwen_timing/complete.json').exists():
        from scripts.tessfold_accel.qwen_timing_stats import compute as compute_timing
        timing_statistics=compute_timing()
        sections+=['','### 8B同题配对速度及分辨率分层','',
            '同题三次先平均，再计算参考/候选时延总和比；页面bootstrap10000次。short/medium/long沿用原固定90题分层；原生8B的视觉token数可能与UND不同。全量数值包含fixed64及post-front，见statistics.json。','',
            '|模型|分层|模式|E2E加速|E2E 95%页面CI|TTFT加速|TTFT 95%页面CI|','|---|---|---|---:|---|---:|---|']
        for v in timing_statistics['comparisons']:
            if v['mode']=='natural_eos' or v['stratum']=='all':
                ci=v['e2e_bootstrap_95_ci'];ti=v['ttft_bootstrap_95_ci']
                sections.append(f'|{v["model"]}|{v["stratum"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
    if (BASE/'qwen_timing/complete.json').exists():
        from scripts.tessfold_accel.boundary_stats import compute as boundary_stats
        boundary=boundary_stats();sections+=['','## 输入token规模与同边界post-tower加速','',
            '同90题的输入视觉token[min,p50,p90,max]：'+json.dumps(boundary['token_scales'],ensure_ascii=False)+'。原生8B经过官方merger后的输入token规模不同，不能只按LLM参数量外推速度。下表只比较REF/FAST：两者都在原始tower返回后、merger之前切分，包含各自merger至首token；不是端到端TTFT。S/DENSE不在此表混用边界。','',
            '|方法|保留率|模式|post-tower配对加速|95%页面CI|','|---|---:|---|---:|---|']
        for v in boundary['strict_und_post_tower_pairs']:
            ci=v['document_bootstrap_95_ci'];sections.append(f'|{v["method"]}|{v["ratio"]:.0%}|{v["mode"]}|{v["post_tower_ttft_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|')
    ceiling=load(BASE/'qwen_timing/amdahl_front_ceiling.json')
    if ceiling:
        sections+=['','### 8B未达预期加速的测量边界','',
            f'REF在同90题上的前端（processor+完整原生vision/mergers）均值{ceiling["reference_front_mean_ms"]:.2f}ms、TTFT均值{ceiling["reference_ttft_mean_ms"]:.2f}ms。保持这个前端不变，即使后端所有工作耗时降至零，理想TTFT上限也仅{ceiling["ideal_zero_postfront_ttft_upper_bound"]:.3f}倍。这是本工作负载的Amdahl上界，不是实测加速或更大模型的通用上界；不能从LLM参数量直接预言2.2–3.1倍。']
    if (BASE/'qwen_memory/complete.json').exists() and (BASE/'qwen_timing/complete.json').exists():
        memory=[r for p in (BASE/'qwen_memory').glob('rows.rank*.jsonl') for r in lines(p)]
        timing=[r for p in (BASE/'qwen_timing').glob('rows.rank*.jsonl') for r in lines(p) if r['mode']=='natural_eos']
        index={(r['id'],r['model'],r['repeat']):r for r in timing};assert len(index)==len(memory)==1350
        grouped=collections.defaultdict(list)
        for r in memory:
            assert r['generated_ids']==index[(r['id'],r['model'],r['repeat'])]['generated_ids'],'Independent memory pass changed outputs'
            assert sha(ROOT/r['trace'])==r['trace_sha256']
            grouped[r['model']].append(r)
        sections+=['','## 8B独立分阶段内存与逻辑视觉KV','',
            '1350次独立内存测量，90题×5模型×3次，natural cap256；逐题token必须与无采样时延pass相同。峰值来自allocator，均值来自1ms时间加权采样。原生encoder与merger合并为前端；downstream含visual setup、prefill和generation，所有数值都含常驻模型权重。逻辑视觉KV不能换算为相同比例的全设备峰值或吞吐。','',
            '|模型|全流程最大GiB|前端最大GiB|下游最大GiB|prefill采样均值GiB|逻辑视觉KV中位MiB|','|---|---:|---:|---:|---:|---:|']
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
        sections+=['','[注意力范围缺省的直接算子证据与双向修复](TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md)。']
    from scripts.tessfold_accel.training_diagnostics import main as training_diagnostics
    training_diagnostics()
    plots();sections+=['','## 曲线与参考','', '[逐epoch损失、梯度和生成分数诊断](TESSFOLD_8B_TRAINING_DIAGNOSTICS.md)。','',
        '纯压缩耗时图：[PDF](../artifacts/tessfold_acceleration_20261115/figures/compressor_wall.pdf)。各8B训练版本的逐epoch训练/验证KL、生成分数和逐step梯度范数曲线位于`artifacts/tessfold_acceleration_20261115/figures/*_training.pdf/png`。训练KL取该epoch实际批次均值，dev KL是冻结checkpoint完整56题的均值，二者不是相同数据集。','',
        '压缩矩阵与预测KL参考：[Wang等，Rethinking Token Reduction for Large Vision-Language Models，CVPR2026（MetaCompress）](../Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf)。本轮沿用prediction KL，但不是复现其P生成器。','',
        '原生8B架构参考：[Qwen3-VL-8B-Instruct官方模型及配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，固定revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。','',
        '本轮未修改原TeX/PDF稿件，不删除旧稠密P与峰值负结果。可采用的新论文表述必须区分严格稠密软读出、近似稀疏、独立hard变体、特征缓存与近似KV缓存，不把解析FLOPs或缓存字节比例写成实测吞吐。']
    path=ROOT/'docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md';path.write_text(path.read_text().split('\n\n## 已完成阶段的实测汇总')[0]+'\n\n## 已完成阶段的实测汇总\n\n'+'\n'.join(sections)+'\n')
    expected=['system_exact','interleaved','formal_equivalence','prefix_formal','multi_deployment','batched_serving','qwen_formal','qwen_timing','qwen_memory','hard_formal','bidirectional_formal','bidirectional_timing','bidir_interleaved']
    state={p:(BASE/p/'complete.json').exists() for p in expected};save(BASE/'final_report_status.json',dict(time=now(),stages=state,all_complete=all(state.values())))
    if a.require_complete:assert all(state.values()),state
    print(json.dumps(dict(report=str(path),stages=state)))

if __name__=='__main__':main()
