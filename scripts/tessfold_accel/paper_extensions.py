"""Standalone paper tables/figures for measured8B and bidirectional variants."""
import collections,json
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def tex_table(path,caption,columns,header,rows):
    text=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{'+caption+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',
        ' & '.join(header)+r' \\',r'\midrule']
    text+=[' & '.join(row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    path.write_text('\n'.join(text)+'\n')


def label(model):
    return {'REF':'Native REF','Column_r0.25':r'Column 25\%','Tree_r0.25':r'Tree 25\%',
        'Tree_r0.5':r'Tree 50\%','Column_r0.5':r'StableColumn 50\%'}.get(model,model.replace('_',r'\_'))


def main():
    directory=BASE/'paper';directory.mkdir(exist_ok=True)
    native=load(BASE/'qwen_formal/complete.json');assert native and load(BASE/'final_audit.json')['all_checks_passed']
    tasks=['DocVQA','InfoVQA','ChartQA','OCRBench','DeepForm','KLC','WTQ','DT-VQA']
    methods=['REF','Column_r0.25','Tree_r0.25','Column_r0.5','Tree_r0.5']
    quality=[]
    for method in methods:
        score=native['native_metric_results'][method]
        quality.append([label(method)]+[f'{score["by_task"][t]:.2f}' for t in tasks]+[f'{score["macro"]:.2f}'])
    tex_table(directory/'native8b_quality.tex',
        r'Frozen native Qwen3-VL-8B with trained spatial token compression, evaluated on the fixed 23,565-question eight-task split. DocVQA/InfoVQA use ANLS, ChartQA relaxed accuracy, DT-VQA accANLS, WTQ denotation accuracy, OCRBench its native score, and KLC/DeepForm DUE corpus F1. Macro weights tasks equally. All base weights remain frozen. The same compression matrix acts on final visual and three DeepStack branches after native mergers. Models are selected using the disjoint fixed development set; StableColumn is the explicit fixed-detail-allocation variant.',
        'l'+'r'*9,['Model']+tasks+['Macro'],quality)
    controls=native['initial_control_results']['comparisons'];control_rows=[]
    for model in methods[1:]:
        v=controls[model];ci=v['shared_fieldEM_task_stratified_document_bootstrap_95_ci']
        control_rows.append([label(model),f'{v["initial"]["native_macro"]:.2f}',f'{v["selected"]["native_macro"]:.2f}',f'{v["native_macro_delta"]:+.2f}',
            f'{v["shared_fieldEM_macro_delta"]:+.2f}',f'[{ci[0]:+.2f},{ci[1]:+.2f}]'])
    tex_table(directory/'native8b_initial_controls.tex',
        r'Independent matched before/after control on 512 preregistered document clusters (2,404 questions), with all questions retained for each selected document. Initial and development-selected routers use the identical subset. Native macro uses corpus F1 for KLC/DeepForm. The separate shared field-EM macro replaces those two metrics with field EM; its 95\% confidence interval uses 10,000 task-stratified document bootstrap samples. This control subset is never used to select models and is not the full-test macro.',
        'lrrrrr',['Model','Initial native','Trained native',r'$\Delta$ native',r'$\Delta$ shared EM',r'Shared EM 95\% CI'],control_rows)
    timing=[r for p in (BASE/'qwen_timing').glob('rows.rank*.jsonl') for r in lines(p) if r['mode']=='natural_eos']
    stats=load(BASE/'qwen_timing/statistics.json');memory=load(BASE/'qwen_memory/summary.json')['results'];deployment=[]
    for method in methods:
        rr=[r for r in timing if r['model']==method]
        speed=next((v for v in stats['comparisons'] if v['model']==method and v['mode']=='natural_eos' and v['stratum']=='all'),None)
        ci=speed['ttft_bootstrap_95_ci'] if speed else [1.,1.]
        deployment.append([label(method),f'{np.median([r["ttft_ms"] for r in rr]):.2f}',f'{np.median([r["end_to_end_ms"] for r in rr]):.2f}',
            f'{speed["ttft_speedup"] if speed else 1.:.3f}',f'[{ci[0]:.3f},{ci[1]:.3f}]',f'{memory[method]["full_max_gib"]:.3f}',f'{memory[method]["visual_kv_median_mib"]:.2f}'])
    tex_table(directory/'native8b_deployment.tex',
        r'Native8B deployment on the same 90 requests with natural EOS and a 256-token cap, three repeats, including processor, full native vision/mergers, compression and decoding. TTFT ends after the first LM head and token selection. Displayed latencies are medians; paired TTFT speedups use means per request and document-bootstrap intervals, not ratios of medians. Peak memory comes from 1,350 independent 1-ms sampled passes, whose generated tokens match the corresponding uninstrumented timing. Logical visual KV is not total device memory. The unchanged native front end limits achievable cold-request acceleration.',
        'lrrrrrr',['Model','TTFT p50 ms','E2E p50 ms',r'TTFT speedup',r'95\% CI','Full peak GiB','Visual KV MiB'],deployment)
    bidir=load(BASE/'bidirectional_formal/complete.json');paired=load(BASE/'bidir_interleaved/statistics.json')
    old=load(BASE/'formal_equivalence/complete.json')['results'];oldteacher=load(ROOT/'artifacts/compression_lab_v2/full/full_col_adaptive_pre-merger_r0.25/benchmark/summary.json')['macro']['teacher']
    models={'REF':'REF','C_FAST':'full_col_adaptive_pre-merger_r0.25','T_FAST':'full_quadtree_question_pre-merger_r0.25'}
    inter_gate=load(BASE/'bidir_interleaved/complete.json')['prior_token_mismatches'];safe=not any(inter_gate.values())
    variant=[]
    for backend in ['EAGER','BIDIR']:
      for method in ['REF','C_FAST','T_FAST']:
        m=models[method];phase=next(v for v in paired['natural_phase_means'] if v['backend']==backend and v['method']==method)
        score=(oldteacher if method=='REF' else old[m]['macro']) if backend=='EAGER' else bidir['shared_fieldEM_results'][m]['macro']
        # Context-dependent generation changes prohibit inheriting full scores
        # for this particular timing workload; retain '--' instead.
        variant.append([backend+'/'+method.replace('_',r'\_'),f'{score:.2f}' if safe else '--',f'{phase["encoder_mean_ms"]:.2f}',f'{phase["prefill_mean_ms"]:.2f}',f'{phase["ttft_mean_ms"]:.2f}',f'{phase["e2e_mean_ms"]:.2f}'])
    tex_table(directory/'und_bidirectional_interleaved.tex',
        r'Same-question randomized comparison of eager and explicit bidirectional PFA deployment on frozen UND at the 25\% compression budget. Each request/mode/repeat randomizes six configurations; 90 questions, natural EOS and fixed64, two repeats (2,160 observations). This table shows natural-EOS means. Full eight-task quality uses the shared field-EM protocol and comes from independent full evaluations of each numerical variant; full quality is omitted if the interleaved outputs fail their prior-output checks. EAGER and BIDIR are not bit-exact alternatives. All encoder costs remain included and both references receive the same backend privileges.',
        'lrrrrr',['Backend/model','Full macro','Encoder mean ms','Prefill mean ms','TTFT mean ms','E2E mean ms'],variant)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=BASE/'figures';fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    values=[native['native_metric_results'][m]['macro'] for m in methods]
    names=['REF','Column25','Tree25','StableCol50','Tree50']
    axs[0].bar(names,values,color=['#6b7280','#2563eb','#059669','#60a5fa','#34d399']);axs[0].set(ylabel='Full-test native macro (%)',ylim=(0,100));axs[0].tick_params(axis='x',rotation=22)
    for i,v in enumerate(values):axs[0].text(i,v+.7,f'{v:.1f}',ha='center',fontsize=8)
    xx=np.arange(4);axs[1].bar(xx-.18,[controls[m]['initial']['native_macro'] for m in methods[1:]],width=.36,label='Initial')
    axs[1].bar(xx+.18,[controls[m]['selected']['native_macro'] for m in methods[1:]],width=.36,label='Trained')
    axs[1].set(xticks=xx,xticklabels=names[1:],ylabel='Matched 2,404-QA native macro (%)',ylim=(0,100));axs[1].tick_params(axis='x',rotation=22);axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(figures/f'native8b_quality_and_controls.{ext}',dpi=180)
    plt.close(fig)
    save(directory/'extension_table_data.json',dict(time=now(),native_quality_rows=quality,native_controls_rows=control_rows,native_deployment_rows=deployment,
        und_variant_rows=variant,interleaved_prior_outputs_all_equal=safe,full_native_metrics='DUE-F1 version',und_metrics='Shared-fieldEM version'))
    (ROOT/'docs/TESSFOLD_PAPER_EXTENSION_TABLES.md').write_text('# Measured Manuscript Tables for Native 8B and Bidirectional Fusion\n\n**Entry point**: `.venv/bin/python -m scripts.tessfold_accel.paper_extensions`, requiring all final audits to pass.\n\n- [Full native 8B task scores](../artifacts/tessfold_acceleration_20261115/paper/native8b_quality.tex)\n- [Before/after comparison on 2,404 fixed questions](../artifacts/tessfold_acceleration_20261115/paper/native8b_initial_controls.tex)\n- [Full 8B deployment and independent memory](../artifacts/tessfold_acceleration_20261115/paper/native8b_deployment.tex)\n- [Same-question interleaved UND eager/bidirectional comparison](../artifacts/tessfold_acceleration_20261115/paper/und_bidirectional_interleaved.tex)\n- [Full-evaluation and independent-control figure PDF](../artifacts/tessfold_acceleration_20261115/figures/native8b_quality_and_controls.pdf)\n\nNative-8B full-test/subset scores remain separate, as do UND shared field-EM and native DUE-F1 metrics. Bidirectional fusion is a separate numerical variant; earlier strict results/PDFs remain.\n')
    print(json.dumps(dict(directory=str(directory),tables=4,figure='native8b_quality_and_controls.pdf')))

if __name__=='__main__':main()
