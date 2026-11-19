"""Evidence-only paper table/snippet; no guessed FLOPs, latency or cache gains.

Produces standalone LaTeX, never edits the historic manuscript PDFs. Strict
accuracy is inherited only AFTER full exact generation regression succeeds.
"""
import collections,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def analytic():
    cfg=load(ROOT/'models/MonkeyOCRv2-B-Und/config.json');d=cfg['hidden_size'];heads=cfg['num_attention_heads'];kvheads=cfg['num_key_value_heads'];head=cfg['head_dim'];ff=cfg['intermediate_size'];layers=cfg['num_hidden_layers'];v=cfg['vision_config']['embed_dim']
    linear=2*(d*heads*head+2*d*kvheads*head+heads*head*d+3*d*ff)
    def decoder(length,layers_):
        return layers_*(linear*length+2*length*(length+1)*d)
    def merger(n):return 2*n*(v*v+v*d)
    details=[]
    rows=lines(BASE/'system_exact/timing_rows.jsonl')
    times=[r for f in (BASE/'system_exact/single').glob('results.rank*.jsonl') for r in lines(f)]
    for row in rows:
        teacher=torch.load(ROOT/'artifacts/meta_compress_strict/benchmark_npu_cache/teachers'/f'{hashlib.sha256(row["id"].encode()).hexdigest()}.pt',map_location='cpu',weights_only=True)
        n=row['original_tokens'];text=teacher['input_ids'].shape[1]-n;reference=decoder(n+text,layers)+merger(n)
        for method in ['REF','C_FAST','T_FAST','T_HARD','R','S']:
            ratio=1. if method=='REF' else .25
            actual=[r['visual_tokens'] for r in times if r['id']==row['id'] and r['method']==method and r['ratio']==ratio];assert actual and len(set(actual))==1
            m=actual[0]
            value=merger(n)+decoder(n+text,1)+decoder(m+text,layers-1) if method=='S' else merger(m)+decoder(m+text,layers)
            details.append(dict(id=row['id'],method=method,n=n,m=m,text_tokens=text,merger_and_prefill_flops=value,ratio=value/reference))
    result=dict(time=now(),rows=details,formula=dict(decoder_per_layer_linear_flops_per_token=linear,
        attention='2*L*(L+1)*hidden for causal active-pair proxy; actual fused kernels may execute block padding',
        merger='2*N*(768^2+768*2048), linear layers only; ignores norms/GELU',
        exclusions='Processor, encoder, P generator/partition/readout, normalization/softmax/rotary, decode, hardwarecounter are NOT included.',
        type='Analytical operation-count proxy, never measured hardware FLOPs'))
    save(BASE/'paper/analytic_postvision_flops.json',result)
    return {m:float(np.median([r['ratio'] for r in details if r['method']==m])) for m in ['REF','C_FAST','T_FAST','T_HARD','R','S']}


def main():
    from scripts.tessfold_accel.system_common import verify
    verify();assert (BASE/'formal_equivalence/complete.json').exists() and (BASE/'system_exact/complete.json').exists() and (BASE/'multi_deployment/complete.json').exists()
    directory=BASE/'paper';directory.mkdir(exist_ok=True)
    formal=load(BASE/'formal_equivalence/complete.json');stats=load(BASE/'system_exact/single/summary.json')['rows'];memory=load(BASE/'system_exact/single/memory_summary.json')
    multi=load(BASE/'multi_deployment/summary.json')['rows']
    oldcol=load(ROOT/'artifacts/compression_lab_v2/full/full_col_adaptive_pre-merger_r0.25/benchmark/summary.json')
    accuracy={'REF':oldcol['macro']['teacher']}
    for method,kind in [('C_FAST','col_adaptive'),('T_FAST','quadtree_question')]:
        matches=[r for name,r in formal['results'].items() if kind in name and 'r0.25' in name];assert len(matches)==1;accuracy[method]=matches[0]['macro']
    hard=load(BASE/'hard_formal/complete.json')
    accuracy['T_HARD']=hard.get('macro')
    for m in ['R','S']:accuracy[m]=load(ROOT/f'artifacts/compression_supplement_20261111/runs/{m}_r0.25/benchmark/summary.json')['variants']['trained']['macro']
    # Verify unchanged R/S natural outputs on every90timingQA before combining
    # historical accuracy and fresh profiling. Hard uses its own full eval.
    cohort={r['id'] for r in lines(BASE/'system_exact/timing_rows.jsonl')}
    alltimes=[r for f in (BASE/'system_exact/single').glob('results.rank*.jsonl') for r in lines(f)]
    for method in ['R','S']:
        previous={}
        for p in (ROOT/f'artifacts/compression_supplement_20261111/runs/{method}_r0.25/benchmark').glob('predictions.rank*.jsonl'):
            with p.open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['id'] in cohort:previous[r['id']]=r['student_ids']
        assert len(previous)==90
        for r in alltimes:
            if r['method']==method and r['ratio']==.25 and r['mode']=='natural_eos':assert r['generated_ids']==previous[r['id']],f'{method} natural output changed'
    flops=analytic();rows=[]
    for method in ['REF','R','S','T_FAST','T_HARD','C_FAST']:
        ratio=1. if method=='REF' else .25
        time_=next(r for r in stats if r['method']==method and r['ratio']==ratio and r['mode']=='natural_eos')
        mem=next(r for r in memory if r['method']==method and r['ratio']==ratio)
        mm={'REF':'REF','T_FAST':'Tree','C_FAST':'Column'}.get(method)
        cache=next((r for r in multi if r['method']==mm and r['scenario']=='feature_text_only' and r['k']==5),None) if mm else None
        rows.append(dict(method=method,accuracy=accuracy[method],analytic_flops_ratio=flops[method],compress_and_merger_ms=None if method=='S' else time_['phase_wall_medians_ms']['compression_and_merger'],
            post_vision_ttft_p50_ms=time_['post_encoder_ttft_p50_ms'],e2e_ttft_p50_ms=time_['ttft_p50_ms'],downstream_peak_gib=mem['decoder_peak_max_gib'],
            visual_kv_mib=time_['visual_kv_mib'],k5_feature_text_only_mean_ms=cache['mean_ms'] if cache else None))
    save(directory/'table.json',dict(time=now(),rows=rows,limitations='Strict C/T fully regressed. R/S frozen historical accuracy with90QA fresh natural-output gate. Hard accuracy only its new full eval. Feature cache includes same privileges forREF. FLOPs analytic, actual KV forS unavailable.'))
    tex=[r'\begin{table*}[t]',r'\centering',r'\small',
        r'\caption{Measured system efficiency at the 25\% target visual budget on Ascend. Accuracy uses the fixed shared eight-task protocol (field EM for KLC/DeepForm), seed 20261105; it is not an official corpus-F1 leaderboard score. FLOPs are analytical merger+causal-prefill operation counts, excluding the encoder, compressor, and decoding. Compression+merger is a combined measured stage; S pruning occurs inside the decoder and is not independently separated. Post-tower excludes the original full merger for S but includes compressed merger for FAST/REF. KV bytes are logical visual positions, not total device memory. K=5 amortization includes first ingestion and fair feature caching for all methods; no approximate prefix KV is used in this column.}',
        r'\begin{tabular}{lrrrrrrr}',r'\toprule',
        r'Method & Macro & FLOPs ratio & Comp.+merger(ms) & Post-tower TTFT(ms) & Decoder peak(GiB) & Visual KV(MiB) & K=5(ms/q) \\',r'\midrule']
    def fmt(v):return '--' if v is None else f'{v:.2f}'
    for r in rows:
        tex.append(r['method'].replace('_',r'\_')+' & '+' & '.join([fmt(r['accuracy']),f'{r["analytic_flops_ratio"]:.3f}',fmt(r['compress_and_merger_ms']),fmt(r['post_vision_ttft_p50_ms']),fmt(r['downstream_peak_gib']),fmt(r['visual_kv_mib']),fmt(r['k5_feature_text_only_mean_ms'])])+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}'];(directory/'measured_efficiency_table.tex').write_text('\n'.join(tex)+'\n')
    paragraph=r'''\paragraph{Exact refactoring and deployment boundaries.}
We remove redundant full-length alignment, cache deterministic geometry, compile
the original float32 partitioning with NumPy-compatible summation rules, and
sort exactly representable representative indices in float32 to avoid an integer
sort CPU fallback. The strict soft path retains the original float32 masked
softmax and dense GEMM: segmented reductions are evaluated as numerical variants,
not asserted to be bit-exact. All 23,565 questions at four method/budget settings
match their original compression matrices, features, and generated tokens.
Feature reuse across independent questions additionally avoids repeated image
preprocessing while retaining the original expanded text prompt. The uncompressed
reference receives the same reuse rights. Prefix-KV reuse is reported separately
because split-prefill arithmetic can change token predictions; no answer-history
KV is reused across questions. The unchanged visual encoder still bounds the
cold-request peak memory, and visual KV reduction alone does not establish a
fourfold concurrent serving throughput gain.
'''
    (directory/'measured_efficiency_paragraph.tex').write_text(paragraph)
    (ROOT/'docs/TESSFOLD_PAPER_EFFICIENCY_TABLE.md').write_text('# Measured Efficiency Tables and English Manuscript Text\n\n**Entry point**: `.venv/bin/python -m scripts.tessfold_accel.paper_tables`. \n\n[LaTeX tables](../artifacts/tessfold_acceleration_20261115/paper/measured_efficiency_table.tex) · [English manuscript text](../artifacts/tessfold_acceleration_20261115/paper/measured_efficiency_paragraph.tex) · [Raw values and measurement definitions](../artifacts/tessfold_acceleration_20261115/paper/table.json). \n\nTables combine only regression-validated implementations and available full-task results, not expected values. Primary experiments use strict soft aggregation; Hard uses its own complete formal scores. Missing independent measurements are shown as --. Old DENSE/S post-tower timing includes the full-length merger and is not the same pure-ViT boundary as FAST/REF. Label those columns post-tower or independently measure merger costs before manuscript use. No original TeX manuscript was available; historical PDFs and negative results remain unchanged.\n')
    print(json.dumps(dict(table=str(directory/'measured_efficiency_table.tex'),rows=len(rows))))

if __name__=='__main__':main()
