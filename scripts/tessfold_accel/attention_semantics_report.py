"""Measured operator semantics and corrected vision pilot, standalone report."""
from scripts.tessfold_repair.common import ROOT,load,save,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    r=load(BASE/'attention_semantics.json');assert len(r['rows'])==8
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=BASE/'figures';directory.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
    for ax,dtype in zip(axs,['torch.bfloat16','torch.float16']):
        rows=[x for x in r['rows'] if x['dtype']==dtype]
        for key,label in [('default_mask','Default next=0 + mask'),('explicit_bidirectional','Explicit bidirectional + mask'),('no_mask','No mask')]:
            ax.plot([x['length'] for x in rows],[x['variants'][key]['full_rms'] for x in rows],marker='o',label=label)
        ax.set(xlabel='Sequence length',ylabel='RMS error vs manual full attention',yscale='log',title=dtype.replace('torch.',''));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(directory/f'attention_window_semantics.{ext}',dpi=180)
    plt.close(fig)
    old=load(BASE/'fused_vision_pilot/complete.json');new=load(BASE/'bidirectional_vision_pilot/complete.json')
    doc=['# Attention-Range Defaults in Ascend Fused Vision: Operator Evidence and Repair Validation','',f'UTC: {now()}. ','',
        '**Entry point**: `sbatch scripts/tessfold_accel/attention_semantics.sh`; this report: `.venv/bin/python -m scripts.tessfold_accel.attention_semantics_report`. ','',
        '[Raw operator numerics](../artifacts/tessfold_acceleration_20261115/attention_semantics.json) · [Independent model pilot](../artifacts/tessfold_acceleration_20261115/bidirectional_vision_pilot/complete.json) · [Error-plot PDF](../artifacts/tessfold_acceleration_20261115/figures/attention_window_semantics.pdf). ','',
        '## Problem Diagnosis','',
        'UND visual attention is bidirectional within each image: patches may read earlier and later patches. The official Ascend branch calls `npu_prompt_flash_attention`supplies an image-segmentation mask without explicitly specifying `next_tokens`. The installed default is 0. Documentation states that sparse_mode=0 ignores pre/next **without a mask**, but uses those bounds for sparse computation **with a mask**. Inconsistent bounds/masks cannot be assumed to produce full bidirectional attention.','',
        'The original-call/eager difference is therefore not entirely rounding. Official files/weights remain unchanged; the separate candidate explicitly sets `pre_tokens=next_tokens=2147483647`, allowing every same-image position while retaining cross-image masks.','',
        '## Direct Verification','',
        'Fixed random Q/K/V: 12 heads, head_dim=64, lengths 128/257/513/1024 in BF16 and FP16. Manually compute QK, scaling, FP32 softmax cast back to the original dtype, and AV, constructing full bidirectional, strictly lower-triangular, and block-causal references at block sizes 64/128/256. Apply default-mask, explicit-bidirectional-mask, and unmasked calls to identical Q/K/V. This directly measures output RMS differences without model weights or task answers.','',
        'The first diagnostic assertion equating the default call with strict causality failed; logs/sources remain. Broader references show kernel-block-dependent behavior, not universal tokenwise causality.','',
        '|dtype|Length|Default-mask RMS relative to full attention|Explicit-bidirectional RMS relative to full attention|Unmasked RMS relative to full attention|','|---|---:|---:|---:|---:|']
    for row in r['rows']:
        v=row['variants'];doc.append(f'|{row["dtype"]}|{row["length"]}|{v["default_mask"]["full_rms"]:.7f}|{v["explicit_bidirectional"]["full_rms"]:.7f}|{v["no_mask"]["full_rms"]:.7f}|')
    doc+=['','At length 128, all three calls are close, so a small sanity check can miss the issue. At length 1024, the default call resembles the block-causal reference with block size 128, while explicit bidirectional and unmasked calls resemble full attention. In BF16, default-to-full RMS is about 0.066886, default-to-block128 about 0.000396, and explicit-bidirectional-to-full about 0.000253. Behavior at nonmultiple lengths 257/513 is not captured by a fixed block128 causal rule; the complete hardware tiling behavior has not been derived.','',
        'In this Ascend/CANN/torch_npu combination and call pattern, default bounds affect allowed attention blocks; explicit bidirectional bounds repair the main semantic difference. This does not generalize to every device/version or the authors\' training implementation. Even with corrected bounds, fused/eager FP16/BF16 rounding differs; the paths are not bit-exact.','',
        '## Impact on the Actual Frozen UND Backbone','',
        'Fix 56 QA and all encoder/merger/LLM/P parameters. Compare original-author and explicit-bidirectional calls separately to eager; results do not adjust parameters/questions.','',
        '|Call|Median encoder speedup|Change in REF mean per-question score (pp)|REF generation changes|Passes original quality/speed criterion|','|---|---:|---:|---:|---|',
        f'|Original author implementation|{old["median_encoder_speedup"]:.3f}×|{old["ref_mean_qa_delta_pp"]:+.4f}|{old["token_changes"]["REF"]}/56|False|',
        f'|Explicit bidirectional|{new["median_encoder_speedup"]:.3f}×|{new["ref_mean_qa_delta_pp"]:+.4f}|{new["token_changes"]["REF"]}/56|{new["expansion_gate_passed"]}|','',
        'The original call has median final-visual-feature cosine ≈0.9515. Explicit bidirectional attention on 56 questions yields ≈0.999962, but is not bit-exact; median/max absolute errors are 0.375 / 1.1015625. The gate requires median encoder speedup ≥1.5× and REF development-score loss ≤2 pp. Only after passing does evaluation expand to 23,565 QA × REF and four compressors, plus 2,700 timing / 450 memory measurements. The 56-question trend is not final accuracy before formal evaluation completes.','',
        '## Scope of the TessFold Conclusions','',
        'The strict eager experiment remains unchanged, with all 94,260 old/new compression regression records matching. Explicit bidirectional fusion is a separate numerical deployment candidate; all task scores are regenerated. REF and students use the same bidirectional frontend when measuring compression benefit, avoiding attribution of encoder fusion speed to P. Comparing complete eager and bidirectional deployments requires both quality changes and full cold-request timings.','',
        'Reference: local torch_npu  `_op_plugin_docs.py`and codegen schema paths, SHA-256, and source excerpts: `bidirectional_attention_hypothesis.json`; original official Ascend forward: `models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2_vision.py`, unchanged in this round.']
    (ROOT/'docs/TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md').write_text('\n'.join(doc)+'\n')
    print('docs/TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md')

if __name__=='__main__':main()
