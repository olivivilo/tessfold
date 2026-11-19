"""Evidence-driven live report for compilation, prefix and native8B extension."""
import collections,json,subprocess
from pathlib import Path
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,save,now
from scripts.tessfold_accel.live_io import live_lines as lines
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'

def main():
    queue=subprocess.run(['squeue','-h','-u','jchen524','-o','%i|%j|%T|%M|%R'],capture_output=True,text=True).stdout.strip()
    doc=['# TessFold Round Two: Acceleration, KV Repair, and Native 8B Compression','',f'Updated (UTC):{now()}. ','',
        '**Main entry point**: `.venv/bin/python -m scripts.tessfold_accel.progress`. This report reconstructs actual logs and completion markers; submitted/running is not completed. Original 20261114 and repaired 20261115 experiments, official UND, and generator weights remain.','',
        '## Targets and Version Boundaries','',
        'Continue tessfold.md by reducing compression overhead, validating visual-prefix reuse, expanding numerical/task checks, and training compressors on the newer, similarly sized Qwen3-VL-8B extension. UND execution optimization remains training-free. Native 8B uses its own visual encoder/alignment, freezes the complete backbone, and trains only the new P generator. Scores from these two routes are not mixed.','',
        '## Compiled Partitioning and Compressor-Only Overhead','',
        'The original implementation regenerates/integer-sorts grid indices (CPU fallback on Ascend), partitions in Python loops, and constructs dense P. The new strict path caches official 2×2 block geometry. Its C++ implementation retains the FP32 integral image, NumPy pairwise summation, and cut/tie ordering. Unique representative indices below 2^24 are exactly sorted as FP32 to avoid int64 fallback. Original FP32 softmax/GEMM remain; fast-math and FMA reassociation are disabled.','']
    cpu=load(BASE/'native/cpu_partition_tests.json');doc+=[f'Independent CPU comparison against original NumPy: {cpu.get("cases",0)} cases; partition mismatches {cpu.get("mismatches","Incomplete")}; includes near-uniform, lognormal, and single-row/column cases. Entry: `.venv/bin/python -m scripts.test_tessfold_native_partition`. ','']
    for label in ['pilot1','validation1']:
        rr=[r for f in (BASE/label).glob('rows.rank*.jsonl') for r in lines(f)]
        if not rr:continue
        doc+=[f'### {label}','',f'Submitted{len(rr)} cases; completion marker: {(BASE/label/"complete.json").exists()}. ','',
            '|Path|Features are not bit-exact|Generated-token changes|Median compressor wall time (ms)|','|---|---:|---:|---:|']
        for b in ['dense_exact','bucket_fp32','bucket_fold','hard_linear']:
            a=[r['variants'][b] for r in rr]
            doc.append(f'|{b}|{sum(not x["feature_equal"] for x in a)}|{sum(not x["generation_equal"] for x in a)}|{np.median([x["compressor_wall_ms"] for x in a]):.4f}|')
        doc+=['','Pure wall time repeatedly calls the entire warmed compressor, including host transfer, CPU partitioning, and NPU aggregation but excluding the merger. Synchronized stage times are reported separately; their sum is not presented as unperturbed total latency. The bucket candidate has O(N) support but changes reduction rounding and currently incurs high scheduling overhead, so it is not promoted to the strict path. Hard is a separate model variant.','']
    doc+=['### Graph Replay and Truly Linear-Aggregation Candidates','',
        'Capture scoring/readout in an NPU graph while retaining exact CPU partitioning between them. Record compilation/capture and resident memory per shape separately; do not hide startup costs. Default efficiency uses the compiled strict path without graph capture.','',
        '|Experiment|Cases|Completed|Median compressor wall time (ms)|Minimum / maximum (ms)|','|---|---:|---|---:|---|']
    for label in ['graph_full1','graph_full2','sparse2','packed1','host_pool1','hard_graph1']:
        rr=[r for f in (BASE/label).glob('rows.rank*.json') for r in load(f,[])]
        if not rr:continue
        v=[r['pure_compressor_ms'] for r in rr]
        doc.append(f'|{label}|{len(rr)}|{(BASE/label/"complete.json").exists()}|{np.median(v):.4f}|{min(v):.4f}/{max(v):.4f}|')
    doc+=['','sparse1 produced nonfinite values and no success marker; its failure logs remain. After retaining captured constants and changing the reduction call, sparse2 independently checked finiteness, repeated execution, and scores. Some index_add reductions were nondeterministic, so it was not promoted as bit-exact. packed1 uses deterministic locally padded GEMM; host_pool1 uses CPU SIMD with both transfer directions counted. Neither improves overall performance over the strict path; both remain documented negative results.','',
        '## Full Efficiency Remeasurement and Full-Set Equivalence','']
    n=sum(len(lines(p)) for p in (BASE/'system_exact/single').glob('results.rank*.jsonl'))
    formal=sum(len(lines(p)) for p in (BASE/'formal_equivalence').glob('rows.rank*.jsonl'))
    doc += [f'Full compiled-strict efficiency: {n}/9450 records, completed and audited: {(BASE/"system_exact/complete.json").exists()}. Original 90 QA / 15 configurations, natural/fixed64 × 3 repeats, plus 1,350 memory records; questions and weights unchanged. Entry point: `sbatch scripts/tessfold_accel/system.sh`; after completion `.venv/bin/python -m scripts.tessfold_accel.system_report`. ','',
        f'Full formal equivalence: {formal}/94260 records (23,565 QA × Column/Tree × 25%/50%). P, PH, and generated tokens are checked against original full results case by case; failure never relaxes eligibility automatically. Entry point: `sbatch scripts/tessfold_accel/formal.sh`. ','',
        '## Visual KV: Diagnosis, Repair, and Limitations','',
        'The original prefix stopped at the final image placeholder. Extending it through official vision_end/following separators restores the known KLC/REF case, but Tree/DeepForm still differs. Layerwise KV comparisons can first differ at different layers; matrix shapes and attention backends alter rounding, and eager does not guarantee native SDPA output equality. This is not contamination by previous-answer KV.','',
        'The new fork copies DynamicCache layer objects while sharing read-only prefix tensors. DynamicLayer.update concatenates into new storage; each question verifies unchanged base-cache length, pointer, and version. This avoids deepcopying the full prefix. It supports standard DynamicLayer only, not a paged-KV serving framework. Feature-cached questions expand image placeholders exactly from the original template without rereading images or rerunning the processor.','',
        '|Scenario|Questions|Completed|Generated-token changes|Per-question score changes and |','|---|---:|---|---:|---:|']
    for name in ['REF','Column']:
        p=BASE/'prefix_pilot1'/name;rr=[r for f in p.glob('rows.rank*.jsonl') for r in lines(f)]
        if rr:doc.append(f'|{name}|{len(rr)}|{(p/"complete.json").exists()}|{sum(not r["tokens_equal"] for r in rr)}|{sum(r["cached_score"]-r["reference_score"] for r in rr):.4f}|')
    doc+=['','prefix_pilot1 contains 24 actual pages with 10 questions each and two historical failure pages, not all eight tasks. Output differences prohibit a strict lossless-KV claim and require separate full-task impact evaluation with fair REF/Column comparison. Tree cannot reuse question-dependent visual prefixes across different questions. Both prefix boundaries are causally valid; extending the boundary changes kernel shapes and fixes a known case, but does not establish cross-question contamination at the original boundary.','',
        '## Qwen3-VL-8B Compression Architecture','',
        'Earlier checks used only uncompressed native inference on a small validation set; this round adds trainable P after native visual mergers and before the text decoder. Final 4,096-dimensional features and 3 DeepStack feature branches share P. M=round(rN), with nonnegative unit-sum rows and disjoint regions. Preserve highest-weight representatives\' original three-dimensional MRoPE coordinates and unchanged question/answer positions. NativeSpatialRouter uses 4,096-dimensional inputs, LayerNorm→Linear(4096,96)→depthwise/pointwise spatial convolution→scoring. Tree additionally uses 4,096-dimensional question embeddings with a trainable gate; Column ignores questions.','',
        'This differs substantially from UND\'s 768-dimensional pre-merger path. Native 8B saves no visual encoder/native-merger work, and old UND alignment is not assumed transferable. The identity path is checked against official HF tokens on 8 questions from the native 56-question set. KL gradients are tested at two budgets/two architectures; all backbone parameter gradients remain absent.','',
        '### Training Protocol and Pilot Eligibility Criteria','',
        'Training images and the fixed 56-question development set are document-disjoint, using the original 20k pool. The 512-question pilot prefers one question/image and balances tasks. Teacher is the same frozen 8B model generating short greedy answers (64 tokens). Full-vocabulary KL uses shared teacher prefixes, not gold labels. AdamW, weight_decay=0, clip=5, 10% warmup then cosine to 0.1 LR; log pre-clipping norms/multipliers and save optimizer state. Defaults: batch=32, 3 epochs, LR=3e-4; failed architectures additionally try 1e-4 and 1e-3.','',
        'Expansion requires a single pilot checkpoint satisfying development KL≤0.95×initial and generation macro gain ≥0.25 points (allowing teacher−0.25 when already near teacher). Selection uses fixed development data only, never formal tests. Full training uses original 20k, 4 epochs, the passing LR/architecture, and a frozen backbone.','',
        '|Training version|Architecture / budget|Training-log steps|Completed|Initial development KL / score|Final development KL / score|Pilot eligibility criterion|','|---|---|---:|---|---|---|---|']
    for label in ['qwen_pilot1','qwen_pilot_low','qwen_pilot_high','qwen_pilot_stable','qwen_full_pass1','qwen_full_tree25','qwen_full_stable50']:
        for out in sorted((BASE/label).glob('*_r*')):
            init=load(out/'initial.summary.json');end=load(out/'complete.json');epochs=sorted(out.glob('epoch*.summary.json'),key=lambda p:int(p.name.split('.')[0][5:]))
            last=load(epochs[-1]) if epochs else {}
            def val(r):return f'{r["kl"]:.4f}/{r["macro"]:.3f}' if r else 'Not completed'
            doc.append(f'|{label}|{out.name}|{len(lines(out/"training.jsonl"))}|{bool(end)}|{val(init)}|{val(last)}|{end.get("pilot_gate_passed","Pending evaluation")}|')
    dataset=load(BASE/'dataset_audit.json')
    if dataset:
        doc+=['','### Actual Dataset Size','', '|Split|QA|Images|Task distribution|','|---|---:|---:|---|']
        for name,r in dataset['splits'].items():doc.append(f'|{name}|{r["questions"]}|{r["images"]}|{json.dumps(r["by_task"],ensure_ascii=False)}|')
        doc+=['','Train/validation/formal document-ID intersections are zero. [File signatures and audit](../artifacts/tessfold_acceleration_20261115/dataset_audit.json). ']
    doc+=['','StableColumn is a separate architectural test: support allocation depends only on image detail and is not perturbed by learned scores; local weights and temperature remain trainable. It diagnoses detached split instability, not a lossless optimization of the original rule. Full 20k training requires passing its pilot.','',
        'Native-8B seven-task development uses DUE corpus F1 for DeepForm/KLC; the old shared UND full benchmark uses field EM, so their macros cannot be subtracted directly. Final 8B formal reporting must include metric definitions and uncompressed teacher; small-validation scores are not formal compression results.','',
        'Microbatch training is about 1.46–1.50× faster, but a fixed-pair KL difference .050002 exceeds the .05 gate. It is not used formally; the gate is not relaxed for speed, and failure records remain.','']
    ceiling=load(BASE/'system_exact/single/amdahl_front_ceiling.json')
    if ceiling:
        v=ceiling['natural_eos'];doc += ['## Frontend Bottlenecks and the Author-Provided Fused-Attention Candidate','',f'On these 90 cold requests, REF frontend (processor + complete encoder) mean {v["reference_front_mean_ms"]:.2f} ms; mean TTFT {v["reference_ttft_mean_ms"]:.2f} ms; with frontend unchanged and all backend work free, ideal TTFT speedup is at most {v["ideal_zero_postfront_ttft_upper_bound"]:.3f}×. Amdahl bound for this limited workload, not measured speedup or a device-independent conclusion.','']
    fused=load(BASE/'fused_vision_pilot/complete.json')
    doc+=['A separate candidate uses the same frozen vision weights and temporarily binds the author\'s existing VisionAscendAttention.forward (npu_prompt_flash_attention) in place of deployed eager V2, without editing official sources. The call omits next_tokens; subsequent interface documentation and bidirectional tests require checking its attention extent rather than assuming rounding-only changes. Errors, generations, and scores are reported independently and cannot inherit bit-exact claims. Expansion requires median encoder speedup ≥1.5× and REF development loss ≤2 points.','',
        f'Fused-attention pilot: `{json.dumps(fused,ensure_ascii=False) if fused else "Not completed"}`. ','',
        '## Additional Controls in the Final Validation','',
        'The 8B primary table contains 23,565 questions × REF plus four dev-selected compressors: 117,825 records. Four initial P models are additionally evaluated on a hash-fixed 512-image/2,404-question subset: 9,616 records for matched before/after generalization checks, giving 127,441 formal records in total. This subset is not mixed into the primary table. Another 1,350 independent 1 ms memory measurements are separate from 2,700 timing records and must match natural-timing tokens exactly. The 15 pilot technical checks do not select models.','',
        '[Final validation protocol](TESSFOLD_FINAL_VALIDATION_PROTOCOL.md) · [Target-by-target comparison against tessfold.md](TESSFOLD_TARGET_GAP.md). ','',
        '## Current Jobs','', '```text',queue or 'Queue empty','```','',
        '## Claims Not Supported for the Manuscript','',
        'Before measurement, do not claim submillisecond full compression, lossless O(N) soft replacement, necessarily lossless KV reuse, fourfold concurrency, or inevitable 2.2–3.1× 8B speedup. Full-request peaks retain encoder occupancy; analytical FLOPs remain estimates. New results require stage completion/audits.']
    if (BASE/'all_stages_complete.json').exists():
        doc[1:1]=['','**All 13 stages and the final audit are complete.** [Final-results overview](TESSFOLD_FINAL_RESULTS.md) · [Target gaps](TESSFOLD_TARGET_GAP.md) · [8B/fusion manuscript tables](TESSFOLD_PAPER_EXTENSION_TABLES.md). All conclusions below use completed original records.']
    report=ROOT/'docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md'
    marker='\n\n## Measurements for Completed Stages'
    previous=report.read_text() if report.exists() else ''
    retained=marker+previous.split(marker,1)[1] if marker in previous else ''
    report.write_text('\n'.join(doc)+'\n'+retained)
    save(BASE/'progress.json',dict(time=now(),system_records=n,formal_records=formal,queue=queue))
    print(json.dumps(dict(report='docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md',system_records=n,formal_records=formal)))

if __name__=='__main__':main()
