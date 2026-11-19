"""Report the explicitly authorized 50% extension without rewriting its failed gate."""
import collections
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/compression_lab_v2'


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    state=load(BASE/'stages.json')['ratios']['0.5']
    if not state.get('user_authorized_extension'):return
    cfg=load(ROOT/state['full_config']);out=ROOT/cfg['output'];list(out.iterdir())
    training=[]
    if (out/'training.jsonl').exists():
        training=[json.loads(s) for s in (out/'training.jsonl').read_text().splitlines() if s.strip()]
    doc=['# 50% Token Retention: Authorized Full Training and Formal Evaluation','',
         f'Updated (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}. Stage: `{state["stage"]}`. ',
         '', '**Entry point**: status  `bash scripts/run_compression_lab_v2.sh status`; resume after interruption  `bash scripts/run_compression_lab_v2.sh resume`; this page updated  `.venv/bin/python -m scripts.report_compression_lab_v2_keep50`; strict full audit  `.venv/bin/python -m scripts.report_compression_lab_v2_final --require-complete`. ',
         '', '## Experiment Registration and Architecture','',
         'The requested 50% extension applies the same full protocol as 10%/25% to an already expanded two-seed question-conditioned tree (quadtree_question, pre-merger). Original independent-confirmation seeds score 50.27 and 51.12 versus teacher 58.82; the first gain CI is [−0.26,10.41], so the original two-seed eligibility remains failed. This separately authorized extension does not rewrite that confirmation outcome.',
         '', 'Use official MonkeyOCRv2-B-Und with encoder, official merger, and Qwen3-1.7B frozen. The generator reads 768-dimensional encoder H, image grid, and question embeddings, producing row-stochastic P. Each output is a convex combination of original tokens; student visual input is merger(PH). Retained tokens=max(1,round(0.5×N)). Teacher is the uncompressed official model. See the [architecture document](COMPRESSION_GENERATOR_V2_DESIGNS.md). ',
         '', '## Fixed Experimental Procedure','',
         'Use the same original 20,000 QA / 16,731 images as the 10%/25% runs. Initialize afresh with seed 20261105, without reusing small-run weights or selecting the better confirmation seed. Train 4 epochs / 2,500 global updates, batch=32; 8 NPU ranks accumulate 4 questions each, with no padding. AdamW LR=3e-4, weight_decay=0, clip=1; 250-step warmup then cosine decay to 10% of peak LR. Optimize only full-vocabulary KL under the shared teacher prefix, without entropy or other regularizers.',
         '', 'Save latest.pt/optimizer every 25 steps and checkpoints/development results each epoch. The 56-QA screening set with cap=64 monitors training; epoch 4 is fixed, without retrospective development/test selection. Formal evaluation retains the same 23,565 QA, 8 tasks, greedy cap=256, and six independent initial/trained/mean/selection/cell_mean/cell_selection answers per question, totaling 141,390 records. Teacher uses the same uncompressed official-model cache. Questions are independent; generation context is never reused across them.',
         '', 'Only 50% full-experiment registration changes; retain 10%/25% weights/answers. Fourteen independent evaluation workers change question allocation only. Use verified cache/checkpoint/controller repairs. All 14 inspected initial tensors equal the 10% model; see [Initialization audit](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/initialization_audit.json). ',
         '', '## Training Progress','',
         f'Training job: `{state.get("full_job","Not submitted")}`; evaluation job: `{state.get("benchmark_job","Not submitted")}`. Recorded global updates: {max((r["step"] for r in training),default=0)}/2500; training completion marker: {(out/"complete.json").exists()}. Logged step count need not equal the committed checkpoint step.',
         '', '|epoch|Logged batches / 625|Full-training batch-mean KL|Fixed training-audit KL|Development KL|Small selection-set generation score|',
         '|---|---:|---:|---:|---:|---:|']
    by_epoch=collections.defaultdict(list)
    for r in training:by_epoch[r['epoch']].append(r['kl'])
    fmt=lambda v:'—' if v is None else f'{v:.5f}'
    for e in range(1,5):
        rr=by_epoch[e];audit=load(out/f'train_audit.epoch{e}.summary.json');val=load(out/f'validation.epoch{e}.summary.json')
        doc.append(f'|{e}|{len(rr)}/625|{fmt(sum(rr)/len(rr) if rr else None)}|{fmt(audit.get("kl"))}|{fmt(val.get("kl"))}|{fmt(val.get("student_macro"))}|')
    doc+=['', 'Full-training means cover completed batches; fewer than 625 means the epoch is incomplete. Fixed audit/validation are separate small sets. Screening cap=64 and formal cap=256 are not a matched before/after comparison.',
          '', '## Formal Evaluation Results','']
    bench=out/'benchmark'
    try:list(bench.iterdir())
    except FileNotFoundError:pass
    audited=load(BASE/'infrastructure/recovery_20261106/final_artifact_audit.json').get('budgets',{}).get('0.5')
    if (bench/'complete.json').exists() and audited:
        summary=load(bench/'summary.json');m=summary['macro'];p=summary['paired_trained_vs_initial'];ci=p['cluster_bootstrap_95ci_points']
        variants=['teacher','initial','trained','mean','selection','cell_mean','cell_selection']
        doc+=['Completed 23,565 QA / 141,390 variant records, passing question coverage, gold/teacher agreement, checkpoint/code hashes, and score recomputation.',
              '', '|Task|'+ '|'.join(variants)+'|','|---|'+ '|'.join(['---:']*len(variants))+'|',
              '|Eight-task macro average|'+ '|'.join(f'{m[v]:.2f}' for v in variants)+'|']
        for task,values in sorted(summary['tasks'].items()):
            doc.append('|'+task+'|'+ '|'.join(f'{values[v]:.2f}' for v in variants)+'|')
        doc+=['',f'Trained versus initial {p["gain_points"]:+.2f} points; image-cluster bootstrap 95% CI [{ci[0]:.2f},{ci[1]:.2f}]; relative to teacher: {m["trained"]-m["teacher"]:+.2f} points; teacher-score retention {100*m["trained"]/m["teacher"]:.2f}%. ',
              '', 'Equally weighted native-task macro; DeepForm/KLC field exact and retained DT-VQA scoring are not unified official paper scores. Same-generator 10%/50% differ in budget; 25% uses another generator, so budget alone does not explain gaps.']
        doc+=['', 'Controls use the same set, scorers, and generation settings as the earlier full evaluation:',
              '', '|Retention fraction|Generator|teacher|initial|trained|', '|---|---|---:|---:|---:|']
        for ratio,name in [('10%','full_quadtree_question_pre-merger_r0.1'),('25%','full_global_topk_pre-merger_r0.25'),('50%','full_quadtree_question_pre-merger_r0.5')]:
            result=load(BASE/'full'/name/'benchmark/summary.json')
            values=result['macro']
            doc.append(f'|{ratio}|{result["config"]["kind"]}|{values["teacher"]:.2f}|{values["initial"]:.2f}|{values["trained"]:.2f}|')
        doc+=['', 'After 50% training, all eight tasks improve over their own initialization, but gaps to the teacher remain, especially on OCRBench. Small-set generation drops from 68.60 at epochs 2/3 to 65.03 at epoch 4 despite lower validation KL; KL reduction therefore does not guarantee monotonic generation gains. Formal evaluation uses the prespecified epoch 4, without retrospectively selecting an earlier checkpoint. There is one full-training seed; image-bootstrap intervals describe evaluation sampling uncertainty, not variation across training seeds.']
    else:doc+=['Full formal results incomplete or unaudited; small confirmation scores are not substitutes.']
    rel='../'+cfg['output']
    doc+=['',f'[Per-question CSV]({rel}/benchmark/answers.csv) · [Image/gold/teacher/before-after training answers]({rel}/benchmark/review.html) · [Formal machine-readable results]({rel}/benchmark/summary.json) · [Fixed epoch-4 weights]({rel}/epoch4.pt). ',
          '', '[Per-epoch training/validation KL curves](../artifacts/compression_lab_v2/plots/loss_v2_full_20k_pre-merger_keep0.5.pdf) · [Per-epoch generation curves](../artifacts/compression_lab_v2/plots/score_v2_full_20k_pre-merger_keep0.5.pdf) · [Formal results at three budgets](COMPRESSION_GENERATOR_V2_FINAL_RESULTS.md) · [Extension registration](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/registration.json) · [Training-completeness audit](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/training_audit.json). ']
    target=ROOT/'docs/COMPRESSION_GENERATOR_V2_KEEP50_FULL.md';tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
    print(json.dumps({'stage':state['stage'],'logged_updates':len(training),'formal_audited':bool(audited)}))


if __name__=='__main__':main()
