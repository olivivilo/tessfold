"""Report only results actually produced by the formal run; never substitute paper scores."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/full'
PAPER={'DocVQA':79.3,'InfoVQA':46.3,'DeepForm':65.8,'KLC':38.2,'WTQ':43.2,'ChartQA':62.0,'DT-VQA':64.3,'OCRBench':58.1}

def main():
    cfg=json.loads((ROOT/'configs/full.json').read_text())
    manifest=json.loads((ROOT/'data/full/manifest.json').read_text()) if (ROOT/'data/full/manifest.json').exists() else {}
    status=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {'state':'waiting_for_resources_or_preparation'}
    if not (OUT/'status.json').exists() and (ROOT/'data/full/ALL_TASKS_READY').exists() and (ROOT/'artifacts/lm_backend_verification.json').exists():
        status={'state':'waiting_for_accelerator_resources'}
    lines=['# Full Training and Standard Evaluation', '',f"Current status: `{status['state']}`. ", '',
           'Configuration: original pretrained MonkeyOCRv2-B, tokenwise two-layer MLP, original Qwen3-1.7B; visual backbone frozen. '
           'Stage 1: LR=1e-3, MLP-only. Stage 2: LR=1e-5, full MLP/LLM updates. Effective batch=128, max_pixels=1003520.', '',
           'Each stage passes once through the full public training mixture, using 5% warmup + cosine. '
           'Exact training proportions and stage epochs are undisclosed; these choices do not reproduce hidden author settings.', '',
           'Training entry: `sbatch scripts/slurm_full_npu.sh`. Automatically performs distributed feature extraction, two-stage training, complete held-out evaluation, and report generation. '
           'Rerunning resumes  `artifacts/full/resume/` model, optimizer, scheduler, and optimizer step.', '',
           '| Benchmark | Split in this run | Measured score (%) | Metric | Paper Table 8 reference (%) |',
           '|---|---|---:|---|---:|']
    measured={}
    for split in ['val','test']:
        p=OUT/f'final_{split}_metrics.json'
        if p.exists():
            for task,result in json.loads(p.read_text())['benchmarks'].items():
                if task not in measured or split=='test':measured[task]=(split,result)
    for task,paper in PAPER.items():
        if task in measured:
            split,result=measured[task];lines.append(f"| {task} | {split} | {result['score_percent']:.2f} | {result['metric']} | {paper} |")
        else:lines.append(f'| {task} | Not completed | Pending measured evaluation | — | {paper} |')
    if len(measured)==8:
        average=sum(r['score_percent'] for _,r in measured.values())/8
        lines.extend(['',f'The arithmetic mean of these eight measured scores is  {average:.2f}%. '
                      'DocVQA/InfoVQA use public validation; DeepForm/KLC use converted original Monkey single-page QA, '
                      'Paper test splits and processing details can differ; exact protocol reproduction of Table 8 is not claimed.'])
        if average<PAPER.get('Overall',57.2)-10:
            lines.extend(['','**Training and the execution pipeline are complete, but measured quality remains far below the paper; effective alignment close to its results has not been achieved.**'
                          'Pipeline checks confirm consistent training/save/reload/scoring, not successful alignment quality.'])
    if (ROOT/'docs/DOCVQA_REVIEW.md').exists():
        lines.extend(['','## DocVQA Predictions and Gold Answers','',
                      'Entry point: [Comparison notes and first 60 examples](DOCVQA_REVIEW.md). '
                      'All 5,349 QA: [Filterable image viewer](../artifacts/full/docvqa_review.html), '
                      '[Complete TSV](../artifacts/full/docvqa_comparison.tsv), '
                      '[Complete JSONL](../artifacts/full/docvqa_comparison.jsonl). '
                      'Retain verbatim predictions, all gold answers, original-image paths, and prediction-file line numbers.'])
    if (ROOT/'docs/CONTEXT_AUDIT.md').exists():
        lines.extend(['','[Repeated-name and cross-question context audit](CONTEXT_AUDIT.md): '
                      'All 226 questions: reordered/cleared-feature-cache controls, initial KV lengths, fresh processes, KV-disabled generation, and image swaps.'])
    if (ROOT/'docs/ALIGNMENT_DIAGNOSIS.md').exists():
        lines.extend(['','[Alignment failure diagnosis and next steps](ALIGNMENT_DIAGNOSIS.md): '
                      'Stage-1/2 comparison on matched train/validation questions, correct/wrong-image controls, and actual eight-image shared-question training.'])
    if (ROOT/'docs/TRAINING_LOSS_AUDIT.md').exists():
        lines.extend(['','[Encoder, training-data, and loss-curve audit](TRAINING_LOSS_AUDIT.md): '
                      'Author-pretrained Base identity, actual VQA mixture, historical curves, and post-hoc fixed training/validation losses.'])
    if (ROOT/'docs/PAPER_REPRODUCTION_GAPS.md').exists():
        lines.extend(['','[Original-paper protocol versus this training setup](PAPER_REPRODUCTION_GAPS.md): '
                      'Paper requirements, undisclosed recipe, and official UND continued fine-tuning versus raw-component alignment.'])
    lines.extend(['', 'Paper scores are references only; pending evaluation is never filled with estimates or released-model scores.', '',
                  'Complete dataset splits and counts:', '', '```json',json.dumps(manifest,indent=2,ensure_ascii=False),'```', '',
                  'Evidence files:', '', '- `data/full/download_manifest.json`: download revisions, sizes, and LFS SHA-256.',
                  '- `data/full/manifest.json`: full dataset counts and training/evaluation image deduplication.',
                  '- `artifacts/full/training.jsonl`: loss, LR, and gradient norm per optimizer step.',
                  '- `artifacts/full/final_{val,test}.rank*.jsonl`: original generations and references per question.',
                  '- `artifacts/full/final_{val,test}_metrics.json`: complete standard-task metrics.',
                  '- `artifacts/full/stage{1,2}/`: training checkpoints.',
                  '- `artifacts/full/queue.json`: Slurm submissions for this run.', '',
                  'Scoring: raw-character ANLS for DocVQA/InfoVQA; recommended AccANLS for DT-VQA with ANLS/accuracy also retained; ChartQA 5% relaxed accuracy, split human/synthetic; '
                  'WTQ uses official DUE denotation; DeepForm/KLC use official DUE corpus set F1, '
                  'Convert original Monkey per-field QA to DU-schema, treating None as absent. OCRBench uses original substring scoring with task breakdowns.'])
    diagnostics={}
    for stage in (1,2):
        p=OUT/f'stage{stage}_alignment_val_metrics.json'
        if p.exists():diagnostics[stage]=json.loads(p.read_text())['benchmarks']
    if diagnostics:
        lines.extend(['','## Measured MLP-Only Versus Joint Alignment Comparison','',
                      'Both stages use the same fixed validation questions (up to 64/task). This is alignment diagnosis, not a replacement for the full benchmark.','',
                      '| Task | Stage 1: MLP only (%) | Stage 2: MLP + Qwen (%) |','|---|---:|---:|'])
        for task in sorted({t for results in diagnostics.values() for t in results}):
            values=[f"{diagnostics[s][task]['score_percent']:.2f}" if task in diagnostics.get(s,{}) else 'Pending evaluation' for s in (1,2)]
            lines.append(f'| {task} | {values[0]} | {values[1]} |')
    verification=OUT/'verification.json'
    if verification.exists():
        result=json.loads(verification.read_text())
        lines.extend(['','## End-to-End Verification','',
                      f"Original encoder-weight checks, prediction coverage, per-question/aggregate score recalculation: `{result['passed']}`. ",
                      f"Saved-model single-image entry check: `{result.get('public_infer_entry',{}).get('passed',False)}`. ",
                      'Detailed evidence: `artifacts/full/verification.json`. '])
    (ROOT/'docs/FULL_TRAINING_RESULTS.md').write_text('\n'.join(lines)+'\n')
    readme=ROOT/'README.md'
    if readme.exists():
        text=readme.read_text()
        start,end='<!-- FULL_STATUS_START -->','<!-- FULL_STATUS_END -->'
        if start in text and end in text:
            if status['state']=='complete':
                banner='Formal two-stage training and full evaluation are complete. Measured metrics: [docs/FULL_TRAINING_RESULTS.md](docs/FULL_TRAINING_RESULTS.md). '
                if len(measured)==8:
                    banner+=f'Eight-task mean in this round  {average:.2f}%; paper reference: 57.2%. The target of approaching the paper is not yet achieved.'
            elif status['state'] in ('failed','interrupted'):
                banner=f"Formal-job status: `{status['state']}`. Checkpoints are resumable; errors/progress:  `artifacts/full/status.json` and Slurm logs."
            elif status['state'] in ('training','caching','evaluating','verifying'):
                banner=f"Formal-job status: `{status['state']}`. Full metrics are incomplete; actual progress:  `artifacts/full/status.json`. "
            elif status['state'] in ('waiting_for_accelerator_resources','cached_waiting_for_training'):
                banner='Full-data and numerical/recovery/saved-model inference checks passed. Formal training awaits NPU resources; no formal evaluation scores yet. Job records: `artifacts/full/queue.json`. '
            else:
                banner='Full data are ready; the submitted formal job awaits NPU validation/resources. No formal evaluation scores yet. Job records: `artifacts/full/queue.json`. '
            before=text.split(start,1)[0];after=text.split(end,1)[1]
            readme.write_text(before+start+'\n'+banner+'\n'+end+after)
    print(ROOT/'docs/FULL_TRAINING_RESULTS.md')

if __name__=='__main__':main()
