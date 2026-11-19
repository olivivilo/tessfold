"""Create a report exclusively from measured run artifacts."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/alignment.json')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    out = ROOT / cfg['output_dir']
    results = json.loads((out / 'results.json').read_text())
    logs = [json.loads(x) for x in (out / 'train.jsonl').read_text().splitlines()]
    environment = json.loads((out / 'environment.json').read_text())
    lines = ['# Measured Run Results', '',
             'Results use QA derived from local MonkeyDocv2 English nonsemantic tables. '
             'These are not reproduced scores on the paper’s eight standard benchmarks. Small-sample checks validate alignment, training, generation, and checkpoint recovery execution.', '',
             '| Stage / split | Questions | EM (%) | ANLS (%) | Relaxed accuracy (%) | Answer NLL |',
             '|---|---:|---:|---:|---:|---:|']
    for key in ('baseline_test', 'stage1_val', 'final_test'):
        m = results[key]
        lines.append(f"| {key} | {m['samples']} | {m['exact_match']:.2f} | {m['anls']:.2f} | {m['relaxed_accuracy']:.2f} | {m['answer_nll']:.4f} |")
    lines.extend(['', 'Final test scores by task:', '',
                  '| Task | Questions | EM (%) | ANLS (%) | Relaxed accuracy (%) |',
                  '|---|---:|---:|---:|---:|'])
    for task, m in results['final_test']['by_task'].items():
        lines.append(f"| {task} | {m['samples']} | {m['exact_match']:.2f} | {m['anls']:.2f} | {m['relaxed_accuracy']:.2f} |")
    lines.extend(['', f"Runtime: `{environment['device']}` / `{environment['dtype']}`, {cfg['threads']} CPU threads. ",
                  f"Visual encoder frozen; MLP parameters  {environment['projector_parameters']:,}; language-model parameters  {environment['language_parameters']:,}. ",
                  f"Input cap  {cfg['max_pixels']}  pixels; greedy decoding; Qwen3 thinking disabled; at most  {cfg['max_new_tokens']} tokens. ", '',
                  'Training-stage records:', ''])
    if environment.get('slurm_job_id'):
        lines.insert(-2, f"Slurm job `{environment['slurm_job_id']}`, node `{environment['hostname']}`, requesting 16 CPUs / 64 GB RAM.")
    for stage in (1,2):
        rows = [x for x in logs if x['stage']==stage]
        if rows:
            lines.append(f"- Stage {stage}: {len(rows)} optimizer steps, LR {rows[0]['lr']}, "
                         f"First/last training-example loss {rows[0]['loss']:.4f}/{rows[-1]['loss']:.4f}, "
                         f"Elapsed {rows[-1]['elapsed_seconds']:.1f} s. Steps use different examples, so first/last sample losses do not directly quantify convergence.")
    baseline, final = results['baseline_test'], results['final_test']
    lines.extend(['', f"Token-weighted answer NLL on the same test set: {baseline['answer_nll']:.4f} → {final['answer_nll']:.4f}. ",
                  'Answer NLL uses teacher forcing. Generated EM/ANLS use independent greedy decoding without reference answers as inputs.', '',
                  f"Checkpoint save/reload consistency: {results['checkpoint_roundtrip']['passed']}(identical greedy output on one held-out example).", '',
                  'Raw evidence (repository-relative):', '',
                  f"- `{cfg['output_dir']}/results.json`: all measured metrics.",
                  f"- `{cfg['output_dir']}/final_test_predictions.jsonl`: per-question predictions, references, and metrics.",
                  f"- `{cfg['output_dir']}/train.jsonl`: loss/gradient norms for all updates.",
                  f"- `artifacts/{Path(args.config).stem}_pipeline.log`: complete execution output (smoke first-run log:  `artifacts/pipeline.log`). ", '',
                  'Paper Table 8 for MonkeyOCRv2-B: Overall 57.2, DocVQA 79.3, InfoVQA 46.3, DF 65.8, '
                  'KLC 38.2, WTQ 43.2, ChartQA 62.0, DT-VQA 64.3, OCRBench 58.1. These are paper reference values, '
                  'Not measured here and not directly comparable to this subset.', '',
                  'Low/zero scores remain. Thirty-two low-resolution tables and few updates cannot establish paper-level document understanding.'])
    predictions = [json.loads(line)['prediction'] for line in (out / 'final_test_predictions.jsonl').read_text().splitlines()]
    if len(set(predictions)) == 1:
        lines.extend(['', f"All test questions generate `{predictions[0]}`, indicating fixed-answer collapse. "
                      'Lower NLL shows a changed answer distribution, not necessarily effective visual–language alignment. '
                      'This completes an executable two-stage pipeline, not usable VQA capability.'])
    verification = out / 'verification.json'
    if verification.exists():
        v = json.loads(verification.read_text())
        lines.extend(['', f"Independent metric recomputation passed `{v['metrics_recalculated']}`; "
                      f"Single image `infer` entry rereads the original image, extracts features, and loads Stage 2: `{v['public_infer_entry']}`. ",
                      f"Evidence: `{cfg['output_dir']}/verification.json`. "])
        if 'llm_weight_check' in v:
            w = v['llm_weight_check']
            lines.extend(['', f"Joint-fine-tuning actual weight update: `{w['tensor']}` mean absolute update:  {w['mean_absolute_update']:.8f}, "
                          f"maximum absolute update  {w['max_absolute_update']:.8f}; Stage-2 MLP also differs from Stage 1."])
    (ROOT / 'docs/EVAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(ROOT / 'docs/EVAL_RESULTS.md')

if __name__ == '__main__':
    main()
