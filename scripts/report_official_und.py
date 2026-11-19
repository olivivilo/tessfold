"""Score native official-model predictions and maintain a provenance-aware report."""
import argparse
import collections
import json
from pathlib import Path

from vqa.benchmark_metrics import due_field_f1, dtvqa_scores, native_score
from scripts.eval_official_und import OUT, ROOT, rows_for

PAPER = {'DocVQA': 79.3, 'InfoVQA': 46.3, 'DeepForm': 65.8, 'KLC': 38.2,
         'WTQ': 43.2, 'ChartQA': 62.0, 'DT-VQA': 64.3, 'OCRBench': 58.1}


def score(args):
    expected = rows_for(args.mode, args.limit_per_task)
    files = sorted(OUT.glob(f'{args.label}.rank*.jsonl'))
    assert files, 'No predictions'
    reference = {r['id']: r for r in expected}
    assert len(reference) == len(expected), 'Duplicate IDs in reference dataset'
    records = {}
    for path in files:
        assert path.with_suffix('.complete.json').exists(), f'Rank incomplete: {path}'
        completion = json.loads(path.with_suffix('.complete.json').read_text())
        assert completion['world'] == len(files), 'Missing or extra rank shards'
        rank = int(path.stem.rsplit('rank', 1)[1])
        shard_ids = set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['id'] not in records, f'Duplicate prediction {row["id"]}'
            gold = reference[row['id']]
            assert all(row[k] == gold[k] for k in ('question', 'answers', 'task', 'image', 'document_id'))
            source = str(args.adapter) if args.adapter else 'Author full MonkeyOCRv2-B-Und'
            assert row['source'] == source, f'Model source mismatch: {row["source"]} vs {source}'
            assert abs(native_score(row['prediction'], row) - row['score']) < 1e-8
            records[row['id']] = row
            shard_ids.add(row['id'])
        assert len(shard_ids) == completion['samples']
        assert shard_ids == {r['id'] for r in expected[rank::len(files)]}, 'Rank partition differs from reference'
    assert set(records) == {r['id'] for r in expected}, f'Coverage {len(records)}/{len(expected)}'
    groups = collections.defaultdict(list)
    for row in records.values():
        groups[row['task']].append(row)
    result = {'label': args.label, 'mode': args.mode, 'limit_per_task': args.limit_per_task,
              'samples': len(records), 'adapter': args.adapter, 'benchmarks': {},
              'pipeline': 'Author AutoProcessor + process_vision_info + full remote-code model.generate',
              'hardware_choice': 'Ascend BF16, author eager vision attention, Qwen SDPA, greedy, independent questions',
              'hit_token_limit': sum(r['runtime']['hit_token_limit'] for r in records.values())}
    for task, rows in groups.items():
        item = {'samples': len(rows), 'score_percent': 100 * sum(r['score'] for r in rows) / len(rows),
                'metric': 'ANLS' if task in ('DocVQA', 'InfoVQA') else 'AccANLS' if task == 'DT-VQA' else
                'relaxed_accuracy' if task == 'ChartQA' else 'WTQ denotation accuracy' if task == 'WTQ' else 'OCRBench_accuracy'}
        if task in ('DeepForm', 'KLC'):
            item.update(due_field_f1(rows))
        if task == 'DT-VQA':
            scores = [dtvqa_scores(r['prediction'], r['answers']) for r in rows]
            item['component_scores'] = {k: 100 * sum(r[k] for r in scores) / len(scores) for k in ('ANLS', 'accuracy', 'accanls')}
        if task in ('ChartQA', 'OCRBench'):
            by_type = collections.defaultdict(list)
            for r in rows:
                by_type[r.get('question_type', 'unspecified')].append(r)
            item['by_type'] = {k: {'samples': len(v), 'score_percent': 100 * sum(r['score'] for r in v) / len(v)} for k, v in by_type.items()}
        result['benchmarks'][task] = item
    result['task_macro_mean_percent'] = sum(r['score_percent'] for r in result['benchmarks'].values()) / len(groups)
    (OUT / f'{args.label}_metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))
    return result


def report():
    def get(label):
        path = OUT / f'{label}_metrics.json'
        return json.loads(path.read_text()) if path.exists() else None
    baseline, tuned = get('baseline_full'), get('finetuned_full')
    val0, val1 = get('baseline_validation'), get('finetuned_validation')
    lines = ['# Official MonkeyOCRv2-B-Und: Baseline and Fine-Tuning', '',
             '**Start from the authors’ complete aligned UND checkpoint; report official baseline, subsequent LoRA fine-tuning, and earlier custom-alignment results separately.**', '',
             'Main entry: `scripts/run_official_und.sh`; official source: `vendor/MonkeyOCRv2/understanding/`; '
             'Official training framework: `vendor/MonkeyOCRv2/parsing/train/ms-swift/`. ', '',
             'Native inference retains the author AutoProcessor, process_vision_info, ChatML, image-token substitution, and full-model generate. '
             'Ascend uses the authors’ eager visual attention and Qwen SDPA; BF16, max_pixels=1,003,520, '
             'Greedy independent generation, at most 256 new tokens, with a fresh independent KV cache for each question.', '',
             '| Task | Paper Table 8 | Official UND baseline | Official UND + current fine-tuning |', '|---|---:|---:|---:|']
    if baseline:
        lines[4:4] = [f"**Full official baseline: {baseline['samples']:,}  questions; eight-task mean  {baseline['task_macro_mean_percent']:.4f}; paper reports 57.2.**"
                      'Evaluation of released complete weights, not from-scratch realignment; protocol differences below.', '']
    if baseline and tuned:
        lines[6:6] = [f"**Fine-tuning complete: 30,000 training QA, 1 epoch, 235 steps; full eight-task mean  {tuned['task_macro_mean_percent']:.4f}, "
                      f"Change from baseline {tuned['task_macro_mean_percent'] - baseline['task_macro_mean_percent']:+.4f}. **"
                      'Full evaluation improves slightly, but fixed small validation does not; selection retains baseline while both results remain reported.', '']
    for task, paper in PAPER.items():
        values = [f"{r['benchmarks'][task]['score_percent']:.2f}" if r and task in r['benchmarks'] else 'Not completed' for r in (baseline, tuned)]
        lines.append(f'| {task} | {paper:.1f} | {values[0]} | {values[1]} |')
    means = [f"{r['task_macro_mean_percent']:.2f}" if r else 'Not completed' for r in (baseline, tuned)]
    lines.append(f'| Eight-task mean | 57.2 | {means[0]} | {means[1]} |')
    if baseline:
        cap = str(tuned['hit_token_limit']) if tuned else 'Pending'
        lines.extend(['', f"Questions reaching the 256-token cap: official baseline **{baseline['hit_token_limit']} / {baseline['samples']}**; "
                      f'Fine-tuned **{cap}**. Original truncated outputs remain without manual completion or rewriting.'])
    lines.extend(['', 'DocVQA/InfoVQA use public validation; the other six use existing local test splits. '
                  'DeepForm/KLC retain the original Monkey single-page-QA-to-DUE conversion protocol. '
                  'Hidden test/document processing may differ from the paper; exact protocol identity is not claimed. '
                  'Overlap between original author pretraining and public splits is not fully audited; new training strictly uses public train only.', '',
                  '## Fine-Tuning Protocol and Selection Criteria', '',
                  'ms-swift in the released author repository, `monkeyocrv2_und` template and official LoRA setup; '
                  'Rank 8, alpha 32, all-linear; encoder frozen, LoRA permitted in LLM and aligner; '
                  'LR 1e-5, batch 128, 1 epoch, 5% warmup＋cosine. '
                  'Disable CUDA flash-only padding_free and DeepSpeed; use Ascend DDP and gradient accumulation, with hardware adaptations documented separately.', '',
                  'Fixed 30,000 training QA: DocVQA 8,000; InfoVQA/ChartQA/WTQ/KLC/DT-VQA 4,000 each; DeepForm 2,000. '
                  'Fixed validation: 256 questions, 64 distinct documents each from DocVQA/InfoVQA/ChartQA/DT-VQA, one question/image. '
                  'Log evaluation loss/save checkpoints every 50 updates; select candidates by validation loss. '
                  'Compare original and candidate generation on the same 256 questions; tests never select models.', '',
                  '| Same fixed validation set | Original UND | Fine-tuning candidate |', '|---|---:|---:|'])
    tasks = ('DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA')
    for task in tasks:
        values = [f"{r['benchmarks'][task]['score_percent']:.2f}" if r else 'Not completed' for r in (val0, val1)]
        lines.append(f'| {task} | {values[0]} | {values[1]} |')
    means = [f"{r['task_macro_mean_percent']:.2f}" if r else 'Not completed' for r in (val0, val1)]
    lines.append(f'| Four-task mean | {means[0]} | {means[1]} |')
    selection = OUT / 'model_selection.json'
    if selection.exists():
        s = json.loads(selection.read_text())
        lines.extend(['', f"Selected by fixed-development generation: **{s['selected']}**; {s['reason']}. "])
        if s['selected'] == 'baseline':
            lines.extend(['', 'LoRA improves validation NLL but reduces fixed-validation generation macro, so its adapter is not selected. '
                          'This only shows no overall validation gain from this continued-fine-tuning mixture/settings; '
                          'This neither invalidates all LLM fine-tuning nor identifies encoder capacity as the cause.'])
            if tuned:
                lines.extend(['', 'Full-evaluation and fixed small-validation trends differ, especially for ChartQA and DT-VQA. '
                              'Only 64 questions/task gives sampling uncertainty; retain the prespecified selection, '
                              'No model is retrospectively selected by test score. The full fine-tuned model and adapter remain available for independent validation.'])
    lines.extend(['', '## Training Clues in the Official Model Release', '',
                  '`models/MonkeyOCRv2-B-Und/args.json` Includes formula, table, transcription, block-recognition, and multiple QA data paths, '
                  'It also records continuation from checkpoint-4688, LR=2e-5, and related settings. '
                  'These release files provide training clues; filenames cannot reconstruct content, exact mixture, or both stages, '
                  'Nor does the arguments file establish exact correspondence to every Table-8 setting. It corrects the earlier overstatement that no training-data clues were public.', '',
                  '## Reproduction and Evidence', '', '```bash',
                  'bash scripts/run_official_und.sh',
                  '# See docs/OFFICIAL_UND.md for single-image native inference.', '```', '',
                  '- Weights: `models/MonkeyOCRv2-B-Und/source_manifest.json` and `VERIFIED`, fixed revision and SHA-256 per file.',
                  '- Data: `data/official_und/manifest.json`, official messages/images format in train/validation JSONL.',
                  '- Original answers: `artifacts/official_und/{label}.rank*.jsonl`, including gold, predictions, scores, token counts, and model provenance.',
                  '- Fine-tuning log: `artifacts/official_und/sft/`; train/eval losses and epochs remain in trainer_state / logging.jsonl.',
                  '- [Measured train/eval loss curves and DocVQA gold/answer comparison](OFFICIAL_UND_TRAINING.md). ',
                  '- Official version and hardware adaptation: [OFFICIAL_UND.md](OFFICIAL_UND.md). ',
                  '- Earlier custom-alignment results: [FULL_TRAINING_RESULTS.md](FULL_TRAINING_RESULTS.md), without overwriting.'])
    (ROOT / 'docs/OFFICIAL_UND_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label')
    parser.add_argument('--mode', default='benchmark')
    parser.add_argument('--limit-per-task', type=int, default=0)
    parser.add_argument('--adapter')
    args, _ = parser.parse_known_args()
    if args.label:
        score(args)
    report()


if __name__ == '__main__':
    main()
