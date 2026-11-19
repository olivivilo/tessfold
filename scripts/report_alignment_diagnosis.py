"""Summarize completed alignment diagnostics without changing formal eval results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/full/alignment_diagnosis'


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    overfit = json.loads((OUT / 'overfit/summary.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    # Predictions are append-only; use the most recent record per measured case.
    records = {}
    for line in (OUT / 'predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        records[row['stage'], row['split'], row['id']] = row
    for stage in (1, 2):
        for split in ('train', 'val'):
            rows = [records[stage, split, row['id']] for row in selection[split]]
            actual = summary['stages'][str(stage)][split]
            assert len(rows) == actual['samples'] == 32
            assert abs(100 * sum(row['correct_image']['anls'] for row in rows) / 32 - actual['correct_image_anls_percent']) < 1e-8
            assert abs(sum(row['wrong_minus_correct_nll'] for row in rows) / 32 - actual['mean_wrong_minus_correct_nll']) < 1e-8
    for branch, result in overfit.items():
        final = result['final']
        assert final == json.loads((OUT / f"overfit/{branch}_step{final['step']}.json").read_text())
        assert final['exact_lowercase'] == sum(any(row['prediction'].lower() == a.lower() for a in row['gold_answers']) for row in final['records'])

    lines = [
        '# Alignment Failure: Diagnosis and Next Steps', '',
        '**Prioritize visual alignment and training-protocol diagnosis; current evidence does not justify a larger LLM first or attributing failure solely to unfreezing Qwen.**'
        'Full DocVQA validation ANLS remains 11.52. Added results are diagnostic, not replacements for the formal benchmark.', '',
        'Paper Section 4.7 / Table 8 uses the same Qwen3-1.7B and explicitly jointly trains MLP/LLM in Stage 2. '
        'Thus 1.7B is not an established capacity barrier to the target scores, and the paper does not freeze its LLM throughout. '
        'The undisclosed exact training mixture/schedule remains unreconstructed; matching backbone, LR, and batch is not an identical full protocol.', '',
        '## Stage 1 Versus Stage 2 on the Same Questions', '',
        'Deterministically select 32 distinct documents each from DocVQA train/val, one question/image. '
        'Evaluate the same questions with Stage 1 (original Qwen + aligned MLP) and Stage 2 (jointly trained Qwen + MLP). '
        'No weight updates or test usage.', '',
        '| Actual model | Training-question ANLS (%) | Development-question ANLS (%) | Training-question answer NLL | Development-question answer NLL |',
        '|---|---:|---:|---:|---:|',
    ]
    for stage, label in [('1', 'Stage 1: MLP only'), ('2', 'Stage 2: MLP＋Qwen')]:
        train, val = (summary['stages'][stage][split] for split in ('train', 'val'))
        lines.append(f"| {label} | {train['correct_image_anls_percent']:.2f} | {val['correct_image_anls_percent']:.2f} | {train['mean_correct_image_answer_nll']:.3f} | {val['mean_correct_image_answer_nll']:.3f} |")
    lines.extend([
        '', 'Training answers themselves are weak; this is not simply learned training data with validation overfitting. '
        'Stage 2 improves generation scores and correct-answer probabilities on these train/validation samples. '
        'On the earlier fixed 64-question/task validation, DocVQA also rises 11.77→15.98 and all other three tasks improve. '
        'Small samples do not validate every unfreezing setting or isolate freezing at matched training budgets.', '',
        '## Does the Correct Image Increase Gold-Answer Likelihood?', '',
        'Keep question/gold fixed and swap another real image from the same split, preferably matching visual-token count. '
        'NLL uses the first gold answer and EOS; lower is better. Teacher-forced token accuracy is not free-generation accuracy.', '',
        '| Model / sample | Correct-image ANLS (%) | Wrong-image ANLS (%) | Wrong-image NLL − correct-image NLL | Questions with lower NLL for the correct image |',
        '|---|---:|---:|---:|---:|',
    ])
    for stage in ('1', '2'):
        for split, name in [('train', 'Training'), ('val', 'Validation')]:
            r = summary['stages'][stage][split]
            lines.append(f"| Stage {stage} / {name} 32 questions | {r['correct_image_anls_percent']:.2f} | {r['wrong_image_anls_percent']:.2f} | {r['mean_wrong_minus_correct_nll']:.3f} | {r['correct_image_lower_nll_samples']}/32 |")
    lines.extend([
        '', 'The model uses some image information, and joint training increases average gold-likelihood image dependence, but correct-image answers remain poor. '
        'These findings neither establish fine-grained text reading nor isolate encoder, MLP, or optimizer as the cause. '
        'Visual-token counts match for 28/32 training and 29/32 validation controls; remaining length differences are retained per record.', '',
        '## Eight-Image Training Check with a Shared Question', '',
        'Select 8 distinct-page DocVQA **training** images only, all with question  `What is the page number?`. '
        'With identical questions, correct answers require distinguishing images. Both independent branches start from Stage-1 MLP + original Qwen, '
        'Encoder features remain frozen. Each branch permits 80 updates, effective batch=8, generation every 20 steps, with early stop at 8/8 correct.', '',
        '| Training branch | LR | Update step | Correct initially | Correct after training | Final answer NLL | Answers donor-image gold after swap |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for branch, label in [('mlp_only', 'MLP only'), ('mlp_and_llm', 'MLP＋Qwen')]:
        r = overfit[branch]
        initial, final = r['initial'], r['final']
        lines.append(f"| {label} | {r['lr']:g} | {r['steps']} | {initial['exact_lowercase']}/8 | {final['exact_lowercase']}/8 | {final['mean_answer_nll']:.4g} | {final['wrong_image_matches_donor_gold']}/8 |")
    mlp_pass = overfit['mlp_only']['final']['exact_lowercase'] == 8
    joint_pass = overfit['mlp_and_llm']['final']['exact_lowercase'] == 8
    if mlp_pass or joint_pass:
        passed_label = 'Both branches ' if mlp_pass and joint_pass else 'At least one branch'
        lines.extend(['', passed_label + 'Memorizing eight image–answer pairs shows trainable image discrimination along features→MLP→Qwen. '
        'It rules out complete inability to learn this path, **not establishes general OCR, localization, or unseen-document generalization**. '
        'The model may memorize layout/feature/length shortcuts rather than actually read page numbers.'])
    else:
        lines.extend(['', 'Neither LR branch reaches 8/8 within these limited updates; first inspect small-sample convergence, gradients, and numerical execution. '
                      'Limited training does not prove an untrainable architecture or insufficient LLM capacity.'])
    lines.extend([
        '', 'The branches use different LRs and a tiny memorization set; they cannot rank frozen/unfrozen generalization. '
        'Formal weights/evaluation files remain unchanged; diagnostic weights are not saved as formal candidates.', '',
        '| Question ID | Gold (first) | Final MLP-only answer | Final MLP+Qwen answer |',
        '|---|---|---|---|',
    ])
    joint = {r['id']: r for r in overfit['mlp_and_llm']['final']['records']}
    def cell(text):
        return str(text).replace('|', '\\|').replace('\n', '<br>')
    for row in overfit['mlp_only']['final']['records']:
        lines.append(f"| {row['id']} | {cell(row['gold_answers'][0])} | {cell(row['prediction'])} | {cell(joint[row['id']]['prediction'])} |")
    lines.extend([
        '', '## Recommended Next Steps', '',
        '1. Retain the authors’ encoder and Qwen3-1.7B; begin with a DocVQA-only comparison in a separate output directory. Fix the training subset, '
        'Independent validation documents, matched steps/generation; compare continued MLP-only, MLP + low-LR full Qwen, and MLP + LoRA; '
        'Select by validation ANLS and correct/wrong-image sensitivity, not training loss or test scores.',
        '2. If small samples can be memorized but unseen documents remain weak, prioritize whether supervision teaches text reading and localization. '
        'Current mixture: DT-VQA 120,444/243,652 (about 49.4%), DocVQA 39,457 (about 16.2%); '
        'Task proportions merit controlled tests but are not established causes. An OCR-transcription alignment warmup can be tested separately, '
        'Then add VQA and inspect resolution, feature scale, template, and answer-termination tokens. These are proposed tests, not a disclosed author recipe.',
        '3. Change unfreezing timing, LR, or LoRA only if frozen controls consistently outperform unfrozen controls on validation at matched update counts. '
        'Evidence does not justify removing Stage 2 directly. Establish reliable image reading/alignment before a larger-LLM capacity comparison.', '',
        'Diagnosis priorities are established, but no unique cause or paper-level performance is established. '
        'Undisclosed author data/schedule and numerical/preprocessing/native equivalence still require controlled validation.', '',
        '## Reproduction Commands and Evidence', '',
        '```bash',
        '# NPU compute node; the script sets up venv/CANN. Do not submit duplicate diagnostics.',
        'sbatch scripts/slurm_alignment_diagnosis.sh',
        'sbatch scripts/slurm_overfit_probe.sh',
        '# Rebuild this report after both jobs succeed and verify every per-question metric.',
        '.venv/bin/python -m scripts.report_alignment_diagnosis',
        '```', '',
        '- Training/validation comparison: Slurm  `147042`, 128 sample/model combinations and 256 generations.',
        '- Eight-image training check: Slurm  `147044`. First  `147043` failed before updating because the diagnostic cached inference-mode tensors; '
        'Rerun no-gradient evaluation. Formal training already used no-grad and was unaffected by this diagnostic-only issue.',
        '- [selection.json](../artifacts/full/alignment_diagnosis/selection.json): fixed train/validation samples. '
        'Created automatically once; reruns verify identical deterministic source selection.',
        '- [summary.json](../artifacts/full/alignment_diagnosis/summary.json), '
        '[predictions.jsonl](../artifacts/full/alignment_diagnosis/predictions.jsonl): correct/wrong-image answers, gold, NLL, and token accuracy.',
        '- [Eight-image summary](../artifacts/full/alignment_diagnosis/overfit/summary.json), '
        '[Eight-image samples](../artifacts/full/alignment_diagnosis/overfit/selection.json), '
        '[Stepwise training records](../artifacts/full/alignment_diagnosis/overfit/training.jsonl): branch initialization, LR, steps, and evaluation details.',
        '- [Cross-question context audit](CONTEXT_AUDIT.md): completed input-isolation evidence for 226 questions.',
        '- [Formal evaluation results](FULL_TRAINING_RESULTS.md), [All DocVQA answers](DOCVQA_REVIEW.md). ',
    ])
    path = ROOT / 'docs/ALIGNMENT_DIAGNOSIS.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path)


if __name__ == '__main__':
    main()
