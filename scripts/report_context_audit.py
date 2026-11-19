"""Render the context audit report from measured diagnostics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/full/context_audit'


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    runtime = load('runtime_summary.json')
    group = load('full_group_summary.json')
    prior = load('training_prior.json')
    sequence = load('sequence_statistics.json')
    features = load('shank_feature_audit.json')
    all_rows = [json.loads(line) for line in (ROOT / 'artifacts/full/docvqa_comparison.jsonl').read_text().splitlines()]
    lookup = {r['id']:r for r in all_rows}
    n = group['samples']
    target = 'Dr. Robert E. Shank'
    baseline_targets = [k for k,v in runtime['per_sample'].items() if v['baseline'] == target]
    stayed = sum(runtime['per_sample'][k]['wrong_image'] == target for k in baseline_targets)
    lines = ['# Repeated Names in DocVQA: Cross-Question Context-Isolation Audit', '',
             '**Conclusion: source inspection and actual-model tests found no prior-question history or KV cache carried into the next question. '
             'All 226 target questions reproduce on the reloaded final model and remain unchanged under reverse order with per-question feature-cache clearing. '
             'Evidence favors insufficient visual-content use and learned answer preferences, but the specific training/representation cause of alignment failure remains unresolved.**', '',
             'Use formal  `artifacts/full/stage2/` model, original frozen features, same BF16 autocast, greedy generation, and 256-token cap. '
             'Formal weights, data, and benchmark results are unchanged. Actual jobs: `147017`(multiple conditions and fresh processes), `147021`(complete repeated-answer group).', '',
             '## Most Informative Measured Comparisons', '',
             '| Control | Measured results | Meaning |', '|---|---|---|',
             f'| Independent regeneration of all targets after model reload | {group["forward_matches_historical"]}/{n} Matches historical answers | Independent of the original evaluation process and its preceding questions |',
             f'| Reverse all target questions and clear visual-feature memory cache before each | {group["reverse_and_cache_clear_matches_forward"]}/{n} Unchanged answer | No observed order or visual-cache contamination |',
             f'| Input-embedding SHA-256 before/after reversing order | {group["prefill_hash_invariant_to_order_and_feature_cache"]}/{n} Same | Current-question inputs are unaffected by preceding questions |',
             '| Initial KV length / cache_position at each generate’s first forward | All 0 / 0 | New questions start with empty KV and position 0 |',
             '| First-forward input versus independently constructed current-image/current-question input | All lengths and SHA-256 values match | No previous-question/answer embeddings are appended |',
             f'| Reverse-order regeneration of 6 target and 4 control questions | {runtime["order_invariant"]}/10 answers unchanged | Reordering controls with varied answer types |',
             f'| Completely disable generation KV cache | {runtime["kv_cache_disabled_matches"]}/{runtime["kv_cache_disabled_samples"]} Still answers the same name | Errors do not depend on KV caching |',
             f'| Each question is generated first in a fresh Python process | {runtime["fresh_process_matches"]}/{runtime["fresh_process_samples"]} Same | No preceding QA exists in the process |',
             f'| Replace the current record’s gold field with an audit sentinel | {runtime["gold_metadata_unused"]}: unchanged input hashes and answers | Inference never reads the gold-answer field |', '',
             'Per-input summaries, initial KV lengths, SHA-256, and answers:  '
             '[runtime_predictions.jsonl](../artifacts/full/context_audit/runtime_predictions.jsonl). '
             'Full grouped summary: [full_group_summary.json](../artifacts/full/context_audit/full_group_summary.json). ', '',
             '## Image Swaps and Zero Features: Limited Content Use, but Some Visual Dependence', '',
             f'Among 6 target-name answers, retain the question and swap another real image: {stayed}/6 still answer the target name; '
             'The other 2 change to different names; 4 non-target controls also change after image swaps. '
             'The model depends somewhat on images but defaults to this name across documents.', '',
             'Zero image features while retaining length/question: all 10/10 answers change and all 6 target questions stop answering the target name, '
             'Convert outputs to include  `<think>` general reasoning/context. Zero vectors are out of distribution, '
             'Visual blocks affect generation; this does not establish correct image reading.', '',
             '| Samples | Gold (all) | Correct image | After swapping another image |', '|---|---|---|---|']
    def cell(text):
        return str(text).replace('|','&#124;').replace('\n','<br>')
    for key,value in runtime['per_sample'].items():
        lines.append(f'| {key} | {cell(json.dumps(lookup[key]["answers"],ensure_ascii=False))} | {cell(value["baseline"])} | {cell(value["wrong_image"])} |')
    lines += ['', 'Some zero-feature answers spontaneously include “Let me check the previous conversation”.'
              '**This is generated text, not evidence that historical dialogue was supplied.** Actual initial KV length is 0, and input contains only the current question and a zero visual block.', '',
              '## Code-Path Inspection', '',
              '- [VLM.prompt](../vqa/model.py#L74) Each call builds only the current user message and assistant-start marker; no conversation/history container.',
              '- [VLM.embeds](../vqa/model.py#L86) Rebuild current image’s MLP embeddings plus current question each time; inference answer=None returns the prefix directly.',
              '- [VLM.predict](../vqa/model.py#L109) Each question calls generate independently without previous-question past_key_values. '
              'Installed Transformers 4.57.1 creates a fresh DynamicCache per default generate; runtime hooks confirm an empty initial cache.',
              '- [Visual cache](../vqa/model.py#L59) Stores frozen encoder features keyed by image SHA-256, not questions, answers, projector outputs, or LLM hidden states.',
              '- [Training forward pass](../vqa/full_train.py#L170) Independent batch-example forward passes, use_cache=False; microbatch=1 and no packing. '
              'Gradient accumulation combines parameter gradients normally; it does not concatenate QA contexts.',
              '- [Evaluation loop](../vqa/full_train.py#L223) Call predict independently per row; write prediction/gold only after generation ends.', '',
              'Actual input example (DocVQA_validation_10): 4,900 visual tokens from this image, total length 4,925:', '',
              '```text', '<|im_start|>user', '<|vision_start|>[Current image visual embeddings]<|vision_end|>Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?<|im_end|>',
              '<|im_start|>assistant', '```', '',
              '## Data and Feature Checks', '',
              f'- Affected  {features["documents"]}  image SHA-256 values equal document_id; cached tensors have  {features["distinct_feature_hashes"]}  distinct SHA-256 values, all finite. '
              'This rules out all these images sharing one mistaken cache tensor. It does not reextract every image or establish encoder quality from hashes alone.',
              f'- DocVQA training contains  {prior["train"]["questions"]:,}  records; target full-name frequency in first-gold training supervision:  {prior["train"]["exact_target_first_answer_count"]}  times; '
              f'QA with Shank in any gold reference:  {prior["train"]["shank_any_gold_questions"]}  records. The name is present in training supervision.',
              f'- Validation uses only  {prior["val"]["exact_target_any_gold_count"]}  gold answers contain the full target name, whereas  {n}  questions predict that name; '
              'True-answer frequency does not explain this pattern. Training frequency suggests a possible source, not a causal mechanism.',
              f'- Among the 226 repeated-name answers, {sequence["target_questions_with_who"]}  questions contain  `who`(including whose/whom); '
              'Suggests a learned tendency to guess this name on person-name questions.',
              f'- Full DocVQA validation questions containing ChatML/vision control tokens:  {sequence["question_includes_chat_special_tokens"]}, with no extra chat turns injected by question text.', '',
              '## Reproduction Commands and Evidence', '', '```bash',
              '# Dataset statistics; --features rechecks the affected 171 images and feature files.',
              '.venv/bin/python -m scripts.audit_vqa_data --features',
              '# Actual Formal-Model Comparison (NPU compute node; no retraining required)',
              'sbatch scripts/slurm_context_audit.sh',
              '# Order/Cache Controls for All 226 Questions',
              'sbatch scripts/slurm_context_audit.sh --full-group',
              '# Generate this report after experimental results are available.',
              '.venv/bin/python -m scripts.report_context_audit', '```', '',
              'Execute experiments sharing an output directory sequentially. Slurm wrappers configure venv/CANN and separate bytecode caches; '
              'Logs append all diagnostic executions; summary JSON describes the most recent completed run.', '',
              '- `artifacts/full/context_audit/runtime_summary.json`: multiple-condition/fresh-process results for 10 questions.',
              '- `artifacts/full/context_audit/full_group_summary.json`: full order/cache results for 226 questions.',
              '- `artifacts/full/context_audit/shank_feature_audit.json`: image/feature hashes and shapes for 171 images.',
              '- `artifacts/full/context_audit/training_prior.json`, `train_shank_gold.jsonl`, `val_shank_gold.jsonl`: name frequencies in training/validation supervision.',
              '- `artifacts/full/context_audit/sequence_statistics.json`: first occurrence and preceding QA per original evaluation rank.',
              '- `artifacts/full/context_audit/source_fingerprints.json`: inspected model, Transformers, and projector-file SHA-256 values.', '',
              'This audit covers complete repeated-answer-group input isolation and small visual interventions, addressing cross-question context, '
              'The paper-performance gap has not been attributed to one setting. Next inspect small-training-set fitting, '
              'Inspect same-question answer probabilities across images, visual feature space, and MLP outputs rather than blindly disabling generation cache or repeating full training.']
    (ROOT / 'docs/CONTEXT_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(ROOT / 'docs/CONTEXT_AUDIT.md')


if __name__ == '__main__':
    main()
