"""Paired interpretation of the raw and uniformly short native8B prompts."""
from pathlib import Path
import json
from scripts.tessfold_repair.common import ROOT,load,lines,save,now


def main():
    roots=[ROOT/'artifacts'/name for name in ['tessfold_qwen3vl_extension_20261115','tessfold_qwen3vl_short_extension_20261115']]
    reports=[load(p/'complete.json') for p in roots];assert all(reports)
    records=[{r['id']:r for f in p.glob('predictions.rank*.jsonl') for r in lines(f)} for p in roots]
    assert len(records[0])==len(records[1])==56 and records[0].keys()==records[1].keys()
    assert reports[0]['revision']==reports[1]['revision']
    counts=[sum(r['hit_token_limit'] for r in rr.values()) for rr in records]
    result=dict(time=now(),questions=56,revision=reports[0]['revision'],weights_frozen=True,
        raw=reports[0],short=reports[1],raw_truncated=counts[0],short_truncated=counts[1],
        limitations='Exploratory reused development cohort, 7tasksx8, no OCRBench or compression. Short prompt motivated by observed verbose/truncated outputs, not a pre-registered unseen-test win or evidence of an aligned UND transplant.')
    save(roots[1]/'prompt_comparison.json',result)
    doc=['# Qwen3-VL-8B: Native Execution and Short-Answer Prompt Comparison','',f'Generated (UTC):{now()}. ','',
        '**Entry point**: `.venv/bin/python -m scripts.report_qwen3vl_prompt_comparison`. Original entry points for both conditions: `sbatch scripts/slurm_qwen3vl_extension.sh`and`sbatch scripts/slurm_qwen3vl_short_extension.sh`. ','',
        '## Conclusions and Scope','',
        'Official Qwen3-VL-8B-Instruct runs on the cluster\'s Ascend hardware as a newer, similarly sized independent alternative to the proposed Qwen2.5-VL-7B. Its revision is fixed and weights pass official SHA-256 checks. All parameters are frozen; native visual encoder, merger, MRoPE, and DeepStack are retained.','',
        '**This is not direct attachment of a newer LLM to the MonkeyOCRv2 encoder.** Its 2,048-dimensional alignment/question features cannot directly enter the newer 4,096-dimensional space. Earlier compressors are not transferred or retrained; the main UND experiment still uses official Qwen3-1.7B.','',
        '## Rationale for the Second Prompt Condition','',
        'The first condition uses original questions. The newer model often generates explanations, sometimes reaching the 64-token cap. Even a passage containing the correct answer can score zero under ANLS, ChartQA relaxed accuracy, or WTQ denotation. To separate execution feasibility from response-format effects, retain that condition and add one identical instruction to all 56 questions:','',
        '> Answer the question with only a short answer. Do not provide any explanation or Markdown formatting.','',
        'No training, gold-based answer extraction, question-specific rewriting, or longer generation is used. Both conditions use greedy decoding, max_new_tokens=64, max_pixels=1003520, and the same fixed 56 questions (8 each from 7 tasks, excluding OCRBench). The second condition follows inspection of the first outputs, so it is exploratory development-set formatting analysis, not preregistered improvement on an unseen test set.','',
        '## Measured Results','',
        '|Task|Questions|Metric|Original prompt (%)|Uniform short-answer prompt (%)|','|---|---:|---|---:|---:|']
    for task,m in reports[0]['metrics'].items():
        n=reports[1]['metrics'][task];assert m['metric']==n['metric'] and m['questions']==n['questions']
        doc.append(f'|{task}|{m["questions"]}|{m["metric"]}|{m["score_percent"]:.4f}|{n["score_percent"]:.4f}|')
    doc += [f'|Seven-task macro average|56|Exploratory mean|{reports[0]["macro_score_percent"]:.4f}|{reports[1]["macro_score_percent"]:.4f}|','',
        f'Reaching 64-token cap: original prompt **{counts[0]}/56**; short-answer condition **{counts[1]}/56**. This table is not directly comparable to the 23,565-question/eight-task benchmark and does not establish compressed 8B quality.','',
        '## Inspectable Answer Examples','',
        'Use the first two DocVQA questions sorted by ID, not score-selected examples. Complete records remain in both JSONL sets.','']
    selected=sorted(k for k,r in records[0].items() if r['task']=='DocVQA')[:2]
    for id_ in selected:
        a,b=records[0][id_],records[1][id_]
        doc += [f'### {id_}','',f'Question:{a["question"]}','',f'Gold: {json.dumps(a["answers"],ensure_ascii=False)}','',
            f'Original-prompt output (score {a["score"]:.4f}): ','',a['prediction'],'',
            f'Short-answer output (score {b["score"]:.4f}): ','',b['prediction'],'']
    doc += ['## Implications for Subsequent Compression Experiments','',
        'Native inference and prompt-format adaptation work, but the uncompressed teacher still needs independent formal evaluation. Retaining the MonkeyOCRv2 encoder requires new visual–language alignment. Full Qwen3-VL-8B compression requires adapting 1,152-dimensional visual inputs, 4,096-dimensional question features, multilevel visual branches, and spatial positions, then training a new compressor. Reusing old P cannot be called a lossless replacement.','',
        '[Full original-prompt report](TESSFOLD_QWEN3VL_8B_NATIVE_BASELINE.md) · [Full short-answer report](TESSFOLD_QWEN3VL_8B_SHORT_BASELINE.md) · [UND repair and validation](TESSFOLD_REPAIR_MEASURED_RESULTS.md) · [Official model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)']
    (ROOT/'docs/TESSFOLD_QWEN3VL_8B_PROMPT_COMPARISON.md').write_text('\n'.join(doc)+'\n')
    print(json.dumps(dict(raw_macro=reports[0]['macro_score_percent'],short_macro=reports[1]['macro_score_percent'],raw_truncated=counts[0],short_truncated=counts[1])))


if __name__=='__main__':main()
