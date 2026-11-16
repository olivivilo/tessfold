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
    doc=['# 新版 Qwen3-VL-8B：原生可运行性与短答案格式对照','',f'生成时间 UTC：{now()}。','',
        '**入口**：`.venv/bin/python -m scripts.report_qwen3vl_prompt_comparison`。两组原始入口为`sbatch scripts/slurm_qwen3vl_extension.sh`和`sbatch scripts/slurm_qwen3vl_short_extension.sh`。','',
        '## 结论和范围','',
        '已在本集群 Ascend 上跑通官方 Qwen3-VL-8B-Instruct，作为原建议 Qwen2.5-VL-7B 的更新、规模接近的独立扩展候选。使用固定 revision 和通过官方 SHA-256 的权重，全部参数冻结，保留原生视觉编码器、merger、MRoPE 与 DeepStack。','',
        '**这不是将新版 LLM 直接接入 MonkeyOCRv2 encoder。** 后者的 2048 维对齐与问题特征不能直接接到新模型的 4096 维空间。旧压缩器也未迁移或重训；本轮 UND 压缩主实验仍使用原官方 Qwen3-1.7B。','',
        '## 为什么增加第二组','',
        '第一组直接使用原问题，新模型常输出解释性段落，部分达到64 token上限。即便回答中出现正确答案，整段文本也可能在ANLS、ChartQA relaxed accuracy或WTQ denotation下得0分。为区分模型能否运行与答题格式问题，保留原组，另增加一条对所有56题相同的输入指令：','',
        '> Answer the question with only a short answer. Do not provide any explanation or Markdown formatting.','',
        '没有训练、使用gold抽取答案、按题目改写规则或提高生成长度。两组均为greedy、max_new_tokens=64、max_pixels=1003520，复用既有固定56题（7任务各8，不包含OCRBench）。第二组的动机来自第一组已观察到的输出，因此属于开发集探索性格式对照，不能宣称未见测试集上的预注册提升。','',
        '## 实测结果','',
        '|任务|题数|指标|原问题（%）|统一短答案（%）|','|---|---:|---|---:|---:|']
    for task,m in reports[0]['metrics'].items():
        n=reports[1]['metrics'][task];assert m['metric']==n['metric'] and m['questions']==n['questions']
        doc.append(f'|{task}|{m["questions"]}|{m["metric"]}|{m["score_percent"]:.4f}|{n["score_percent"]:.4f}|')
    doc += [f'|七任务宏平均|56|探索性均分|{reports[0]["macro_score_percent"]:.4f}|{reports[1]["macro_score_percent"]:.4f}|','',
        f'达到64 token上限：原问题 **{counts[0]}/56**，短答案 **{counts[1]}/56**。本表不能与23565题、八任务正式评测直接比较，也没有证明8B压缩后的分数。','',
        '## 可复查的作答示例','',
        '以下固定取ID排序最前的两道DocVQA题，不按得分挑选；全部逐题记录保留在两组JSONL中。','']
    selected=sorted(k for k,r in records[0].items() if r['task']=='DocVQA')[:2]
    for id_ in selected:
        a,b=records[0][id_],records[1][id_]
        doc += [f'### {id_}','',f'问题：{a["question"]}','',f'Gold：{json.dumps(a["answers"],ensure_ascii=False)}','',
            f'原问题输出（得分 {a["score"]:.4f}）：','',a['prediction'],'',
            f'短答案输出（得分 {b["score"]:.4f}）：','',b['prediction'],'']
    doc += ['## 对后续压缩实验的含义','',
        '新主干的原生推理与格式适配可行，但仍需独立确认其未压缩teacher在正式数据上的表现。若保留MonkeyOCRv2 encoder，需新的视觉语言对齐；若用完整Qwen3-VL-8B，需适配1152维视觉输入、4096维问题特征、多层视觉支路和空间位置，再训练压缩器。不能直接继承旧P生成器并称为无损替换。','',
        '[原问题组完整报告](TESSFOLD_QWEN3VL_8B_NATIVE_BASELINE.md) · [短答案组完整报告](TESSFOLD_QWEN3VL_8B_SHORT_BASELINE.md) · [UND修复闭环](TESSFOLD_REPAIR_MEASURED_RESULTS.md) · [官方模型](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)']
    (ROOT/'docs/TESSFOLD_QWEN3VL_8B_PROMPT_COMPARISON.md').write_text('\n'.join(doc)+'\n')
    print(json.dumps(dict(raw_macro=reports[0]['macro_score_percent'],short_macro=reports[1]['macro_score_percent'],raw_truncated=counts[0],short_truncated=counts[1])))


if __name__=='__main__':main()
