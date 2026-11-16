"""Separate frozen native Qwen3-VL-8B baseline, not an UND LLM transplant.

The existing 56-question fixed development cohort is used unchanged. Never
attribute its scores to MonkeyOCRv2 or to an adapted TessFold compressor.
"""
import json,os,time,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from scripts.train_meta_compress import init_runtime
from scripts.download_qwen3vl_extension import ROOT,MODEL,REVISION
BASE=ROOT/'artifacts/tessfold_qwen3vl_short_extension_20261115'
from scripts.tessfold_repair.common import save,load,lines,sha,now
from vqa.benchmark_metrics import native_score,due_field_f1


def report():
    rows=[r for p in BASE.glob('predictions.rank*.jsonl') for r in lines(p)]
    assert len(rows)==56 and len({r['id'] for r in rows})==56
    grouped={task:[r for r in rows if r['task']==task] for task in sorted({r['task'] for r in rows})}
    metrics={}
    for task,rr in grouped.items():
        if task in ['KLC','DeepForm']:m=due_field_f1(rr)
        else:
            label={'DocVQA':'ANLS','InfoVQA':'ANLS','DT-VQA':'AccANLS','ChartQA':'relaxed accuracy','WTQ':'denotation accuracy'}[task]
            m=dict(score_percent=sum(r['score'] for r in rr)/len(rr)*100,metric=label)
        metrics[task]=dict(questions=len(rr),**m)
    result=dict(time=now(),model='Qwen/Qwen3-VL-8B-Instruct',revision=REVISION,questions=56,
        weights_frozen=True,training=False,compression=False,metrics=metrics,
        macro_score_percent=sum(m['score_percent'] for m in metrics.values())/len(metrics),
        prompt_variant='uniform_short_answer_instruction',scope='56 fixed development questions (7 tasks x8), native Qwen visual+language model; exploratory, not the23565QA benchmark and not an aligned UND transplant.',
        download_manifest_sha256=sha(BASE/'download_complete.json'))
    save(BASE/'complete.json',result)
    doc=['# 新版约7B扩展：Qwen3-VL-8B统一短答案推理对照','',f'更新时间UTC：{now()}。','',
         '**入口**：`sbatch scripts/slurm_qwen3vl_short_extension.sh`；报告重建：`.venv/bin/python -m scripts.report_qwen3vl_short_extension`。','',
         '## 实际做了什么','',
         '将原建议的Qwen2.5-VL-7B扩展候选改为更新、规模相近且当前Transformers支持的官方Qwen3-VL-8B-Instruct。下载固定revision，LFS权重逐文件核对SHA256，在Ascend上加载完整原生VLM并冻结全部参数。未训练、未使用adapter、未接入旧压缩器。','',
         f'模型revision：`{REVISION}`。它使用自己的视觉编码器和已训练视觉语言对齐；**不是把MonkeyOCRv2的encoder接到新版LLM上，也不是本轮UND零重训等价实验的替代结果**。','',
         '## 数据与配置','',
         '复用`artifacts/compression_lab_v2/validation.jsonl`的固定56题：7个任务各8题，不包含OCRBench。统一追加指令`Answer the question with only a short answer. Do not provide any explanation or Markdown formatting.`，只改变答题格式，不进行训练，也不按gold抽取答案。保留原问题组作对照；仅为新环境原生推理的探索性验证；不得与23565题正式评测的均分直接比较。每题独立图文对话，greedy、use_cache=True、max_new_tokens=64、max_pixels=1003520。使用官方processor、16像素patch的qwen_vl_utils，保留原生MRoPE和DeepStack。不同主干的实际图像缩放、视觉token数不同。','',
         '对DocVQA/InfoVQA使用原ANLS，DT-VQA使用AccANLS，ChartQA使用relaxed accuracy，WTQ使用官方denotation，KLC/DeepForm使用DUE corpus F1。小样本分数不能证明新主干上的压缩器有效。','',
         '## 实际结果','',
         '|任务|题数|指标|分数（%）|','|---|---:|---|---:|']
    for task,m in metrics.items():doc.append(f'|{task}|{m["questions"]}|{m["metric"]}|{m["score_percent"]:.4f}|')
    doc += ['',f'七任务探索性均分：**{result["macro_score_percent"]:.4f}**。','',
        '原始逐题结果包含问题、gold、prediction、token IDs、视觉token数和截断标记，位于`artifacts/tessfold_qwen3vl_short_extension_20261115/predictions.rank*.jsonl`。','',
        '## 下一步的明确边界','',
        '原生推理通过只证明该模型可作为新版约7B扩展基模。若要保留MonkeyOCRv2 encoder接入新版LLM，仍需训练新的跨模态对齐；若采用完整Qwen3-VL-8B开展压缩，需要适配1152维视觉输入、4096维问题特征、多层DeepStack和空间位置，并重新训练压缩器。本轮没有声称这些工作已经完成。','',
        '[UND修复闭环](TESSFOLD_SYSTEM_REPAIR_20261115.md) · [新版模型兼容性检查](TESSFOLD_QWEN_UPGRADE_FEASIBILITY.md) · [官方模型](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)']
    path=ROOT/'docs/TESSFOLD_QWEN3VL_8B_SHORT_BASELINE.md';path.write_text('\n'.join(doc)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


@torch.no_grad()
def main():
    assert (BASE/'download_complete.json').exists()
    rank,world,_,device=init_runtime('npu',4)
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    source=ROOT/'artifacts/compression_lab_v2/validation.jsonl';rows=lines(source)[rank::world]
    runtime=dict(time=now(),rank=rank,world=world,job=os.environ.get('SLURM_JOB_ID'),
        revision=REVISION,source_sha256=sha(Path(__file__)),data_sha256=sha(source),
        torch=torch.__version__,device=torch.npu.get_device_name(),dtype='bfloat16',attention='sdpa',
        max_new_tokens=64,max_pixels=1003520,do_sample=False,use_cache=True,prompt_variant='uniform_short_answer_instruction',
        parameters=sum(p.numel() for p in model.parameters()),
        weights_frozen=all(not p.requires_grad for p in model.parameters()))
    runtime_path=BASE/f'runtime.rank{rank}.json'
    previous=load(runtime_path)
    if previous:
        stable=['rank','world','revision','source_sha256','data_sha256','torch','dtype','attention',
                'max_new_tokens','max_pixels','do_sample','use_cache','parameters','weights_frozen']
        assert all(previous[k]==runtime[k] for k in stable),'Resume signature changed; create a new version'
        runtime['previous_jobs']=previous.get('previous_jobs',[])+[previous['job']]
    save(runtime_path,runtime)
    path=BASE/f'predictions.rank{rank}.jsonl';done={r['id'] for r in lines(path)}
    with path.open('a') as f:
        for row in rows:
            if row['id'] in done:continue
            messages=[{'role':'user','content':[{'type':'image','image':str(ROOT/row['image']),'max_pixels':1003520},
                {'type':'text','text':row['question']+'\nAnswer the question with only a short answer. Do not provide any explanation or Markdown formatting.'}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            images,videos=process_vision_info(messages,image_patch_size=16)
            batch=processor(text=prompt,images=images,videos=videos,padding=True,return_tensors='pt')
            batch={k:v.to(device) for k,v in batch.items()}
            assert not any(k in batch for k in ['past_key_values','cache_position'])
            torch.npu.synchronize();start=time.perf_counter()
            output=model.generate(**batch,do_sample=False,use_cache=True,max_new_tokens=64)
            torch.npu.synchronize();elapsed=time.perf_counter()-start
            generated=output[0,batch['input_ids'].shape[1]:].cpu().tolist()
            prediction=processor.decode(generated,skip_special_tokens=True)
            record=dict(row,prediction=prediction,score=native_score(prediction,row),generated_ids=generated,
                per_question_metric='normalized exact match diagnostic; official corpus F1 reported separately' if row['task'] in ['KLC','DeepForm'] else 'native task metric',
                input_tokens=batch['input_ids'].shape[1],visual_tokens=int((batch['input_ids']==model.config.image_token_id).sum()),
                output_tokens=len(generated),hit_token_limit=len(generated)==64,inference_seconds=elapsed,
                model='Qwen3-VL-8B-Instruct-native',compression=False)
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            print(json.dumps(dict(rank=rank,id=row['id'],prediction=prediction,score=record['score'])),flush=True)
    dist.barrier()
    if rank==0:report()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
