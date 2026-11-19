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
    doc=['# Newer ~7B-Class Extension: Qwen3-VL-8B with a Uniform Short-Answer Prompt','',f'Updated (UTC):{now()}. ','',
         '**Entry point**: `sbatch scripts/slurm_qwen3vl_short_extension.sh`; rebuild report: `.venv/bin/python -m scripts.report_qwen3vl_short_extension`. ','',
         '## Work Performed','',
         'Replace the proposed Qwen2.5-VL-7B extension with the newer, similarly sized official Qwen3-VL-8B-Instruct supported by the current Transformers version. Download a fixed revision and verify each LFS weight\'s SHA-256. Load the full native VLM on Ascend with all parameters frozen, without training, adapters, or the old compressor.','',
         f'Model revision: `{REVISION}`. It uses its own visual encoder and learned alignment; **it does not attach the MonkeyOCRv2 encoder to a newer LLM, nor replace this round’s no-retraining UND equivalence results**.','',
         '## Data and Configuration','',
         'Reuse`artifacts/compression_lab_v2/validation.jsonl`fixed 56 questions (8/task, 7 tasks, no OCRBench), appending instruction `Answer the question with only a short answer. Do not provide any explanation or Markdown formatting.` , changing response format only, without training or gold-based answer extraction. Retain original questions as a control. This explores native inference in the new environment and is not directly comparable to the 23,565-question formal mean. Each question is an independent image/text dialogue: greedy, use_cache=True, max_new_tokens=64, max_pixels=1003520, official processor, and qwen_vl_utils with 16-pixel patches, retaining native MRoPE/DeepStack. Actual resizing and visual-token counts differ across backbones.','',
         'Use original ANLS for DocVQA/InfoVQA, AccANLS for DT-VQA, relaxed accuracy for ChartQA, official denotation for WTQ, and DUE corpus F1 for KLC/DeepForm. Small-sample scores do not establish effective compression on the new backbone.','',
         '## Measured Results','',
         '|Task|Questions|Metric|Score (%)|','|---|---:|---|---:|']
    for task,m in metrics.items():doc.append(f'|{task}|{m["questions"]}|{m["metric"]}|{m["score_percent"]:.4f}|')
    doc += ['',f'Exploratory seven-task mean: **{result["macro_score_percent"]:.4f}**. ','',
        'Raw per-question records include question, gold, prediction, token IDs, visual-token count, and truncation flag, at `artifacts/tessfold_qwen3vl_short_extension_20261115/predictions.rank*.jsonl`. ','',
        '## Scope of the Next Steps','',
        'Successful native inference establishes only feasibility of this newer, similarly sized ~7B-class backbone. Keeping the MonkeyOCRv2 encoder with a newer LLM still requires new cross-modal alignment. Compressing complete Qwen3-VL-8B requires adapting 1,152-dimensional visual inputs, 4,096-dimensional question features, multiple DeepStack levels, and spatial positions, followed by compressor retraining. This round does not claim completion of those tasks.','',
        '[UND repair and validation](TESSFOLD_SYSTEM_REPAIR_20261115.md) · [New-backbone compatibility check](TESSFOLD_QWEN_UPGRADE_FEASIBILITY.md) · [Official model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)']
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
