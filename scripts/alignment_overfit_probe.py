"""Isolated 8-image/one-question training sanity check; never updates formal weights."""
import gc
import hashlib
import json
import os

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from vqa.full_train import TrainableVLM, set_runtime, amp, atomic_json
from vqa.model import resolve
from vqa.benchmark_metrics import benchmark_anls

OUT = resolve('artifacts/full/alignment_diagnosis/overfit')


def select():
    candidates=[]
    for line in (resolve('data/full')/'train.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['task']=='DocVQA' and row['question'].strip().lower()=='what is the page number?' and row['answers'][0].strip().isdigit():
            candidates.append(row)
    candidates.sort(key=lambda r:hashlib.sha256(('overfit-sanity-'+r['id']).encode()).hexdigest())
    rows=[];seen_answers=set();seen_docs=set()
    for row in candidates:
        answer=row['answers'][0].strip()
        if answer in seen_answers or row['document_id'] in seen_docs:continue
        # Require exactly the same question token sequence in all eight samples.
        seen_answers.add(answer);seen_docs.add(row['document_id'])
        rows.append({**row,'source_question':row['question'],'question':'What is the page number?'})
        if len(rows)==8:break
    assert len(rows)==8
    return rows


# Features remain cached when training resumes; inference_mode would create
# inference tensors that LayerNorm cannot save for backward.
@torch.no_grad()
def evaluate(model,cfg,rows,branch,step):
    model.eval()
    records=[]
    for i,row in enumerate(rows):
        donor=rows[(i+1)%len(rows)]
        wrong={**row,'document_id':donor['document_id'],'image':donor['image']}
        with amp(cfg):
            prediction=model.predict(row)
            wrong_prediction=model.predict(wrong)
            loss,_=model.loss(row)
        records.append({'id':row['id'],'gold_answers':row['answers'],'prediction':prediction,
                        'anls':benchmark_anls(prediction,row['answers']),
                        'wrong_image_prediction':wrong_prediction,'wrong_image_donor_gold':donor['answers'],
                        'wrong_image_anls_vs_original_gold':benchmark_anls(wrong_prediction,row['answers']),
                        'wrong_image_anls_vs_donor_gold':benchmark_anls(wrong_prediction,donor['answers']),
                        'answer_nll':float(loss)})
    result={'branch':branch,'step':step,'samples':len(rows),'anls_percent':100*sum(r['anls'] for r in records)/len(rows),
            'exact_lowercase':sum(any(r['prediction'].lower()==a.lower() for a in r['gold_answers']) for r in records),
            'mean_answer_nll':sum(r['answer_nll'] for r in records)/len(rows),
            'wrong_image_matches_donor_gold':sum(any(r['wrong_image_prediction'].lower()==a.lower() for a in r['wrong_image_donor_gold']) for r in records),
            'records':records}
    atomic_json(OUT/f'{branch}_step{step}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(resolve('configs/full.json').read_text())
    cfg['max_new_tokens']=32
    set_runtime(cfg)
    rows=select()
    atomic_json(OUT/'selection.json',{'source':'Only actual DocVQA training split, no validation/test training',
                                    'samples':rows,'scope':'Eight distinct documents, same exact question, eight distinct numeric answers; memorization/gradient sanity, not benchmark'})
    summary={}
    for branch in ('mlp_only','mlp_and_llm'):
        torch.manual_seed(cfg['seed'])
        model=TrainableVLM(cfg)
        model.projector.load_state_dict(torch.load(resolve('artifacts/full/stage1/projector.pt'),map_location='cpu',weights_only=True))
        model.lm.requires_grad_(branch=='mlp_and_llm')
        initial=evaluate(model,cfg,rows,branch,0)
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        params=[p for p in model.parameters() if p.requires_grad]
        lr=1e-3 if branch=='mlp_only' else 1e-5
        optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.0,foreach=False)
        final=initial
        for step in range(1,81):
            model.train();optimizer.zero_grad(set_to_none=True)
            losses=[]
            for row in rows:
                with amp(cfg):loss=model([row])
                assert torch.isfinite(loss)
                (loss/len(rows)).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(params,1.0,foreach=False)
            assert torch.isfinite(norm)
            optimizer.step()
            record={'branch':branch,'step':step,'loss':sum(losses)/len(losses),'gradient_norm':float(norm),'lr':lr,'effective_batch':8}
            with (OUT/'training.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
            if step%20==0:
                final=evaluate(model,cfg,rows,branch,step)
                if final['exact_lowercase']==len(rows):break
        summary[branch]={'initial':initial,'final':final,'lr':lr,'steps':step,'initial_checkpoint':'stage1 MLP + original Qwen3-1.7B'}
        del model,optimizer,params,loss,norm
        gc.collect();torch.npu.empty_cache()
    atomic_json(OUT/'summary.json',summary)


if __name__=='__main__':main()
