"""Compare stage-1 and stage-2 training fit and dependence on the current image."""
import gc
import hashlib
import json
import os

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from vqa.model import VLM, resolve
from vqa.full_train import set_runtime, amp, atomic_json
from vqa.benchmark_metrics import benchmark_anls

OUT = resolve('artifacts/full/alignment_diagnosis')


def select():
    """Persist a deterministic, distinct-document sample from each source split."""
    selection = {}
    for split in ('train', 'val'):
        rows = [json.loads(line) for line in (resolve('data/full') / f'{split}.jsonl').read_text().splitlines() if line]
        candidates = sorted((row for row in rows if row['task'] == 'DocVQA'),
                            key=lambda row: hashlib.sha256(('diagnosis-20261015-' + row['id']).encode()).hexdigest())
        selected, seen = [], set()
        for row in candidates:
            if row['document_id'] in seen:
                continue
            selected.append(row)
            seen.add(row['document_id'])
            if len(selected) == 32:
                break
        assert len(selected) == 32
        selection[split] = selected
    path = OUT / 'selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == selection, 'Saved selection differs from current corpus; investigate before rerunning'
    else:
        atomic_json(path, selection)
    return selection


@torch.inference_mode()
def measure(model, cfg, row, donor=None):
    sample = row if donor is None else {**row, 'document_id': donor['document_id'], 'image': donor['image']}
    with amp(cfg):
        embeds, targets, prefix = model.embeds(sample, row['answers'][0])
        states = model.lm.model(inputs_embeds=embeds, use_cache=False, return_dict=True).last_hidden_state
        logits = model.lm.lm_head(states[:, prefix-1:-1]).float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        predicted_ids = logits.argmax(-1)
        # EOS is excluded from token accuracy, but included in NLL as in training.
        token_count = max(1, targets.numel()-1)
        accuracy = float((predicted_ids[:, :token_count] == targets[:, :token_count]).float().mean())
        first = int(predicted_ids[0, 0])
        del states, logits, embeds
        prediction = model.predict(sample)
    return {'prediction': prediction, 'anls': benchmark_anls(prediction, row['answers']),
            'exact_lowercase': any(prediction.lower() == a.lower() for a in row['answers']),
            'answer_nll': float(loss), 'answer_tokens_including_eos': targets.numel(),
            'teacher_forced_answer_token_accuracy': accuracy,
            'first_token_correct': first == int(targets[0, 0]),
            'first_token': model.tokenizer.decode([first]),
            'used_document_id': sample['document_id'], 'visual_tokens': model.feature(sample).shape[0]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = select()
    cfg = json.loads(resolve('configs/full.json').read_text())
    set_runtime(cfg)
    # Save deterministic selection and make clear this is a diagnostic sample.
    atomic_json(OUT / 'manifest.json', {'selection': 'SHA256(diagnosis-20261015- + id), first 32 distinct documents per split',
                                       'train_samples': 32, 'val_samples': 32, 'stages': [1, 2],
                                       'test_used': False, 'weights_modified': False,
                                       'wrong_image_policy': 'Nearest visual token count from another selected document; report actual lengths'})
    all_results = []
    for stage in (1, 2):
        stage_cfg = {**cfg, 'language_model': 'models/Qwen3-1.7B' if stage == 1 else 'artifacts/full/stage2/language_model'}
        model = VLM(stage_cfg)
        model.projector.load_state_dict(torch.load(resolve(f'artifacts/full/stage{stage}/projector.pt'), map_location='cpu', weights_only=True))
        model.eval()
        for split, rows in selection.items():
            shapes = {r['document_id']: model.feature(r).shape[0] for r in rows}
            for i, row in enumerate(rows):
                donors = [r for r in rows if r['document_id'] != row['document_id']]
                donor = min(donors, key=lambda r:(abs(shapes[r['document_id']]-shapes[row['document_id']]), r['id']))
                actual = measure(model, stage_cfg, row)
                wrong = measure(model, stage_cfg, row, donor)
                result = {'stage': stage, 'split': split, 'id': row['id'], 'question': row['question'],
                          'gold_answers': row['answers'], 'document_id': row['document_id'],
                          'correct_image': actual, 'wrong_image': wrong,
                          'wrong_minus_correct_nll': wrong['answer_nll']-actual['answer_nll']}
                all_results.append(result)
                with (OUT / 'predictions.jsonl').open('a') as handle:
                    handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                print(json.dumps({'stage':stage,'split':split,'done':i+1,'anls':actual['anls'],
                                   'nll':actual['answer_nll'],'wrong_minus_correct_nll':result['wrong_minus_correct_nll']},ensure_ascii=False),flush=True)
        del model
        gc.collect()
        torch.npu.empty_cache()
    summary = {'samples_per_split':32,'scope':'Small training-fit and validation diagnostic, not a new benchmark score','stages':{}}
    for stage in (1,2):
        summary['stages'][stage]={}
        for split in selection:
            rr=[r for r in all_results if r['stage']==stage and r['split']==split]
            def mean(fn):return sum(fn(r) for r in rr)/len(rr)
            summary['stages'][stage][split]={'samples':len(rr), 'correct_image_anls_percent':100*mean(lambda r:r['correct_image']['anls']),
                                           'wrong_image_anls_percent':100*mean(lambda r:r['wrong_image']['anls']),
                                           'exact_lowercase_percent':100*mean(lambda r:r['correct_image']['exact_lowercase']),
                                           'mean_correct_image_answer_nll':mean(lambda r:r['correct_image']['answer_nll']),
                                           'mean_wrong_image_answer_nll':mean(lambda r:r['wrong_image']['answer_nll']),
                                           'mean_wrong_minus_correct_nll':mean(lambda r:r['wrong_minus_correct_nll']),
                                           'correct_image_lower_nll_samples':sum(r['wrong_minus_correct_nll']>0 for r in rr),
                                           'same_prediction_after_image_swap':sum(r['correct_image']['prediction']==r['wrong_image']['prediction'] for r in rr),
                                           'teacher_forced_answer_token_accuracy_percent':100*mean(lambda r:r['correct_image']['teacher_forced_answer_token_accuracy']),
                                           'first_token_accuracy_percent':100*mean(lambda r:r['correct_image']['first_token_correct']),
                                           'same_visual_token_count_samples':sum(r['correct_image']['visual_tokens']==r['wrong_image']['visual_tokens'] for r in rr)}
    atomic_json(OUT / 'summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
