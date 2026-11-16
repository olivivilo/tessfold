"""Measure saved-stage answer NLL on fixed matched train/validation task samples."""
import gc
import hashlib
import json
import os

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from vqa.full_train import amp, atomic_json, read_split, set_runtime
from vqa.model import VLM, resolve

OUT = resolve('artifacts/full/loss_audit')
TASKS = ('DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA')


def selection(cfg):
    result = {}
    for split in ('train', 'val'):
        rows = read_split(cfg, split)
        selected = []
        for task in TASKS:
            candidates = sorted((r for r in rows if r['task'] == task),
                                key=lambda r: hashlib.sha256(('loss-audit-v1-' + r['id']).encode()).hexdigest())
            seen = set()
            for row in candidates:
                if row['document_id'] in seen:
                    continue
                seen.add(row['document_id'])
                selected.append(row)
                if len(seen) == 64:
                    break
            assert len(seen) == 64
        result[split] = selected
    assert not ({r['document_id'] for r in result['train']} & {r['document_id'] for r in result['val']})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(resolve('configs/full.json').read_text())
    set_runtime(cfg)
    selected = selection(cfg)
    selection_path = OUT / 'selection.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == selected
    atomic_json(selection_path, selected)
    atomic_json(OUT / 'manifest.json', {
        'kind': 'Post-hoc saved-checkpoint loss evaluation; not historical intermediate eval loss',
        'job_id': os.environ.get('SLURM_JOB_ID'), 'tasks': TASKS, 'samples_per_task_per_split': 64,
        'stages': [1, 2], 'test_used': False, 'weights_modified': False,
        'selection': 'SHA256(loss-audit-v1- + id), first 64 distinct documents per task per split',
        'loss': 'Causal CE on first gold answer and EOS only; per-sample and token-weighted aggregation',
        'config': cfg,
    })
    records_path = OUT / 'endpoint_records.jsonl'
    records = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            row = json.loads(line)
            records[row['stage'], row['split'], row['id']] = row
    for stage in (1, 2):
        stage_cfg = {**cfg, 'language_model': 'models/Qwen3-1.7B' if stage == 1 else 'artifacts/full/stage2/language_model'}
        model = VLM(stage_cfg)
        model.projector.load_state_dict(torch.load(resolve(f'artifacts/full/stage{stage}/projector.pt'), map_location='cpu', weights_only=True))
        model.eval()
        with torch.no_grad(), records_path.open('a') as handle:
            for split, rows in selected.items():
                for index, row in enumerate(rows):
                    key = (stage, split, row['id'])
                    if key in records:
                        continue
                    with amp(stage_cfg):
                        loss, tokens = model.loss(row)
                    assert torch.isfinite(loss)
                    result = {'stage': stage, 'split': split, 'id': row['id'], 'task': row['task'],
                              'document_id': row['document_id'], 'answer_nll': float(loss),
                              'answer_tokens_including_eos': tokens}
                    handle.write(json.dumps(result) + '\n')
                    handle.flush()
                    records[key] = result
                    if (index + 1) % 32 == 0:
                        print(json.dumps({'stage': stage, 'split': split, 'done': index + 1, 'total': len(rows)}), flush=True)
        del model
        gc.collect()
        torch.npu.empty_cache()
    def aggregate(rows):
        tokens = sum(r['answer_tokens_including_eos'] for r in rows)
        return {'samples': len(rows), 'answer_tokens_including_eos': tokens,
                'sample_mean_nll': sum(r['answer_nll'] for r in rows) / len(rows),
                'token_weighted_nll': sum(r['answer_nll'] * r['answer_tokens_including_eos'] for r in rows) / tokens}
    summary = {'scope': 'Post-hoc fixed 256 train + 256 val samples; four common validation tasks, equally weighted',
               'stages': {}}
    for stage in (1, 2):
        summary['stages'][stage] = {}
        for split, rows in selected.items():
            rr = [records[stage, split, row['id']] for row in rows]
            summary['stages'][stage][split] = {
                **aggregate(rr), 'tasks': {task: aggregate([r for r in rr if r['task'] == task]) for task in TASKS}}
    atomic_json(OUT / 'endpoint_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
