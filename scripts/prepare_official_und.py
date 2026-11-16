"""Deterministic train-only SFT mix and heldout validation in official Swift format."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/official_und'
BUDGET = {'DocVQA': 8000, 'InfoVQA': 4000, 'ChartQA': 4000, 'WTQ': 4000,
          'KLC': 4000, 'DeepForm': 2000, 'DT-VQA': 4000}


def ordered(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256((salt + r['id']).encode()).hexdigest())


def write(path, rows):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def swift(row):
    return {'messages': [{'role': 'user', 'content': '<image>' + row['question']},
                         {'role': 'assistant', 'content': row['answers'][0]}],
            'images': [str(ROOT / row['image'])]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = {split: [json.loads(line) for line in (ROOT / f'data/full/{split}.jsonl').read_text().splitlines()]
              for split in ('train', 'val')}
    train = []
    for task, budget in BUDGET.items():
        train.extend(ordered([r for r in source['train'] if r['task'] == task], 'official-sft-v1-')[:budget])
    validation = []
    for task in ('DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA'):
        candidates = ordered([r for r in source['val'] if r['task'] == task], 'official-val-v1-')
        seen = set()
        for row in candidates:
            if row['document_id'] in seen:
                continue
            validation.append(row)
            seen.add(row['document_id'])
            if len(seen) == 64:
                break
        assert len(seen) == 64
    assert not ({r['document_id'] for r in train} & {r['document_id'] for r in source['val']})
    train = ordered(train, 'official-sft-order-v1-')
    write(OUT / 'train_index.jsonl', train)
    write(OUT / 'validation_index.jsonl', validation)
    write(OUT / 'train.jsonl', map(swift, train))
    write(OUT / 'validation.jsonl', map(swift, validation))
    manifest = {'train': len(train), 'validation': len(validation), 'train_tasks': dict(collections.Counter(r['task'] for r in train)),
                'validation_tasks': dict(collections.Counter(r['task'] for r in validation)),
                'training_source': 'Only original public training splits, deterministic hash selection',
                'test_used_for_training_or_selection': False,
                'purpose': 'Bounded official-checkpoint fine-tuning experiment, not reconstruction of author private mixture',
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.jsonl')}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
