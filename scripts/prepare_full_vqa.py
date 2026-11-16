"""Convert complete official train/validation/test splits without synthetic QA."""
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/full'

def convert():
    image_dir = OUT / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    sources = [('lmms-lab/DocVQA/DocVQA', 'DocVQA'),
               ('lmms-lab/DocVQA/InfographicVQA', 'InfoVQA'),
               ('ahmed-masry/ChartQA/data', 'ChartQA'), ('echo840/OCRBench/data', 'OCRBench')]
    records = collections.defaultdict(list)
    counts = collections.Counter()
    for source, task in sources:
        paths = sorted((OUT / 'raw' / source).glob('*.parquet'))
        if not paths:
            raise RuntimeError(f'Missing {source}')
        for path in paths:
            original_split = path.name.split('-')[0]
            split = 'val' if original_split in ('val','validation') else original_split
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                for index, row in enumerate(batch.to_pylist()):
                    blob = row['image']['bytes'] if isinstance(row['image'],dict) else row['image']
                    digest = hashlib.sha256(blob).hexdigest()
                    dest = image_dir / f'{digest}.img'
                    if not dest.exists():
                        dest.write_bytes(blob)
                    question = row.get('question') or row.get('query')
                    answers = row.get('answers') or row.get('label') or row.get('answer')
                    if isinstance(answers, str):
                        answers = [answers]
                    if not question or not answers:
                        raise ValueError(f'Missing supervision {task} {path.name}: {list(row)}')
                    record = {'id': f'{task}_{original_split}_{counts[(task,split)]}', 'document_id': digest,
                              'image': str(dest.relative_to(ROOT)), 'question': question,
                              'answers': [str(a) for a in answers], 'task': task,
                              'source_repo': source, 'source_split': original_split,
                              'source_id': str(row.get('questionId', row.get('id', '')))}
                    if task=='ChartQA':
                        record['question_type'] = row.get('type', row.get('human_or_machine','unspecified'))
                    if task=='OCRBench':
                        record['question_type'] = row.get('question_type',row.get('type',''))
                        record['dataset_name'] = row.get('dataset',row.get('dataset_name',''))
                    records[split].append(record)
                    counts[(task,split)] += 1
            print(task, original_split, counts[(task,split)], flush=True)
    # Official test remains separate. DocVQA/InfoVQA public test labels are unavailable.
    # Hold validation out of training; don't re-randomize official question splits.
    ids = {split: {r['document_id'] for r in rows} for split,rows in records.items()}
    overlaps = {f'{a}-{b}': len(ids[a] & ids[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    # A content-identical image can appear in public training and validation. Remove training
    # records for such images to give conservative evaluation, never move validation into train.
    heldout = ids['val'] | ids['test']
    dropped = sum(r['document_id'] in heldout for r in records['train'])
    records['train'] = [r for r in records['train'] if r['document_id'] not in heldout]
    for split, rows in records.items():
        with (OUT / f'{split}.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = {'scope': 'Full public official splits: DocVQA/InfoVQA/ChartQA training; DocVQA/InfoVQA validation, ChartQA val/test, OCRBench test.',
                'selection': 'All available records, no training subset limit', 'original_image_overlaps': overlaps,
                'training_records_removed_for_heldout_image_overlap': dropped,
                'splits': {s: {'questions': len(rows), 'documents': len({r['document_id'] for r in rows}),
                               'tasks': dict(collections.Counter(r['task'] for r in rows))} for s,rows in records.items()}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'PREPARE_COMPLETE').write_text('All official records converted\n')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    if not (OUT / 'DOWNLOAD_COMPLETE').exists():
        raise RuntimeError('Wait for verified complete download')
    convert()
