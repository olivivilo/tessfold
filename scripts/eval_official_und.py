"""Distributed native-author-pipeline evaluation; each question is independent."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from vqa.official_und import ROOT, OfficialUND
from vqa.benchmark_metrics import native_score
from vqa.jsonl import recover_predictions

OUT = ROOT / 'artifacts/official_und'
TASK_SPLITS = {'DocVQA': 'val', 'InfoVQA': 'val', 'ChartQA': 'test', 'DT-VQA': 'test',
               'DeepForm': 'test', 'KLC': 'test', 'WTQ': 'test', 'OCRBench': 'test'}


def rows_for(mode, limit):
    if mode == 'validation':
        path = ROOT / 'data/official_und/validation_index.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        rows = []
        for split in ('val', 'test'):
            rows.extend(r for r in map(json.loads, (ROOT / f'data/full/{split}.jsonl').read_text().splitlines())
                        if TASK_SPLITS.get(r['task']) == split)
    if limit:
        counts, selected = {}, []
        for row in sorted(rows, key=lambda r: hashlib.sha256(('official-quick-' + r['id']).encode()).hexdigest()):
            task = row['task']
            if counts.get(task, 0) < limit:
                counts[task] = counts.get(task, 0) + 1
                selected.append(row)
        rows = selected
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--mode', choices=['benchmark', 'validation'], default='benchmark')
    parser.add_argument('--limit-per-task', type=int, default=0)
    parser.add_argument('--adapter')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{args.label}.rank{rank}.jsonl'
    rows = rows_for(args.mode, args.limit_per_task)
    config = {**vars(args), 'world_size': world,
              'adapter': str(Path(args.adapter).resolve()) if args.adapter else None,
              'data_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              'max_pixels': 1003520, 'do_sample': False, 'dtype': 'bfloat16', 'use_cache': True}
    config_path = path.with_suffix('.config.json')
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Cannot reuse a label with different model/data/generation settings'
    else:
        assert not path.exists(), 'Legacy label lacks a config manifest; use a fresh label to avoid mixing results'
        config_path.write_text(json.dumps(config, indent=2))
    done = recover_predictions(path)
    engine = OfficialUND(adapter=args.adapter)
    start = time.monotonic()
    with path.open('a') as handle:
        for index, row in enumerate(rows[rank::world]):
            if row['id'] in done:
                continue
            prediction, runtime = engine.predict(row, args.max_new_tokens)
            result = {**row, 'prediction': prediction, 'score': native_score(prediction, row),
                      'runtime': runtime, 'source': 'Author full MonkeyOCRv2-B-Und' if not args.adapter else str(args.adapter)}
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            if index % 20 == 0:
                print(json.dumps({'label': args.label, 'rank': rank, 'done': index + 1,
                                  'total': len(rows[rank::world]), 'elapsed_seconds': time.monotonic() - start,
                                  'prediction': prediction, 'gold': row['answers']}), flush=True)
    (OUT / f'{args.label}.rank{rank}.complete.json').write_text(json.dumps({'samples': len(rows[rank::world]), 'world': world}))


if __name__ == '__main__':
    main()
