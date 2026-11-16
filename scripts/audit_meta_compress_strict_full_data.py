"""Prepare the already-declared full cohort and check image/grid compatibility.

Reads benchmark image dimensions for interface compatibility only. No model,
answer scoring, filtering by correctness, or hyperparameter choice is involved.
"""
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from scripts.report_meta_compress_strict import ROOT, BASE, read, save


def main():
    forbidden = {r['document_id'] for name in ['validation', 'confirm'] for r in read(BASE/f'{name}.jsonl')}
    grouped = defaultdict(list)
    for r in read(ROOT/'data/full/train.jsonl'):
        if r['document_id'] not in forbidden:
            grouped[r['task']].append(r)
    required = read(BASE/'train.jsonl')
    selected = []
    for task, rows in sorted(grouped.items()):
        keep = [r for r in required if r['task'] == task]
        ids = {r['id'] for r in keep}
        extra = sorted((r for r in rows if r['id'] not in ids),
                       key=lambda r: hashlib.sha256(('strict_full:'+r['id']).encode()).hexdigest())
        selected += keep+extra[:max(0, 2857-len(keep))]
    path = BASE/'full_train.jsonl'
    value = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected)
    if path.exists():
        assert path.read_text() == value
    else:
        tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
        tmp.write_text(value)
        tmp.replace(path)
    benchmark = [r for r in read(ROOT/'data/full/val.jsonl') if r['task'] in ['DocVQA', 'InfoVQA']]
    benchmark += read(ROOT/'data/full/test.jsonl')
    assert len(benchmark) == 23565
    assert not ({r['document_id'] for r in selected} & {r['document_id'] for r in benchmark})
    assert not ({r['document_id'] for r in selected} & forbidden)
    docs = {}
    for cohort, rows in [('full_train', selected), ('benchmark', benchmark)]:
        for r in rows:
            docs.setdefault(r['document_id'], (cohort, r))

    def inspect(item):
        doc, (cohort, row) = item
        try:
            with Image.open(ROOT/row['image']) as im:
                w, h = im.size
            nh, nw = smart_resize(h, w, factor=28, min_pixels=3136, max_pixels=1003520)
            return {'document_id': doc, 'cohort': cohort, 'height': h, 'width': w,
                    'grid_h': nh//14, 'grid_w': nw//14}
        except Exception as error:
            return {'document_id': doc, 'cohort': cohort, 'error': repr(error)}

    with ThreadPoolExecutor(8) as pool:
        geometry = list(pool.map(inspect, docs.items()))
    errors = [r for r in geometry if 'error' in r]
    valid = [r for r in geometry if 'error' not in r]
    overflow = [r for r in valid if max(r['grid_h'], r['grid_w']) > 512]
    out = BASE/'full_data_preflight'
    out.mkdir(exist_ok=True)
    (out/'image_geometry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in geometry))
    result = {'full_training_qa': len(selected), 'full_training_images': len({r['document_id'] for r in selected}),
              'full_training_task_qa': dict(Counter(r['task'] for r in selected)),
              'benchmark_qa': len(benchmark), 'benchmark_images': len({r['document_id'] for r in benchmark}),
              'image_records': len(geometry), 'errors': errors, 'grid_overflow': overflow,
              'max_grid_h': max(r['grid_h'] for r in valid), 'max_grid_w': max(r['grid_w'] for r in valid),
              'train_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': {'training_benchmark_image_overlap': 0, 'training_validation_confirmation_overlap': 0,
                         'required_672_training_images_retained': {r['document_id'] for r in required} <= {r['document_id'] for r in selected}},
              'scope': 'Data preparation/metadata only; no full model has been selected or trained by this script.'}
    save(out/'RESULTS.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    assert not errors and not overflow


if __name__ == '__main__':
    main()
