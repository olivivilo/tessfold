"""Export actual DocVQA validation predictions, golds and an image review page.

Entry: .venv/bin/python -m scripts.export_docvqa_review
"""
import collections
import csv
import json
import math
from pathlib import Path

from PIL import Image
from vqa.benchmark_metrics import benchmark_anls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/full'


def cell(value):
    return str(value).replace('|', '&#124;').replace('\r', '').replace('\n', '<br>')


def main():
    source = {}
    for line in (ROOT / 'data/full/val.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['task'] == 'DocVQA':
            source[row['id']] = row
    rows = {}
    for path in sorted(OUT.glob('final_val.rank*.jsonl')):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            if row['task'] != 'DocVQA':
                continue
            assert row['id'] not in rows, f'Duplicate {row["id"]}'
            gold = source[row['id']]
            assert all(row[key] == gold[key] for key in ('question', 'answers', 'image', 'document_id'))
            score = benchmark_anls(row['prediction'], row['answers'])
            assert math.isclose(score, row['score'], abs_tol=1e-12), row['id']
            exact = any(row['prediction'].lower() == a.lower() for a in row['answers'])
            rows[row['id']] = {**row, 'gold_answers': row['answers'], 'anls_official': score,
                               'review_status': 'exact' if exact else 'partial' if score > 0 else 'zero',
                               'image_absolute': str(ROOT / row['image']),
                               'source_prediction_file': str(path.relative_to(ROOT)),
                               'source_prediction_line': number}
    assert set(source) == set(rows), 'Incomplete DocVQA coverage'
    rows = sorted(rows.values(), key=lambda r: int(r['id'].rsplit('_', 1)[1]))
    mean = 100 * sum(r['anls_official'] for r in rows) / len(rows)
    reported = json.loads((OUT / 'final_val_metrics.json').read_text())['benchmarks']['DocVQA']
    assert reported['samples'] == len(rows) and math.isclose(mean, reported['score_percent'], abs_tol=1e-10)
    (OUT / 'docvqa_comparison.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    fields = ['id', 'source_id', 'question', 'prediction', 'gold_answers', 'anls_official',
              'review_status', 'image_absolute', 'source_prediction_file', 'source_prediction_line']
    with (OUT / 'docvqa_comparison.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'gold_answers': json.dumps(row['answers'], ensure_ascii=False)})
    image_dir = OUT / 'docvqa_images'
    image_dir.mkdir(exist_ok=True)
    images = {}
    for row in rows:
        doc = row['document_id']
        if doc not in images:
            original = ROOT / row['image']
            with Image.open(original) as image:
                suffix = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tiff', 'WEBP': '.webp', 'GIF': '.gif'}.get(image.format)
                size = list(image.size)
                if suffix in ('.tiff', None):
                    name = doc + '.png'
                    image.convert('RGB').save(image_dir / name)
                else:
                    name = doc + suffix
                    target = image_dir / name
                    if not target.exists():
                        target.symlink_to(Path('../../../') / row['image'])
            images[doc] = {'image_url': 'docvqa_images/' + name, 'image_size': size}
        row.update(images[doc])
    counts = dict(collections.Counter(r['review_status'] for r in rows))
    common = []
    for prediction, count in collections.Counter(r['prediction'] for r in rows).most_common(15):
        group = [r for r in rows if r['prediction'] == prediction]
        common.append({'prediction': prediction, 'questions': count,
                       'distinct_documents': len({r['document_id'] for r in group}),
                       'exact_matches': sum(r['review_status'] == 'exact' for r in group)})
    summary = {'split': 'DocVQA official public validation', 'samples': len(rows), 'anls_percent': mean,
               'status_counts': counts, 'distinct_documents': len(images), 'most_common_predictions': common,
               'note': 'exact means raw lowercase equality to any reference; partial means 0 < official ANLS < 1. Descriptive evidence only, no causal attribution.'}
    (OUT / 'docvqa_review_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    template = (ROOT / 'docs/docvqa_review_template.html').read_text()
    payload = json.dumps({'rows': rows, 'summary': summary}, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (OUT / 'docvqa_review.html').write_text(template.replace('/* REVIEW_PAYLOAD */null', payload))
    lines = ['# DocVQA Predictions and Gold Answers', '',
             f'Final Stage-2 model; complete public validation **{len(rows):,}** records; ANLS **{mean:.4f}%**. ', '',
             '**Entry point: `artifacts/full/docvqa_review.html`(all QA, filterable/searchable with original images).**', '',
             '- Complete TSV: `artifacts/full/docvqa_comparison.tsv`, suitable for spreadsheet or IDE inspection.',
             '- Complete JSONL: `artifacts/full/docvqa_comparison.jsonl`, retaining all gold, absolute original-image paths, and original prediction-file/line provenance.',
             '- Original results: `artifacts/full/final_val.rank{0,1,2,3}.jsonl`, screening  `task == "DocVQA"`. ', '',
             'Regeneration entry: `.venv/bin/python -m scripts.export_docvqa_review`. No model loading or new inference is required.', '',
             'If HTML preview is unavailable, run from project root:', '', '```bash',
             '.venv/bin/python -m http.server 8765 --bind 127.0.0.1', '```', '',
             'Access through this server:  `/artifacts/full/docvqa_review.html`; remote use requires IDE port forwarding. Original-resolution images support actual-size view or separate opening.', '',
             '## Full-Evaluation Observations', '',
             f'- Case-insensitive exact matches: {counts.get("exact", 0)}; partial credit: {counts.get("partial", 0)}; zero-ANLS count: {counts.get("zero", 0)}. ',
             '- ANLS retains original whitespace/punctuation, lowercases only, and takes the maximum across gold references; outputs are neither truncated nor cleaned.',
             '- Repeated identical answers across images merit inspection but do not alone establish an encoder/projector bug. Compare correct, absent, and wrong images on the same questions to test visual dependence.', '',
             '| Frequent answer (verbatim) | Questions | Distinct images | Exact-match question count |', '|---|---:|---:|---:|']
    for item in common:
        lines.append(f'| {cell(item["prediction"])} | {item["questions"]} | {item["distinct_documents"]} | {item["exact_matches"]} |')
    if (ROOT / 'docs/CONTEXT_AUDIT.md').exists():
        lines += ['', '## Repeated-Name Answers and Context-Isolation Audit', '',
                  '[Investigation report](CONTEXT_AUDIT.md): all 226 fixed-name questions checked under reordering/cache clearing/actual-input inspection, '
                  'Fresh-process, KV-disabled, and image-swap controls also find no cross-question context contamination; see measurements.']
    lines += ['', '## Examples Checked Against the Original Images', '',
              '`DocVQA_validation_1`, `_3`, `_4` The same memo is used. Its heading states UNIVERSITY OF CALIFORNIA, SAN DIEGO, and its To field says Paul. The model answers University of California for the institution (score 1), but Baltimore for the city and Dr. Shank for the recipient (both score 0). These errors concern entities in the image, not merely response formatting.', '',
              '`DocVQA_validation_0` has gold 0.28 and prediction 1.0; the plotted 1975 curve appears near 0.24, warranting gold review. Preserve original gold/scoring. One case does not explain overall weakness.', '',
              '## First 60 Records in Original Data Order', '',
              'Includes correct, incorrect, and partial-credit cases without score selection. All 5,349 records are in the full files; original-image links retain full resolution.', '',
              '| ID | Question | Reproduced answer (verbatim) | All gold answers | ANLS | Original image |', '|---|---|---|---|---:|---|']
    for row in rows[:60]:
        path = row['image_url']
        lines.append(f'| {row["id"]} | {cell(row["question"])} | {cell(row["prediction"])} | {cell(json.dumps(row["answers"], ensure_ascii=False))} | {row["score"]:.4f} | [Original image](../artifacts/full/{path}) |')
    (ROOT / 'docs/DOCVQA_REVIEW.md').write_text('\n'.join(lines) + '\n')
    assert all((OUT / r['image_url']).exists() for r in rows), 'Broken image links'
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
