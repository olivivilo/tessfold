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
    lines = ['# DocVQA 实际作答与 Gold 对照', '',
             f'正式第二阶段最终模型；公开 validation 全部 **{len(rows):,}** 条，ANLS **{mean:.4f}%**。', '',
             '**入口：`artifacts/full/docvqa_review.html`（全部问答，可筛选、搜索和看原图）。**', '',
             '- 完整 TSV：`artifacts/full/docvqa_comparison.tsv`，适合表格或 IDE 查看。',
             '- 完整 JSONL：`artifacts/full/docvqa_comparison.jsonl`，保留所有 gold、原图绝对路径和原始预测文件/行号。',
             '- 原始结果：`artifacts/full/final_val.rank{0,1,2,3}.jsonl`，筛选 `task == "DocVQA"`。', '',
             '再生成入口：`.venv/bin/python -m scripts.export_docvqa_review`。无需加载模型或重新推理。', '',
             '如本地浏览器不能直接预览 HTML，可在项目根目录运行：', '', '```bash',
             '.venv/bin/python -m http.server 8765 --bind 127.0.0.1', '```', '',
             '通过该服务器访问 `/artifacts/full/docvqa_review.html`；远程环境需使用 IDE 的端口转发。图片使用原始分辨率，预览可切换原尺寸或单独打开。', '',
             '## 全量观察', '',
             f'- 忽略大小写后完全匹配：{counts.get("exact", 0)}；部分得分：{counts.get("partial", 0)}；ANLS 为 0：{counts.get("zero", 0)}。',
             '- ANLS 保留原始空格/标点、只转小写，对所有 gold 取最大得分；不截断或清理模型作答。',
             '- 相同回答跨图像反复出现是值得检查的现象，不能据此直接断言 encoder 或 projector 出错；需要有图/空图/错图的同题对照来确认视觉依赖。', '',
             '| 高频作答（原文） | 题数 | 不同图像数 | 完全匹配题数 |', '|---|---:|---:|---:|']
    for item in common:
        lines.append(f'| {cell(item["prediction"])} | {item["questions"]} | {item["distinct_documents"]} | {item["exact_matches"]} |')
    if (ROOT / 'docs/CONTEXT_AUDIT.md').exists():
        lines += ['', '## 固定人名与上下文调查', '',
                  '[调查报告](CONTEXT_AUDIT.md)：已对全部 226 道固定人名作答题做换序、清空视觉缓存和实际输入核验，'
                  '另完成独立进程、关闭 KV cache 及换图对照。结果未发现跨问答上下文串扰，详见实测证据。']
    lines += ['', '## 已结合原图核对的样例', '',
              '`DocVQA_validation_1`、`_3`、`_4` 使用同一张便笺。页眉明确为 UNIVERSITY OF CALIFORNIA, SAN DIEGO；To 一栏为 Paul。模型对大学名称回答 University of California（得分 1），但城市回答 Baltimore、收件人回答 Dr. Shank（均为 0 分）。这两个错误涉及原图中的实体，不只是回答格式不一致。', '',
              '`DocVQA_validation_0` 的 gold 为 0.28，模型为 1.0；原图 1975 年实际曲线看起来约为 0.24，gold 本身也值得人工核查。对照导出保留原始 gold，不修改标准评测。这只是个例，不能解释整体低分。', '',
              '## 按原始数据顺序的前 60 条', '',
              '这里包含对/错/部分得分，未按效果挑选；全部 5,349 条在上述完整文件中。原图链接指向保留完整分辨率的文件。', '',
              '| ID | 问题 | 复现作答（原文） | 全部 Gold answers | ANLS | 原图 |', '|---|---|---|---|---:|---|']
    for row in rows[:60]:
        path = row['image_url']
        lines.append(f'| {row["id"]} | {cell(row["question"])} | {cell(row["prediction"])} | {cell(json.dumps(row["answers"], ensure_ascii=False))} | {row["score"]:.4f} | [原图](../artifacts/full/{path}) |')
    (ROOT / 'docs/DOCVQA_REVIEW.md').write_text('\n'.join(lines) + '\n')
    assert all((OUT / r['image_url']).exists() for r in rows), 'Broken image links'
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
