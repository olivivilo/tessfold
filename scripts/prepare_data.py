"""Stream a tiny real subset out of the 81 GB English MonkeyDocv2 table archive."""
import argparse
import os
import hashlib
import io
import json
import random
import re
import tarfile
from html.parser import HTMLParser
from pathlib import Path

import ijson
import zstandard
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

class Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cell, self.active, self.spans = [], '', False, False
    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.rows.append([])
        if tag in ('td', 'th'):
            self.active, self.cell = True, ''
            self.spans |= any(k in ('rowspan', 'colspan') and v != '1' for k, v in attrs)
    def handle_data(self, text):
        if self.active:
            self.cell += text
    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.active:
            if self.rows:
                self.rows[-1].append(' '.join(self.cell.split()))
            self.active = False

def qa_from_html(html, rng):
    table = Table()
    table.feed(html)
    rows = table.rows
    if not rows or len(rows) > 30:
        return []
    qa = [{'task': 'row_count', 'question': 'How many rows are in this table? Count all rows, including any header. Answer with a number only.',
           'answers': [str(len(rows))]}]
    cells = [(r, c, s) for r, row in enumerate(rows) for c, s in enumerate(row)
             if s and len(s) <= 20 and s.isascii() and not s.startswith('$')]
    if cells and not table.spans:
        r, c, s = rng.choice(cells)
        qa.append({'task': 'cell_read', 'question': f'What is the text in row {r+1}, column {c+1} of this table? Count from the top left, including headers. Answer with the cell text only.',
                   'answers': [s]})
    return qa

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default=os.environ.get('DATASET_ROOT', 'datasets/monkeydocv2'))
    p.add_argument('--documents', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    out = ROOT / 'data'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    archive = Path(args.dataset_root) / 'synth40m/table/en.tar.zst'
    labels = {}
    documents = []
    seen_hashes = set()
    with archive.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for member in tar:
                if member.name == 'en/html.json':
                    # Parse only the first candidates; tar skips the remainder sequentially.
                    for n, entry in enumerate(ijson.items(tar.extractfile(member), 'item')):
                        image = entry['images'][0]
                        relative = image.split('/table/', 1)[-1]
                        if relative.startswith('en/'):
                            labels[relative] = entry['messages'][-1]['content']
                        if n >= max(args.documents * 20, 1024):
                            break
                    print(f'Loaded {len(labels)} candidate labels', flush=True)
                if member.name not in labels or not member.isfile():
                    continue
                qa = qa_from_html(labels[member.name], rng)
                if len(qa) != 2:
                    continue
                blob = tar.extractfile(member).read()
                digest = hashlib.sha256(blob).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                image = Image.open(io.BytesIO(blob))
                image.verify()
                path = out / 'images' / f'{digest}.png'
                path.write_bytes(blob)
                documents.append({'document_id': digest, 'image': str(path.relative_to(ROOT)),
                                  'source_member': member.name, 'source_archive': str(archive),
                                  'qas': qa})
                if len(documents) >= args.documents:
                    break
    if len(documents) < args.documents:
        raise RuntimeError(f'Only {len(documents)} matched images. Check source label paths.')
    rng.shuffle(documents)
    n_eval = max(4, args.documents // 8)
    groups = {'test': documents[:n_eval], 'val': documents[n_eval:2*n_eval], 'train': documents[2*n_eval:]}
    manifest = {'seed': args.seed, 'source': str(archive), 'split_unit': 'image SHA256',
                'scope': 'MonkeyDocv2 synthetic English nonsemantic table derived QA; not official VQA benchmarks',
                'splits': {}}
    for split, docs in groups.items():
        records = []
        for doc in docs:
            for i, qa in enumerate(doc['qas']):
                records.append({**{k: v for k, v in doc.items() if k != 'qas'},
                                **qa, 'id': f"{doc['document_id']}_{i}"})
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
        manifest['splits'][split] = {'documents': len(docs), 'questions': len(records)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
