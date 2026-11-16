"""Configuration and dataset validation without loading a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TASKS = {'DocVQA', 'InfoVQA', 'ChartQA', 'OCRBench', 'DeepForm', 'KLC', 'WTQ', 'DT-VQA'}
DEFAULTS = dict(seed=20261115, epochs=4, batch_size=32, lr=3e-4,
                clip=5.0, weight_decay=0.0, warmup_fraction=0.1, lr_floor=0.1,
                teacher_cap=64, dev_cap=64, eval_cap=256, cache_features=True,
                cache_logits=False, cache_memory_entries=2, checkpoint_every=25,
                threads=4, pilot_limit=512, pilot_epochs=3, require_pilot=True,
                min_kl_reduction=0.05, min_macro_gain=0.25)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(4 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    given = json.loads(path.read_text())
    cfg = dict(DEFAULTS, **given)
    if cfg.get('backend') not in {'native8b', 'und'}:
        raise ValueError('backend must be native8b or und')
    if cfg.get('kind') not in {'Tree', 'Column', 'StableColumn'}:
        raise ValueError('kind must be Tree, Column or StableColumn')
    if cfg['backend'] == 'und' and cfg['kind'] == 'StableColumn':
        raise ValueError('StableColumn is a distinct native8b variant')
    if cfg.get('insertion') not in {'pre-merger', 'post-merger'}:
        raise ValueError('insertion must be pre-merger or post-merger')
    if cfg['backend'] == 'native8b' and cfg['insertion'] != 'post-merger':
        raise ValueError('native8b requires post-merger (including all DeepStack branches)')
    if not 0 < cfg.get('ratio', 0) <= 1:
        raise ValueError('ratio is the token retention fraction, in (0,1]')
    for k in ('epochs', 'batch_size', 'teacher_cap', 'dev_cap', 'eval_cap',
              'checkpoint_every', 'pilot_limit', 'pilot_epochs', 'threads'):
        if not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] <= 0:
            raise ValueError(f'{k} must be a positive integer')
    if cfg['lr'] <= 0 or cfg['clip'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('lr/clip must be positive; weight_decay must be nonnegative')
    if not 0 < cfg['warmup_fraction'] < 1 or not 0 <= cfg['lr_floor'] <= 1:
        raise ValueError('Invalid scheduler settings')
    if cfg['cache_memory_entries'] < 0:
        raise ValueError('cache_memory_entries must be nonnegative')
    if not 0 < cfg['min_kl_reduction'] < 1 or cfg['min_macro_gain'] < 0:
        raise ValueError('Invalid development gate settings')
    # Relative paths are explicitly relative to the invocation directory.
    for key in ('model', 'train_rows', 'dev_rows', 'eval_rows', 'output', 'cache', 'data_root'):
        if key not in cfg:
            if key in {'cache', 'data_root'}:
                continue
            raise ValueError(f'Missing config field: {key}')
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    cfg.setdefault('data_root', str(Path.cwd()))
    cfg.setdefault('cache', str(Path(cfg['output']) / 'cache'))
    return cfg


def rows(path, data_root):
    result = []
    for line, text in enumerate(Path(path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        for key in ('id', 'document_id', 'image', 'question', 'answers', 'task'):
            if key not in row:
                raise ValueError(f'{path}:{line}: missing {key}')
        if not all(isinstance(row[k], str) and row[k] for k in ('id','document_id','image','question','task')):
            raise ValueError(f'{path}:{line}: id/document/image/question/task must be nonempty strings')
        if not row['question'].strip():
            raise ValueError(f'{path}:{line}: question must contain text')
        if row['task'] not in TASKS:
            raise ValueError(f'Unsupported task: {row["task"]}')
        if not isinstance(row['answers'], list) or not row['answers'] or not all(isinstance(a,str) for a in row['answers']):
            raise ValueError(f'{path}:{line}: answers must be a nonempty string list')
        image = Path(row['image'])
        row['image'] = str((image if image.is_absolute() else Path(data_root)/image).resolve())
        result.append(row)
    if not result or len({r['id'] for r in result}) != len(result):
        raise ValueError(f'{path}: empty dataset or duplicate question IDs')
    return result


def validate(cfg, check_images=True):
    splits = {s: rows(cfg[f'{s}_rows'], cfg['data_root']) for s in ('train','dev','eval')}
    # Compare both supplied document identifiers and canonical image paths.
    # Identical image content under different filenames is checked when requested.
    for field in ('id', 'document_id', 'image'):
        keys = {s:{r[field] for r in rr} for s,rr in splits.items()}
        for a,b in (('train','dev'),('train','eval'),('dev','eval')):
            if keys[a] & keys[b]:
                raise ValueError(f'{a}/{b} overlap in {field}: {len(keys[a]&keys[b])}')
    image_hashes = {}
    if check_images:
        hashes = {}
        for s, rr in splits.items():
            for r in rr:
                if not Path(r['image']).is_file():
                    raise FileNotFoundError(r['image'])
                image_hashes.setdefault(r['image'], None)
                if image_hashes[r['image']] is None:
                    image_hashes[r['image']] = sha(r['image'])
            hashes[s] = {image_hashes[r['image']] for r in rr}
        for a,b in (('train','dev'),('train','eval'),('dev','eval')):
            if hashes[a] & hashes[b]:
                raise ValueError(f'{a}/{b} contain identical image bytes under different IDs')
    # A document may have multiple questions, but a cache key must identify one image.
    docs = {}
    for rr in splits.values():
        for r in rr:
            image_id = image_hashes.get(r['image'], r['image'])
            if r['document_id'] in docs and docs[r['document_id']] != image_id:
                raise ValueError(f'Document ID maps to different images: {r["document_id"]}')
            docs[r['document_id']] = image_id
    summary = {s:dict(questions=len(rr), images=len({r['document_id'] for r in rr}),
                     tasks={t:sum(r['task']==t for r in rr) for t in sorted({r['task'] for r in rr})},
                     manifest_sha256=sha(cfg[f'{s}_rows'])) for s,rr in splits.items()}
    summary['image_content_sha256'] = image_hashes
    return splits, summary
