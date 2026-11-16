"""Shared immutable teacher-cache write coalescing; model math is unchanged.

Some shared-filesystem rename races report EEXIST rather than replacing a
concurrently committed file. Accept only an exactly identical payload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from scripts import train_meta_compress as common

ORIGINAL_SAVE = common.atomic_save


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and
                a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys() and
                all(identical(a[k], b[k]) for k in a))
    if isinstance(a, (tuple, list)):
        return (type(a) is type(b) and len(a) == len(b) and
                all(identical(x, y) for x, y in zip(a, b)))
    return type(a) is type(b) and a == b


def guarded_save(path, obj):
    try:
        ORIGINAL_SAVE(path, obj)
    except FileExistsError:
        immutable = (path.parent.name in {'features', 'teachers', 'logits'} and
                     path.parent.parent.name.endswith('_cache'))
        if not immutable:
            raise
        winner = torch.load(path, map_location='cpu', weights_only=True)
        if not identical(obj, winner):
            raise RuntimeError(f'Concurrent immutable cache payload differs: {path}')
        path.with_name(path.name + f'.{os.getpid()}.tmp').unlink(missing_ok=True)
        print(json.dumps({'immutable_cache_eexist_identical': str(path),
                          'rank': os.environ.get('RANK')}), flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / args.config).read_text())
    output = root / cfg['output']
    output.mkdir(parents=True, exist_ok=True)
    audit = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'scope': 'only EEXIST on immutable feature/teacher/logit cache writes',
             'fallback': 'require all tensors and metadata to match bitwise',
             'scientific_sources_unchanged': True,
             'job': os.environ.get('SLURM_JOB_ID')}
    target = output / f'io_runtime.rank{os.environ.get("RANK", "0")}.json'
    if target.exists():
        assert json.loads(target.read_text())['source_sha256'] == audit['source_sha256']
    common.atomic_json(target, audit)
    common.atomic_save = guarded_save
    from scripts.train_compression_lab_v2_distributed import main as train
    train()


if __name__ == '__main__':
    main()
