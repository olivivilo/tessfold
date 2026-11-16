"""Versioned read-visibility retry, layered over identical-payload cache writes."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from scripts import compression_lab_v2_io as io

ORIGINAL_LOAD = torch.load


def visible_load(file, *args, **kwargs):
    if not isinstance(file, (str, os.PathLike)):
        return ORIGINAL_LOAD(file, *args, **kwargs)
    path = Path(file)
    immutable = (path.parent.name in {'features', 'teachers', 'logits'} and
                 path.parent.parent.name.endswith('_cache'))
    for attempt in range(12):
        try:
            if immutable:
                # Keep one open inode through the NPU ZIP deserializer instead
                # of asking the shared filesystem to resolve the path twice.
                with path.open('rb') as stream:
                    return ORIGINAL_LOAD(stream, *args, **kwargs)
            return ORIGINAL_LOAD(file, *args, **kwargs)
        except FileNotFoundError:
            if not immutable or attempt == 11:
                raise
            # Readdir invalidates a stale negative lookup on the shared FS.
            list(path.parent.iterdir())
            list(path.parent.parent.iterdir())
            time.sleep(min(2., .2*(attempt+1)))
            print(json.dumps({'immutable_cache_read_retry': str(path),
                              'attempt': attempt+1, 'rank': os.environ.get('RANK')}), flush=True)


def install():
    global ORIGINAL_LOAD
    # Ascend replaces torch.load when imported. Install AFTER that patch, or
    # init_runtime would silently remove our retry wrapper on real NPU jobs.
    if os.environ.get('SLURM_JOB_ID'):
        import torch_npu  # noqa: F401
    if torch.load is not visible_load:
        ORIGINAL_LOAD = torch.load
    torch.load = visible_load
    io.common.atomic_save = io.guarded_save


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/args.config).read_text())
    out = root/cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    io.common.atomic_json(out/f'io_retry_runtime.rank{os.environ.get("RANK", "0")}.json', {
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'job': os.environ.get('SLURM_JOB_ID'), 'changes_model_math': False,
        'scope': 'bounded ENOENT retry on immutable teacher caches only; other errors propagate'})
    install()
    io.main()


if __name__ == '__main__':
    main()
