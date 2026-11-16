"""Formal benchmark with the identical-payload cache rename guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from scripts import compression_lab_v2_io as io


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / args.config).read_text())
    output = root / cfg['output']
    from scripts import compression_lab_v2_io_retry as retry
    files = [Path(__file__), Path(io.__file__), Path(retry.__file__)]
    io.common.atomic_json(output / f'benchmark_io_runtime.rank{os.environ.get("RANK", "0")}.json',
        {'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'scope': 'only identical-payload EEXIST on immutable teacher cache writes',
         'changes_model_math': False, 'job': os.environ.get('SLURM_JOB_ID')})
    retry.install()
    from scripts.benchmark_compression_lab_v2 import main as benchmark
    benchmark()


if __name__ == '__main__':
    main()
