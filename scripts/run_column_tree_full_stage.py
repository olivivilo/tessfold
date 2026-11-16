"""NPU stage wrapper: audit immutable registration and call unchanged v2 math."""
import argparse
import os
import sys
from scripts.column_tree_full_suite import ROOT, SUITE, verify, save, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train', 'benchmark'], required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    reg = verify(args.config)
    import json
    cfg = json.loads((ROOT / args.config).read_text())
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    save(out / f'suite_preflight.{args.stage}.rank{os.environ.get("RANK", "0")}.json', dict(
        checked_utc=now(), registration=str((SUITE / 'registration.json').relative_to(ROOT)),
        data_and_scientific_source_hashes_verified=True, job=os.environ.get('SLURM_JOB_ID'),
        world=int(os.environ.get('WORLD_SIZE', 1)), protocol=reg['protocol']))
    sys.argv = [sys.argv[0], '--config', args.config]
    if args.stage == 'train':
        from scripts.compression_lab_v2_io_retry import main as run
    else:
        from scripts.benchmark_compression_lab_v2_io import main as run
    run()


if __name__ == '__main__':
    main()
