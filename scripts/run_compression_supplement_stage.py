"""Frozen source preflight and explicit per-stage dispatch, no legacy edits."""
import argparse
import os
import sys
from scripts.supplement_common import ROOT,BASE,verify,load,save,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['smoke','train','eval','efficiency']);p.add_argument('--config',required=True);a=p.parse_args()
    reg=verify();cfg=load(ROOT/a.config)
    out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    save(out/f'preflight.{a.stage}.rank{os.environ.get("RANK",0)}.json',dict(checked_utc=now(),stage=a.stage,job=os.environ.get('SLURM_JOB_ID'),registration_sha256=__import__('hashlib').sha256((BASE/'registration.json').read_bytes()).hexdigest(),source_and_data_hashes_verified=True))
    from scripts.compression_lab_v2_io_retry import install
    install()
    sys.argv=[sys.argv[0],'--config',a.config]
    if a.stage=='smoke':from scripts.smoke_compression_supplement import main as run
    elif a.stage=='train':from scripts.train_compression_supplement import main as run
    elif a.stage=='eval':from scripts.eval_compression_supplement import main as run
    else:from scripts.benchmark_compression_supplement import main as run
    run()


if __name__=='__main__':main()
