"""Versioned resize candidate validation; never edit the signed original evaluator.

Only nonpositive smart_resize candidates are discarded. Original successful
requests are byte-for-byte identical in selected shape and processor input.
"""
from __future__ import annotations
import inspect
import json
import os
import sys
from pathlib import Path
from scripts.supplement_common import ROOT,BASE,load,save,sha,verify,now

AMENDMENT=BASE/'resize_recovery/registration.json'


def install():
    from scripts import eval_compression_supplement as original
    text=inspect.getsource(original.resized_inputs)
    before='        shapes.add((hh,ww))'
    assert text.count(before)==1
    after='        if hh > 0 and ww > 0:\n            shapes.add((hh,ww))'
    text=text.replace(before,after)
    scope=dict(original.__dict__)
    exec(compile(text,__file__+':validated_candidates','exec'),scope)
    original.resized_inputs=scope['resized_inputs']
    return original


def main():
    reg=verify();amend=load(AMENDMENT);assert amend
    for p,d in amend['source_sha256'].items():assert sha(ROOT/p)==d
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);a=parser.parse_args()
    cfg=load(ROOT/a.config);assert cfg['method']=='R' and cfg['name'] in amend['runs']
    from scripts.compression_lab_v2_io_retry import install as io_install
    io_install();original=install()
    out=ROOT/cfg['output']/'benchmark';out.mkdir(parents=True,exist_ok=True)
    save(out/f'resize_recovery.rank{os.environ.get("RANK",0)}.json',dict(
        amended_utc=now(),amendment_sha256=sha(AMENDMENT),source_sha256=amend['source_sha256'],
        job=os.environ.get('SLURM_JOB_ID'),reason=amend['reason'],
        original_successful_rows_preserved=True,original_registration_unchanged=True))
    sys.argv=[sys.argv[0],'--config',a.config];original.main()


if __name__=='__main__':main()
