"""No-training stage entry, with frozen independent registration verification."""
import argparse,sys,os
from scripts.tessfold_system_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validate','single','multi'],required=True);a=p.parse_args();reg=verify()
 save(BASE/'preflight'/f'{a.stage}.rank{os.environ.get("RANK",0)}.json',dict(time=now(),job=os.environ.get('SLURM_JOB_ID'),reg_sha256=sha(REG)))
 from scripts.compression_lab_v2_io_retry import install
 install()
 if a.stage=='validate':
  from scripts.validate_tessfold_npu import main as run
  sys.argv=[sys.argv[0]]
 else:
  from scripts.benchmark_tessfold_system import main as run
  sys.argv=[sys.argv[0],'--mode',a.stage]
 run()
if __name__=='__main__':main()
