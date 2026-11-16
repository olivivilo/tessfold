"""Verify frozen new registration and dispatch unchanged training or new profiling."""
import argparse,sys,os
from scripts.seed_memory_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();reg=verify()
 stage=next(s for s in reg['stages'] if s['id']==a.stage)
 save(BASE/'preflight'/f'{a.stage}.rank{os.environ.get("RANK",0)}.json',dict(time=now(),registration_sha256=sha(REG),job=os.environ.get('SLURM_JOB_ID')))
 from scripts.compression_lab_v2_io_retry import install
 install()
 if stage['phase']=='train':
  m=next(m for m in reg['models'] if m['id']==stage['model']);sys.argv=[sys.argv[0],'--config',m['config']]
  from scripts.compression_lab_v2_io_retry import main as run
 elif stage['phase']=='eval':
  sys.argv=[sys.argv[0],'--model',stage['model']]
  from scripts.eval_seed_memory import main as run
 elif stage['phase']=='profile':
  sys.argv=[sys.argv[0]]
  from scripts.profile_seed_memory import main as run
 else:
  sys.argv=[sys.argv[0]]
  from scripts.smoke_seed_memory import main as run
 run()
if __name__=='__main__':main()
