"""Persistent CPU-side controller; advances registered gates, never weakens them."""
import fcntl,json,subprocess,time,os
from pathlib import Path
from scripts.report_compression_lab import ROOT,BASE
TERMINAL={'no_pilot_pass','expanded_failed_no_large_training','confirmation_failed_no_large_training','benchmark_complete'}
def main():
 with (BASE/'controller.lock').open('w') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('Existing controller retained',flush=True);return
  (BASE/'controller_job.json').write_text(json.dumps({'job':os.environ.get('SLURM_JOB_ID'),'pid':os.getpid(),'entry':'scripts/control_compression_lab.py'})+'\n')
  while True:
   result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.advance_compression_lab','--submit'],cwd=ROOT,text=True,capture_output=True)
   (BASE/'controller_last.log').write_text(result.stdout+result.stderr)
   if result.returncode:raise RuntimeError(result.stderr)
   state=json.loads((BASE/'stages.json').read_text());stages={k:v.get('stage','waiting') for k,v in state['ratios'].items()};print(json.dumps(stages),flush=True)
   if len(stages)==3 and all(v in TERMINAL for v in stages.values()):break
   time.sleep(30)
  subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.report_compression_lab'],cwd=ROOT,check=True)
  subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.plot_compression_lab'],cwd=ROOT,check=True)
  (BASE/'controller_complete.json').write_text(json.dumps({'stages':stages,'job':os.environ.get('SLURM_JOB_ID')})+'\n')
if __name__=='__main__':main()
