"""Persistent CPU controller; terminal failures stop rather than relaxing gates."""
import fcntl,json,os,subprocess,time
from scripts.prepare_compression_lab_v2 import ROOT,BASE
from scripts.advance_compression_lab_v2 import TERMINAL
def main():
 with (BASE/'controller.lock').open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('Existing controller retained');return
  (BASE/'controller_job.json').write_text(json.dumps(dict(job=os.environ.get('SLURM_JOB_ID'),pid=os.getpid()))+'\n')
  while True:
   # Refresh directory entries before children import newly deployed control/report code.
   for folder in ['scripts','vqa']:list((ROOT/folder).iterdir())
   result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.advance_compression_lab_v2','--submit'],cwd=ROOT,text=True,capture_output=True)
   (BASE/'controller_last.log').write_text(result.stdout+result.stderr)
   if result.returncode:raise RuntimeError(result.stderr)
   # Recover interrupted infrastructure jobs; completed/rejected runs stay fixed.
   recovery=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.resume_compression_lab_v2','--no-controller'],cwd=ROOT,text=True,capture_output=True)
   if recovery.stdout:print(recovery.stdout,flush=True)
   if recovery.returncode:raise RuntimeError(recovery.stderr)
   stages={k:v.get('stage') for k,v in json.loads((BASE/'stages.json').read_text())['ratios'].items()}
   print(json.dumps(stages),flush=True)
   if len(stages)==3 and all(s in TERMINAL for s in stages.values()):break
   time.sleep(30)
  for module in ['scripts.report_compression_lab_v2','scripts.plot_compression_lab_v2','scripts.review_compression_lab_v2_benchmark','scripts.report_compression_lab_v2_final','scripts.report_compression_lab_v2_keep50']:
   command=[str(ROOT/'.venv/bin/python'),'-m',module]
   if module=='scripts.report_compression_lab_v2_final':command.append('--require-complete')
   subprocess.run(command,cwd=ROOT,check=True)
  (BASE/'controller_complete.json').write_text(json.dumps(dict(stages=stages,job=os.environ.get('SLURM_JOB_ID')))+'\n')
if __name__=='__main__':main()
