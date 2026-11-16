"""Versioned shared-FS marker visibility recovery; immutable science unchanged.

Only refresh containing directories before old controller checks Path.exists().
A missing directory is still missing; no completion marker is fabricated.
"""
from pathlib import Path
import os
from scripts import column_tree_seed_memory_suite as suite
from scripts.seed_memory_common import BASE,REG,ROOT,load,save,sha,now
ORIGINAL_MARKER=suite.marker

def refresh_marker(stage,reg):
 path=ORIGINAL_MARKER(stage,reg)
 # Ascending parents plus immediate parent readdir refresh stale negative dentries.
 for parent in reversed(path.parents):
  if parent==ROOT.parent:continue
  if parent==ROOT or ROOT in parent.parents:
   try:list(parent.iterdir())
   except FileNotFoundError:continue
 return path

def main():
 record=load(BASE/'visibility_recovery_20261114/registration.json')
 assert record['original_registration_sha256']==sha(REG)
 for rel,digest in record['recovery_source_sha256'].items():assert sha(ROOT/rel)==digest
 suite.marker=refresh_marker
 save(BASE/'visibility_recovery_20261114/runtime.json',dict(time=now(),job=os.environ.get('SLURM_JOB_ID'),changes_model_math=False))
 suite.control()

if __name__=='__main__':main()
