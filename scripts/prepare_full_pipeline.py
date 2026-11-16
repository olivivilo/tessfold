"""Wait for independent downloads/extraction, then build the immutable full corpus."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
required=['data/full/DOWNLOAD_COMPLETE','data/full/monkey_original/train/EXTRACT_COMPLETE',
          'data/full/monkey_original/test/EXTRACT_COMPLETE','data/full/raw/shuozhang2/DT-VQA/COMPLETE']
start=time.monotonic()
while True:
    missing=[name for name in required if not (ROOT/name).exists()]
    if not missing:break
    print(json.dumps({'phase':'waiting_for_source_data','missing':missing,'elapsed_seconds':time.monotonic()-start}),flush=True)
    if time.monotonic()-start>5*3600:raise TimeoutError('Source data incomplete after five hours; inspect source logs')
    time.sleep(30)
for script in ['prepare_full_vqa.py','append_due_vqa.py','append_dtvqa.py','preflight_full_data.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
manifest=json.loads((ROOT/'data/full/manifest.json').read_text())
print('FULL_CORPUS_READY',json.dumps(manifest),flush=True)
