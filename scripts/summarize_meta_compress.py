"""Summarize MetaCompress training/eval artifacts and update report tables."""
import argparse,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',default='artifacts/meta_compress'); a=p.parse_args(); root=Path(a.root)
 tr=[]
 f=root/'training.jsonl'
 if f.exists():
  tr=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
 print('training_records',len(tr))
 if tr:
  for ep in sorted({x['epoch'] for x in tr}):
   z=[x for x in tr if x['epoch']==ep]
   print('epoch',ep,'n',len(z),*(f'{k}={statistics.mean(x[k] for x in z):.6g}' for k in ['prediction','entropy','collapse','loss']))
 for f in sorted(root.glob('eval/metrics.*.json')):
  print(f); print(f.read_text())
if __name__=='__main__': main()
