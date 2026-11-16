"""Download and verify the pinned original Monkey VQA training annotations."""
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV='dbff559fd451ae5e46e108cc4797fe0a86a6a05f'
SIZE=534066989
SHA='a0c75c2ff8a149ea4a21fa5cbbcacd503a86e9dfb658bc28949492a528d65b18'
def main():
    out=ROOT/'data/full/raw/echo840/Monkey_Data';out.mkdir(parents=True,exist_ok=True)
    dest=out/'train_monkey.json'
    if not dest.exists() or dest.stat().st_size!=SIZE:
        subprocess.run(['curl','-sSLf','--retry','5','--connect-timeout','30','--speed-limit','1024','--speed-time','120','-C','-','-o',str(dest),f'https://hf-mirror.com/datasets/echo840/Monkey_Data/resolve/{REV}/train_monkey.json'],check=True)
    h=hashlib.sha256()
    with dest.open('rb') as handle:
        for b in iter(lambda:handle.read(8*1024*1024),b''):h.update(b)
    if dest.stat().st_size!=SIZE or h.hexdigest()!=SHA:raise ValueError('Monkey annotation SHA256/size mismatch')
    (out/'annotation_manifest.json').write_text(json.dumps({'revision':REV,'path':'train_monkey.json','size':SIZE,'sha256':SHA},indent=2))
    print('VERIFIED train_monkey.json')
if __name__=='__main__':main()
