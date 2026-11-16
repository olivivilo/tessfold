"""Download pinned DT-VQA original train/test corpus and verify SHA256."""
import concurrent.futures
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/full/raw/shuozhang2/DT-VQA'
BASE='https://hf-mirror.com'
def api(url):return json.loads(subprocess.check_output(['curl','-sSLf','--retry','4',url]))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rev=api(BASE+'/api/models/shuozhang2/DT-VQA')['sha']
    entries=api(BASE+f'/api/models/shuozhang2/DT-VQA/tree/{rev}?recursive=true')
    selected=[x for x in entries if x['type']=='file' and (x['path']=='image.zip' or x['path'].endswith('.json'))]
    (OUT/'source_manifest.json').write_text(json.dumps({'revision':rev,'files':selected},indent=2))
    def fetch(entry):
        dest=OUT/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or dest.stat().st_size!=entry['size']:
            subprocess.run(['curl','-sSLf','--retry','5','-C','-','-o',str(dest),f'{BASE}/shuozhang2/DT-VQA/resolve/{rev}/{entry["path"]}'],check=True)
        if dest.stat().st_size!=entry['size']:raise RuntimeError('Incomplete DT-VQA file')
        expected=entry.get('lfs',{}).get('oid')
        if expected:
            h=hashlib.sha256()
            with dest.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            if h.hexdigest()!=expected:raise RuntimeError('DT-VQA checksum failure')
        print(f'VERIFIED {entry["path"]}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,selected))
    with zipfile.ZipFile(OUT/'image.zip') as archive:
        for entry in archive.infolist():
            dest=OUT/entry.filename
            if not dest.resolve().is_relative_to(OUT.resolve()):raise ValueError('Unsafe ZIP member')
            if entry.is_dir():continue
            if not dest.exists() or dest.stat().st_size!=entry.file_size:
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.read(entry))
    (OUT/'COMPLETE').write_text('All DT-VQA files verified/extracted\n')
if __name__=='__main__':main()
