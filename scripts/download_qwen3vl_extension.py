"""Pinned public Qwen3-VL-8B weights; isolate from the native UND experiment."""
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/tessfold_qwen3vl_extension_20261115'
MODEL=ROOT/'models/Qwen3-VL-8B-Instruct'
REVISION='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
REPO='Qwen/Qwen3-VL-8B-Instruct'


def main():
    BASE.mkdir(exist_ok=True);MODEL.mkdir(exist_ok=True)
    api=requests.get(f'https://hf-mirror.com/api/models/{REPO}/revision/{REVISION}',timeout=30)
    api.raise_for_status();info=api.json()
    names=[r['rfilename'] for r in info['siblings'] if '/' not in r['rfilename'] and
        (r['rfilename'].endswith(('.json','.safetensors')) or r['rfilename'] in ['merges.txt','README.md','LICENSE'])]
    assert 'model.safetensors.index.json' in names
    (BASE/'model_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    start=time.time()
    def download(name):
        target=MODEL/name
        url=f'https://hf-mirror.com/{REPO}/resolve/{REVISION}/{name}'
        expected=None;expected_git=None
        head=requests.head(url,allow_redirects=False,timeout=30);head.raise_for_status()
        if head.headers.get('x-linked-etag'):
            linked=head.headers['x-linked-etag'].strip('"')
            if len(linked)==64:expected=linked
            elif len(linked)==40:expected_git=linked
        for attempt in range(3):
            digest=hashlib.sha256();part=target.with_name(name+'.part')
            try:
                if target.exists():
                    with target.open('rb') as f:
                        for b in iter(lambda:f.read(4*1024**2),b''):digest.update(b)
                    if expected is None or digest.hexdigest()==expected:
                        return dict(file=name,bytes=target.stat().st_size,sha256=digest.hexdigest(),expected_lfs_sha256=expected,reused=True)
                    digest=hashlib.sha256()
                offset=part.stat().st_size if part.exists() and expected else 0
                if offset:
                    with part.open('rb') as f:
                        for b in iter(lambda:f.read(4*1024**2),b''):digest.update(b)
                with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code!=206:offset=0;digest=hashlib.sha256()
                    with part.open('ab' if offset else 'wb') as f:
                        for b in response.iter_content(4*1024**2):
                            if b:f.write(b);digest.update(b)
                if expected is not None:assert digest.hexdigest()==expected,f'Hash mismatch: {name}'
                if expected_git is not None:
                    blob=hashlib.sha1(b'blob '+str(part.stat().st_size).encode()+b'\0'+part.read_bytes()).hexdigest()
                    assert blob==expected_git,f'Git blob mismatch: {name}'
                part.replace(target)
                result=dict(file=name,bytes=target.stat().st_size,sha256=digest.hexdigest(),expected_lfs_sha256=expected,reused=False)
                print(json.dumps(result),flush=True);return result
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(download,names))
    result=dict(repo=REPO,revision=REVISION,endpoint='https://hf-mirror.com',files=results,
        total_bytes=sum(r['bytes'] for r in results),seconds=time.time()-start,
        scope='Separate native VLM feasibility and baseline only; not a drop-in replacement of UND LLM or aligned compressor.')
    (BASE/'download_complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(download_complete=True,bytes=result['total_bytes'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
