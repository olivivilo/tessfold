"""Resumable bounded parallel HTTP ranges for the pinned optional8B weights."""
import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
import requests
from scripts.download_qwen3vl_extension import ROOT,BASE,MODEL,REPO,REVISION,main as finish_manifest


def main():
    index=requests.get(f'https://hf-mirror.com/{REPO}/raw/{REVISION}/model.safetensors.index.json',timeout=30)
    index.raise_for_status();names=sorted(set(index.json()['weight_map'].values()))
    directory=MODEL/'.download_ranges';directory.mkdir(exist_ok=True)
    plans=[];tasks=[]
    for name in names:
        url=f'https://hf-mirror.com/{REPO}/resolve/{REVISION}/{name}'
        head=requests.head(url,allow_redirects=False,timeout=30);head.raise_for_status()
        size=int(head.headers['x-linked-size']);expected=head.headers['x-linked-etag'].strip('"')
        target=MODEL/name;part=MODEL/(name+'.part')
        if target.exists():continue
        offset=part.stat().st_size if part.exists() else 0
        assert offset<=size
        chunks=[]
        for lo in range(offset,size,128*1024**2):
            hi=min(size-1,lo+128*1024**2-1);path=directory/f'{name}.{lo}-{hi}'
            item=dict(url=url,lo=lo,hi=hi,path=path,name=name)
            chunks.append(item);tasks.append(item)
        plans.append(dict(name=name,size=size,sha256=expected,part=part,target=target,offset=offset,chunks=chunks))
    def fetch(t):
        expected=t['hi']-t['lo']+1
        for attempt in range(4):
            offset=t['path'].stat().st_size if t['path'].exists() else 0
            if offset==expected:return
            assert offset<expected
            lo=t['lo']+offset
            try:
                with requests.get(t['url'],headers={'Range':f'bytes={lo}-{t["hi"]}'},stream=True,timeout=(30,60)) as response:
                    response.raise_for_status()
                    assert response.status_code==206,(t['name'],response.status_code)
                    assert response.headers['Content-Range'].startswith(f'bytes {lo}-{t["hi"]}/')
                    with t['path'].open('ab') as f:
                        for b in response.iter_content(1024**2):
                            if b:f.write(b)
                assert t['path'].stat().st_size==expected
                return
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    print(json.dumps(dict(range_tasks=len(tasks),workers=32,remaining_bytes=sum(p['size']-p['offset'] for p in plans))),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures=[pool.submit(fetch,t) for t in tasks]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result()
            if i%8==0:print(json.dumps(dict(completed_ranges=i,total=len(tasks))),flush=True)
    for p in plans:
        assert (p['part'].stat().st_size if p['part'].exists() else 0)==p['offset']
        with p['part'].open('ab') as out:
            for t in p['chunks']:
                with t['path'].open('rb') as f:
                    for b in iter(lambda:f.read(4*1024**2),b''):out.write(b)
        assert p['part'].stat().st_size==p['size']
        digest=hashlib.sha256()
        with p['part'].open('rb') as f:
            for b in iter(lambda:f.read(4*1024**2),b''):digest.update(b)
        assert digest.hexdigest()==p['sha256'],p['name']
        p['part'].replace(p['target'])
        for t in p['chunks']:t['path'].unlink()
        print(json.dumps(dict(verified_weight=p['name'],sha256=digest.hexdigest())),flush=True)
    finish_manifest()


if __name__=='__main__':main()
