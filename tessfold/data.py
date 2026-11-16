"""Reconstruct exact cohorts using public source row indexes, never bundled QA."""
import json
from pathlib import Path
from .config import sha


def download_sources(endpoint='https://huggingface.co'):
    """Fetch pinned sources; raw model-independent dataset files remain ignored."""
    import hashlib
    import os
    import shutil
    from urllib.request import urlopen
    import zipfile
    root=Path(__file__).resolve().parents[1]
    specs=Path(__file__).with_name('sources')
    entries=json.loads((specs/'datasets.files.json').read_text())
    annotation=json.loads((specs/'monkey_annotation.source.json').read_text())
    dt=json.loads((specs/'dtvqa.source.json').read_text())

    def fetch(url,dest,size,sha256=None,git_sha1=None):
        def valid(path):
            if not path.is_file() or path.stat().st_size!=size:return False
            if sha256:return sha(path)==sha256
            h=hashlib.sha1(f'blob {size}\0'.encode())
            with path.open('rb') as f:
                while chunk:=f.read(8*1024**2):h.update(chunk)
            return h.hexdigest()==git_sha1
        if valid(dest):return
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_name(dest.name+f'.{os.getpid()}.partial')
        with urlopen(url,timeout=180) as src,temp.open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
        if not valid(temp):raise RuntimeError(f'Dataset download checksum mismatch: {dest}')
        temp.replace(dest);print('Verified',dest.relative_to(root),flush=True)

    for e in entries:
        fetch(f"{endpoint}/datasets/{e['repo']}/resolve/{e['revision']}/{e['path']}",
              root/'data/full/raw'/e['repo']/e['path'],e['size'],
              e.get('lfs',{}).get('oid'),None if e.get('lfs') else e.get('oid'))
    ann=root/'data/full/raw/echo840/Monkey_Data'
    fetch(f"{endpoint}/datasets/echo840/Monkey_Data/resolve/{annotation['revision']}/{annotation['path']}",
          ann/annotation['path'],annotation['size'],annotation['sha256'])
    dest=root/'data/full/raw/shuozhang2/DT-VQA'
    for e in dt['files']:
        fetch(f"{endpoint}/shuozhang2/DT-VQA/resolve/{dt['revision']}/{e['path']}",
              dest/e['path'],e['size'],e.get('lfs',{}).get('oid'),None if e.get('lfs') else e.get('oid'))
    with zipfile.ZipFile(dest/'image.zip') as archive:
        for info in archive.infolist():
            path=dest/info.filename
            if not path.resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe archive path')
            if not info.is_dir() and (not path.is_file() or path.stat().st_size!=info.file_size):
                path.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(info) as src,path.open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
    (dest/'COMPLETE').write_text('Pinned DT-VQA sources downloaded\n')
    (root/'data/full/DOWNLOAD_COMPLETE').write_text('Pinned parquet sources downloaded\n')
    # Dataset repo uses large multipart ZIPs. Use pinned range extraction rather
    # than shipping archives in Git or requiring their entire local expansion.
    from scripts import remote_monkey_zip
    import sys
    original_argv=sys.argv[:];original_base=remote_monkey_zip.BASE
    try:
        remote_monkey_zip.BASE=f'{endpoint}/datasets/echo840/Monkey_Data/resolve/{remote_monkey_zip.REVISION}/'
        for flags in [[],['--test']]:
            sys.argv=['remote_monkey_zip',*flags];remote_monkey_zip.main()
    finally:
        sys.argv=original_argv;remote_monkey_zip.BASE=original_base
    print('Public sources ready. Run the conversion steps in docs/DATA.md.')


def prepare_cohorts(root='.', output_root='.'):
    root=Path(root).resolve();output_root=Path(output_root).resolve()
    spec=json.loads((Path(__file__).with_name('sources')/'cohorts.json').read_text())
    sources={}
    for name,meta in spec['sources'].items():
        path=root/name
        if sha(path)!=meta['sha256']:
            raise ValueError(f'Source differs from pinned public conversion: {name}')
        sources[name]=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if len(sources[name])!=meta['rows']:raise ValueError(f'Wrong row count: {name}')
    for split,recipe in spec['recipes'].items():
        rows=[sources[name][index] for name,index in recipe['entries']]
        blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
        import hashlib
        if hashlib.sha256(blob).hexdigest()!=recipe['sha256']:raise ValueError(f'Cohort differs: {split}')
        dest=output_root/recipe['output']
        if dest.exists() and dest.read_bytes()!=blob:raise FileExistsError(f'Refuse to overwrite different cohort: {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
        print(f'Verified {split}: {len(rows)} QA -> {dest}')
