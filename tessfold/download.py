"""Explicit opt-in download; pinned versions, no weights in the source repository."""
from __future__ import annotations
import json
from pathlib import Path
from urllib.request import urlopen
import os
import shutil
from .config import sha


def download(cfg):
    target=Path(cfg['model']);target.mkdir(parents=True,exist_ok=True)
    if cfg['backend']=='native8b':
        from huggingface_hub import snapshot_download
        snapshot_download('Qwen/Qwen3-VL-8B-Instruct',
                          revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                          local_dir=str(target))
    else:
        recipe=Path(__file__).with_name('sources')/'monkeyocrv2_und.files.json'
        for entry in json.loads(recipe.read_text()):
            rel=Path(entry['Name'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe model filename')
            dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and dest.stat().st_size==entry['Size'] and sha(dest)==entry['Sha256']:
                continue
            url=f"https://modelscope.cn/models/zenosai/MonkeyOCRv2-B-Und/resolve/{entry['Revision']}/{entry['Name']}"
            temp=dest.with_name(dest.name+f'.{os.getpid()}.partial')
            with urlopen(url,timeout=120) as src,temp.open('wb') as out:
                shutil.copyfileobj(src,out,8*1024**2)
            if temp.stat().st_size!=entry['Size'] or sha(temp)!=entry['Sha256']:
                temp.unlink(missing_ok=True);raise RuntimeError(f'Model checksum mismatch: {dest.name}')
            temp.replace(dest)
            print('Verified',dest.name,flush=True)
    print('Model stored outside Git tracking at',target)
