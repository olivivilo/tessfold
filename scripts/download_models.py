"""Download the two original pretrained models; no understanding checkpoint is used."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def download(repo, directory):
    url = f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true'
    manifest = json.load(urlopen(url, timeout=60))['Data']['Files']
    target = ROOT / 'models' / directory
    target.mkdir(parents=True, exist_ok=True)
    (target / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    for entry in manifest:
        name = entry['Name']
        if name.startswith('.') or entry.get('Type') == 'tree':
            continue
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == entry['Size']:
            continue
        subprocess.run(['curl', '--fail', '-L', '--retry', '3', '--connect-timeout', '30',
                        '-C', '-', '-o', str(path),
                        f'https://modelscope.cn/models/{repo}/resolve/master/{name}'], check=True)
        if path.stat().st_size != entry['Size']:
            raise RuntimeError(f'Incomplete download: {path}')
        expected = entry.get('Sha256') or entry.get('SHA256')
        if expected:
            hasher = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected:
                raise RuntimeError(f'Checksum mismatch: {path}')
    print(f'Ready: {repo}', flush=True)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, r, d) for r, d in [
            ('zenosai/MonkeyOCRv2-B', 'MonkeyOCRv2-B'), ('Qwen/Qwen3-1.7B', 'Qwen3-1.7B')]]
        for f in futures:
            f.result()
