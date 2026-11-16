"""Fetch complete public VQA splits, pin revisions and verify LFS SHA256."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://hf-mirror.com'
SOURCES = [('lmms-lab/DocVQA', 'DocVQA', ('train-', 'validation-')),
           ('lmms-lab/DocVQA', 'InfographicVQA', ('train-', 'validation-')),
           ('ahmed-masry/ChartQA', 'data', ('train-', 'val-', 'test-')),
           ('echo840/OCRBench', 'data', ('test-',))]

def api(url):
    return json.loads(subprocess.check_output(['curl', '-sSLf', '--retry','4', '--max-time','120',url]))

def fetch(task):
    repo, revision, entry = task
    dest = ROOT / 'data/full/raw' / repo / entry['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != entry['size']:
        url = f"{BASE}/datasets/{repo}/resolve/{revision}/{entry['path']}"
        subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '5', '--connect-timeout', '30',
                        '-C', '-', '-o', str(dest), url], check=True)
    if dest.stat().st_size != entry['size']:
        raise RuntimeError(f'Incomplete: {dest}')
    expected = entry.get('lfs', {}).get('oid')
    if expected:
        h = hashlib.sha256()
        with dest.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError(f'SHA256 mismatch: {dest}')
    print(f'VERIFIED {repo}/{entry["path"]}', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--monkey', action='store_true', help='Also download original Monkey/TextMonkey full train/test images and annotations')
    args = p.parse_args()
    tasks, manifests = [], []
    revisions = {}
    for repo, folder, prefixes in SOURCES:
        if repo not in revisions:
            revisions[repo] = api(f'{BASE}/api/datasets/{repo}')['sha']
        revision = revisions[repo]
        entries = api(f'{BASE}/api/datasets/{repo}/tree/{revision}/{folder}?recursive=true')
        for entry in entries:
            if entry['type']=='file' and entry['path'].endswith('.parquet') and Path(entry['path']).name.startswith(prefixes):
                tasks.append((repo, revision, entry))
    if args.monkey:
        repo = 'echo840/Monkey_Data'
        revision = api(f'{BASE}/api/datasets/{repo}')['sha']
        entries = api(f'{BASE}/api/datasets/{repo}/tree/{revision}?recursive=true')
        for entry in entries:
            if entry['path'].startswith(('train_monkey.json','Monkey_Train_data.zip','Monkey_Test_data.zip')):
                tasks.append((repo, revision, entry))
    root = ROOT / 'data/full'
    root.mkdir(parents=True, exist_ok=True)
    manifest = [{'repo': repo, 'revision': rev, **entry} for repo, rev, entry in tasks]
    (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(tasks)} files, {sum(t[2]["size"] for t in tasks)/1e9:.2f} GB', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, tasks):
            pass
    (root / 'DOWNLOAD_COMPLETE').write_text('All selected files size/SHA256 verified\n')

if __name__ == '__main__':
    main()
