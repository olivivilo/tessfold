"""Download and SHA256-verify the author's full Understanding checkpoint."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models/MonkeyOCRv2-B-Und'
REPO = 'zenosai/MonkeyOCRv2-B-Und'


def fetch(entry):
    name = entry['Name']
    if entry.get('Type') == 'tree' or name.startswith('.'):
        return
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != entry['Size']:
        subprocess.run(['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error', '--retry', '5', '--retry-all-errors',
                        '--speed-time', '60', '--speed-limit', '1024',
                        '--connect-timeout', '30', '--continue-at', '-', '--output', str(path),
                        f"https://modelscope.cn/models/{REPO}/resolve/{entry['Revision']}/{name}"], check=True)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    assert path.stat().st_size == entry['Size'], name
    assert h.hexdigest() == entry['Sha256'], name
    print(json.dumps({'file': name, 'size': entry['Size'], 'sha256': h.hexdigest(), 'verified': True}), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'source_manifest.json'
    if not manifest.exists():
        files = json.load(urlopen(f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true', timeout=60))['Data']['Files']
        manifest.write_text(json.dumps(files, indent=2))
    files = json.loads(manifest.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    (OUT / 'VERIFIED').write_text('All downloaded files match pinned per-file source SHA256.\n')


if __name__ == '__main__':
    main()
