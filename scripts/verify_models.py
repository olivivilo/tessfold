"""Verify downloaded weights against the ModelScope per-file SHA256 manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = []
for name in ('MonkeyOCRv2-B', 'Qwen3-1.7B'):
    directory = ROOT / 'models' / name
    entries = json.loads((directory / 'source_manifest.json').read_text())
    for entry in entries:
        if not entry['Name'].endswith('.safetensors'):
            continue
        path = directory / entry['Name']
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(chunk)
        expected = entry['Sha256']
        if digest.hexdigest() != expected:
            raise RuntimeError(f'Model SHA256 mismatch: {path}')
        row = {'model': name, 'file': entry['Name'], 'sha256': digest.hexdigest(),
               'revision': entry['Revision'], 'size': path.stat().st_size, 'passed': True}
        results.append(row)
        print(json.dumps(row), flush=True)
(ROOT / 'artifacts/model_verification.json').write_text(json.dumps(results, indent=2))
