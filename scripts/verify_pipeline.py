"""Validate saved results and the public single-image inference entry."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vqa.metrics import anls, exact_match, relaxed_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/alignment.json')
args = parser.parse_args()
cfg = json.loads((ROOT / args.config).read_text())
out = ROOT / cfg['output_dir']
results = json.loads((out / 'results.json').read_text())
rows = [json.loads(line) for line in (out / 'final_test_predictions.jsonl').read_text().splitlines()]
for metric, scorer in [('exact_match', exact_match), ('anls', anls), ('relaxed_accuracy', relaxed_accuracy)]:
    scores = [scorer(row['prediction'], row['answers']) for row in rows]
    assert math.isclose(100*sum(scores)/len(scores), results['final_test'][metric])
assert results['checkpoint_roundtrip']['passed']
first = rows[0]
command = [sys.executable, '-m', 'vqa.pipeline', '--config', args.config, 'infer', '--stage', '2',
           '--image', first['image'], '--question', first['question']]
completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
(out / 'infer_stdout.txt').write_text(completed.stdout)
(out / 'infer_stderr.txt').write_text(completed.stderr)
actual = completed.stdout.strip()
if actual != first['prediction']:
    raise RuntimeError(f"Public inference changed prediction: {actual!r} != {first['prediction']!r}")
validation = {'metrics_recalculated': True, 'checkpoint_roundtrip': True,
              'public_infer_entry': True, 'sample_id': first['id'], 'prediction': actual}
import torch
from safetensors import safe_open
key = 'model.layers.0.self_attn.q_proj.weight'
original_dir = ROOT / cfg['language_model']
saved_dir = out / 'stage2/language_model'
def read_weight(directory):
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    with safe_open(directory / index['weight_map'][key], framework='pt', device='cpu') as handle:
        return handle.get_tensor(key).float()
delta = (read_weight(saved_dir) - read_weight(original_dir)).abs()
assert float(delta.max()) > 0
validation['llm_weight_check'] = {'tensor': key, 'mean_absolute_update': float(delta.mean()),
                                'max_absolute_update': float(delta.max()), 'passed': True}
stage1 = torch.load(out / 'stage1/projector.pt', map_location='cpu', weights_only=True)
stage2 = torch.load(out / 'stage2/projector.pt', map_location='cpu', weights_only=True)
assert any(not torch.equal(stage1[k], stage2[k]) for k in stage1)
validation['projector_updated_in_stage2'] = True
(out / 'verification.json').write_text(json.dumps(validation, indent=2))
print(json.dumps(validation, indent=2))
