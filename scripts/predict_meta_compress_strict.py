"""Single-image inference for the new strict checkpoint format and placements."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch

from scripts.train_meta_compress import init_runtime, atomic_json, sha
from scripts.benchmark_meta_compress_strict import ROOT, build, project
from scripts.eval_token_proxy import generate, shortened_inputs
from vqa.official_und import OfficialUND, MODEL


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--teacher', action='store_true')
    p.add_argument('--save-json')
    p.add_argument('--save-matrix')
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    rank, world, _, device = init_runtime('npu', a.threads)
    assert world == 1, 'This entry takes one image; launch without torchrun.'
    checkpoint = ROOT/a.checkpoint
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    assert 'ratio' in cfg and cfg['output'].startswith('artifacts/meta_compress_strict/')
    engine = OfficialUND(device=str(device))
    engine.model.eval().requires_grad_(False)
    module = build(cfg).to(device).eval()
    module.load_state_dict(state['generator'])
    with torch.no_grad():
        inputs = engine.inputs({'image': a.image, 'question': a.question})
        grid = inputs['image_grid_thw']
        x, raw = engine.model.vision_tower(inputs['pixel_values'], grid)
        sp, pmat = project(engine, module, raw, grid, x, cfg)
        si = shortened_inputs(inputs, len(sp), engine.model.config.image_token_id)
        answer, runtime = generate(engine, si, a.max_new_tokens, sp, raw)
        result = {'image': a.image, 'question': a.question, 'answer': answer,
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                  'base_model': str(MODEL), 'placement': cfg['insertion'],
                  'keep_ratio': cfg['ratio'], 'original_tokens': len(raw),
                  'compressed_tokens': len(sp), 'row_sum_max_error': float((pmat.sum(-1)-1).abs().max()),
                  'generated_ids': runtime['generated_ids'], 'runtime': runtime,
                  'frozen_backbone_trainable_parameters': sum(p.numel() for p in engine.model.parameters() if p.requires_grad)}
        if a.teacher:
            teacher, tr = generate(engine, inputs, a.max_new_tokens, x, raw)
            result.update(teacher_answer=teacher, teacher_generated_ids=tr['generated_ids'])
        if a.save_matrix:
            path = ROOT/a.save_matrix
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'P': pmat.cpu(), 'grid': grid.cpu(), 'checkpoint_sha256': result['checkpoint_sha256']}, path)
            result['matrix_file'] = str(path)
    if a.save_json:
        path = ROOT/a.save_json
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
