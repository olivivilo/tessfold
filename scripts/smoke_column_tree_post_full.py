"""Real UND placement/gradient checks; eight fresh independent configurations.

The small before/after scores are diagnostics, never a configuration selector.
Smoke weights are discarded. Full training initializes again in another job.
"""
from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist

from scripts.column_tree_post_full_suite import ROOT, SUITE, load, save, verify, now
from scripts import train_compression_lab_v2 as lab
from scripts.train_meta_compress import init_runtime
from scripts.train_compression_lab_v2_repaired import build
from vqa.official_und import OfficialUND


def pick(rows):
    selected = []
    for row in rows:
        if not selected or row['task'] != selected[0]['task']:
            selected.append(row)
        if len(selected) == 2:
            return selected
    raise AssertionError('Need two task-distinct diagnostic questions')


def main():
    reg = verify()
    from scripts.compression_lab_v2_io_retry import install
    install()
    rank, world, _, device = init_runtime('npu', 4)
    assert world == len(reg['runs']) == 8
    record = reg['runs'][rank]
    cfg = load(ROOT / record['config'])
    assert cfg['insertion'] == 'post-merger'
    engine = OfficialUND(device=str(device))
    engine.model.eval().requires_grad_(False)
    merger = engine.model.vision_tower.merger
    assert merger.hidden_size == 768  # Official constructor uses merge_size=1.
    data = lab.Data(engine)
    torch.manual_seed(cfg['seed'])
    module = build(cfg).to(device)
    out = SUITE / 'smoke' / record['name']
    out.mkdir(parents=True, exist_ok=True)
    train = pick(lab.read(ROOT / cfg['train_rows']))
    validation = pick(lab.read(ROOT / cfg['validation_rows']))
    # Materialize all immutable teachers before checking the compression route.
    for row in train + validation:
        data.get_all(row)
    train_before = lab.evaluate(engine, module, train, data, cfg, out, 'before', 'train', False)
    validation_before = lab.evaluate(engine, module, validation, data, cfg, out, 'before')
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg['lr'], weight_decay=0.)
    updates, norms, placement_error = [], [], 0.
    module.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for row in train:
            losses, (p, sp, _, raw, _, _, _) = lab.student(engine, module, row, data, cfg)
            x = data.get_all(row)[3]
            assert len(raw) == len(x) == p.shape[1]
            assert p.shape[0] == max(1, round(len(x) * cfg['ratio']))
            assert bool(torch.isfinite(p).all()) and float(p.min()) >= 0
            assert float((p.sum(-1) - 1).abs().max()) < 1e-4
            error = float((sp.detach().float() - (p.detach() @ x.float()).to(raw.dtype).float()).abs().max())
            assert error == 0., 'Qwen input must be P @ merger(H)'
            placement_error = max(placement_error, error)
            assert bool(torch.isfinite(losses['total']))
            (losses['total'] / len(train)).backward()
            del losses, p, sp, x, raw
        norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), cfg['clip']))
        assert math.isfinite(norm) and norm > 0
        before = [p.detach().clone() for p in module.parameters()]
        optimizer.step()
        delta = math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(module.parameters(), before)))
        assert math.isfinite(delta) and delta > 0
        assert all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
        norms.append(norm)
        updates.append(delta)
    train_after = lab.evaluate(engine, module, train, data, cfg, out, 'after', 'train', False)
    validation_after = lab.evaluate(engine, module, validation, data, cfg, out, 'after')
    result = dict(kind=cfg['kind'], ratio=cfg['ratio'], passed=True, rank=rank,
                  train_before=train_before, train_after=train_after,
                  validation_before=validation_before, validation_after=validation_after,
                  gradient_norm=norms[-1], gradient_norms=norms, update_l2=updates,
                  placement_max_error=placement_error, trainable_backbone=0,
                  full_training_reuses_smoke_weights=False,
                  train_ids=[r['id'] for r in train], validation_ids=[r['id'] for r in validation])
    save(SUITE / 'smoke' / f'rank{rank}.json', result)
    dist.barrier()
    if rank == 0:
        results = [load(SUITE / 'smoke' / f'rank{i}.json') for i in range(world)]
        assert all(r['passed'] for r in results)
        save(SUITE / 'smoke/complete.json', dict(passed=True, cases=8, results=results,
             completed_utc=now(), job=os.environ.get('SLURM_JOB_ID'),
             scientific_source_sha256=reg['scientific_source_sha256']))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
