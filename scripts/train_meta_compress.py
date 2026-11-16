"""Train only MetaCompress; official UND encoder/projector/LLM stay frozen.

Entry: python -m scripts.train_meta_compress --device cpu|npu --batch-size 36
Fresh raw features are extracted from B-Und, NEVER from artifacts/full/features.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vqa.meta_compress import MetaCompress, meta_losses
from vqa.official_und import ROOT, OfficialUND
from vqa.decoder_dependent_loss import common_prefix_logits
from scripts.eval_token_proxy import generate, shortened_inputs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def atomic_save(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(obj, tmp); tmp.replace(path)


def read_rows(split):
    return [json.loads(x) for x in (ROOT/'data/full'/f'{split}.jsonl').read_text().splitlines() if x]


def choose_rows(rows, limit, seed=42):
    rows = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    return rows[:limit] if limit else rows


def init_runtime(device_name, threads):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(threads)
    if device_name == 'npu':
        import torch_npu  # registers torch.npu before it is used
        torch.npu.set_device(local)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        device = torch.device(f'npu:{local}'); backend = 'hccl'
    else:
        device = torch.device('cpu'); backend = 'gloo'
    if world > 1:
        dist.init_process_group(backend, timeout=datetime.timedelta(hours=6))
    torch.manual_seed(20261026); random.seed(20261026)
    return rank, world, local, device


def barrier(world):
    if world > 1: dist.barrier()


def feature_for(engine, inputs, row, cache_dir):
    """All fresh features use the actual UND tower and its official input path."""
    path = cache_dir/'features'/f"{row['document_id']}.pt"
    signature = 'MonkeyOCRv2-B-Und-native-bf16-v1'
    if path.exists():
        f = torch.load(path, map_location='cpu', weights_only=True)
        if f.get('source') != signature: raise RuntimeError('Incompatible feature cache')
        raw = f['features'].to(engine.device)
        grid = f['grid'].to(engine.device)
        with torch.no_grad(): projected = engine.model.vision_tower.merger(raw)
    else:
        with torch.no_grad():
            projected, raw = engine.model.vision_tower(inputs['pixel_values'], inputs['image_grid_thw'])
        grid = inputs['image_grid_thw']
        atomic_save(path, {'features': raw.cpu(), 'grid': grid.cpu(), 'source': signature})
    n = int((inputs['input_ids'] == engine.model.config.image_token_id).sum())
    assert len(raw) == n == int(grid.prod())
    assert torch.equal(grid, inputs['image_grid_thw'])
    return raw, grid, projected


def prepare_example(engine, row, cache_dir, max_new_tokens):
    """Teacher text is model-generated, not gold. Cache includes prompt and raw features."""
    path = cache_dir/'teachers'/f"{hashlib.sha256(row['id'].encode()).hexdigest()}.pt"
    fpath = cache_dir/'features'/f"{row['document_id']}.pt"
    if path.exists() and fpath.exists():
        record = torch.load(path, map_location='cpu', weights_only=True)
        assert record['id'] == row['id'] and record['max_new_tokens'] == max_new_tokens
        prompt = record['input_ids'].to(engine.device)
        f = torch.load(fpath, map_location='cpu', weights_only=True)
        assert f['source'] == 'MonkeyOCRv2-B-Und-native-bf16-v1'
        raw, grid = f['features'].to(engine.device), f['grid'].to(engine.device)
        assert len(raw) == int((prompt == engine.model.config.image_token_id).sum())
        with torch.no_grad(): projected = engine.model.vision_tower.merger(raw)
        return prompt, raw, grid, projected, record['generated_ids']
    inputs = engine.inputs(row)
    raw, grid, projected = feature_for(engine, inputs, row, cache_dir)
    with torch.no_grad():
        prediction, rt = generate(engine, inputs, max_new_tokens, projected, raw)
    response = rt['generated_ids']
    if not response: raise RuntimeError(f"Empty teacher response: {row['id']}")
    atomic_save(path, {'id': row['id'], 'input_ids': inputs['input_ids'].cpu(),
                      'generated_ids': response, 'prediction': prediction,
                      'max_new_tokens': max_new_tokens})
    return inputs['input_ids'], raw, grid, projected, response


def teacher_logits_for(engine, row, prompt, projected, response, cache_dir):
    # Frozen, deterministic teacher: reuse exactly the same FP32 logits in
    # epoch two, with no quantization of the KL target.
    path=cache_dir/'logits'/f"{hashlib.sha256(row['id'].encode()).hexdigest()}.pt"
    if path.exists():
        item=torch.load(path,map_location='cpu',weights_only=True)
        assert item['response']==response
        return item['logits'].to(engine.device)
    with torch.no_grad():
        logits=common_prefix_logits(engine.model,prompt,projected[None],response)
    atomic_save(path,{'response':response,'logits':logits.cpu()})
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='artifacts/meta_compress')
    p.add_argument('--cache', default='artifacts/meta_compress_cache')
    p.add_argument('--device', choices=['npu', 'cpu'], default='npu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--max-items', type=int, default=20000)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=36)
    p.add_argument('--keep-ratio', type=float, default=.10)
    p.add_argument('--insertion', choices=['post-merger', 'pre-merger'], default='post-merger')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grad-clip', type=float, default=1e-2)
    p.add_argument('--checkpoint-interval', type=int, default=10)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    rank, world, local, device = init_runtime(a.device, a.threads)
    if a.batch_size % world: raise ValueError('batch-size must be divisible by world-size')
    accum = a.batch_size // world
    out, cache = ROOT/a.output, ROOT/a.cache
    out.mkdir(parents=True, exist_ok=True)
    for name in ['features', 'teachers', 'logits']: (cache/name).mkdir(parents=True, exist_ok=True)
    rows = choose_rows(read_rows('train'), 2 if a.smoke else a.max_items)
    if a.smoke: a.epochs = 1
    heldout = {r['document_id'] for split in ['val', 'test'] for r in read_rows(split)}
    assert not any(r['document_id'] in heldout for r in rows), 'Train/eval image overlap'
    source_files = ['vqa/meta_compress.py', 'scripts/train_meta_compress.py',
                    'vqa/official_und.py', 'vqa/decoder_dependent_loss.py']
    config = {'paper': 'Wang et al. CVPR 2026 MetaCompress Eq8+Eq10',
              'model': 'models/MonkeyOCRv2-B-Und', 'raw_features': 'fresh official UND BF16',
              'rows': len(rows), 'images': len({r['document_id'] for r in rows}),
              'world': world, 'device': a.device, 'epochs': a.epochs, 'batch_size': a.batch_size,
              'keep_ratio': a.keep_ratio, 'max_new_tokens': a.max_new_tokens,
              'lr': a.lr, 'grad_clip': a.grad_clip, 'compression_dim': 64,
              'pool': 'fractional_max_kernel3_fixed_offset0.5', 'insertion': a.insertion,
              'task_counts': dict(__import__('collections').Counter(r['task'] for r in rows)),
              'train_sha256': sha(ROOT/'data/full/train.jsonl'),
              'code_sha256': {f: sha(ROOT/f) for f in source_files},
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    if rank == 0:
        if (out/'protocol.json').exists():
            old = json.loads((out/'protocol.json').read_text())
            for k in ['rows','batch_size','keep_ratio','lr','epochs','insertion','code_sha256']:
                if old[k] != config[k]: raise RuntimeError(f'Incompatible restart: {k}')
        atomic_json(out/'protocol.json', config)
        (out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        for f in source_files:
            dest=out/'source_snapshots'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    barrier(world)
    print(json.dumps({'phase':'load_model','rank':rank,'device':str(device)}),flush=True)
    engine = OfficialUND(device=str(device)); engine.model.eval().requires_grad_(False)
    config['feature_dim'] = engine.model.config.hidden_size if a.insertion=='post-merger' else 768
    generator = MetaCompress(feature_dim=config['feature_dim']).to(device=device)
    if rank==0: atomic_json(out/'protocol.json',config)
    optimizer = torch.optim.SGD(generator.parameters(), lr=a.lr)
    first_step = 0
    ckpath = out/'meta_compress.pt'
    if ckpath.exists():
        saved = torch.load(ckpath,map_location='cpu',weights_only=False)
        generator.load_state_dict(saved['generator']); optimizer.load_state_dict(saved['optimizer'])
        first_step = saved['step']
    if rank == 0 and first_step == 0:
        atomic_save(out/'initial.pt', {'generator':generator.state_dict(),'config':config})
    wrapped = DDP(generator, device_ids=[local] if a.device=='npu' else None,
                  broadcast_buffers=False) if world>1 else generator
    steps_per_epoch = math.ceil(len(rows)/a.batch_size)
    total_steps = steps_per_epoch*a.epochs
    start = time.monotonic()
    print(json.dumps({'phase':'train_start','rank':rank,'steps':total_steps,'resume_step':first_step,
                      'parameters':sum(v.numel() for v in generator.parameters())}),flush=True)
    for global_step in range(first_step,total_steps):
        epoch = global_step//steps_per_epoch
        in_epoch = global_step%steps_per_epoch
        order = list(range(len(rows))); random.Random(42+epoch).shuffle(order)
        # Pad the final global batch; this adds at most batch_size-1 samples/epoch.
        padded = [order[i%len(order)] for i in range(steps_per_epoch*a.batch_size)]
        batch = padded[in_epoch*a.batch_size:(in_epoch+1)*a.batch_size]
        optimizer.zero_grad(set_to_none=True)
        values = torch.zeros(6,device=device,dtype=torch.float64)
        for micro in range(accum):
            row = rows[batch[micro*world+rank]]
            tick=time.monotonic()
            prompt, raw_bf16, grid, teacher_projected, response = prepare_example(engine,row,cache,a.max_new_tokens)
            teacher_logits=teacher_logits_for(engine,row,prompt,teacher_projected,response,cache)
            context = wrapped.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            with context:
                source=teacher_projected if a.insertion=='post-merger' else raw_bf16
                compressed, pmat = wrapped(source.float(),grid,a.keep_ratio)
                student_projected = (compressed.to(dtype=raw_bf16.dtype) if a.insertion=='post-merger'
                                     else engine.model.vision_tower.merger(compressed.to(dtype=raw_bf16.dtype)))
                sprompt=shortened_inputs({'input_ids':prompt},len(compressed),engine.model.config.image_token_id)['input_ids']
                student_logits = common_prefix_logits(engine.model,sprompt,student_projected[None],response)
                losses=meta_losses(pmat,teacher_logits,student_logits)
                if not torch.isfinite(losses['total']): raise RuntimeError(f'Nonfinite loss {row["id"]}')
                if a.smoke and world==1 and micro==0 and global_step==0:
                    pg=torch.autograd.grad(losses['prediction'],tuple(generator.parameters()),retain_graph=True)
                    pg_norm=float(torch.stack([v.float().norm().square() for v in pg]).sum().sqrt())
                    if not math.isfinite(pg_norm) or pg_norm==0:raise RuntimeError('KL gradient does not reach generator')
                    atomic_json(out/'gradient_audit.json',{'prediction_gradient_norm':pg_norm,
                                'frozen_model_trainable_parameters':sum(v.numel() for v in engine.model.parameters() if v.requires_grad)})
                (losses['total']/accum).backward()
            entry={'epoch':epoch+1,'step':global_step+1,'id':row['id'],'task':row['task'],
                   **{k:float(v.detach()) for k,v in losses.items()},'tokens':len(raw_bf16),
                   'compressed_tokens':len(compressed),'seconds':time.monotonic()-tick,
                   'p_row_sum_error':float((pmat.detach().sum(-1)-1).abs().max())}
            with (out/f'examples.rank{rank}.jsonl').open('a') as h:h.write(json.dumps(entry)+'\n')
            values += torch.tensor([entry[k] for k in ['prediction','entropy','collapse','total','tokens','compressed_tokens']],device=device,dtype=torch.float64)
            if a.smoke: print(json.dumps({'rank':rank,**entry}),flush=True)
            del teacher_logits,student_logits,teacher_projected,student_projected,losses,pmat,compressed
        grad_norm=torch.nn.utils.clip_grad_norm_(generator.parameters(),a.grad_clip)
        if not torch.isfinite(grad_norm): raise RuntimeError('Nonfinite gradient')
        if a.smoke and float(grad_norm)==0: raise RuntimeError('No generator gradient')
        optimizer.step()
        if world>1:dist.all_reduce(values)
        values/=a.batch_size
        if rank==0:
            entry={'epoch':epoch+1,'step':global_step+1,'total_steps':total_steps,
                   **dict(zip(['prediction','entropy','collapse','loss','tokens','compressed_tokens'],values.tolist())),
                   'grad_norm':float(grad_norm),'elapsed_seconds':time.monotonic()-start}
            with (out/'training.jsonl').open('a') as h:h.write(json.dumps(entry)+'\n')
            print(json.dumps(entry),flush=True)
            if (global_step+1)%a.checkpoint_interval==0 or global_step+1==total_steps:
                atomic_save(ckpath,{'generator':{k:v.detach().cpu() for k,v in generator.state_dict().items()},
                                   'optimizer':optimizer.state_dict(),'step':global_step+1,'epoch':epoch+1,'config':config})
    barrier(world)
    if rank==0:
        final={'steps':total_steps,'rows':len(rows),'epochs':a.epochs,
               'sample_presentations':total_steps*a.batch_size,'checkpoint':str(ckpath),
               'seconds':time.monotonic()-start,'checkpoint_sha256':sha(ckpath)}
        atomic_json(out/'complete.json',final)
    barrier(world)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
