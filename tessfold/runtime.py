"""Device setup and exact gradient weighting for accumulated distributed batches."""
from __future__ import annotations
import datetime
import math
import os
import random


def initialize(name, threads=4):
    import torch
    import torch.distributed as dist
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local=int(os.environ.get('LOCAL_RANK',0));torch.set_num_threads(threads)
    if name=='auto':
        name='cuda' if torch.cuda.is_available() else 'cpu'
    if name=='npu':
        import torch_npu
        torch.npu.set_device(local)
        torch.npu.matmul.allow_hf32=False
        torch.npu.conv.allow_hf32=False
        device=torch.device(f'npu:{local}');backend='hccl'
    elif name=='cuda':
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32=False
        device=torch.device(f'cuda:{local}');backend='nccl'
    else:
        device=torch.device('cpu');backend='gloo'
    if world>1:
        dist.init_process_group(backend,timeout=datetime.timedelta(hours=6))
    return rank,world,local,device


def synchronize(device):
    import torch
    if device.type=='cuda':torch.cuda.synchronize(device)
    elif device.type=='npu':torch.npu.synchronize(device)


def barrier(world):
    if world>1:
        import torch.distributed as dist
        dist.barrier()


def seed(value):
    import numpy as np
    import torch
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def lr_factor(step,total,warmup_fraction=0.1,floor=0.1):
    warm=max(1,round(total*warmup_fraction))
    return (step+1)/warm if step<warm else floor+(1-floor)*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2


def shard_batch(indices,rank,world):
    """Equal forward counts with zero-weight padding; DDP averages gradients.

    Real examples each receive world/actual_batch_size before the DDP mean,
    so a short final batch is not duplicated or biased toward a smaller rank.
    """
    count=math.ceil(len(indices)/world)
    return [(indices[min(i,len(indices)-1)],world/len(indices) if i<len(indices) else 0.)
            for i in range(rank,count*world,world)]
