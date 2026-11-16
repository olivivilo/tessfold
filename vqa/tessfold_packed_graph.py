"""Deterministic padded-local GEMM, avoids all global scatter-add reductions.

Numerically approximate soft readout. It stores M*capacity support slots, not
M*N. Capacity is the next power of two above actual largest leaf, not truncated.
Strict worst-case is NOT globally O(N): report actual slots/N and fallback.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from vqa.tessfold_fast_runtime import ScoreGraph,geometry
from vqa.tessfold_native_partition import partition,support

class PackedGraph:
    @torch.no_grad()
    def __init__(self,module,raw,question,shape,ratio):
        self.module=module;self.shape=shape;self.n=len(raw);self.m=max(1,min(self.n,round(self.n*ratio)))
        before=int(torch.npu.memory_allocated());torch.npu.synchronize();ts=time.perf_counter()
        self.score=ScoreGraph(module,raw,question,*shape)
        self.host_energy=torch.empty(self.n,dtype=torch.float32,pin_memory=True);self.done=torch.npu.Event()
        _,energy=self.score(raw,question);self.host_energy.copy_(energy,non_blocking=True);self.done.record();self.done.synchronize()
        groups=self.partition();_,offset=support(groups,self.m);self.capacity=1<<(int(np.diff(offset).max())-1).bit_length()
        self.host_indices=torch.empty((self.m,self.capacity),dtype=torch.long,pin_memory=True)
        self.host_mask=torch.empty((self.m,self.capacity),dtype=torch.bool,pin_memory=True)
        self.indices=self.host_indices.to(raw.device);self.mask=self.host_mask.to(raw.device)
        self.pack(groups)
        self.tau=.25+F.softplus(module.temperature)
        def readout():
            logits=(self.score.outputs[0][self.indices]/self.tau).masked_fill(~self.mask,-1e4)
            weights=logits.softmax(-1)
            values=self.score.raw[self.indices].float()
            output=torch.bmm(weights[:,None],values).squeeze(1).to(raw.dtype)
            rep=self.indices.gather(1,weights.argmax(-1)[:,None]).squeeze(1)
            order=rep.float().argsort(stable=True)
            return output[order],rep[order]
        for _ in range(2):readout()
        torch.npu.synchronize();self.graph=torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):self.output,self.reps=readout()
        torch.npu.synchronize();self.setup_ms=(time.perf_counter()-ts)*1e3
        self.persistent_allocated_bytes=int(torch.npu.memory_allocated())-before

    def partition(self):
        h,w=self.shape;block,inverse,_,_=geometry(h,w,str(self.score.raw.device))
        raster=partition(self.host_energy.numpy()[inverse].reshape(h,w),self.m,'col' if self.module.kind=='col_adaptive' else 'tree')
        return np.ascontiguousarray(raster[block])

    def pack(self,groups):
        permutation,offset=support(groups,self.m);sizes=np.diff(offset)
        if sizes.max()>self.capacity:raise OverflowError('Leaf capacity changed: recapture a larger plan, never truncate support')
        positions=offset[:-1,None]+np.arange(self.capacity)[None]
        valid=positions<offset[1:,None]
        self.host_indices.numpy()[:]=np.where(valid,permutation[np.minimum(positions,self.n-1)],0)
        self.host_mask.numpy()[:]=valid
        self.indices.copy_(self.host_indices,non_blocking=True);self.mask.copy_(self.host_mask,non_blocking=True)

    @torch.no_grad()
    def __call__(self,raw,question):
        _,energy=self.score(raw,question);self.host_energy.copy_(energy,non_blocking=True);self.done.record();self.done.synchronize()
        self.pack(self.partition());self.graph.replay()
        return self.output,dict(representatives=self.reps,n=self.n,m=self.m,support_slots=self.m*self.capacity)
