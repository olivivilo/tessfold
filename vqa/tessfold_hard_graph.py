"""Linear-support top-token variant, separate from the original soft model."""
import time
import numpy as np
import torch
from vqa.tessfold_fast_runtime import ScoreGraph,geometry
from vqa.tessfold_native_partition import partition

class HardGraph:
    @torch.no_grad()
    def __init__(self,module,raw,question,shape,ratio):
        self.module=module;self.shape=shape;self.n=len(raw);self.m=max(1,min(self.n,round(self.n*ratio)))
        before=int(torch.npu.memory_allocated());torch.npu.synchronize();ts=time.perf_counter()
        self.score=ScoreGraph(module,raw,question,*shape)
        self.host_energy=torch.empty(self.n,dtype=torch.float32,pin_memory=True)
        self.host_score=torch.empty(self.n,dtype=torch.float32,pin_memory=True)
        self.host_reps=torch.empty(self.m,dtype=torch.long,pin_memory=True)
        self.reps=self.host_reps.to(raw.device);self.done=torch.npu.Event()
        self.update(raw,question)
        for _ in range(2):raw.index_select(0,self.reps)
        torch.npu.synchronize();self.graph=torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):self.output=self.score.raw.index_select(0,self.reps)
        torch.npu.synchronize();self.setup_ms=(time.perf_counter()-ts)*1e3
        self.persistent_allocated_bytes=int(torch.npu.memory_allocated())-before

    def update(self,raw,question):
        score,energy=self.score(raw,question)
        self.host_score.copy_(score,non_blocking=True);self.host_energy.copy_(energy,non_blocking=True)
        self.done.record();self.done.synchronize()
        h,w=self.shape;block,inverse,_,_=geometry(h,w,str(raw.device))
        raster=partition(self.host_energy.numpy()[inverse].reshape(h,w),self.m,'col' if self.module.kind=='col_adaptive' else 'tree')
        group=raster[block];scores=self.host_score.numpy();maxima=np.full(self.m,-np.inf,np.float32)
        np.maximum.at(maxima,group,scores)
        rep=np.full(self.m,self.n,np.int64)
        np.minimum.at(rep,group,np.where(scores==maxima[group],np.arange(self.n,dtype=np.int64),self.n));rep.sort()
        self.host_reps.numpy()[:]=rep;self.reps.copy_(self.host_reps,non_blocking=True)

    @torch.no_grad()
    def __call__(self,raw,question):
        self.update(raw,question);self.graph.replay()
        return self.output,dict(representatives=self.reps,n=self.n,m=self.m)
