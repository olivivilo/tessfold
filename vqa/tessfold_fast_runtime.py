"""Versioned compiled-partition inference, exact and approximate paths separated.

Exact: retain original FP32 score/dense readout arithmetic; cache deterministic
geometry, compile NumPy-compatible partition, sort exactly representable unique
representative indices as FP32 to avoid NPU's int64 argsort CPU fallback.
Bucket/Hard: separate candidates, no inherited bit-exact/accuracy claims.
"""
from functools import lru_cache
import time
import numpy as np
import torch
from torch.nn import functional as F
from vqa.tessfold_native_partition import partition,support


@lru_cache(maxsize=128)
def geometry(h,w,device):
    assert h%2==w%2==0
    block=np.arange(h*w,dtype=np.int64).reshape(h//2,2,w//2,2).transpose(0,2,1,3).reshape(-1).copy()
    inverse=np.empty_like(block);inverse[block]=np.arange(h*w,dtype=np.int64)
    return block,inverse,torch.from_numpy(block).to(device),torch.from_numpy(inverse).to(device)


def dimensions(grid):
    t,h,w=map(int,grid.detach().cpu().reshape(-1,3)[0].tolist());assert t==1
    return h,w


def score_features(module,raw,question,h,w,fold_zero=False):
    block,inverse,b,i=geometry(h,w,str(raw.device));n=len(raw);raw=raw.float()
    def image(x):return x[i].reshape(h,w,-1).permute(2,0,1)[None]
    def sequence(x):return x[0].permute(1,2,0).reshape(n,-1)[b]
    if fold_zero:
        z=F.gelu(F.linear(module.hnorm(raw),module.input.weight[:,2048:],module.input.bias))
    else:
        zero=torch.zeros((n,2048),dtype=torch.float32,device=raw.device)
        z=F.gelu(module.input(torch.cat([module.xnorm(zero),module.hnorm(raw)],-1)))
    z=z+sequence(module.conv(image(z)));score=module.score(z).squeeze(-1)
    if module.kind=='quadtree_question':
        q=F.gelu(module.qproj(question.float()[:64]))
        score=score+(F.normalize(module.word_key(z),dim=-1)@F.normalize(q,dim=-1).T).max(-1).values
    low=sequence(F.avg_pool2d(image(raw[:,:64]),3,1,1,count_include_pad=False))
    detail=(raw[:,:64]-low).square().mean(-1).sqrt().detach()
    detail=detail/detail.mean().clamp_min(1e-6)
    energy=(detail+.1)*score.detach().sigmoid().mul(2).add(.1)
    return score,energy


class ScoreGraph:
    """Optional shape-specialized NPU graph; setup and persistent memory measured."""
    @torch.no_grad()
    def __init__(self,module,raw,question,h,w):
        self.raw=raw.clone();self.question=question.clone() if question is not None else None
        geometry(h,w,str(raw.device));torch.npu.synchronize();start=time.perf_counter()
        stream=torch.npu.Stream();stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):score_features(module,self.raw,self.question,h,w)
        torch.npu.current_stream().wait_stream(stream);torch.npu.synchronize()
        self.graph=torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):self.outputs=score_features(module,self.raw,self.question,h,w)
        torch.npu.synchronize();self.setup_ms=(time.perf_counter()-start)*1e3

    @torch.no_grad()
    def __call__(self,raw,question):
        self.raw.copy_(raw)
        if self.question is not None:self.question.copy_(question)
        self.graph.replay()
        return self.outputs


def dense_readout(raw,score,groups,m,tau):
    n=len(raw);mask=torch.arange(m,device=raw.device)[:,None]==groups[None]
    p=(score[None].expand(m,n)/tau).masked_fill(~mask,-1e4).softmax(-1)
    rep=p.argmax(-1)
    assert n<2**24 # integers are exact FP32, each disjoint leaf has a unique rep
    order=rep.float().argsort(stable=True)
    p=p[order]
    return (p@raw.float()).to(raw.dtype),rep[order],p


class GraphCompressor:
    """Two graphs separated by exact CPU partition, no operator approximation.

    Persistent plan memory and capture cost must be reported. A plan serves only
    its fixed geometry/question length; callers may use bounded plan caches.
    """
    @torch.no_grad()
    def __init__(self,module,raw,question,shape,ratio):
        h,w=shape;self.shape=shape;self.ratio=ratio;self.module=module
        self.n=len(raw);self.m=max(1,min(self.n,round(self.n*ratio)))
        before=int(torch.npu.memory_allocated());torch.npu.synchronize();start=time.perf_counter()
        self.score=ScoreGraph(module,raw,question,h,w)
        _,energy=self.score(raw,question)
        block,inverse,_,_=geometry(h,w,str(raw.device))
        raster=partition(energy.cpu().numpy()[inverse].reshape(h,w),self.m,'col' if module.kind=='col_adaptive' else 'tree')
        self.groups=torch.from_numpy(np.ascontiguousarray(raster[block])).to(raw.device)
        self.host_pinned=True
        try:
            self.host_energy=torch.empty(self.n,dtype=torch.float32,pin_memory=True)
            self.host_groups=torch.empty(self.n,dtype=torch.long,pin_memory=True)
        except RuntimeError:
            self.host_pinned=False
            self.host_energy=torch.empty(self.n,dtype=torch.float32)
            self.host_groups=torch.empty(self.n,dtype=torch.long)
        self.energy_numpy=self.host_energy.numpy();self.groups_numpy=self.host_groups.numpy()
        self.transfer_done=torch.npu.Event()
        tau=.25+F.softplus(module.temperature)
        self.temperature=tau
        for _ in range(2):dense_readout(self.score.raw,self.score.outputs[0],self.groups,self.m,tau)
        torch.npu.synchronize();self.readout=torch.npu.NPUGraph()
        with torch.npu.graph(self.readout):
            self.output,self.representatives,self.weights=dense_readout(self.score.raw,self.score.outputs[0],self.groups,self.m,tau)
        torch.npu.synchronize();self.setup_ms=(time.perf_counter()-start)*1e3
        self.persistent_allocated_bytes=int(torch.npu.memory_allocated())-before

    @torch.no_grad()
    def __call__(self,raw,question):
        _,energy=self.score(raw,question);h,w=self.shape
        block,inverse,_,_=geometry(h,w,str(raw.device))
        self.host_energy.copy_(energy,non_blocking=self.host_pinned)
        self.transfer_done.record();self.transfer_done.synchronize()
        raster=partition(self.energy_numpy[inverse].reshape(h,w),self.m,'col' if self.module.kind=='col_adaptive' else 'tree')
        self.groups_numpy[:]=raster[block]
        self.groups.copy_(self.host_groups,non_blocking=self.host_pinned)
        self.readout.replay()
        return self.output,dict(representatives=self.representatives,weights=self.weights,n=self.n,m=self.m)


class SparseGraphCompressor:
    """O(N*d) grouped FP32 normalization and native index-add, numerical variant.

    Avoid unsupported scatter-reduce maxima: positive-leaf metadata is computed
    on CPU beside the exact compiled partition; features never leave the NPU.
    Native index-add reduction order can differ from dense GEMM; not bit-exact.
    """
    @torch.no_grad()
    def __init__(self,module,raw,question,shape,ratio):
        h,w=shape;self.shape=shape;self.module=module;self.n=len(raw);self.m=max(1,min(self.n,round(self.n*ratio)))
        before=int(torch.npu.memory_allocated());torch.npu.synchronize();start=time.perf_counter()
        self.score=ScoreGraph(module,raw,question,h,w)
        self.host_energy=torch.empty(self.n,dtype=torch.float32,pin_memory=True)
        self.host_score=torch.empty(self.n,dtype=torch.float32,pin_memory=True)
        self.host_groups=torch.empty(self.n,dtype=torch.long,pin_memory=True)
        self.host_maxima=torch.empty(self.m,dtype=torch.float32,pin_memory=True)
        self.host_order=torch.empty(self.m,dtype=torch.long,pin_memory=True)
        self.group=self.host_groups.to(raw.device);self.maximum=self.host_maxima.to(raw.device);self.order=self.host_order.to(raw.device)
        self.transfer_done=torch.npu.Event()
        self.score(raw,question);self.metadata()
        tau=.25+F.softplus(module.temperature)
        self.temperature=tau
        def readout():
            logits=self.score.outputs[0]/tau
            ex=(logits-self.maximum[self.group]/tau).exp()
            denominator=torch.zeros(self.m,device=raw.device,dtype=torch.float32).index_add(0,self.group,ex)
            weights=ex/denominator[self.group]
            output=torch.zeros((self.m,raw.shape[1]),device=raw.device,dtype=torch.float32).index_add(0,self.group,self.score.raw.float()*weights[:,None])
            return output[self.order].to(raw.dtype),weights
        for _ in range(2):readout()
        torch.npu.synchronize();self.readout=torch.npu.NPUGraph()
        with torch.npu.graph(self.readout):self.output,self.weights=readout()
        torch.npu.synchronize();self.setup_ms=(time.perf_counter()-start)*1e3
        self.persistent_allocated_bytes=int(torch.npu.memory_allocated())-before

    def metadata(self):
        self.host_energy.copy_(self.score.outputs[1],non_blocking=True);self.host_score.copy_(self.score.outputs[0],non_blocking=True)
        self.transfer_done.record();self.transfer_done.synchronize()
        h,w=self.shape;block,inverse,_,_=geometry(h,w,str(self.score.raw.device))
        raster=partition(self.host_energy.numpy()[inverse].reshape(h,w),self.m,'col' if self.module.kind=='col_adaptive' else 'tree')
        groups=self.host_groups.numpy();groups[:]=raster[block];scores=self.host_score.numpy()
        maxima=self.host_maxima.numpy();maxima.fill(-np.inf);np.maximum.at(maxima,groups,scores)
        rep=np.full(self.m,self.n,np.int64)
        np.minimum.at(rep,groups,np.where(scores==maxima[groups],np.arange(self.n,dtype=np.int64),self.n))
        order=self.host_order.numpy();order[:]=rep.argsort(kind='stable');self.representatives_cpu=rep[order].copy()
        self.group.copy_(self.host_groups,non_blocking=True);self.maximum.copy_(self.host_maxima,non_blocking=True);self.order.copy_(self.host_order,non_blocking=True)

    @torch.no_grad()
    def __call__(self,raw,question):
        self.score(raw,question);self.metadata();self.readout.replay()
        return self.output,dict(weights=self.weights,groups=self.group,representatives_cpu=self.representatives_cpu,n=self.n,m=self.m)


def bucket_readout(raw,score,groups_cpu,m,tau):
    """Power-of-two support buckets: padded supports total <2N; no scatter max.

    Same convex aggregation in real arithmetic; reduction/softmax rounding can
    differ from dense GEMM. Treat and evaluate as a separate numerical variant.
    """
    permutation,offsets=support(groups_cpu,m);sizes=np.diff(offsets)
    output=torch.empty((m,raw.shape[1]),device=raw.device,dtype=raw.dtype)
    representatives=torch.empty(m,device=raw.device,dtype=torch.long)
    padded=0
    for power in range(int(sizes.max()).bit_length()+1):
        width=1<<power;rows=np.flatnonzero((sizes>(width//2 if power else 0))&(sizes<=width))
        if not len(rows):continue
        positions=offsets[rows,None]+np.arange(width)[None]
        valid=positions<offsets[rows+1,None]
        idx=np.where(valid,permutation[np.minimum(positions,len(raw)-1)],0)
        ids=torch.from_numpy(idx).to(raw.device);valid_tensor=torch.from_numpy(valid).to(raw.device)
        rr=torch.from_numpy(rows).to(raw.device);padded+=len(rows)*width
        if width==1:
            value=raw.index_select(0,ids[:,0]);rep=ids[:,0]
        else:
            logits=(score[ids]/tau).masked_fill(~valid_tensor,-1e4)
            weights=logits.softmax(-1)
            value=torch.bmm(weights[:,None],raw[ids].float()).squeeze(1).to(raw.dtype)
            rep=ids.gather(1,weights.argmax(-1)[:,None]).squeeze(1)
        output.index_copy_(0,rr,value);representatives.index_copy_(0,rr,rep)
    assert padded<2*len(raw) or padded==len(raw)
    order=representatives.float().argsort(stable=True)
    return output[order],representatives[order],dict(padded_support=padded,logical_nonzeros=len(raw))


@torch.no_grad()
def compress(module,raw,grid,question,ratio,backend='dense_exact',shape=None,score_graph=None,profile=False):
    h,w=shape or dimensions(grid);n=len(raw);assert h*w==n
    block,inverse,_,_=geometry(h,w,str(raw.device));m=max(1,min(n,round(n*ratio)))
    timings={}
    def begin():
        if profile:torch.npu.synchronize();return time.perf_counter()
    def end(name,start):
        if profile:torch.npu.synchronize();timings[name]=(time.perf_counter()-start)*1e3
    ts=begin()
    score,energy=score_graph(raw,question) if score_graph is not None else score_features(module,raw,question,h,w,backend=='bucket_fold')
    end('score',ts);ts=begin()
    transfer=torch.stack([energy,score]).cpu().numpy() if backend=='hard_linear' else energy.cpu().numpy()[None]
    end('device_to_host',ts);ts=begin()
    raster=partition(transfer[0,inverse].reshape(h,w),m,'col' if module.kind=='col_adaptive' else 'tree')
    group_cpu=np.ascontiguousarray(raster[block]);end('compiled_partition',ts);ts=begin()
    tau=.25+F.softplus(module.temperature);extra={}
    if backend=='dense_exact':
        groups=torch.from_numpy(group_cpu).to(raw.device);c,rep,p=dense_readout(raw,score,groups,m,tau)
    elif backend in ['bucket_fp32','bucket_fold']:
        c,rep,extra=bucket_readout(raw,score,group_cpu,m,tau);p=None
    elif backend=='hard_linear':
        maxima=np.full(m,-np.inf,np.float32);np.maximum.at(maxima,group_cpu,transfer[1])
        candidates=np.where(transfer[1]==maxima[group_cpu],np.arange(n,dtype=np.int64),n)
        representatives=np.full(m,n,np.int64);np.minimum.at(representatives,group_cpu,candidates)
        representatives.sort();rep=torch.from_numpy(representatives).to(raw.device)
        c=raw.index_select(0,rep);p=None
    else:raise ValueError(backend)
    end('readout',ts)
    return c,dict(representatives=rep,weights=p,groups_cpu=group_cpu,score=score,energy=energy,n=n,m=m,backend=backend,timings_ms=timings,**extra)
