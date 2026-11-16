"""Second-round convex compressors: spatial support, sparse selection, adaptive allocation.

Only P is learned; the values passed to UND are always P @ its original features.
Discrete support/ordering decisions are detached. Hard choices use an explicitly
biased straight-through soft routing derivative, never a soft-only evaluation.
"""
from __future__ import annotations
import heapq
import math
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vqa.compression_generators import coordinates

NAMES = [
    'cell_soft', 'cell_hard', 'row_adaptive', 'col_adaptive',
    'quadtree_detail', 'quadtree_question', 'density_xy', 'deformable_2d',
    'edge_anchor', 'global_topk', 'stratified_topk', 'hybrid_uniform_detail',
    'hybrid_uniform_question', 'similarity_medoid', 'sparse_relu', 'relation_axis',
    'region_attention', 'sinkhorn_spatial', 'prototype_cells', 'axial_deformable',
    'dual_representative', 'moment_pair', 'select_merge_gate', 'multiscale_question',
]

def small_head(d, out=1):
    net = nn.Sequential(nn.Linear(d, 96), nn.GELU(), nn.Linear(96, out))
    nn.init.normal_(net[-1].weight, std=.01)
    nn.init.zeros_(net[-1].bias)
    return net

@lru_cache(maxsize=256)
def spatial_partition(h, w, m):
    """Exactly m nonempty disjoint compact rectangles, independent of content."""
    groups = np.empty((h, w), dtype=np.int64)
    counter = [0]
    def divide(y0, y1, x0, x1, count):
        if count == 1:
            groups[y0:y1, x0:x1] = counter[0]; counter[0] += 1
            return
        hh, ww = y1-y0, x1-x0
        vertical = ww > hh
        if ww == 1: vertical = False
        if hh == 1: vertical = True
        length, other = (ww, hh) if vertical else (hh, ww)
        left = count//2
        cut = min(length-1, max(1, round(length*left/count)))
        left = min(cut*other, max(left, count-(length-cut)*other))
        left = min(count-1, max(1,left))
        assert 0 < cut < length
        if vertical:
            divide(y0,y1,x0,x0+cut,left); divide(y0,y1,x0+cut,x1,count-left)
        else:
            divide(y0,y0+cut,x0,x1,left); divide(y0+cut,y1,x0,x1,count-left)
    divide(0,h,0,w,m)
    assert counter[0] == m
    return groups.reshape(-1)

def adaptive_rectangles(energy, h, w, m, mode='tree'):
    """Detached greedy weighted bisection; no claim of differentiable tree splits."""
    image = energy.detach().float().cpu().numpy().reshape(h,w)
    integral = np.pad(image, ((1,0),(1,0))).cumsum(0).cumsum(1)
    def mass(y0,y1,x0,x1):
        return float(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])
    heap=[];leaves={};serial=[0]
    def add(rect):
        key=serial[0];serial[0]+=1;leaves[key]=rect
        y0,y1,x0,x1=rect
        if (y1-y0)*(x1-x0)>1:
            heapq.heappush(heap,(-mass(*rect),key))
    add((0,h,0,w))
    while len(leaves)<m:
        _,key=heapq.heappop(heap);y0,y1,x0,x1=leaves.pop(key)
        hh,ww=y1-y0,x1-x0
        vertical=(ww>hh) if mode=='tree' else mode=='col'
        if ww==1:vertical=False
        if hh==1:vertical=True
        if vertical:
            weights=image[y0:y1,x0:x1].sum(0);cut=int(np.searchsorted(weights.cumsum(),weights.sum()/2))+1
            cut=max(1,min(ww-1,cut));add((y0,y1,x0,x0+cut));add((y0,y1,x0+cut,x1))
        else:
            weights=image[y0:y1,x0:x1].sum(1);cut=int(np.searchsorted(weights.cumsum(),weights.sum()/2))+1
            cut=max(1,min(hh-1,cut));add((y0,y0+cut,x0,x1));add((y0+cut,y1,x0,x1))
    groups=np.empty((h,w),dtype=np.int64)
    for i,rect in enumerate(sorted(leaves.values())):
        y0,y1,x0,x1=rect;groups[y0:y1,x0:x1]=i
    return groups.reshape(-1)

def hard_st(soft, indices=None, train=True):
    if indices is None: indices=soft.argmax(-1)
    hard=F.one_hot(indices,soft.shape[-1]).to(soft.dtype)
    return hard+(soft-soft.detach()) if train else hard

class SpatialGenerator(nn.Module):
    def __init__(self,kind,hidden=96):
        super().__init__();assert kind in NAMES;self.kind=kind;d=hidden
        self.hnorm=nn.LayerNorm(768,elementwise_affine=False)
        self.xnorm=nn.LayerNorm(2048,elementwise_affine=False)
        self.input=nn.Linear(2816,d);self.qproj=nn.Linear(2048,d)
        self.conv=nn.Sequential(nn.Conv2d(d,d,3,padding=1,groups=d),nn.GELU(),nn.Conv2d(d,d,1),nn.GELU())
        self.score=small_head(d)
        if kind in ['quadtree_question','hybrid_uniform_question','multiscale_question']:
            self.word_key=nn.Linear(d,d,bias=False)
        if kind in ['density_xy','deformable_2d','edge_anchor','axial_deformable']:
            self.offset=small_head(d,2)
        if kind in ['region_attention','sinkhorn_spatial']:
            self.key=nn.Linear(d,d,bias=False);self.query=nn.Linear(d,d,bias=False)
        if kind=='prototype_cells':self.prototypes=nn.Parameter(torch.randn(32,d)*.1);self.proto_score=small_head(2*d)
        if kind in ['relation_axis','axial_deformable']:
            self.axis=nn.Linear(3*d,d)
        if kind=='multiscale_question':
            self.dilated=nn.Conv2d(d,d,3,padding=3,dilation=3,groups=d);self.multiscale=nn.Linear(3*d,d)
        if kind=='select_merge_gate':self.gate=small_head(d)
        if kind in ['dual_representative','moment_pair']:self.second=small_head(d)
        if kind=='similarity_medoid':self.metric=nn.Parameter(torch.zeros(d))
        self.temperature=nn.Parameter(torch.tensor(0.))

    def forward(self,x,grid,keep_ratio=.1,raw=None,question=None):
        assert raw is not None and question is not None
        x=x.float();raw=raw.float();n=len(x);m=max(1,min(n,round(n*keep_ratio)))
        h,w,block,inverse=coordinates(grid,x.device);assert n==h*w
        def image(v):return v[inverse].reshape(h,w,-1).permute(2,0,1)[None]
        def sequence(v):return v[0].permute(1,2,0).reshape(n,-1)[block]
        z=F.gelu(self.input(torch.cat([self.xnorm(x),self.hnorm(raw)],-1)))
        z=z+sequence(self.conv(image(z)))
        q=F.gelu(self.qproj(question.float()[:64]));k=self.kind
        if k in ['relation_axis','axial_deformable']:
            im=image(z);rr=sequence(im.mean(3,keepdim=True).expand_as(im));cc=sequence(im.mean(2,keepdim=True).expand_as(im))
            z=z+F.gelu(self.axis(torch.cat([z,rr,cc],-1)))
        if k=='multiscale_question':
            im=image(z);coarse=sequence(F.avg_pool2d(im,5,1,2,count_include_pad=False));wide=sequence(self.dilated(im))
            z=z+F.gelu(self.multiscale(torch.cat([z,coarse,wide],-1)))
        score=self.score(z).squeeze(-1)
        if k in ['quadtree_question','hybrid_uniform_question','multiscale_question']:
            # Learned visual/text metric; no decoder invocation in the router.
            relevance=(F.normalize(self.word_key(z),dim=-1)@F.normalize(q,dim=-1).T).max(-1).values
            score=score+relevance
        # Raw feature high-pass is a texture heuristic, not OCR supervision.
        rawsmall=raw[:,:64]
        low=sequence(F.avg_pool2d(image(rawsmall),3,1,1,count_include_pad=False))
        detail=(rawsmall-low).square().mean(-1).sqrt().detach()
        detail=detail/detail.mean().clamp_min(1e-6)
        group_cpu=spatial_partition(h,w,m)
        if k in ['quadtree_detail','quadtree_question','row_adaptive','col_adaptive']:
            energy=detail+.1
            if k in ['row_adaptive','col_adaptive','quadtree_question']:
                energy=energy*score.detach().sigmoid().mul(2).add(.1)
            group_cpu=adaptive_rectangles(energy[inverse],h,w,m,{'row_adaptive':'row','col_adaptive':'col'}.get(k,'tree'))
        groups=torch.as_tensor(group_cpu.copy(),device=x.device)[block]
        base=(torch.arange(m,device=x.device)[:,None]==groups[None]).float()
        normbase=base/base.sum(-1,keepdim=True)
        # Coordinates measured in patch units; all routing respects the official block order.
        raster=torch.arange(n,device=x.device);xy=torch.stack([(raster//w).float(),(raster%w).float()],-1)[block]
        centers=normbase@xy;means=normbase@z
        temperature=.25+F.softplus(self.temperature)
        def local(logits,mask=base):
            return (logits[None].expand_as(mask)/temperature).masked_fill(mask==0,-1e4).softmax(-1)
        p=None
        if k in ['cell_soft','row_adaptive','col_adaptive','quadtree_detail','quadtree_question','relation_axis','multiscale_question']:
            p=local(score)
        elif k in ['cell_hard','stratified_topk']:
            logits=score if k=='cell_hard' else score+.2*detail
            p=hard_st(local(logits),train=self.training)
        elif k=='sparse_relu':
            weights=(1+score/temperature).relu().square()[None]*base
            p=weights/(weights.sum(-1,keepdim=True)+1e-12)
            empty=weights.sum(-1,keepdim=True)<1e-10;p=torch.where(empty,normbase,p)
        elif k=='similarity_medoid':
            metric=self.metric.sigmoid();central=(z-(normbase@z)[groups]).square().mul(metric).mean(-1)
            p=hard_st(local(score-central+.1*detail),train=self.training)
        elif k=='prototype_cells':
            context=(z@self.prototypes.T/math.sqrt(z.shape[-1])).softmax(-1)@self.prototypes
            p=local(score+self.proto_score(torch.cat([z,context],-1)).squeeze(-1))
        elif k=='select_merge_gate':
            soft=local(score);hard=hard_st(soft,train=self.training)
            # Starts near exact selection; a learned gate can request local aggregation.
            gate=(self.gate(means)+3).sigmoid();p=gate*hard+(1-gate)*soft
        elif k in ['global_topk','hybrid_uniform_detail','hybrid_uniform_question']:
            saliency=score+.2*detail
            if k=='global_topk':indices=saliency.topk(m).indices
            else:
                fixed_count=max(1,m//2)
                anchors=((torch.arange(fixed_count,device=x.device)+.5)*n/fixed_count-.5).round().long()
                allowed=saliency.clone();allowed[anchors]=-1e4
                extras=allowed.topk(m-fixed_count).indices
                indices=torch.cat([anchors,extras])
            indices=indices.sort().values
            # Backward surrogate uses neighborhoods of the actual selected coordinates.
            dist=((xy[None]-xy[indices,None]).square()).sum(-1)
            soft=(saliency[None]/temperature-dist/8).softmax(-1)
            p=hard_st(soft,indices,self.training)
        elif k in ['density_xy','deformable_2d','edge_anchor','axial_deformable','region_attention','sinkhorn_spatial']:
            # Two-dimensional neighborhoods, no one-dimensional bin prior.
            scale=math.sqrt(n/m)
            if k in ['density_xy','deformable_2d','edge_anchor','axial_deformable']:
                offset=self.offset(means).tanh()*scale
                if k=='density_xy':
                    density=F.softplus(score)+.01
                    weighted=(base*density[None]);density_center=(weighted@xy)/weighted.sum(-1,keepdim=True)
                    centers=.5*centers+.5*density_center
                if k=='edge_anchor':
                    edge_center=local(detail)@xy;centers=.5*centers+.5*edge_center
                centers=centers+offset
            dist=((centers[:,None]-xy[None])/scale).square().sum(-1)
            logits=score[None]/temperature-2*dist
            if k in ['region_attention','sinkhorn_spatial']:
                queries=F.normalize(self.query(means),dim=-1);keys=F.normalize(self.key(z),dim=-1)
                logits=logits+2*(queries@keys.T)
            if k=='sinkhorn_spatial':
                for _ in range(4):
                    logits=logits-torch.logsumexp(logits,dim=1,keepdim=True)
                    logits=logits-torch.logsumexp(logits,dim=0,keepdim=True)+math.log(m/n)
            p=logits.softmax(-1)
        elif k in ['dual_representative','moment_pair']:
            # Two distinct representatives per larger spatial cell, then exact-budget truncation.
            cells=math.ceil(m/2);gg=torch.as_tensor(spatial_partition(h,w,cells).copy(),device=x.device)[block]
            support=(torch.arange(cells,device=x.device)[:,None]==gg[None]).float()
            first=local(score,support);firstidx=first.argmax(-1)
            secondscore=self.second(z).squeeze(-1)
            if k=='moment_pair':
                # Opposite local feature moment keeps both sides of a stroke/boundary.
                moment=z[:,0]-((support@z[:,0])/support.sum(-1))[gg]
                first=local(score+moment,support);firstidx=first.argmax(-1)
                secondscore=secondscore-moment
            remaining=support.clone();remaining[torch.arange(cells,device=x.device),firstidx]=0
            # For a singleton region the two representatives coincide; avoid all-masked softmax.
            remaining=torch.where(remaining.sum(-1,keepdim=True)>0,remaining,support)
            second=local(secondscore,remaining)
            a=hard_st(first,firstidx,self.training);b=hard_st(second,train=self.training)
            p=torch.stack([a,b],dim=1).reshape(-1,n)[:m]
        else:raise ValueError(k)
        # Maintain original token reading order after data-dependent region/selection decisions.
        indices=p.detach().argmax(-1).argsort(stable=True);p=p[indices]
        return p@x,p

class SpatialFixed(nn.Module):
    def __init__(self,kind):super().__init__();assert kind in ['cell_mean','cell_selection'];self.kind=kind
    def forward(self,x,grid,keep_ratio,**kwargs):
        h,w,block,_=coordinates(grid,x.device);n=len(x);m=max(1,min(n,round(n*keep_ratio)))
        group=torch.as_tensor(spatial_partition(h,w,m).copy(),device=x.device)[block]
        base=(torch.arange(m,device=x.device)[:,None]==group[None]).float()
        if self.kind=='cell_mean':p=base/base.sum(-1,keepdim=True)
        else:
            # Spatial center of each true 2-D cell, not middle of a serial bin.
            ii=torch.arange(n,device=x.device);xy=torch.stack([(ii//w).float(),(ii%w).float()],-1)[block]
            center=(base@xy)/base.sum(-1,keepdim=True)
            dist=(xy[None]-center[:,None]).square().sum(-1).masked_fill(base==0,1e9)
            p=F.one_hot(dist.argmin(-1),n).float()
        p=p[p.argmax(-1).argsort(stable=True)]
        return p@x.float(),p
