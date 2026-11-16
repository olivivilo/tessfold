"""Versioned feedback ablations; legacy generator/source files stay unchanged."""
from __future__ import annotations

import heapq
import math
import numpy as np
import torch
from torch.nn import functional as F
from vqa.compression_generators_v2 import SpatialGenerator, adaptive_rectangles
from vqa.compression_generators import coordinates
from vqa.meta_compress import MetaCompress


def score_parts(module, x, raw, grid, question):
    """Reproduce exactly the active legacy C/T branches, exposing their signals."""
    x, raw = x.float(), raw.float()
    h, w, block, inverse = coordinates(grid, x.device)
    n = h*w
    assert len(x) == len(raw) == n
    def image(v):
        return v[inverse].reshape(h,w,-1).permute(2,0,1)[None]
    def sequence(v):
        return v[0].permute(1,2,0).reshape(n,-1)[block]
    z = F.gelu(module.input(torch.cat([module.xnorm(x),module.hnorm(raw)],-1)))
    z = z + sequence(module.conv(image(z)))
    score = module.score(z).squeeze(-1)
    rho = torch.zeros_like(score)
    if module.kind == 'quadtree_question':
        q = F.gelu(module.qproj(question.float()[:64]))
        rho = (F.normalize(module.word_key(z),dim=-1) @ F.normalize(q,dim=-1).T).max(-1).values
        score = score + rho
    low = sequence(F.avg_pool2d(image(raw[:,:64]),3,1,1,count_include_pad=False))
    detail = (raw[:,:64]-low).square().mean(-1).sqrt().detach()
    detail = detail/detail.mean().clamp_min(1e-6)
    energy = (detail+.1)*(2*score.detach().sigmoid()+.1)
    return score, rho, energy, (h,w,block,inverse)


def midpoint_rectangles(energy,h,w,m):
    """J: same max-mass leaf/axis/ties; only the split point becomes floor-mid."""
    im=energy.detach().float().cpu().numpy().reshape(h,w)
    integral=np.pad(im,((1,0),(1,0))).cumsum(0).cumsum(1)
    heap=[];leaves={};serial=0
    def add(rect):
        nonlocal serial
        k=serial;serial+=1;leaves[k]=rect
        a,b,c,d=rect
        if (b-a)*(d-c)>1:
            mass=integral[b,d]-integral[a,d]-integral[b,c]+integral[a,c]
            heapq.heappush(heap,(-float(mass),k))
    add((0,h,0,w))
    while len(leaves)<m:
        _,k=heapq.heappop(heap);a,b,c,d=leaves.pop(k)
        hh,ww=b-a,d-c;vertical=ww>hh
        if ww==1:vertical=False
        if hh==1:vertical=True
        if vertical:
            cut=c+ww//2;add((a,b,c,cut));add((a,b,cut,d))
        else:
            cut=a+hh//2;add((a,cut,c,d));add((cut,b,c,d))
    g=np.empty((h,w),dtype=np.int64)
    for k,(a,b,c,d) in enumerate(sorted(leaves.values())):g[a:b,c:d]=k
    return g.reshape(-1)


def make_groups(energy,shape,m,mode='tree',midpoint=False):
    h,w,block,inverse=shape
    fn=midpoint_rectangles if midpoint else adaptive_rectangles
    arr=fn(energy[inverse],h,w,m) if midpoint else fn(energy[inverse],h,w,m,mode)
    return torch.as_tensor(arr.copy(),device=energy.device)[block]


def weights(module,score,groups,m,order_score=None,aggregation='soft'):
    base=torch.arange(m,device=score.device)[:,None]==groups[None]
    tau=.25+F.softplus(module.temperature)
    soft=(score[None].expand_as(base)/tau).masked_fill(~base,-1e4).softmax(-1)
    if order_score is None:
        representative=soft.detach().argmax(-1)
    else:
        ref=(order_score[None].expand_as(base)/tau).masked_fill(~base,-1e4).softmax(-1)
        representative=ref.detach().argmax(-1)
    order=representative.argsort(stable=True)
    if aggregation=='hard':p=F.one_hot(soft.argmax(-1),len(score)).float()
    elif aggregation=='mean':p=base.float()/base.sum(-1,keepdim=True)
    else:p=soft
    return p[order],order


class AblationSpatial(SpatialGenerator):
    def __init__(self,method):
        super().__init__('col_adaptive' if method=='V' else 'quadtree_question')
        self.method=method

    def forward(self,x,grid,keep_ratio=.25,raw=None,question=None):
        m=max(1,min(len(x),round(len(x)*keep_ratio)))
        s,_,e,shape=score_parts(self,x,raw,grid,question)
        g=make_groups(e,shape,m,midpoint=self.method=='J')
        p,_=weights(self,s,g,m)
        return p@x.float(),p


def build(cfg):
    if cfg['method']=='M':return MetaCompress(feature_dim=2048,compression_dim=64)
    if cfg['method'] in ['V','J']:return AblationSpatial(cfg['method'])
    return SpatialGenerator('quadtree_question' if cfg['method'] in ['T','TP'] else 'col_adaptive')


def intervention(module,raw,x,grid,q,qwrong,ratio,variant):
    """Pre-merger T with explicit real leaf IDs, independently routed Q/W signals."""
    n=len(raw);m=max(1,min(n,round(n*ratio)));zero=torch.zeros_like(x,dtype=torch.float32)
    s,_,e,shape=score_parts(module,zero,raw,grid,q)
    sw,_,ew,_=score_parts(module,zero,raw,grid,qwrong)
    wrong_partition=variant in ['WQ','WW']
    g=make_groups(ew if wrong_partition else e,shape,m)
    selected=sw if variant in ['QW','WW'] else s
    aggregation={'H':'hard','A':'mean'}.get(variant,'soft')
    p,order=weights(module,selected,g,m,order_score=s,aggregation=aggregation)
    gq=make_groups(e,shape,m)
    h,w,_,inverse=shape;a=gq[inverse].reshape(h,w);b=g[inverse].reshape(h,w)
    changed=((a[1:]!=a[:-1])!=(b[1:]!=b[:-1])).sum()+((a[:,1:]!=a[:,:-1])!=(b[:,1:]!=b[:,:-1])).sum()
    ref,_=weights(module,s,g,m,order_score=s)
    detail=dict(boundary_change=float(changed)/max(1,(h-1)*w+h*(w-1)),
                weight_l1_mean=float((p-ref).abs().sum(-1).mean()),
                leaf_ids=g.detach().cpu().tolist(),leaf_order=order.detach().cpu().tolist())
    return p,detail


def coarse_regions(h,w,m,side=4):
    gh=min(side,h);gw=min(side,w)
    while gh*gw>m:
        if gw>=gh and gw>1:gw-=1
        else:gh-=1
    yy=np.rint(np.linspace(0,h,gh+1)).astype(int)
    xx=np.rint(np.linspace(0,w,gw+1)).astype(int)
    return [(int(yy[i]),int(yy[i+1]),int(xx[j]),int(xx[j+1])) for i in range(gh) for j in range(gw)]


def allocate(scores,capacities,m,minimum=1):
    """S2 appendix A.3: iterative capacity-constrained largest remainder."""
    c=np.asarray(scores,dtype=np.float64);cap=np.asarray(capacities,dtype=int)
    b=np.full(len(cap),minimum,dtype=int)
    assert np.isfinite(c).all() and (c>=0).all() and m>=b.sum() and m<=cap.sum()
    while b.sum()<m:
        active=np.flatnonzero(b<cap);left=int(m-b.sum());ww=c[active].copy()
        if ww.sum()<=np.finfo(np.float64).eps:ww[:]=1
        q=left*ww/ww.sum();extra=np.minimum(np.floor(q).astype(int),cap[active]-b[active])
        b[active]+=extra;left=int(m-b.sum())
        order=sorted(range(len(active)),key=lambda i:(-(q[i]-math.floor(q[i])),-c[active[i]],int(active[i])))
        for i in order:
            g=active[i]
            if left>0 and b[g]<cap[g]:b[g]+=1;left-=1
    assert b.sum()==m and (b<=cap).all()
    return b


def grid_star(module,raw,x,grid,q,ratio):
    s,rho,_,shape=score_parts(module,torch.zeros_like(x),raw,grid,q)
    h,w,block,inverse=shape;n=len(raw);m=max(1,round(n*ratio))
    regions=coarse_regions(h,w,m,4);raster=torch.arange(n,device=x.device).reshape(h,w)
    members=[raster[a:b,c:d].reshape(-1) for a,b,c,d in regions]
    raster_s=s[inverse];raster_r=rho[inverse]
    rr=torch.stack([raster_r[ix].mean() for ix in members])
    alloc=allocate(rr.softmax(0).detach().cpu().numpy(),[len(ix) for ix in members],m)
    selected=[]
    for ix,k in zip(members,alloc):
        # Stable ties by raster index inside each fixed cell.
        chosen=raster_s[ix].argsort(descending=True,stable=True)[:int(k)]
        selected.append(inverse[ix[chosen]])
    idx=torch.cat(selected).sort().values
    p=F.one_hot(idx,n).float()
    return p,dict(coarse_regions=regions,quotas=alloc.tolist(),region_relevance=rr.detach().cpu().tolist(),selected_indices=idx.cpu().tolist())
