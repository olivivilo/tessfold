"""24 frozen-backbone convex routing hypotheses. See docs/COMPRESSION_GENERATOR_DESIGNS.md."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

NAMES=['local_mlp','spatial_cnn','dilated_cnn','window_attention','question_gate','question_cross',
       'anchor_resampler','balanced_transport','soft_kmeans','local_medoid','novelty_gate','raw_detail',
       'channel_uncertainty','pyramid_context','local_global_mixture','hard_local_selection',
       'monotone_density','hierarchical_merge','axial_context','highpass_gate','spatial_graph',
       'question_moe','multihead_transport','prototype_memory']

def mlp(d,out=1):
    net=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,out))
    nn.init.normal_(net[-1].weight,std=.001);nn.init.zeros_(net[-1].bias)
    return net

def coordinates(grid,device):
    t,h,w=map(int,grid.reshape(-1,3)[0].detach().cpu().tolist());assert t==1 and h%2==w%2==0
    idx=torch.arange(h*w,device=device).reshape(h,w)
    block=idx.reshape(h//2,2,w//2,2).permute(0,2,1,3).reshape(-1)
    return h,w,block,torch.argsort(block)

def group_mean(z,m):
    # Matrix aggregation is deterministic on NPU and matches the exact n/m bins.
    n=len(z);g=torch.arange(n,device=z.device)*m//n
    p=(torch.arange(m,device=z.device)[:,None]==g[None]).to(z.dtype)
    return p@z/p.sum(-1,keepdim=True),g,p

class ConvexGenerator(nn.Module):
    def __init__(self,kind,hidden=64):
        super().__init__();assert kind in NAMES;self.kind=kind;d=hidden
        self.norm_x=nn.LayerNorm(2048,elementwise_affine=False)
        self.norm_h=nn.LayerNorm(768,elementwise_affine=False)
        self.input=nn.Linear(2816,d);self.qproj=nn.Linear(2048,d)
        if kind in ['local_mlp','question_gate','question_cross','window_attention','hard_local_selection','monotone_density']:
            self.head=mlp(d)
        elif kind in ['spatial_cnn','dilated_cnn']:
            self.conv=nn.Sequential(nn.Conv2d(d,d,3,padding=1 if kind=='spatial_cnn' else 2,dilation=1 if kind=='spatial_cnn' else 2,groups=d),nn.GELU(),nn.Conv2d(d,d,1))
            if kind=='dilated_cnn':self.conv2=nn.Conv2d(d,d,3,padding=4,dilation=4,groups=d)
            self.head=mlp(d)
        elif kind in ['anchor_resampler','balanced_transport','soft_kmeans','local_global_mixture','multihead_transport']:
            self.key=nn.Linear(d,d,bias=False);self.query=nn.Linear(d,d,bias=False)
            nn.init.normal_(self.key.weight,std=.005);nn.init.normal_(self.query.weight,std=.005)
            if kind=='local_global_mixture':
                self.mix=nn.Linear(d,1);nn.init.zeros_(self.mix.weight);nn.init.constant_(self.mix.bias,-5)
            if kind=='multihead_transport':self.head_mix=nn.Parameter(torch.zeros(4))
        elif kind=='local_medoid':self.metric=nn.Parameter(torch.full((d,),-2.))
        elif kind in ['novelty_gate','highpass_gate']:self.head=mlp(2*d)
        elif kind=='raw_detail':self.detail=nn.Linear(768,d);self.head=mlp(2*d)
        elif kind=='channel_uncertainty':self.head=mlp(d+3)
        elif kind in ['pyramid_context','axial_context']:self.head=mlp(3*d)
        elif kind=='hierarchical_merge':self.head=mlp(d);self.head2=mlp(d)
        elif kind=='spatial_graph':self.message=nn.Linear(2*d,d);self.head=mlp(d)
        elif kind=='question_moe':self.experts=nn.ModuleList([mlp(d),mlp(2*d),mlp(2*d)]);self.router=nn.Linear(d,3)
        elif kind=='prototype_memory':self.prototypes=nn.Parameter(torch.randn(32,d)*.02);self.head=mlp(2*d)
        if kind in ['window_attention','question_cross']:
            self.attn=nn.MultiheadAttention(d,4,dropout=0,batch_first=True)

    @staticmethod
    def local(score,base):
        return (score.reshape(1,-1).expand_as(base).masked_fill(base==0,-1e4)).softmax(-1)

    def forward(self,x,grid,keep_ratio=.1,raw=None,question=None):
        assert raw is not None and question is not None
        x=x.float();raw=raw.float();n=len(x);m=max(1,min(n,round(n*keep_ratio)))
        z=F.gelu(self.input(torch.cat([self.norm_x(x),self.norm_h(raw)],-1)))
        means,g,base=group_mean(z,m);center=means[g];k=self.kind
        q=F.gelu(self.qproj(question.float().mean(0,keepdim=True)))
        h,w,block,inverse=coordinates(grid,x.device)
        def image(v):return v[inverse].reshape(h,w,-1).permute(2,0,1)[None]
        def sequence(v):return v[0].permute(1,2,0).reshape(n,-1)[block]
        def low(v):
            im=image(v);return sequence(F.avg_pool2d(im,3,stride=1,padding=1,count_include_pad=False))
        if k=='local_mlp':score=self.head(z)
        elif k in ['spatial_cnn','dilated_cnn']:
            im=self.conv(image(z));im=self.conv2(F.gelu(im)) if k=='dilated_cnn' else im
            score=self.head(sequence(im))
        elif k=='window_attention':
            padded=F.pad(z,(0,0,0,(-n)%32)).reshape(-1,32,z.shape[-1])
            mask=(torch.arange(padded.shape[0]*32,device=z.device).reshape(-1,32)>=n)
            result,_=self.attn(padded,padded,padded,key_padding_mask=mask,need_weights=False)
            score=self.head(z+result.reshape(-1,z.shape[-1])[:n])
        elif k=='question_gate':score=self.head(z*q)
        elif k=='question_cross':
            words=F.gelu(self.qproj(question.float()[:64]))[None]
            result,_=self.attn(z[None],words,words,need_weights=False);score=self.head(z+result[0])
        elif k in ['anchor_resampler','balanced_transport','soft_kmeans','local_global_mixture','multihead_transport']:
            keys=self.key(z);queries=self.query(means)
            dist=(torch.arange(m,device=x.device)[:,None]-g[None]).abs().float()
            prior=-8*dist
            if k=='soft_kmeans':logits=-(queries.square().sum(-1)[:,None]+keys.square().sum(-1)[None]-2*queries@keys.T)/8
            elif k=='multihead_transport':
                ps=[]
                for head in range(4):ps.append((queries[:,head*16:(head+1)*16]@keys[:,head*16:(head+1)*16].T/4+prior).softmax(-1))
                p=sum(weight*pi for weight,pi in zip(self.head_mix.softmax(0),ps))
                return p@x,p
            else:logits=queries@keys.T/8
            if k=='balanced_transport':
                logp=logits+prior
                for _ in range(6):
                    logp=logp-torch.logsumexp(logp,dim=1,keepdim=True)
                    logp=logp-torch.logsumexp(logp,dim=0,keepdim=True)+math.log(m/n)
                p=logp.softmax(-1)
            else:p=(logits+prior).softmax(-1)
            if k=='local_global_mixture':
                # Global attention component deliberately small at initialization.
                global_p=logits.softmax(-1);mix=self.mix(means).sigmoid()
                p=(1-mix)*base/base.sum(-1,keepdim=True)+mix*global_p
            return p@x,p
        elif k=='local_medoid':score=-((z-center).square()*self.metric.sigmoid()).mean(-1,keepdim=True)
        elif k=='novelty_gate':score=self.head(torch.cat([z-center,(z-center).abs()],-1))
        elif k=='raw_detail':
            rm,rg,_=group_mean(raw,m);detail=F.gelu(self.detail(raw-rm[rg]));score=self.head(torch.cat([z,detail],-1))
        elif k=='channel_uncertainty':
            logp=z.log_softmax(-1);entropy=-(logp.exp()*logp).sum(-1,keepdim=True)
            score=self.head(torch.cat([z,entropy,z.abs().mean(-1,keepdim=True),z.var(-1,keepdim=True)],-1))
        elif k=='pyramid_context':
            contexts=[]
            for size in [4,16]:
                avg,group,_=group_mean(z,max(1,n//size));contexts.append(avg[group])
            score=self.head(torch.cat([z,*contexts],-1))
        elif k=='hard_local_selection':
            score=self.head(z).squeeze(-1)
            # Tie-breaking: choose token nearest the middle of each source bin.
            target=(g.float()+.5)*n/m-.5
            score=score-1e-4*(torch.arange(n,device=x.device)-target).abs()
            soft=self.local(score,base);idx=soft.argmax(-1)
            hard=F.one_hot(idx,n).to(soft.dtype);p=hard+soft-soft.detach() if self.training else hard
            return p@x,p
        elif k=='monotone_density':
            density=F.softplus(self.head(z).squeeze(-1))+1e-4
            cdf=(density.cumsum(0)-density/2)/density.sum()
            targets=(torch.arange(m,device=x.device)+.5)/m
            # Smooth local kernel retains usable derivatives to density/centers.
            logits=-((cdf[None]-targets[:,None])*m).square()*4
            p=logits.softmax(-1);return p@x,p
        elif k=='hierarchical_merge':
            middle=min(n,2*m);mid,gg,b1=group_mean(z,middle)
            p1=self.local(self.head(z),b1)
            _,_,b2=group_mean(mid,m);p2=self.local(self.head2(p1@z),b2)
            p=p2@p1;return p@x,p
        elif k=='axial_context':
            im=image(z);rows=sequence(im.mean(3,keepdim=True).expand_as(im));cols=sequence(im.mean(2,keepdim=True).expand_as(im))
            score=self.head(torch.cat([z,rows,cols],-1))
        elif k=='highpass_gate':score=self.head(torch.cat([z,z-low(z)],-1))
        elif k=='spatial_graph':
            im=image(z);pad=F.pad(im,(1,1,1,1),mode='replicate')
            nei=(pad[:,:,1:-1,:-2]+pad[:,:,1:-1,2:]+pad[:,:,:-2,1:-1]+pad[:,:,2:,1:-1])/4
            msg=F.gelu(self.message(torch.cat([z,sequence(nei)],-1)))
            score=self.head(z+msg)
        elif k=='question_moe':
            scores=[self.experts[0](z),self.experts[1](torch.cat([z,z-center],-1)),self.experts[2](torch.cat([z,low(z)],-1))]
            ps=[self.local(score,base) for score in scores]
            p=sum(weight*pi for weight,pi in zip(self.router(q)[0].softmax(-1),ps));return p@x,p
        elif k=='prototype_memory':
            match=(z@self.prototypes.T/8).softmax(-1);context=match@self.prototypes
            score=self.head(torch.cat([z,context],-1))
        else:raise ValueError(k)
        p=self.local(score,base);return p@x,p

class FixedGenerator(nn.Module):
    def __init__(self,kind):super().__init__();self.kind=kind
    def forward(self,x,grid,keep_ratio,**kwargs):
        n=len(x);m=max(1,round(n*keep_ratio));_,g,base=group_mean(x,m)
        if self.kind=='mean':p=base/base.sum(-1,keepdim=True)
        else:
            target=((torch.arange(m,device=x.device)+.5)*n/m-.5).round().long().clamp(0,n-1)
            p=F.one_hot(target,n).to(x.dtype)
        return p@x,p
