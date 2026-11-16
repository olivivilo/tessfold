"""Fixed image-dependent allocation pilot; local P weights remain trainable.

Addresses an optimization mismatch: detached score-driven split changes are not
covered by the local-weight gradient. This is an explicit new router variant,
not a numerical optimization of the original allocation rule.
"""
import numpy as np
import torch
from torch.nn import functional as F
from vqa.tessfold_qwen3vl import NativeSpatialRouter
from vqa.tessfold_native_partition import partition

class StableColumn(NativeSpatialRouter):
    def forward(self,x,shape,question,ratio):
        assert self.kind=='Column';h,w=shape;n=len(x);m=max(1,min(n,round(n*ratio)));xx=x.float()
        z=F.gelu(self.input(self.norm(xx)));image=z.reshape(h,w,-1).permute(2,0,1)[None]
        z=z+self.conv(image)[0].permute(1,2,0).reshape(n,-1);score=self.score(z).squeeze(-1)
        small=xx[:,:64].reshape(h,w,64).permute(2,0,1)[None]
        low=F.avg_pool2d(small,3,1,1,count_include_pad=False)[0].permute(1,2,0).reshape(n,64)
        detail=(xx[:,:64]-low).square().mean(-1).sqrt().detach();detail=detail/detail.mean().clamp_min(1e-6)
        groups=partition((detail+.1).cpu().numpy().reshape(h,w),m,'col');groups=torch.from_numpy(groups).to(x.device)
        mask=torch.arange(m,device=x.device)[:,None]==groups[None]
        p=(score[None].expand(m,n)/(.25+F.softplus(self.temperature))).masked_fill(~mask,-1e4).softmax(-1)
        reps=p.detach().argmax(-1);order=reps.float().argsort(stable=True)
        return p[order],reps[order]
