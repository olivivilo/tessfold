"""Isolated tuning variants; the original paper reproduction stays untouched."""
import math
import torch
from vqa.meta_compress import MetaCompress


class TunableMetaCompress(MetaCompress):
    def __init__(self, config):
        super().__init__(feature_dim=2048)
        self.tuning_config=dict(config)

    def forward(self,x,grid,keep_ratio=.1):
        c,p=super().forward(x,grid,keep_ratio)
        bias=self.tuning_config.get('local_bias',0.)
        scale=self.tuning_config.get('attention_scale',1.)
        if bias or scale!=1:
            m,n=p.shape
            logits=p.clamp_min(1e-30).log()*scale
            if bias:
                groups=torch.arange(n,device=x.device)*m//n
                local=groups[None,:]==torch.arange(m,device=x.device)[:,None]
                logits=logits+(~local)*(-bias)
            p=logits.softmax(-1);c=p@x
        return c,p


def learning_rate_factor(step,total_steps,config):
    if config.get('scheduler','constant')=='constant':return 1.
    warm=max(1,round(total_steps*config.get('warmup_fraction',.1)))
    if step<warm:return (step+1)/warm
    fraction=(step-warm)/max(1,total_steps-warm-1)
    minimum=config.get('min_lr_fraction',.1)
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(1.,fraction)))


def make_optimizer(module,config):
    if config['optimizer']=='sgd':
        return torch.optim.SGD(module.parameters(),lr=config['lr'],momentum=config.get('momentum',0.))
    if config['optimizer']=='adamw':
        return torch.optim.AdamW(module.parameters(),lr=config['lr'],betas=(.9,.999),
                                 weight_decay=config.get('weight_decay',.01))
    raise ValueError(config['optimizer'])
