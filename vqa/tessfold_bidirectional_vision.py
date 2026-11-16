"""Full bidirectional fused vision candidate, explicit window semantics.

Installed PFA sparse_mode0 honors pre/next token bounds WHEN a mask is supplied.
The author's call supplies a mask but omits next_tokens (default0). Explicitly
allow all future keys for the noncausal vision encoder. Separate new candidate,
never rewrites old fused/eager evidence or claims floating bit equivalence.
"""
from contextlib import contextmanager
from types import MethodType
import sys,torch
import torch_npu


def forward(self,hidden_states,cu_seqlens,rotary_pos_emb=None):
    n=len(hidden_states)
    q,k,v=self.qkv(hidden_states).reshape(n,3,self.num_heads,-1).permute(1,0,2,3).unbind(0)
    rotary=sys.modules[type(self).__module__].apply_rotary_pos_emb_vision
    q=rotary(q.unsqueeze(0),rotary_pos_emb).squeeze(0)
    k=rotary(k.unsqueeze(0),rotary_pos_emb).squeeze(0)
    mask=torch.ones((1,n,n),device=q.device,dtype=torch.bool)
    for i in range(1,len(cu_seqlens)):
        mask[...,cu_seqlens[i-1]:cu_seqlens[i],cu_seqlens[i-1]:cu_seqlens[i]]=False
    q,k,v=[x.transpose(0,1).unsqueeze(0) for x in [q,k,v]]
    out=torch_npu.npu_prompt_flash_attention(q,k,v,atten_mask=mask,num_heads=self.num_heads,input_layout='BNSD',
        scale_value=self.head_dim**-.5,pre_tokens=2147483647,next_tokens=2147483647,sparse_mode=0)
    return self.proj(out.squeeze(0).transpose(0,1).reshape(n,-1))


@contextmanager
def fused_vision(tower):
    states=[]
    try:
        for block in tower.blocks:
            a=block.attn;states.append((a,'forward' in a.__dict__,a.__dict__.get('forward')))
            a.forward=MethodType(forward,a)
        yield
    finally:
        for a,had,value in reversed(states):
            if had:a.forward=value
            else:delattr(a,'forward')
