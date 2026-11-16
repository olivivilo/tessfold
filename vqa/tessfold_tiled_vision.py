"""Query-tiled author eager attention; original rounding boundaries retained.

This is only a numerical candidate until all requested gates pass. Query tiles
keep the entire key axis and author BF16 matmul, divide, FP32-softmax-to-BF16,
and BF16 AV ordering. No fused attention substitution or weight changes.
"""
import math,sys
from contextlib import contextmanager
from types import MethodType
import torch


def forward(self,hidden_states,cu_seqlens,rotary_pos_emb=None):
    n=hidden_states.shape[0]
    q,k,v=self.qkv(hidden_states).reshape(n,3,self.num_heads,-1).permute(1,0,2,3).unbind(0)
    rotary=sys.modules[type(self).__module__].apply_rotary_pos_emb_vision
    q=rotary(q.unsqueeze(0),rotary_pos_emb).squeeze(0)
    k=rotary(k.unsqueeze(0),rotary_pos_emb).squeeze(0)
    lengths=torch.diff(cu_seqlens).tolist();outputs=[]
    for qi,ki,vi in zip(torch.split(q,lengths,0),torch.split(k,lengths,0),torch.split(v,lengths,0)):
        qi,ki,vi=qi.transpose(0,1),ki.transpose(0,1),vi.transpose(0,1)
        kt=ki.transpose(1,2);chunks=[]
        for start in range(0,qi.shape[1],self._tessfold_query_tile):
            weights=torch.matmul(qi[:,start:start+self._tessfold_query_tile],kt)/math.sqrt(self.head_dim)
            weights=torch.nn.functional.softmax(weights,dim=-1,dtype=torch.float32).to(q.dtype)
            chunks.append(torch.matmul(weights,vi))
        outputs.append(torch.cat(chunks,dim=1).transpose(0,1))
    return self.proj(torch.cat(outputs,dim=0).reshape(n,-1))


@contextmanager
def tiled_vision(tower,tile=512):
    assert tile>0;states=[]
    try:
        for block in tower.blocks:
            a=block.attn
            states.append((a,'forward' in a.__dict__,a.__dict__.get('forward')))
            a._tessfold_query_tile=tile;a.forward=MethodType(forward,a)
        yield
    finally:
        for a,had,value in reversed(states):
            del a._tessfold_query_tile
            if had:a.forward=value
            else:delattr(a,'forward')
