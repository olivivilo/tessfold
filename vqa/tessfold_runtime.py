"""Zero-training inference-only TessFold refactor; historical sources immutable.

strict_dense removes dead work but keeps original float32 partition/softmax/GEMM.
segmented_fp32 is an EXPERIMENTAL numerically-close candidate, never bit-exact by
assertion. It must pass independent feature/token regression before promotion.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy,heapq,math,time
import numpy as np
import torch
from torch.nn import functional as F
from vqa.compression_generators import coordinates
from vqa.compression_generators_v2 import adaptive_rectangles


def budget(n,r):return max(1,min(n,round(n*r)))


def partition_integral_candidate(image,m,mode):
 """Proposed float64 integral/binary search; real-arithmetic equivalent only.

Different from the reference float32 cumulative sums and per-rectangle sums.
Kept as a counterexample candidate, not used by strict inference.
 """
 image=np.asarray(image,dtype=np.float32);h,w=image.shape
 integral=np.pad(image.astype(np.float64),((1,0),(1,0))).cumsum(0).cumsum(1)
 def mass(a,b,c,d):return float(integral[b,d]-integral[a,d]-integral[b,c]+integral[a,c])
 heap=[];leaves={};serial=0
 def add(rect):
  nonlocal serial
  k=serial;serial+=1;leaves[k]=rect
  a,b,c,d=rect
  if (b-a)*(d-c)>1:heapq.heappush(heap,(-mass(*rect),k))
 add((0,h,0,w))
 while len(leaves)<m:
  _,k=heapq.heappop(heap);a,b,c,d=leaves.pop(k);hh,ww=b-a,d-c
  vertical=ww>hh if mode=='tree' else True
  if ww==1:vertical=False
  if hh==1:vertical=True
  length=ww if vertical else hh;lo,hi=1,length-1;half=mass(a,b,c,d)*.5
  while lo<hi:
   mid=(lo+hi)//2;value=mass(a,b,c,c+mid) if vertical else mass(a,a+mid,c,d)
   if value>=half:hi=mid
   else:lo=mid+1
  if vertical:add((a,b,c,c+lo));add((a,b,c+lo,d))
  else:add((a,a+lo,c,d));add((a+lo,b,c,d))
 g=np.empty((h,w),dtype=np.int64)
 for k,(a,b,c,d) in enumerate(sorted(leaves.values())):g[a:b,c:d]=k
 return g.reshape(-1)


@torch.no_grad()
def score_energy(module,raw,grid,question,x=None):
 """Same active arithmetic as SpatialGenerator; drop unused coordinates/means.

Pre-merger x-slot remains literal zero passed through original2816wide linear;
we do not fold/slice the linear (which can select different GEMM kernels).
 """
 raw=raw.float();n=len(raw);device=raw.device
 if x is None:x=torch.zeros((n,2048),device=device,dtype=torch.float32)
 else:x=x.float()
 h,w,block,inverse=coordinates(grid,device)
 def image(v):return v[inverse].reshape(h,w,-1).permute(2,0,1)[None]
 def sequence(v):return v[0].permute(1,2,0).reshape(n,-1)[block]
 z=F.gelu(module.input(torch.cat([module.xnorm(x),module.hnorm(raw)],-1)))
 z=z+sequence(module.conv(image(z)))
 score=module.score(z).squeeze(-1)
 if module.kind=='quadtree_question':
  q=F.gelu(module.qproj(question.float()[:64]))
  score=score+(F.normalize(module.word_key(z),dim=-1)@F.normalize(q,dim=-1).T).max(-1).values
 low=sequence(F.avg_pool2d(image(raw[:,:64]),3,1,1,count_include_pad=False))
 detail=(raw[:,:64]-low).square().mean(-1).sqrt().detach()
 detail=detail/detail.mean().clamp_min(1e-6)
 # Preserve literal operator grouping of reference, including detail+.1 rounding.
 energy=(detail+.1)*score.detach().sigmoid().mul(2).add(.1)
 return score,energy,(h,w,block,inverse)


@torch.no_grad()
def dense_readout(raw,score,group,m,tau,hard=False):
 n=len(raw);base=torch.arange(m,device=raw.device)[:,None]==group[None]
 p=(score[None].expand(m,n)/tau).masked_fill(~base,-1e4).softmax(-1)
 rep=p.argmax(-1);order=rep.argsort(stable=True)
 if hard:return raw.index_select(0,rep[order]),rep[order],None
 p=p[order];compressed=(p@raw.float()).to(raw.dtype)
 return compressed,rep[order],p


@torch.no_grad()
def segmented_readout(raw,score,group,m,tau,bf16_accumulation=False):
 """Linear-support candidate. fp32 still has different reduction ordering.

The BF16 accumulation option reproduces the unsafe proposal for diagnosis only.
It is never the strict backend and never silently replaces dense output.
 """
 n=len(raw);logits=score.float()/tau.float()
 top=torch.full((m,),-torch.inf,device=raw.device,dtype=torch.float32)
 top.scatter_reduce_(0,group,logits,reduce='amax',include_self=True)
 ex=(logits-top[group]).exp();total=torch.zeros(m,device=raw.device,dtype=torch.float32)
 total.scatter_add_(0,group,ex);weights=ex/total[group]
 maxima=torch.full((m,),-torch.inf,device=raw.device,dtype=torch.float32)
 maxima.scatter_reduce_(0,group,weights,reduce='amax',include_self=True)
 ii=torch.arange(n,device=raw.device);candidate=torch.where(weights==maxima[group],ii,n)
 rep=torch.full((m,),n,device=raw.device,dtype=torch.long);rep.scatter_reduce_(0,group,candidate,reduce='amin',include_self=True)
 order=rep.argsort(stable=True)
 dtype=raw.dtype if bf16_accumulation else torch.float32
 out=torch.zeros((m,raw.shape[1]),device=raw.device,dtype=dtype)
 out.index_add_(0,group,raw.to(dtype)*weights.to(dtype)[:,None])
 return out[order].to(raw.dtype),rep[order],weights


@torch.no_grad()
def compress(module,raw,grid,question,ratio,backend='strict_dense',hard=False,x=None,partition='reference'):
 score,energy,shape=score_energy(module,raw,grid,question,x)
 h,w,block,inverse=shape;m=budget(len(raw),ratio);mode='col' if module.kind=='col_adaptive' else 'tree'
 if partition=='reference':g=adaptive_rectangles(energy[inverse],h,w,m,mode)
 elif partition=='integral_candidate':g=partition_integral_candidate(energy[inverse].cpu().numpy().reshape(h,w),m,mode)
 elif partition=='jit_reference':
  raise NotImplementedError('No compiled partition has passed exact reference validation; use reference')
 else:raise ValueError(partition)
 group=torch.as_tensor(g.copy(),device=raw.device)[block];tau=.25+F.softplus(module.temperature)
 if backend=='strict_dense' or hard:c,rep,p=dense_readout(raw,score,group,m,tau,hard)
 elif backend in ['segmented_fp32','segmented_bf16']:
  c,rep,p=segmented_readout(raw,score,group,m,tau,backend=='segmented_bf16')
 else:raise ValueError(backend)
 return c,dict(groups=group,representatives=rep,weights=p,score=score,energy=energy,m=m,n=len(raw),backend=backend,partition=partition,hard=hard)


@contextmanager
def raw_vision_only(tower):
 """Skip native full-length merger in pre-merger inference; use no model change.

Original tower still returns(raw,raw) through Identity; no weight/config edit.
Not thread-safe on a shared model; serving engine serializes requests.
 """
 original=tower.merger;tower.merger=torch.nn.Identity()
 try:yield
 finally:tower.merger=original


def kv_visual_bytes(cache,visual_tokens):
 total=0
 layers=cache.layers if hasattr(cache,'layers') else cache
 for layer in layers:
  if hasattr(layer,'keys'):k,v=layer.keys,layer.values
  else:k,v=layer[:2]
  assert k.ndim==v.ndim==4 and k.shape[2]>=visual_tokens
  total+=(k.shape[0]*k.shape[1]*k.shape[3]*k.element_size()+v.shape[0]*v.shape[1]*v.shape[3]*v.element_size())*visual_tokens
 return int(total)


@torch.no_grad()
def build_prefix(model,ids,projected):
 """Exact existing causal prefix up to final visual placeholder, no question.

Returns immutable-by-convention KV; clone before answering each independent QA.
 """
 positions=(ids[0]==model.config.image_token_id).nonzero().flatten();assert len(positions)==len(projected)
 stop=int(positions[-1])+1;prefix=ids[:,:stop]
 embeds=model.get_input_embeddings()(prefix).clone();mask=prefix==model.config.image_token_id;embeds[mask]=projected
 outputs=model(input_ids=prefix,inputs_embeds=embeds,attention_mask=torch.ones_like(prefix),use_cache=True,logits_to_keep=1)
 return outputs.past_key_values,prefix.clone()


@torch.no_grad()
def answer_prefix(model,full_ids,prefix_cache,prefix_ids,cap=256,fixed=False):
 assert torch.equal(full_ids[:,:len(prefix_ids[0])],prefix_ids)
 # HF generate slices the non-empty suffix at cached length and retains full
 # attention/position indexing. Prefix never includes previous question/answer.
 cache=copy.deepcopy(prefix_cache)
 outputs=model.generate(input_ids=full_ids,attention_mask=torch.ones_like(full_ids),past_key_values=cache,
  use_cache=True,do_sample=False,max_new_tokens=64 if fixed else cap,min_new_tokens=64 if fixed else 0)
 return outputs[0,full_ids.shape[1]:].cpu().tolist()
