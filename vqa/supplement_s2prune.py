"""S2Prune-UND adapter: official spatial allocation, original-order layer-0 ERC.

Pinned upstream code is loaded read-only. UND uses Qwen3 1-D RoPE, not the
upstream Qwen2.5-VL M-RoPE. Full layer0 KV is pruned before suffix/decode.
"""
from __future__ import annotations
import importlib.util
import math
import sys
import time
from pathlib import Path
import torch
from PIL import Image
from transformers.cache_utils import DynamicCache
from vqa.compression_generators import coordinates

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'artifacts/compression_supplement_20261111/sources/S2Prune/s2prune/allocation.py'
_allocation=None


def allocation_module():
    global _allocation
    if _allocation is None:
        spec=importlib.util.spec_from_file_location('supplement_pinned_s2_allocation',FILE)
        _allocation=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=_allocation;spec.loader.exec_module(_allocation)
    return _allocation


def select(erc,image,grid,m,ratio):
    h,w,block,inverse=coordinates(grid,erc.device)
    side=min(8 if ratio<=.25 else 9,h,w,math.isqrt(m))
    original=allocation_module().select_s2prune_tokens(erc[inverse].cpu(),image,h,w,m,side)
    idx=inverse[original.selected_indices.to(erc.device)].sort().values
    return idx,dict(coarse_side=side,region_budgets=original.region_budgets,
                    regions=original.regions,raw_complexity=original.raw_complexity.tolist(),
                    normalized_complexity=original.normalized_complexity.tolist(),
                    selected_visual_indices=idx.cpu().tolist())


@torch.no_grad()
def s2_generate(engine,prompt,x,grid,image,ratio,cap=256,fixed_length=False,on_first=None):
    model=engine.model;lm=model.model
    n=len(x);m=max(1,min(n,round(n*ratio)));origlen=prompt.shape[1]
    image_idx=(prompt[0]==model.config.image_token_id).nonzero().flatten()
    assert len(image_idx)==n
    hidden=model.get_input_embeddings()(prompt).clone();hidden[0,image_idx]=x.to(hidden.dtype)
    cache=DynamicCache(config=lm.config)
    pos=torch.arange(origlen,device=prompt.device)[None]
    rope=lm.rotary_emb(hidden,pos)
    before=hidden[0,image_idx].clone()
    hidden=lm.layers[0](hidden,attention_mask=None,position_ids=pos,past_key_values=cache,
                        use_cache=True,cache_position=pos[0],position_embeddings=rope)
    erc=(hidden[0,image_idx]-before).float().norm(dim=-1)
    if m<n:idx,trace=select(erc,image,grid,m,ratio)
    else:idx=torch.arange(n,device=x.device);trace={'selected_visual_indices':idx.cpu().tolist(),'coarse_side':None}
    text=(prompt[0]!=model.config.image_token_id).nonzero().flatten()
    keep=torch.cat([text,image_idx[idx]]).sort().values
    assert len(keep)==origlen-n+m
    hidden=hidden[:,keep];pos=pos[:,keep]
    layer0=cache.layers[0]
    layer0.keys=layer0.keys.index_select(-2,keep);layer0.values=layer0.values.index_select(-2,keep)
    length=len(keep);rope=lm.rotary_emb(hidden,pos)
    for layer in lm.layers[1:]:
        hidden=layer(hidden,attention_mask=None,position_ids=pos,past_key_values=cache,
                     use_cache=True,cache_position=torch.arange(length,device=x.device),position_embeddings=rope)
    for c in cache.layers:assert c.get_seq_length()==length
    logits=model.lm_head(lm.norm(hidden[:,-1:]))[:,-1]
    ids=[];eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    for step in range(cap):
        if fixed_length:logits[:,list(eos)]=-float('inf')
        token=logits.argmax(-1);v=int(token);ids.append(v)
        if step==0 and on_first:on_first()
        if not fixed_length and v in eos:break
        if step==cap-1:break
        emb=model.get_input_embeddings()(token[:,None])
        out=model.model(inputs_embeds=emb,position_ids=torch.tensor([[origlen+step]],device=x.device),
                         cache_position=torch.tensor([length+step],device=x.device),past_key_values=cache,use_cache=True)
        logits=model.lm_head(out.last_hidden_state[:,-1:])[:,-1]
    prediction=engine.processor.decode(ids,skip_special_tokens=True)
    trace.update(first_layer_visual_tokens=n,suffix_visual_tokens=m,original_positions_preserved=True,
                 pruned_layer0_cache=True,generated_ids=ids,hit_token_limit=len(ids)==cap,
                 kept_sequence_indices=keep.cpu().tolist())
    return prediction,trace
