"""Semantic visual prefix and copy-on-write DynamicCache forks for greedy QA.

Only full-attention DynamicLayer is allowed: update() concatenates into a NEW
tensor. Clone layer objects, share read-only prefix storage, never share mutable
layer objects. This is not a paged KV serving implementation.
"""
import copy
import torch
from transformers.cache_utils import DynamicLayer


def snapshot(cache):
    return (int(cache.get_seq_length()),tuple((x.keys.data_ptr(),x.values.data_ptr(),x.keys._version,x.values._version) for x in cache.layers))


def fork(cache):
    assert not getattr(cache,'offloading',False)
    assert all(type(layer) is DynamicLayer for layer in cache.layers),'Only immutable-growing DynamicLayer is supported'
    result=copy.copy(cache);result.layers=[copy.copy(layer) for layer in cache.layers]
    return result


@torch.no_grad()
def build(model,ids,projected):
    mask=ids==model.config.image_token_id;positions=mask[0].nonzero().flatten();assert len(positions)==len(projected)
    stop=int(positions[-1])+1
    if int(ids[0,stop])==151653:stop+=1
    if stop<ids.shape[1] and int(ids[0,stop])==198:stop+=1
    prefix=ids[:,:stop];embeds=model.get_input_embeddings()(prefix)
    m=prefix==model.config.image_token_id
    embeds=embeds.masked_scatter(m.unsqueeze(-1).expand_as(embeds),projected.to(embeds.dtype))
    cache=model(input_ids=prefix,inputs_embeds=embeds,attention_mask=torch.ones_like(prefix),use_cache=True,logits_to_keep=1).past_key_values
    return cache,prefix.clone()


@torch.no_grad()
def answer(model,ids,cache,prefix,cap=256,fixed=False):
    assert torch.equal(ids[:,:prefix.shape[1]],prefix)
    before=snapshot(cache);local=fork(cache)
    output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=local,do_sample=False,use_cache=True,
        max_new_tokens=cap,min_new_tokens=cap if fixed else 0)
    assert snapshot(cache)==before,'Question/answer mutated the shared visual cache'
    return output[0,ids.shape[1]:].cpu().tolist()


def text_inputs(engine,row,visual_tokens):
    """Native image-expanded IDs from cached page geometry, no image reread."""
    from pathlib import Path
    from vqa.official_und import ROOT
    image=Path(row['image']);image=image if image.is_absolute() else ROOT/image
    messages=[{'role':'user','content':[{'type':'image','image':str(image),'max_pixels':1003520},{'type':'text','text':row['question']}]}]
    text=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=engine.processor.tokenizer(text,return_tensors='pt')['input_ids']
    positions=(ids[0]==engine.model.config.image_token_id).nonzero().flatten();assert len(positions)==1
    at=int(positions[0]);ids=torch.cat([ids[:,:at],ids.new_full((1,visual_tokens),engine.model.config.image_token_id),ids[:,at+1:]],dim=1)
    return ids.to(engine.device)
