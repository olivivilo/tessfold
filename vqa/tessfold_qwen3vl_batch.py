"""Microbatch training with identical perQA KL weighting and frozen8B model.

Left padding only; original MRoPE coordinates and4compressed visual branches
are preserved per example. Training arithmetic may differ in BF16 kernels, so
pilot KL/gradient and memory/speed comparison is mandatory before adoption.
"""
import torch
from torch.nn import functional as F
from vqa.tessfold_qwen3vl import NativeSpatialRouter
from vqa.tessfold_qwen3vl_stable import StableColumn

class BatchSpatialRouter(NativeSpatialRouter):
    def forward(self,x,shape=None,question=None,ratio=.25):
        if isinstance(x,list):return [super(BatchSpatialRouter,self).forward(d['x'],d['shape'],d['question'],shape) for d in x]
        return super().forward(x,shape,question,ratio)

class BatchStableColumn(StableColumn):
    def forward(self,x,shape=None,question=None,ratio=.25):
        if isinstance(x,list):return [super(BatchStableColumn,self).forward(d['x'],d['shape'],d['question'],shape) for d in x]
        return super().forward(x,shape,question,ratio)


def batch_kl(engine,examples,visuals,teachers,responses):
    sequences=[]
    for d,(p,reps),response in zip(examples,visuals,responses):
        ids,embeds,positions,mask,deep=engine.visual(d,p,reps)
        response=torch.as_tensor(response,dtype=ids.dtype,device=engine.device)
        if len(response)>1:
            embeds=torch.cat([embeds,engine.model.get_input_embeddings()(response[:-1][None])],dim=1)
            extra=positions[:,:,-1:]+torch.arange(1,len(response),device=engine.device)[None,None,:]
            positions=torch.cat([positions,extra],dim=2)
            mask=torch.cat([mask,torch.zeros((1,len(response)-1),dtype=torch.bool,device=engine.device)],dim=1)
        sequences.append((embeds,positions,mask,deep,len(response)))
    length=max(x[0].shape[1] for x in sequences);embeds=[];positions=[];masks=[];attention=[]
    for e,pos,m,deep,t in sequences:
        missing=length-e.shape[1]
        embeds.append(F.pad(e,(0,0,missing,0)));positions.append(F.pad(pos,(missing,0),value=0));masks.append(F.pad(m,(missing,0),value=False))
        attention.append(F.pad(torch.ones(e.shape[:2],dtype=torch.long,device=engine.device),(missing,0),value=0))
    deep=[torch.cat([s[3][i] for s in sequences],dim=0) for i in range(len(sequences[0][3]))]
    output=engine.model.model.language_model(inputs_embeds=torch.cat(embeds,dim=0),position_ids=torch.cat(positions,dim=1),
        attention_mask=torch.cat(attention,dim=0),visual_pos_masks=torch.cat(masks,dim=0),deepstack_visual_embeds=deep,use_cache=False)
    answer_hidden=torch.cat([output.last_hidden_state[i,-s[4]:] for i,s in enumerate(sequences)],dim=0)
    logits=engine.model.lm_head(answer_hidden).float();offset=0;losses=[]
    for teacher,s in zip(teachers,sequences):
        t=s[4];student=logits[offset:offset+t][None].log_softmax(-1);offset+=t
        losses.append((teacher.exp()*(teacher-student)).sum(-1).mean().clamp_min(0))
    return losses
