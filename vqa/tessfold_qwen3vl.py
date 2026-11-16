"""Frozen native Qwen3-VL extension: one learned P for all four visual branches.

Insertion is AFTER native visual mergers and BEFORE the text decoder. Unlike
UND's pre-merger experiment, no native vision/merger FLOP saving is claimed.
All MRoPE positions retain the selected representative's original coordinates;
question/answer positions retain their original coordinates as well.
"""
import math,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from vqa.tessfold_native_partition import partition

ROOT=Path(__file__).resolve().parents[1]
SHORT='\nAnswer the question with only a short answer. Do not provide any explanation or Markdown formatting.'


class NativeSpatialRouter(nn.Module):
    def __init__(self,kind,hidden=96):
        super().__init__();assert kind in ['Column','Tree'];self.kind=kind
        self.norm=nn.LayerNorm(4096,elementwise_affine=False);self.input=nn.Linear(4096,hidden)
        self.conv=nn.Sequential(nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden),nn.GELU(),nn.Conv2d(hidden,hidden,1),nn.GELU())
        self.score=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.normal_(self.score[-1].weight,std=.001);nn.init.zeros_(self.score[-1].bias)
        self.qproj=nn.Linear(4096,hidden);self.key=nn.Linear(hidden,hidden,bias=False)
        self.question_gate=nn.Parameter(torch.tensor(0.));self.temperature=nn.Parameter(torch.tensor(0.))

    def forward(self,x,shape,question,ratio):
        h,w=shape;n=len(x);assert h*w==n
        m=max(1,min(n,round(n*ratio)));xx=x.float()
        z=F.gelu(self.input(self.norm(xx)));image=z.reshape(h,w,-1).permute(2,0,1)[None]
        z=z+self.conv(image)[0].permute(1,2,0).reshape(n,-1)
        score=self.score(z).squeeze(-1)
        if self.kind=='Tree':
            q=F.normalize(F.gelu(self.qproj(question.float()[:64])),dim=-1)
            score=score+self.question_gate.tanh()*(F.normalize(self.key(z),dim=-1)@q.T).max(-1).values
        small=xx[:,:64].reshape(h,w,64).permute(2,0,1)[None]
        low=F.avg_pool2d(small,3,1,1,count_include_pad=False)[0].permute(1,2,0).reshape(n,64)
        detail=(xx[:,:64]-low).square().mean(-1).sqrt().detach();detail=detail/detail.mean().clamp_min(1e-6)
        energy=(detail+.1)*score.detach().sigmoid().mul(2).add(.1)
        groups=partition(energy.detach().cpu().numpy().reshape(h,w),m,'col' if self.kind=='Column' else 'tree')
        groups=torch.from_numpy(groups).to(x.device)
        mask=torch.arange(m,device=x.device)[:,None]==groups[None]
        p=(score[None].expand(m,n)/(.25+F.softplus(self.temperature))).masked_fill(~mask,-1e4).softmax(-1)
        reps=p.detach().argmax(-1);order=reps.float().argsort(stable=True)
        return p[order],reps[order]


class NativeQwen:
    def __init__(self,device,model_path=None):
        self.device=torch.device(device);path=Path(model_path or os.environ.get('TESSFOLD_NATIVE_MODEL',ROOT/'models/Qwen3-VL-8B-Instruct'))
        self.processor=AutoProcessor.from_pretrained(str(path),local_files_only=True)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(str(path),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)

    def batch(self,row):
        image=Path(row['image']);image=image if image.is_absolute() else ROOT/image
        messages=[{'role':'user','content':[{'type':'image','image':str(image),'max_pixels':1003520},{'type':'text','text':row['question']+SHORT}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        images,videos=process_vision_info(messages,image_patch_size=16)
        batch=self.processor(text=text,images=images,videos=videos,padding=True,return_tensors='pt')
        return {k:v.to(self.device) for k,v in batch.items()}

    @torch.no_grad()
    def from_features(self,row,features):
        """Rebuild the EXACT image-expanded prompt without re-reading the page."""
        image=Path(row['image']);image=image if image.is_absolute() else ROOT/image
        messages=[{'role':'user','content':[{'type':'image','image':str(image),'max_pixels':1003520},{'type':'text','text':row['question']+SHORT}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.processor.tokenizer(text,return_tensors='pt')['input_ids']
        positions=(ids[0]==self.model.config.image_token_id).nonzero().flatten();assert len(positions)==1
        at=int(positions[0]);n=len(features['x'])
        ids=torch.cat([ids[:,:at],ids.new_full((1,n),self.model.config.image_token_id),ids[:,at+1:]],dim=1).to(self.device)
        h,w=features['shape'];grid=ids.new_tensor([[1,h*2,w*2]])
        rope,_=self.model.model.get_rope_index(ids,grid,attention_mask=torch.ones_like(ids))
        qids=self.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(self.device)
        return dict(ids=ids,positions=rope,x=features['x'].to(self.device),deep=[v.to(self.device) for v in features['deep']],shape=(h,w),question=self.model.get_input_embeddings()(qids))

    @torch.no_grad()
    def prepare(self,row):
        batch=self.batch(row)
        features,deep=self.model.get_image_features(batch['pixel_values'],batch['image_grid_thw'])
        x=torch.cat(features,dim=0)
        positions,delta=self.model.model.get_rope_index(batch['input_ids'],batch['image_grid_thw'],attention_mask=batch['attention_mask'])
        t,h,w=map(int,batch['image_grid_thw'][0].cpu().tolist());assert t==1
        question_ids=self.processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(self.device)
        question=self.model.get_input_embeddings()(question_ids)
        return dict(ids=batch['input_ids'],positions=positions,x=x,deep=deep,shape=(h//2,w//2),question=question)

    def visual(self,data,p=None,reps=None):
        ids=data['ids'];positions=data['positions'];x=data['x'];deep=data['deep']
        if p is not None:
            x=(p@x.float()).to(x.dtype);deep=[(p@v.float()).to(v.dtype) for v in deep]
            indices=(ids[0]==self.model.config.image_token_id).nonzero().flatten();lo,hi=int(indices[0]),int(indices[-1])+1
            keep=torch.cat([torch.arange(lo,device=self.device),lo+reps,torch.arange(hi,ids.shape[1],device=self.device)])
            ids=ids.index_select(1,keep);positions=positions.index_select(2,keep)
        embeds=self.model.get_input_embeddings()(ids)
        mask=ids==self.model.config.image_token_id
        embeds=embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds),x.to(embeds.dtype))
        return ids,embeds,positions,mask,deep

    def logits(self,data,response,p=None,reps=None):
        ids,embeds,positions,mask,deep=self.visual(data,p,reps)
        response=torch.as_tensor(response,dtype=ids.dtype,device=self.device)
        if len(response)>1:
            extra=self.model.get_input_embeddings()(response[:-1][None])
            embeds=torch.cat([embeds,extra],dim=1)
            pp=positions[:,:,-1:]+torch.arange(1,len(response),device=self.device)[None,None,:]
            positions=torch.cat([positions,pp],dim=2)
            mask=torch.cat([mask,torch.zeros((1,len(response)-1),dtype=torch.bool,device=self.device)],dim=1)
        out=self.model.model.language_model(inputs_embeds=embeds,position_ids=positions,attention_mask=torch.ones(embeds.shape[:2],device=self.device,dtype=torch.long),
            visual_pos_masks=mask,deepstack_visual_embeds=deep,use_cache=False)
        return self.model.lm_head(out.last_hidden_state[:,-len(response):]).float()

    @torch.no_grad()
    def greedy(self,data,p=None,reps=None,cap=64,fixed=False):
        ids,embeds,positions,mask,deep=self.visual(data,p,reps)
        length=embeds.shape[1];out=self.model.model.language_model(inputs_embeds=embeds,position_ids=positions,
            attention_mask=torch.ones((1,length),dtype=torch.long,device=self.device),use_cache=True,
            visual_pos_masks=mask,deepstack_visual_embeds=deep)
        response=[];last=positions[:,:,-1:];eos=self.model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        for step in range(cap):
            logits=self.model.lm_head(out.last_hidden_state[:,-1:])[:,0]
            if fixed and step<cap-1:logits[:,eos]=-torch.inf
            token=logits.argmax(-1);value=int(token);response.append(value)
            if not fixed and value in eos:break
            if step==cap-1:break
            last=last+1
            out=self.model.model.language_model(input_ids=token[:,None],position_ids=last,past_key_values=out.past_key_values,
                attention_mask=torch.ones((1,length+step+1),dtype=torch.long,device=self.device),use_cache=True)
        return response

    def decode(self,ids):return self.processor.decode(ids,skip_special_tokens=True)
