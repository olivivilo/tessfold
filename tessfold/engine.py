"""Adapters use the original frozen backbones and unmodified routing arithmetic."""
from __future__ import annotations

import torch


class NativeEngine:
    def __init__(self, cfg, device):
        from vqa.tessfold_qwen3vl import NativeQwen
        self.engine = NativeQwen(device, model_path=cfg['model'])
        self.model = self.engine.model
        self.device = self.engine.device

    @torch.no_grad()
    def extract(self, row):
        d = self.engine.prepare(row)
        return dict(x=d['x'].cpu(), deep=[v.cpu() for v in d['deep']], shape=d['shape'])

    def prepare(self, row, f):
        return self.engine.from_features(row, f)

    def route(self, router, data, ratio):
        return router(data['x'], data['shape'], data['question'], ratio)

    def logits(self, data, response, route=None):
        return self.engine.logits(data, response, *(route or (None,None)))

    def greedy(self, data, route=None, cap=64):
        return self.engine.greedy(data, *(route or (None,None)), cap=cap)

    def decode(self, ids):
        return self.engine.decode(ids)

    def token_count(self, data):
        return len(data['x'])


class UNDEngine:
    def __init__(self, cfg, device):
        from vqa.official_und import OfficialUND
        self.engine = OfficialUND(model_path=cfg['model'], device=str(device))
        self.model = self.engine.model.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.insertion = cfg['insertion']

    @torch.no_grad()
    def extract(self, row):
        b = self.engine.inputs(row)
        x, raw = self.model.vision_tower(b['pixel_values'], b['image_grid_thw'])
        return dict(raw=raw.cpu(), x=x.cpu(), grid=b['image_grid_thw'].cpu())

    @torch.no_grad()
    def prepare(self, row, f):
        # UND prompt rebuilding uses the same native image-token expansion.
        # The portable adapter performs it explicitly, avoiding image reprocessing.
        from pathlib import Path
        messages=[{'role':'user','content':[
            {'type':'image','image':str(Path(row['image'])),'max_pixels':1003520},
            {'type':'text','text':row['question']}]}]
        processor = self.engine.processor
        text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids = processor.tokenizer(text,return_tensors='pt')['input_ids']
        positions = (ids[0]==self.model.config.image_token_id).nonzero().flatten()
        if len(positions)!=1:
            raise ValueError('Expected one image placeholder in UND text template')
        at=int(positions[0]);n=len(f['raw'])
        ids=torch.cat([ids[:,:at],ids.new_full((1,n),self.model.config.image_token_id),ids[:,at+1:]],dim=1).to(self.device)
        qids=processor.tokenizer(row['question'],return_tensors='pt',add_special_tokens=False)['input_ids'][0,:64].to(self.device)
        return dict(ids=ids,raw=f['raw'].to(self.device),x=f['x'].to(self.device),
                    grid=f['grid'].to(self.device),question=self.model.get_input_embeddings()(qids))

    def route(self, router, d, ratio):
        if self.insertion=='pre-merger':
            dummy=d['raw'].new_zeros((len(d['raw']),2048),dtype=torch.float32)
            _,p=router(dummy,d['grid'],ratio,raw=d['raw'],question=d['question'])
            projected=self.model.vision_tower.merger((p@d['raw'].float()).to(d['raw'].dtype))
        else:
            projected,p=router(d['x'].float(),d['grid'],ratio,raw=d['raw'],question=d['question'])
            projected=projected.to(d['raw'].dtype)
        from scripts.eval_token_proxy import shortened_inputs
        ids=shortened_inputs({'input_ids':d['ids']},len(p),self.model.config.image_token_id)['input_ids']
        return p,projected,ids

    def logits(self, data, response, route=None):
        from vqa.decoder_dependent_loss import common_prefix_logits
        projected,ids=(data['x'],data['ids']) if route is None else (route[1],route[2])
        return common_prefix_logits(self.model,ids,projected[None],response)

    @torch.no_grad()
    def greedy(self, data, route=None, cap=64):
        from vqa.tessfold_generation_compat import preserve_prefill_ids
        projected,ids=(data['x'],data['ids']) if route is None else (route[1],route[2])
        embeds=self.model.get_input_embeddings()(ids).clone()
        embeds[ids==self.model.config.image_token_id]=projected
        with preserve_prefill_ids(self.model):
            outputs=self.model.generate(input_ids=ids,inputs_embeds=embeds,attention_mask=torch.ones_like(ids),
                                        do_sample=False,use_cache=True,max_new_tokens=cap)
        return outputs[0,ids.shape[1]:].cpu().tolist()

    def decode(self, ids):
        return self.engine.processor.decode(ids,skip_special_tokens=True)

    def token_count(self, data):
        return len(data['raw'])


def create(cfg, device):
    engine=(NativeEngine if cfg['backend']=='native8b' else UNDEngine)(cfg,device)
    engine.model.eval().requires_grad_(False)
    return engine


def router(cfg):
    if cfg['backend']=='native8b':
        from vqa.tessfold_qwen3vl import NativeSpatialRouter
        from vqa.tessfold_qwen3vl_stable import StableColumn
        return StableColumn('Column') if cfg['kind']=='StableColumn' else NativeSpatialRouter(cfg['kind'])
    from vqa.compression_generators_v2 import SpatialGenerator
    return SpatialGenerator('quadtree_question' if cfg['kind']=='Tree' else 'col_adaptive')
