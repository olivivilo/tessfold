"""Cold-visual-cache batch1 timing; weights resident, natural/fixed output separate."""
from __future__ import annotations
import argparse
import io
import json
import os
import platform
import random
import time
import torch
import torch.distributed as dist
from PIL import Image
from scripts.supplement_common import ROOT,BASE,verify,load,save,lines,save_lines,legacy
from scripts.train_meta_compress import init_runtime
from scripts.eval_compression_supplement import load_module,resized_inputs,question_embedding
from scripts.eval_token_proxy import cached_vision,shortened_inputs
from vqa.supplement_generators import grid_star
from vqa.supplement_s2prune import s2_generate
from vqa.official_und import OfficialUND


class FirstToken:
    def __init__(self,callback):self.prompt=True;self.callback=callback;self.delivered=False
    def put(self,token):
        if self.prompt:self.prompt=False;return
        if not self.delivered:self.callback();self.delivered=True
    def end(self):pass


def image_inputs(engine,row,image):
    from qwen_vl_utils import process_vision_info
    messages=[{'role':'user','content':[{'type':'image','image':image,'max_pixels':1003520},{'type':'text','text':row['question']}]}]
    text=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ims,vid=process_vision_info(messages)
    batch=engine.processor(text=text,images=ims,videos=vid,padding=True,return_tensors='pt')
    return {k:v.to(engine.device) for k,v in batch.items()}


@torch.no_grad()
def request(engine,module,row,cfg,image_bytes,n_original,fixed):
    torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
    image=Image.open(io.BytesIO(image_bytes)).convert('RGB')
    torch.npu.synchronize();start=time.perf_counter();first=[None]
    def first_event():
        torch.npu.synchronize();first[0]=time.perf_counter()
    marks={'request_start':start};method=cfg['method'];ratio=cfg['ratio']
    if method=='R':inp,detail=resized_inputs(engine,row,n_original,ratio,image)
    else:inp=image_inputs(engine,row,image);detail={}
    torch.npu.synchronize();marks['processor_end']=time.perf_counter()
    x,raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw']);grid=inp['image_grid_thw']
    assert len(raw)==(n_original if method!='R' else detail['actual_tokens'])
    torch.npu.synchronize();marks['vision_and_native_merger_end']=time.perf_counter()
    m=len(x)
    if method=='S':
        prediction,rt=s2_generate(engine,inp['input_ids'],x,grid,image,ratio,64 if fixed else 256,fixed,on_first=first_event)
        m=rt['suffix_visual_tokens'];generated=rt['generated_ids'];marks['compression_end']=marks['vision_and_native_merger_end']
        detail.update(s2_prefill_note='Compression/ERC occurs inside prefill, not separately timed')
    else:
        if method not in ['REF','R']:
            q=question_embedding(engine,row)
            if method=='M':_,p=module(x.float(),grid,ratio)
            elif method=='Gstar':p,_=grid_star(module,raw,x,grid,q,ratio)
            else:_,p=module(torch.zeros_like(x,dtype=torch.float32),grid,ratio,raw=raw,question=q)
            projected=(p@x.float()).to(raw.dtype) if method=='M' else engine.model.vision_tower.merger((p@raw.float()).to(raw.dtype))
            m=len(projected);si=shortened_inputs(inp,m,engine.model.config.image_token_id)
        else:projected=x;si=inp
        torch.npu.synchronize();marks['compression_end']=time.perf_counter()
        streamer=FirstToken(first_event)
        with cached_vision(engine.model.vision_tower,projected,raw):
            outputs=engine.model.generate(**si,do_sample=False,use_cache=True,max_new_tokens=64 if fixed else 256,
                                          min_new_tokens=64 if fixed else 0,streamer=streamer)
        generated=outputs[0,si['input_ids'].shape[1]:].cpu().tolist()
    torch.npu.synchronize();end=time.perf_counter();assert first[0] is not None
    return dict(id=row['id'],task=row['task'],document_id=row['document_id'],method=method,ratio=ratio,
                mode='fixed64' if fixed else 'natural_eos',tokens=n_original,target_tokens=round(n_original*ratio),actual_tokens=m,
                first_layer_visual_tokens=n_original if method=='S' else m,suffix_visual_tokens=m,
                new_tokens=len(generated),ttft_ms=1000*(first[0]-start),end_to_end_ms=1000*(end-start),
                decode_ms_per_token=1000*(end-first[0])/max(1,len(generated)-1),
                processor_ms=1000*(marks['processor_end']-start),vision_and_native_merger_ms=1000*(marks['vision_and_native_merger_end']-marks['processor_end']),
                compression_extra_ms=1000*(marks['compression_end']-marks['vision_and_native_merger_end']),
                prefill_and_first_token_ms=1000*(first[0]-marks['compression_end']),
                peak_allocated_gib=torch.npu.max_memory_allocated()/1024**3,peak_reserved_gib=torch.npu.max_memory_reserved()/1024**3,
                original_image_bytes_in_memory=True,cold_visual_cache=True,weights_resident=True,detail=detail)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();reg=verify()
    rank,world,_,device=init_runtime('npu',4);engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    out=BASE/'efficiency';out.mkdir(exist_ok=True);rows=lines(BASE/'timing_rows.jsonl')
    cfgs=load(BASE/'timing_methods.json',[]);resident=[];resident_name=[None]
    def get_module(cfg):
        if resident_name[0]!=cfg['method']:
            resident.clear();torch.npu.empty_cache();resident_name[0]=cfg['method']
            if cfg['method'] not in ['REF','R','S']:resident.append(load_module(engine,cfg))
        return resident[0] if resident else None
    save(out/f'runtime.rank{rank}.json',dict(job=os.environ.get('SLURM_JOB_ID'),device=torch.npu.get_device_name(),torch=torch.__version__,platform=platform.platform(),world=world,rank=rank,warmup_requests_per_method=10,source_sha256=reg['source_sha256']))
    path=out/f'timings.rank{rank}.jsonl';items=lines(path);save_lines(path,items)
    done={(r['id'],r['method'],r['mode'],r['repeat']) for r in items}
    mine=rows[rank::world];assert mine
    # Warmup is excluded from timing table and repeated after process restart.
    for cfg in cfgs:
        for ix in range(10):
            row=mine[ix%len(mine)];blob=(ROOT/row['image']).read_bytes()
            request(engine,get_module(cfg),row,cfg,blob,row['original_tokens'],True)
    jobs=[(row,cfg,fixed,repeat) for repeat in range(3) for row in mine for fixed in [False,True] for cfg in cfgs]
    random.Random(20261111+rank).shuffle(jobs)
    with path.open('a') as f:
        for row,cfg,fixed,repeat in jobs:
            mode='fixed64' if fixed else 'natural_eos'
            if (row['id'],cfg['method'],mode,repeat) in done:continue
            r=request(engine,get_module(cfg),row,cfg,(ROOT/row['image']).read_bytes(),row['original_tokens'],fixed);r['repeat']=repeat
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    dist.barrier()
    if rank==0:
        rr=[r for p in out.glob('timings.rank*.jsonl') for r in lines(p)]
        assert len(rr)==len(rows)*len(cfgs)*2*3
        save(out/'complete.json',dict(records=len(rr),questions=len(rows),methods=len(cfgs),repeats=3,modes=['natural_eos','fixed64']))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
