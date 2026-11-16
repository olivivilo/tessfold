"""Localize the two known prefix discrepancies; compare semantic cache cuts.

Always clone KV for independent answers. Eager is an isolated numerical backend
candidate, compared both to its own full-prompt reference and native SDPA.
"""
import copy,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,lines,save,verify,load_module,sha
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.benchmark import clean_generate,get_question
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import raw_vision_only,answer_prefix
from vqa.tessfold_fast_runtime import compress,dimensions
from vqa.benchmark_metrics import native_score


@torch.no_grad()
def main():
    verify();rank,world,_,device=init_runtime('npu',4)
    base=ROOT/'artifacts/tessfold_acceleration_20261115/prefix_probe1';base.mkdir(exist_ok=True)
    engine=OfficialUND(device=str(device));engine.model.eval().requires_grad_(False)
    wanted=[('KLC_test_1569','REF'),('DeepForm_test_1410','Tree')][rank::world]
    pool={r['id']:r for r in lines(ROOT/'artifacts/compression_supplement_20261111/same_page_rows.jsonl')}
    save(base/f'manifest.rank{rank}.json',dict(job=os.environ.get('SLURM_JOB_ID'),source_sha256=sha(__file__)))
    out=[]
    for id_,method in wanted:
        row=pool[id_];inp=engine.inputs(row)
        with raw_vision_only(engine.model.vision_tower):_,raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
        if method=='REF':projected=engine.model.vision_tower.merger(raw);ids=inp['input_ids']
        else:
            module=load_module(engine,'quadtree_question',.25).requires_grad_(False)
            c,_=compress(module,raw,inp['image_grid_thw'],get_question(engine,row),.25,shape=dimensions(inp['image_grid_thw']))
            projected=engine.model.vision_tower.merger(c)
            from scripts.eval_token_proxy import shortened_inputs
            ids=shortened_inputs(inp,len(c),engine.model.config.image_token_id)['input_ids']
        embeds=engine.model.get_input_embeddings()(ids);mask=ids==engine.model.config.image_token_id
        embeds=embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds),projected.to(embeds.dtype))
        native=clean_generate(engine,ids,projected,cap=64)
        visual_stop=int(mask[0].nonzero()[-1])+1
        semantic_stop=visual_stop
        # Include native vision_end and the separating newline, no question text.
        if int(ids[0,semantic_stop])==151653:semantic_stop+=1
        if int(ids[0,semantic_stop])==198:semantic_stop+=1
        for attention in ['sdpa','eager']:
            original=engine.model.config._attn_implementation;engine.model.config._attn_implementation=attention
            try:
                full=engine.model(input_ids=ids,inputs_embeds=embeds,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1).past_key_values
                reference=clean_generate(engine,ids,projected,cap=64)
                for cut,stop in [('last_visual',visual_stop),('complete_visual_block',semantic_stop)]:
                    prefix=ids[:,:stop];pe=embeds[:,:stop]
                    cache=engine.model(input_ids=prefix,inputs_embeds=pe,attention_mask=torch.ones_like(prefix),use_cache=True,logits_to_keep=1).past_key_values
                    layer_errors=[]
                    for a,b in zip(full.layers,cache.layers):
                        layer_errors.append(dict(k_equal=torch.equal(a.keys[:,:,:stop],b.keys),v_equal=torch.equal(a.values[:,:,:stop],b.values),
                            k_max_abs=float((a.keys[:,:,:stop].float()-b.keys.float()).abs().max()),v_max_abs=float((a.values[:,:,:stop].float()-b.values.float()).abs().max())))
                    before=cache.get_seq_length();answer=answer_prefix(engine.model,ids,cache,prefix,cap=64)
                    assert cache.get_seq_length()==before
                    record=dict(id=id_,method=method,attention=attention,cut=cut,prefix_tokens=stop,
                        equals_own_full=answer==reference,equals_native=answer==native,own_full_equals_native=reference==native,
                        native_ids=native,reference_ids=reference,cached_ids=answer,
                        native_prediction=engine.processor.decode(native,skip_special_tokens=True),cached_prediction=engine.processor.decode(answer,skip_special_tokens=True),
                        native_score=native_score(engine.processor.decode(native,skip_special_tokens=True),row),cached_score=native_score(engine.processor.decode(answer,skip_special_tokens=True),row),layer_errors=layer_errors)
                    out.append(record);print(json.dumps({k:v for k,v in record.items() if k!='layer_errors'}),flush=True)
                    del cache
                del full
            finally:engine.model.config._attn_implementation=original
    save(base/f'rows.rank{rank}.json',out);dist.barrier()
    if rank==0:save(base/'complete.json',dict(rows=sum(len(load(base/f'rows.rank{k}.json',[])) for k in range(world))))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
