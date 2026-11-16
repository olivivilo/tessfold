"""Fixed dev, unchanged weights: tiled eager numerical/time/memory pilot."""
import argparse,collections,json,time
from contextlib import nullcontext
import torch
import torch.distributed as dist
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now,verify
from scripts.train_meta_compress import init_runtime
from scripts.tessfold_repair.benchmark import clean_generate
from vqa.official_und import OfficialUND
from vqa.tessfold_runtime import raw_vision_only
from vqa.tessfold_tiled_vision import tiled_vision
from vqa.benchmark_metrics import native_score
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


@torch.no_grad()
def measure(engine,inp,tile):
    context=tiled_vision(engine.model.vision_tower,tile) if tile else nullcontext()
    with context,raw_vision_only(engine.model.vision_tower):
        engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
        torch.npu.synchronize();torch.npu.reset_peak_memory_stats();start=time.perf_counter()
        for _ in range(3):_,raw=engine.model.vision_tower(inp['pixel_values'],inp['image_grid_thw'])
        torch.npu.synchronize();elapsed=(time.perf_counter()-start)*1000/3
        return raw,elapsed,int(torch.npu.max_memory_allocated())


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='tiled_vision_pilot');p.add_argument('--tiles',default='256,512,1024');a=p.parse_args()
    verify();rank,world,_,device=init_runtime('npu',4);group=dist.new_group(backend='gloo')
    base=BASE/a.label;base.mkdir(exist_ok=True);tiles=[int(x) for x in a.tiles.split(',')]
    rows=lines(ROOT/'artifacts/compression_lab_v2/validation.jsonl');assert len(rows)==56
    manifest=dict(rank=rank,world=world,tiles=tiles,questions=56,sources={s:sha(ROOT/s) for s in ['vqa/tessfold_tiled_vision.py','scripts/tessfold_accel/tiled_vision_probe.py']},
        gate='Require raw features and REF generation exact on all56QA before any strict expansion; faster fullencoder timing and smaller memory are independently reported. No model or checkpoint changes.')
    mp=base/f'manifest.rank{rank}.json'
    if mp.exists():assert load(mp)==manifest
    save(mp,manifest)
    engine=OfficialUND(device=str(device));engine.model.requires_grad_(False)
    path=base/f'rows.rank{rank}.jsonl';previous=lines(path);done={(r['id'],r['tile']) for r in previous}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    with path.open('a') as f:
      for row in rows[rank::world]:
        inp=engine.inputs(row);raw,ms,mem=measure(engine,inp,0);reference=raw.cpu()
        projected=engine.model.vision_tower.merger(raw);original_ids=clean_generate(engine,inp['input_ids'],projected,cap=64)
        original_pred=engine.processor.decode(original_ids,skip_special_tokens=True);del raw,projected
        for tile in tiles:
            if (row['id'],tile) in done:continue
            raw,elapsed,peak=measure(engine,inp,tile);cpu=raw.cpu()
            same=torch.equal(cpu,reference);difference=float((cpu.float()-reference.float()).abs().max())
            cosine=float(torch.nn.functional.cosine_similarity(cpu.flatten().float(),reference.flatten().float(),dim=0))
            projected=engine.model.vision_tower.merger(raw);ids=clean_generate(engine,inp['input_ids'],projected,cap=64)
            pred=engine.processor.decode(ids,skip_special_tokens=True)
            record=dict(row,tile=tile,eager_encoder_ms=ms,tiled_encoder_ms=elapsed,eager_peak_bytes=mem,tiled_peak_bytes=peak,
                features_exact=same,max_abs_feature_difference=difference,feature_cosine=cosine,generated_ids=ids,reference_ids=original_ids,
                tokens_exact=ids==original_ids,prediction=pred,reference_prediction=original_pred,score=native_score(pred,row),reference_score=native_score(original_pred,row))
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.add((row['id'],tile))
            print(json.dumps(dict(rank=rank,id=row['id'],tile=tile,feature_equal=same,speedup=ms/elapsed)),flush=True)
            del raw,projected,cpu
    dist.barrier(group=group)
    if rank==0:
        rr=[r for p in base.glob('rows.rank*.jsonl') for r in lines(p)];assert len(rr)==len({(r['id'],r['tile']) for r in rr})==56*len(tiles)
        results={}
        for tile in tiles:
            ss=[r for r in rr if r['tile']==tile]
            results[str(tile)]=dict(questions=56,feature_mismatches=sum(not r['features_exact'] for r in ss),token_mismatches=sum(not r['tokens_exact'] for r in ss),
                median_encoder_speedup=float(np.median([r['eager_encoder_ms']/r['tiled_encoder_ms'] for r in ss])),
                max_eager_peak_gib=max(r['eager_peak_bytes'] for r in ss)/2**30,max_tiled_peak_gib=max(r['tiled_peak_bytes'] for r in ss)/2**30,
                mean_score_delta_pp=float(100*np.mean([r['score']-r['reference_score'] for r in ss])),all56_exact=all(r['features_exact'] and r['tokens_exact'] for r in ss))
        save(base/'complete.json',dict(time=now(),records=len(rr),results=results))
    dist.barrier(group=group);dist.destroy_process_group()

if __name__=='__main__':main()
