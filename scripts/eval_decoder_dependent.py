"""Rescore exact cached random interventions with frozen official UND decoder."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import closing, nullcontext

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import numpy as np
import torch
from vqa.official_und import ROOT, OfficialUND
from vqa.token_proxy import BalancedMatrix
from vqa.decoder_dependent_loss import common_prefix_logits, prediction_kl, row_entropy
from scripts.eval_token_proxy import digest, shortened_inputs, cached_vision, generate

BASE = ROOT/'artifacts/token_proxy/random_v1_20x'
OUT = ROOT/'artifacts/decoder_dependent_loss'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compatible(saved, current):
    """Runner scheduling may change; loss, model wrapper and generator may not."""
    return all(saved[f]==current[f] for f in current if f!='scripts/eval_decoder_dependent.py')


def claimed_rows(samples, out, rank, world):
    """Claim whole images atomically; completed interventions are never replaced."""
    import random
    primary=samples[rank::world]
    primary_ids={r['id'] for r in primary}
    remainder=[r for r in samples if r['id'] not in primary_ids]
    random.Random(20261025+rank).shuffle(remainder)
    locks=out/'claims';locks.mkdir(exist_ok=True)
    for row in primary+remainder:
        if (out/'losses'/f"{row['document_id']}.json").exists():continue
        lock=locks/row['document_id']
        try:lock.mkdir()
        except FileExistsError:continue
        try:
            if not (out/'losses'/f"{row['document_id']}.json").exists():yield row
        finally:lock.rmdir()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0, help='Per-rank pilot image cap; 0 = full cohort')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--verify-generation-tokens',type=int,default=16,
                   help='Reproduce up to this many initial cached generated tokens per rank')
    p.add_argument('--device', default=None, help='cpu for a separate CPU Slurm allocation')
    p.add_argument('--work-queue',action='store_true',help='Atomically claim remaining whole images across workers')
    p.add_argument('--output', default=str(OUT))
    a = p.parse_args()
    out = Path(a.output); (out/'losses').mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    samples = [json.loads(s) for s in (BASE/'samples.jsonl').read_text().splitlines()]
    assert not (a.work_queue and a.limit), 'Queue run uses the whole fixed cohort'
    wanted = {r['id'] for r in (samples if a.work_queue else samples[rank::world][:a.limit or None])}
    teachers, trials = {}, collections.defaultdict(list)
    for path in sorted(BASE.glob('teachers.rank*.jsonl')):
        for line in path.open():
            r = json.loads(line)
            if r['id'] in wanted: teachers[r['id']] = r
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for line in path.open():
            r = json.loads(line)
            if r['sample_id'] in wanted: trials[r['sample_id']].append(r)
    code = {f: sha(ROOT/f) for f in ['vqa/decoder_dependent_loss.py','scripts/eval_decoder_dependent.py',
                                   'vqa/official_und.py','vqa/token_proxy.py']}
    archive=out/'source_snapshots';archive.mkdir(exist_ok=True)
    for f,h in code.items():
        snapshot=archive/f'{h}.py'
        if not snapshot.exists():snapshot.write_bytes((ROOT/f).read_bytes())
    manifest = {'rank':rank,'world':world,'code_sha256':code,'samples_sha256':sha(BASE/'samples.jsonl'),
                'base_protocol_sha256':sha(BASE/'protocol.json'),'model':'models/MonkeyOCRv2-B-Und',
                'adapter':None,'batch_size':a.batch_size,'limit':a.limit,'device':a.device,
                'work_queue':a.work_queue,
                'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    (out/f'run.rank{rank}.json').write_text(json.dumps(manifest, indent=2))
    if all((out/'losses'/f"{r['document_id']}.json").exists() for r in samples if r['id'] in wanted):
        return
    engine = OfficialUND(device=a.device)
    engine.model.requires_grad_(False)
    tower = engine.model.vision_tower
    start = time.monotonic(); checked = False
    queue=claimed_rows(samples,out,rank,world) if a.work_queue else iter(samples)
    with torch.inference_mode(), (closing(queue) if a.work_queue else nullcontext(queue)):
        for row in queue:
            if row['id'] not in wanted: continue
            target = out/'losses'/f"{row['document_id']}.json"
            if target.exists():
                saved=json.loads(target.read_text())
                assert compatible(saved['code_sha256'],code) and saved['sample_id']==row['id'], 'Incompatible resume'
                continue
            tick = time.monotonic(); teacher = teachers[row['id']]
            feature = torch.load(BASE/'features'/f"{row['document_id']}.pt", map_location='cpu', weights_only=True)
            raw_cpu = feature['x']; raw = raw_cpu.to(engine.device)
            assert digest(raw_cpu)==teacher['feature_sha256']==feature['sha256']
            inputs = engine.inputs(row)
            assert int((inputs['input_ids']==engine.model.config.image_token_id).sum())==len(raw_cpu)
            assert inputs['input_ids'].shape[1]==teacher['runtime']['input_tokens']
            assert inputs['image_grid_thw'][0].cpu().tolist()==feature['grid']
            projected = tower.merger(raw)
            response = teacher['runtime']['generated_ids']
            teacher_logits = common_prefix_logits(engine.model, inputs['input_ids'], projected[None], response)
            logp = teacher_logits.log_softmax(-1)
            if a.limit:
                print(json.dumps({'rank':rank,'phase':'teacher_forward','seconds':time.monotonic()-tick}),flush=True)
            audit = {'feature_sha256':feature['sha256'],'response_tokens':len(response),
                     'teacher_first_argmax':int(teacher_logits[0,0].argmax()),
                     'teacher_response_first_id':response[0],
                     'teacher_forced_greedy_agreement':float((teacher_logits[0].argmax(-1).cpu()==torch.tensor(response)).float().mean())}
            if not checked:
                # Independent generation with the exact original image and frozen encoder
                # must still match the prior experiment, before reusing its scores.
                pred, runtime = generate(engine, inputs, a.verify_generation_tokens, projected, raw)
                teacher_match=runtime['generated_ids']==response[:a.verify_generation_tokens]
                if a.device!='cpu':assert teacher_match, 'Cached teacher response no longer reproducible'
                with cached_vision(tower, projected, raw):
                    native = engine.model(**inputs, use_cache=False, logits_to_keep=1).logits.float()
                first = common_prefix_logits(engine.model, inputs['input_ids'], projected[None], response[:1])
                torch.testing.assert_close(native, first, rtol=0, atol=0)
                identity_kl = prediction_kl(logp, teacher_logits)
                assert identity_kl.abs().max()<1e-6
                audit.update({'teacher_generation_prefix_reproduced':teacher_match,'teacher_verified_tokens':len(runtime['generated_ids']),
                              'teacher_regenerated_ids':runtime['generated_ids'],
                              'native_vs_direct_embedding_max_abs':float((native-first).abs().max()),
                              'identity_KL_max_abs':float(identity_kl.abs().max())})
            del teacher_logits
            groups = collections.defaultdict(list)
            for r in trials[row['id']]: groups[int(r['matrix_diagnostics']['m'])].append(r)
            assert sum(map(len,groups.values()))==26
            results=[]
            for m, records in groups.items():
                sinputs=shortened_inputs(inputs,m,engine.model.config.image_token_id)
                for lo in range(0,len(records),a.batch_size):
                    chunk=records[lo:lo+a.batch_size]; projections=[]; metadata=[]
                    for r in chunk:
                        path=ROOT/r['matrix_file']; assert sha(path)==r['matrix_file_sha256']
                        with np.load(path) as d:
                            matrix=BalancedMatrix(torch.from_numpy(d['group']),torch.from_numpy(d['weight']),int(d['m']),float(d['floor']))
                        compressed=matrix.compress(raw_cpu.float()).to(device=engine.device,dtype=raw.dtype)
                        assert digest(compressed)==r['compressed_feature_sha256'], 'Changed compressed intervention'
                        projection=tower.merger(compressed); projections.append(projection)
                        entropy=row_entropy(matrix)
                        metadata.append({'id':r['id'],'sample_id':r['sample_id'],'task':r['task'],
                          'kind':r['kind'],'family':r['family'],'ratio':r['ratio'],'replicate':r['replicate'],
                          'matrix_sha256':r['matrix_file_sha256'],'compressed_feature_sha256':r['compressed_feature_sha256'],
                          'entropy':entropy,'entropy_normalized':entropy/np.log(matrix.n),
                          'teacher_score':r['teacher_score'],'student_score':r['student_score'],'score_drop':r['score_drop']})
                        if not checked and lo==0 and m==next(iter(groups)) and len(metadata)==1:
                            pred, runtime=generate(engine,sinputs,a.verify_generation_tokens,projection,compressed)
                            student_match=runtime['generated_ids']==r['runtime']['generated_ids'][:a.verify_generation_tokens]
                            if a.device!='cpu':assert student_match, 'Cached student response no longer reproducible'
                            audit['student_generation_prefix_reproduced']=student_match
                            audit['student_verified_tokens']=len(runtime['generated_ids'])
                            audit['student_regenerated_ids']=runtime['generated_ids']
                    logits=common_prefix_logits(engine.model,sinputs['input_ids'],torch.stack(projections),response)
                    kl=prediction_kl(logp,logits).cpu()
                    if a.limit:
                        print(json.dumps({'rank':rank,'phase':'student_batch','m':m,'batch':len(chunk),'seconds':time.monotonic()-tick}),flush=True)
                    for meta,k in zip(metadata,kl):
                        meta.update({'prediction_kl':float(k.mean()),'first_token_kl':float(k[0]),
                                     'per_token_kl':k.tolist(),'total':float(k.mean())+meta['entropy']})
                        assert min(k.tolist())>-1e-5 and np.isfinite(meta['total'])
                        results.append(meta)
                    del logits,projections,kl
            checked=True
            saved={'sample_id':row['id'],'document_id':row['document_id'],'task':row['task'],
                   'code_sha256':code,'teacher_generated_ids':response,'audit':audit,'trials':results,
                   'elapsed_seconds':time.monotonic()-tick}
            temporary=target.with_suffix(f'.rank{rank}.tmp');temporary.write_text(json.dumps(saved,ensure_ascii=False,allow_nan=False));temporary.replace(target)
            del logp,raw,projected,inputs
            print(json.dumps({'rank':rank,'sample':row['id'],'task':row['task'],'seconds':saved['elapsed_seconds'],
                              'elapsed_seconds':time.monotonic()-start,'trials':len(results)}),flush=True)
    (out/f'rank{rank}.complete.json').write_text(json.dumps({'images':len(wanted)}))


if __name__=='__main__':main()
