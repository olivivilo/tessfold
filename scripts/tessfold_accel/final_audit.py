"""Fail-closed final audit: frozen signatures, full coverage, dev-only selection."""
import collections,json,math
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
from scripts.tessfold_accel.system_common import verify
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'
STAGES=['system_exact','interleaved','formal_equivalence','prefix_formal','multi_deployment','batched_serving','qwen_formal','qwen_timing','qwen_memory','hard_formal','bidirectional_formal','bidirectional_timing','bidir_interleaved']


def strict_rows(directory,pattern='rows.rank*.jsonl'):
    rows=[]
    for path in sorted(directory.glob(pattern)):
        data=path.read_bytes();assert not data or data.endswith(b'\n'),f'Uncommitted tail: {path}'
        rows.extend(json.loads(line) for line in data.splitlines() if line.strip())
    return rows


def main():
    training_audit=load(BASE/'full_training_audit.json');assert training_audit['all_passed']
    assert training_audit['source_sha256']==sha(ROOT/'scripts/tessfold_accel/training_audit_npu.py')
    verify();assert all((BASE/s/'complete.json').exists() for s in STAGES)
    from scripts.tessfold_accel.system_report import main as system_audit
    system_audit()
    formal=strict_rows(BASE/'formal_equivalence');assert len(formal)==94260
    assert all(r['p_exact'] and r['feature_exact'] and r['generation_exact'] for r in formal)
    checked_sources=0
    for label in ['formal_equivalence','prefix_formal','interleaved','multi_deployment','qwen_formal','qwen_timing','qwen_memory','hard_formal','bidirectional_formal','bidirectional_timing','bidir_interleaved']:
        for p in (BASE/label).glob('manifest.rank*.json'):
            m=load(p)
            sources=m.get('source_sha256',m.get('sources',{}))
            if isinstance(sources,str):sources={f'scripts/tessfold_accel/{label}.py':sources}
            for source,digest in sources.items():
                assert sha(ROOT/source)==digest,f'{label} source changed: {source}';checked_sources+=1
    for label in ['bidirectional_formal','qwen_formal']:
        pr=load(BASE/label/'parallel_registration.json')
        if not pr:continue
        assert sha(ROOT/'scripts/tessfold_accel/inference_parallel.py')==pr['driver_sha256']
        assert sha(ROOT/pr['frozen_evaluator'])==pr['frozen_evaluator_sha256']
        archive=ROOT/pr['archive']
        for name,digest in pr['archived_file_sha256'].items():assert sha(archive/name)==digest
        current={(r['id'],r['model']):r for r in strict_rows(BASE/label)}
        from scripts.tessfold_accel.live_io import live_lines
        previous=[r for p in archive.glob('rows.rank*.jsonl') for r in live_lines(p)]
        assert len(previous)==pr['committed_records_retained']
        assert all(current[(r['id'],r['model'])]==r for r in previous),'Parallelcontinuation changedcommittedrecord'
        metadata=[load(BASE/label/f'parallel.rank{k}.json') for k in range(pr['logical_world'])]
        assert all(m['driver_sha256']==pr['driver_sha256'] and m['logical_rank']==k for k,m in enumerate(metadata))
        assert all((BASE/label/f'parallel.barrier{phase}.rank{k}.json').exists() for phase in [1,2] for k in range(pr['logical_world']))
    train=[];registry=load(BASE/'qwen_selected_runs.json');assert not registry['formal_used_for_selection']
    assert len(registry['runs'])==4
    for meta in registry['runs']:
        out=ROOT/meta['full_run'];manifest=load(out/'manifest.json');completed=load(out/'complete.json');initial=load(out/'initial.summary.json')
        assert manifest['backbone_trainable']==completed['backbone_trainable']==0
        for source,digest in manifest['source_sha256'].items():assert sha(ROOT/source)==digest
        for field,digest in manifest['data_sha256'].items():
            config=load(ROOT/'artifacts/compression_lab_v2/configs/full_quadtree_question_pre-merger_r0.5.json')
            assert sha(ROOT/config[field])==digest
        logs=strict_rows(out,'training.jsonl');steps_per_epoch=math.ceil(manifest['train_questions']/manifest['batch']);total=steps_per_epoch*manifest['epochs']
        assert [r['step'] for r in logs]==list(range(1,total+1))
        warm=max(1,round(total*.1))
        for r in logs:
            step=r['step']-1
            factor=(step+1)/warm if step<warm else .1+.9*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2
            assert math.isfinite(r['kl']) and r['kl']>=0 and math.isfinite(r['gradient_norm'])
            assert abs(r['lr']-manifest['lr']*factor)<1e-12
            assert abs(r['clip_multiplier']-min(1.,5./(r['gradient_norm']+1e-6)))<1e-9
        candidates=[load(out/f'epoch{i}.summary.json') for i in range(1,manifest['epochs']+1)]
        eligible=[r for r in candidates if r['kl']<=initial['kl']*.95 and (r['macro']>=initial['macro']+.25 or (initial['macro']>=initial['teacher']['macro']-.25 and r['macro']>=initial['teacher']['macro']-.25))]
        best=max(eligible,key=lambda r:(r['macro'],-r['kl'])) if eligible else None
        assert best==completed['best']
        assert meta['stage']==(best['stage'] if best else 'initial')
        assert sha(ROOT/meta['checkpoint'])==meta['checkpoint_sha256']
        train.append(dict(model=meta['id'],steps=total,epochs=manifest['epochs'],dev_selected_stage=meta['stage'],
            initial_dev_macro=initial['macro'],selected_dev_macro=(best or initial)['macro'],clip_fraction=sum(r['clip_multiplier']<1 for r in logs)/len(logs)))
    hard=strict_rows(BASE/'hard_formal');assert len(hard)==23565
    hard_index={r['id']:r for r in hard};assert len(hard_index)==23565
    measured=strict_rows(BASE/'system_exact/single','results.rank*.jsonl')
    hard_natural=0
    for r in measured:
        if r['method']=='T_HARD' and r['ratio']==.25 and r['mode']=='natural_eos':
            assert r['generated_ids']==hard_index[r['id']]['generated_ids'],'Hardformal differsfromactualtiming'
            hard_natural+=1
    assert hard_natural==360
    from scripts.tessfold_accel.qwen_control_stats import control_rows,paired_results
    subset,_=control_rows();rows=strict_rows(BASE/'qwen_formal');wanted={r['id'] for r in subset}
    assert len(rows)==127441
    formalqa=strict_rows(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108','formal_benchmark.jsonl')
    ids={r['id'] for r in formalqa};assert len(ids)==23565
    for model in ['REF']+[r['id'] for r in registry['runs']]:
        rr=[r for r in rows if r['model']==model];assert len(rr)==23565 and {r['id'] for r in rr}==ids
    for meta in registry['runs']:
        rr=[r for r in rows if r['model']==meta['id']+'_initial'];assert len(rr)==2404 and {r['id'] for r in rr}==wanted
    assert paired_results(rows,registry['runs'])==load(BASE/'qwen_formal/complete.json')['initial_control_results']
    assert load(BASE/'qwen_timing/statistics.json')['formal_natural_token_matches']==1350
    assert load(BASE/'qwen_memory/summary.json')['all_timing_tokens_equal']
    assert load(BASE/'bidirectional_statistics.json')['formal_natural_token_equal_records']==1800
    from vqa.benchmark_metrics import native_score
    bidir=strict_rows(BASE/'bidirectional_formal');assert len(bidir)==117825
    by_model=collections.defaultdict(set)
    for r in bidir:
        by_model[r['model']].add(r['id'])
        assert abs(r['score']-native_score(r['prediction'],r))<1e-10
    assert len(by_model)==5 and all(value==ids for value in by_model.values())
    prefixes=strict_rows(BASE/'prefix_formal');assert len(prefixes)==47130
    assert {(r['id'],r['method']) for r in prefixes}=={(i,m) for i in ids for m in ['REF','Column']}
    assert all(r['cache_immutable'] for r in prefixes)
    for r in prefixes:
        assert abs(r['cached_score']-native_score(r['cached_prediction'],r))<1e-10
        assert abs(r['reference_score']-native_score(r['reference_prediction'],r))<1e-10
    semantics=load(BASE/'attention_semantics.json')
    assert len(semantics['rows'])==8
    assert semantics['source_sha256']==sha(ROOT/'scripts/tessfold_accel/attention_semantics.py')
    dataset=load(BASE/'dataset_audit.json')
    # Detailed split-intersection audit is kept verbatim with signed files.
    assert dataset['splits']['train']['questions']==20000
    assert all(value==0 for value in dataset['document_overlaps'].values())
    for info in dataset['splits'].values():assert sha(ROOT/info['path'])==info['sha256']
    result=dict(time=now(),all_checks_passed=True,stages=STAGES,checked_source_references=checked_sources,
        hard_timing_natural_equal_records=hard_natural,full_training=train,formal_records=127441,exact_und_records=94260,initial_control_qa=2404,
        source_sha256=sha(__file__),stage_complete_sha256={s:sha(BASE/s/'complete.json') for s in STAGES},
        scopes='Frozen signatures and sources, source-enforced all-backbone-grad-None each training update, exact sequential optimizer logs and LR/clip arithmetic, original dev-only gates, primary/control full coverage, independent native8B timing and memory token/trace gates. Fixed test data never used for checkpoint selection.')
    save(BASE/'final_audit.json',result);print(json.dumps(result))

if __name__=='__main__':main()
