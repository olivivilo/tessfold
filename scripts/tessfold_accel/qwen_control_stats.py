"""Matched preregistered initial-vs-selected controls; no checkpoint selection."""
import collections
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,sha
from vqa.benchmark_metrics import native_score,due_field_f1
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def control_rows():
    registration=load(BASE/'qwen_initial_control_registration.json')
    path=BASE/'qwen_initial_control_rows.jsonl'
    assert registration['data_sha256']==sha(path)
    rows=lines(path)
    assert len(rows)==len({r['id'] for r in rows})==registration['questions']==2404
    assert len({r['document_id'] for r in rows})==512
    return rows,registration


def scores(rows):
    by={};shared={}
    for task in sorted({r['task'] for r in rows}):
        rr=[r for r in rows if r['task']==task]
        shared[task]=100*sum(native_score(r['prediction'],r) for r in rr)/len(rr)
        by[task]=due_field_f1(rr)['score_percent'] if task in ['KLC','DeepForm'] else shared[task]
    return dict(questions=len(rows),native_macro=sum(by.values())/len(by),native_by_task=by,
        shared_fieldEM_macro=sum(shared.values())/len(shared),shared_fieldEM_by_task=shared)


def paired_results(records,runs):
    records=sorted(records,key=lambda r:(r['model'],r['id']))
    subset,registration=control_rows();wanted={r['id'] for r in subset};out={}
    for meta in runs:
        initial_id=meta['id']+'_initial'
        initial=[r for r in records if r['model']==initial_id]
        selected=[r for r in records if r['model']==meta['id'] and r['id'] in wanted]
        assert {r['id'] for r in initial}=={r['id'] for r in selected}==wanted
        assert len(initial)==len(selected)==2404
        a,b=scores(initial),scores(selected)
        by_task={t:b['native_by_task'][t]-a['native_by_task'][t] for t in a['native_by_task']}
        # Cluster bootstrap for shared row metrics. For native DUE corpus F1,
        # give exact matched point estimates; do not average per-QA F1.
        pages=collections.defaultdict(lambda:np.zeros(2));index={r['id']:r for r in initial}
        for r in selected:
            pages[(r['task'],r['document_id'])]+=np.array([native_score(r['prediction'],r)-native_score(index[r['id']]['prediction'],r),1.])
        rng=np.random.default_rng(20261115);boot=[]
        for task in sorted(by_task):
            arr=np.asarray([pages[key] for key in sorted(pages) if key[0]==task])
            sums=arr[rng.integers(len(arr),size=(10000,len(arr)))].sum(1)
            boot.append(100*sums[:,0]/sums[:,1])
        out[meta['id']]=dict(initial=a,selected=b,native_macro_delta=b['native_macro']-a['native_macro'],native_by_task_delta=by_task,
            shared_fieldEM_macro_delta=b['shared_fieldEM_macro']-a['shared_fieldEM_macro'],
            shared_fieldEM_task_stratified_document_bootstrap_95_ci=np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist(),
            confidence_interval_scope='Shared fieldEM macro only; native DUE-F1 macro point estimate reported separately.',
            selected_checkpoint=meta['checkpoint'],initial_checkpoint=str((ROOT/meta['full_run']/'initial.pt').relative_to(ROOT)),
            benchmark_used_for_selection=False)
    return dict(questions=2404,images=512,subset_sha256=registration['data_sha256'],comparisons=out,
        interpretation='Initial and dev-selected routers evaluated on identical fixed test subset, no feedback to training/selection. Not the full23565QA score.')
