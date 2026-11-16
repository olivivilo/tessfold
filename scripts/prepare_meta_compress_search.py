"""Freeze image-disjoint train-only pilot/search/confirmation cohorts and round1."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import hashlib
import json
from collections import Counter
from pathlib import Path
from scripts.train_meta_compress import ROOT,read_rows,sha,atomic_json

BASE=ROOT/'artifacts/meta_compress_search'
TASKS=['ChartQA','DT-VQA','DeepForm','DocVQA','InfoVQA','KLC','WTQ']


def dump_rows(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def main():
    BASE.mkdir(exist_ok=True)
    if (BASE/'protocol.json').exists():raise RuntimeError('Frozen protocol exists')
    original=[json.loads(x) for x in (ROOT/'artifacts/meta_compress/samples.jsonl').read_text().splitlines()]
    all_train=read_rows('train');original_docs={r['document_id'] for r in original}
    official_docs={r['document_id'] for split in ['val','test'] for r in read_rows(split)}
    previous_docs={json.loads(x)['document_id'] for x in (ROOT/'artifacts/meta_compress_diagnosis/samples.jsonl').read_text().splitlines()}
    used=set(previous_docs);cohorts={}
    for name,source,count in [('pilot',original,32),('search',all_train,8),('confirm',all_train,16)]:
        counts=Counter();selected=[]
        for row in sorted(source,key=lambda r:hashlib.sha256(('multimage20261028:'+name+':'+r['id']).encode()).hexdigest()):
            task=row['task'];doc=row['document_id']
            if task not in TASKS or counts[task]>=count or doc in used or doc in official_docs:continue
            if name!='pilot' and doc in original_docs:continue
            selected.append(row);used.add(doc);counts[task]+=1
            if len(selected)==count*len(TASKS):break
        assert dict(counts)=={t:count for t in TASKS},counts
        cohorts[name]=selected;dump_rows(BASE/f'{name}.jsonl',selected)
    dump_rows(BASE/'full_train.jsonl',original)
    rows=cohorts['pilot']+cohorts['search']+cohorts['confirm'];dump_rows(BASE/'cache_rows.jsonl',rows)
    default=dict(epochs=4,batch_size=8,optimizer='sgd',lr=.001,clip=1.,scheduler='warmup_cosine',
                 warmup_fraction=.1,min_lr_fraction=.1,ratio=.1,entropy_weight=1.,collapse_weight=1.,
                 local_bias=0.,attention_scale=1.,seed=20261026)
    settings=[
        dict(name='r1_sgd_original10',clip=.01,scheduler='constant'),
        dict(name='r1_sgd_cos_clip1_10'),
        dict(name='r1_sgd_cos_clip10_25',lr=.003,clip=10.,ratio=.25),
        dict(name='r1_sgd_momentum_cos50',lr=.003,momentum=.9,ratio=.5),
        dict(name='r1_adamw_cos10',optimizer='adamw',lr=1e-4),
        dict(name='r1_adamw_cos25',optimizer='adamw',lr=3e-5,ratio=.25),
        dict(name='r1_adamw_cos50',optimizer='adamw',lr=1e-5,clip=.1,ratio=.5),
        dict(name='r1_adamw_constant25',optimizer='adamw',lr=3e-5,ratio=.25,scheduler='constant'),
    ]
    configs=BASE/'configs';configs.mkdir()
    for setting in settings:
        cfg={**default,**setting,'train_rows':'artifacts/meta_compress_search/pilot.jsonl',
             'search_rows':'artifacts/meta_compress_search/search.jsonl','cohort':'pilot',
             'output':f"artifacts/meta_compress_search/runs/{setting['name']}"}
        atomic_json(configs/f"{cfg['name']}.json",cfg)
    protocol={'date_label':'2026-10-28','model':'models/MonkeyOCRv2-B-Und','backbone_frozen':True,
        'insertion':'post official merger, 2048-dimensional tokens',
        'cohorts':{n:{'questions':len(r),'images':len({x['document_id'] for x in r}),
                     'tasks':dict(Counter(x['task'] for x in r)),'sha256':sha(BASE/f'{n}.jsonl')} for n,r in cohorts.items()},
        'full_train_rows':len(original),'full_train_sha256':sha(BASE/'full_train.jsonl'),
        'selection':'search only, choose fixed epoch checkpoints; confirmation once per advanced candidate; no official benchmark for model choice',
        'screen_gate':{'search_score_gain_points':2.,'search_kl_relative_reduction':.05,
            'confirm_score_gain_points':2.,'confirm_kl_relative_reduction':.05,
            'max_mean_column_mass_fraction':.25},
        'metrics':'per-task native existing scorers; DeepForm/KLC per-field exact, not corpus F1',
        'teacher_selection':'no teacher correctness/answer length/image token filtering',
        'round1':'optimizer, clipping, scheduler, budget only; generator and all paper loss weights unchanged',
        'initial_checkpoint_sha256':sha(ROOT/'artifacts/meta_compress/initial.pt'),
        'cache':'artifacts/meta_compress_cache (official fresh UND BF16 source signature enforced)',
        'round1_configs':[s['name'] for s in settings]}
    atomic_json(BASE/'protocol.json',protocol)
    print(json.dumps(protocol,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
