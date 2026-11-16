"""Register fresh image-disjoint multi-question cohorts before any new result."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/meta_compress_strict'
TASKS=['ChartQA','DT-VQA','DeepForm','DocVQA','InfoVQA','KLC','WTQ']

def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    if (BASE/'protocol.json').exists():
        print('Existing frozen protocol retained'); return
    forbidden=set()
    for path in [ROOT/'data/full/val.jsonl',ROOT/'data/full/test.jsonl',
                 ROOT/'artifacts/meta_compress_search/search.jsonl',
                 ROOT/'artifacts/meta_compress_search/confirm.jsonl',
                 ROOT/'artifacts/meta_compress_diagnosis/samples.jsonl']:
        forbidden.update(r['document_id'] for r in read(path))
    grouped=defaultdict(list)
    for r in read(ROOT/'data/full/train.jsonl'):
        if r['document_id'] not in forbidden:grouped[(r['task'],r['document_id'])].append(r)
    used=set(); metadata={}
    # Confirmation images have never appeared in the old 20k/pilot training.
    old_docs={r['document_id'] for r in read(ROOT/'artifacts/meta_compress/samples.jsonl')}
    specs=[('confirm',32,2),('validation',16,2),('train',96,3)]
    for name,nimages,nquestions in specs:
        selected=[]; imagecounts=Counter()
        keys=sorted(grouped,key=lambda k:hashlib.sha256(f'strict20261101:{name}:{k}'.encode()).hexdigest())
        for task,doc in keys:
            if task not in TASKS or imagecounts[task]>=nimages or doc in used:continue
            if name!='train' and doc in old_docs:continue
            # Do not filter on teacher correctness, token count or gold text.
            rr=sorted(grouped[(task,doc)],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
            selected.extend(rr[:nquestions]); used.add(doc); imagecounts[task]+=1
        assert dict(imagecounts)=={t:nimages for t in TASKS}, imagecounts
        path=BASE/f'{name}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        metadata[name]={'images':sum(imagecounts.values()),'questions':len(selected),
                        'task_images':dict(imagecounts),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if name=='train':
            pilotdocs=set(); counts=Counter();pilot=[]
            for r in selected:
                if r['document_id'] not in pilotdocs and counts[r['task']]<16:
                    pilotdocs.add(r['document_id']);counts[r['task']]+=1
                if r['document_id'] in pilotdocs:pilot.append(r)
            (BASE/'pilot.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pilot))
            audit=[];counts=Counter();auditdocs=set()
            for r in pilot:
                if r['document_id'] not in auditdocs and counts[r['task']]<4:
                    auditdocs.add(r['document_id']);counts[r['task']]+=1
                if r['document_id'] in auditdocs:audit.append(r)
            (BASE/'train_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    protocol={'model':'official MonkeyOCRv2-B-Und, all backbone weights frozen',
        'architecture':'unmodified vqa.meta_compress.MetaCompress; supplemental fractional max kernel3',
        'loss':'unmodified meta_losses: full-vocab teacher-prefix KL + mean row entropy + max column mass, all coefficients 1',
        'allowed_changes':'parameter initialization, pre/post merger insertion, optimizer/scheduler/clipping, training coverage, fixed token budget',
        'forbidden_changes':['local logit bias','logit temperature multiplier','KL-only','regularizer reweighting','question-conditioned generator'],
        'cohorts':metadata,'teacher_max_new_tokens':64,'student_max_new_tokens':64,
        'cache':'fresh device-specific CPU/NPU cache, not mixed old CPU teacher logits with NPU students',
        'stability_gate':{'last_epochs':3,'train_and_validation_total_relative_drop':.03,
            'train_and_validation_KL_relative_drop':.05,'maximum_last_epoch_total_rebound_fraction':.02,
            'minimum_validation_gain_points':2.,'minimum_last3_validation_gain_points':1.,
            'confirmation_gain_cluster_bootstrap_lower_bound':0.,'repeat_seeds':2},
        'selection':'validation only; confirm at most once for the selected family and two predeclared seeds; no benchmark selection',
        'reporting':'Task native macro, field exact for DeepForm/KLC, not official corpus F1. Bootstrap clusters by image.',
        'screen':'112 images, up to 3 questions/image, 4 epochs; expand promising full-loss candidates to 672 images, 12 epochs',
        'failed_screen_contingency':'Expand highest validation-score candidate with >=3% total-loss reduction; report unverified, never relabel as passing.'}
    dump(BASE/'protocol.json',protocol)
    settings=[]
    for insertion in ['pre-merger','post-merger']:
        settings.append(dict(insertion=insertion,initialization='paper',projection_gain=1.))
        settings.append(dict(insertion=insertion,initialization='paper',projection_gain=8.))
        for gain in [.5,1.,2.]:
            settings.append(dict(insertion=insertion,initialization='fourier-position',position_strength=2.,projection_gain=gain))
    defaults=dict(seed=20261101,ratio=.5,optimizer='adamw',lr=1e-4,clip=1.,weight_decay=0.,
        scheduler='warmup_cosine',warmup_fraction=.1,min_lr_fraction=.1,batch_size=8,epochs=4,
        train_rows='artifacts/meta_compress_strict/pilot.jsonl',validation_rows='artifacts/meta_compress_strict/validation.jsonl',
        audit_rows='artifacts/meta_compress_strict/train_audit.jsonl',max_new_tokens=64)
    for i,setting in enumerate(settings):
        name=f'screen{i:02d}_{setting["insertion"]}_{setting["initialization"]}_g{setting["projection_gain"]:g}'
        cfg={**defaults,**setting,'name':name,'output':f'artifacts/meta_compress_strict/runs/{name}'}
        dump(BASE/f'configs/{name}.json',cfg)
    dump(BASE/'screen_configs.json',[str(p.relative_to(ROOT)) for p in sorted((BASE/'configs').glob('screen*.json'))])
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
