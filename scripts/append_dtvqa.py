import collections
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/full'
SOURCE=OUT/'raw/shuozhang2/DT-VQA'

def main():
    if not (OUT/'FULL_CORPUS_READY').exists() or not (SOURCE/'COMPLETE').exists():raise RuntimeError('Complete base corpus/DT-VQA required')
    paths={}
    for path in SOURCE.rglob('*'):
        if path.is_file() and '/image/' in str(path):paths['image/'+str(path).split('/image/',1)[1]]=path
    def image(original):
        key=original.lstrip('./')
        # Official JSON calls this subset DTTabShot; its image.zip directory is DTTabFact.
        key=key.replace('image/DTTabShot/','image/DTTabFact/',1)
        if key not in paths:raise ValueError(f'Missing DT-VQA image {key}')
        blob=paths[key].read_bytes();doc=hashlib.sha256(blob).hexdigest();dest=OUT/'images'/f'{doc}.img'
        if not dest.exists():dest.write_bytes(blob)
        return doc,str(dest.relative_to(ROOT))
    added=collections.defaultdict(list)
    exclusions=[]
    for annotation in sorted(SOURCE.glob('test/*.json')):
        for row in json.loads(annotation.read_text()):
            doc,img=image(row['image_path'])
            added['test'].append({'id':f"DTVQA_{row['dataset_name']}_{row['id']}",'document_id':doc,'image':img,
                                  'question':row['question'],'answers':row['answers'],'task':'DT-VQA',
                                  'dataset_name':row['dataset_name'],'source_split':'test','source_repo':'shuozhang2/DT-VQA'})
    for annotation in sorted(SOURCE.glob('train/*.json')):
        if annotation.name=='DTScene_train_gpt-4v.json':continue # alternative supervision, do not duplicate
        split='val' if annotation.stem.endswith('_val') and not annotation.stem.endswith('_train_val') else 'train'
        for n,row in enumerate(json.loads(annotation.read_text())):
            conv=row['conversations']
            if conv is None:
                exclusions.append({'file':annotation.name,'row':n,'image_id':row['id'],'split':split,'reason':'Official source has conversations=null; no supervised QA'})
                continue
            doc,img=image(row['id'])
            for i in range(0,len(conv),2):
                if i+1>=len(conv):raise ValueError('Unpaired DT-VQA dialogue')
                question=re.sub(r'<img>.*?</img>\s*','',conv[i]['value']).strip()
                question=re.sub(r'\s*Answer:\s*$','',question).strip()
                added[split].append({'id':f'DTVQA_{annotation.stem}_{n}_{i//2}','document_id':doc,'image':img,
                                     'question':question,'answers':[conv[i+1]['value']],'task':'DT-VQA',
                                     'dataset_name':annotation.stem.split('_')[0],'source_split':split,
                                     'source_repo':'shuozhang2/DT-VQA'})
    data={s:[json.loads(l) for l in (OUT/f'{s}.jsonl').read_text().splitlines() if l] for s in ['train','val','test']}
    for s in data:
        data[s]=[r for r in data[s] if r['task']!='DT-VQA']+added[s]
    heldout={r['document_id'] for s in ['val','test'] for r in data[s]}
    before=len(data['train']);data['train']=[r for r in data['train'] if r['document_id'] not in heldout]
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest['dtvqa']={'train_policy':'All default train/train_val annotations, GPT4 alternative excluded; turns flattened to independent QA',
                      'official_image_directory_alias':{'DTTabShot':'DTTabFact'},
                      'unlabelled_source_rows_excluded':dict(collections.Counter(r['file'] for r in exclusions)),
                      'extra_training_records_removed_for_heldout_overlap':before-len(data['train'])}
    manifest['splits']={s:{'questions':len(rr),'documents':len({r['document_id'] for r in rr}),
                           'tasks':dict(collections.Counter(r['task'] for r in rr))} for s,rr in data.items()}
    for s,rr in data.items():
        (OUT/f'{s}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
    (OUT/'dtvqa_exclusions.json').write_text(json.dumps(exclusions,indent=2))
    (OUT/'ANNOTATIONS_READY').write_text('Seven training tasks and eight benchmark families prepared\n')
if __name__=='__main__':main()
