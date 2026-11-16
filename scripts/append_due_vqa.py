"""Add all available original Monkey DUE training/test QA, deduplicate answers."""
import collections
import hashlib
import json
import re
from pathlib import Path
import ijson

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/full'
TASKS={'DeepForm':'DeepForm','KleisterCharity':'KLC','WikiTableQuestions':'WTQ'}
LOOKUP={}

def image(path, split):
    path=path.replace('\\','/')
    relative=path[path.index('DUE_Benchmark/'):]
    if split not in LOOKUP:
        LOOKUP[split]={}
        for candidate in (OUT/'monkey_original'/split).rglob('*'):
            text=str(candidate)
            if candidate.is_file() and 'DUE_Benchmark/' in text:
                LOOKUP[split][text[text.index('DUE_Benchmark/'):]]=candidate
    matches=[LOOKUP[split][relative]] if relative in LOOKUP[split] else []
    if len(matches)!=1:raise ValueError(f'Missing/ambiguous original image {relative}: {matches}')
    blob=matches[0].read_bytes();doc=hashlib.sha256(blob).hexdigest()
    dest=OUT/'images'/f'{doc}.img'
    if not dest.exists():dest.write_bytes(blob)
    return doc,str(dest.relative_to(ROOT))

def main():
    if not (OUT/'PREPARE_COMPLETE').exists():raise RuntimeError('Prepare parquet sources first')
    for split in ('train','test'):
        if not (OUT/'monkey_original'/split/'EXTRACT_COMPLETE').exists():raise RuntimeError(f'Original {split} extraction incomplete')
    originals={}
    annotation=OUT/'raw/echo840/Monkey_Data/train_monkey.json'
    if not annotation.exists():raise RuntimeError(f'Missing {annotation}')
    with annotation.open('rb') as handle:
        for row in ijson.items(handle,'item'):
            path=row['id']
            if not path.startswith('DUE_Benchmark/'):continue
            task=path.split('/')[1]
            if task not in TASKS:continue
            conv=row['conversations'];question=re.sub(r'<img>.*?</img>\s*','',conv[0]['value']).strip()
            question=re.sub(r'\s*Answer:\s*$','',question).strip()
            answer=conv[1]['value']
            key=(task,path,question)
            originals.setdefault(key,[])
            if answer not in originals[key]:originals[key].append(answer)
    added=[]
    for (task,path,question),answers in originals.items():
        doc,img=image(path,'train')
        added.append({'id':'Monkey_'+hashlib.sha256((path+question).encode()).hexdigest(),'document_id':doc,
                      'image':img,'question':question,'answers':answers,'task':TASKS[task],'source_image':path,
                      'source_split':'train','source_repo':'echo840/Monkey_Data'})
    test=[]
    for task in TASKS:
        paths=list((OUT/'monkey_original/test').glob(f'**/test_{task}.jsonl'))
        if len(paths)!=1:raise ValueError(f'Missing test annotation {task}')
        for n,line in enumerate(paths[0].read_text().splitlines()):
            row=json.loads(line);doc,img=image(row['image'],'test')
            answers=row.get('answer',row.get('answers'))
            if isinstance(answers,str):answers=[answers]
            test.append({'id':f'{TASKS[task]}_test_{n}','document_id':doc,'image':img,'question':row['question'],
                         'answers':[str(a) for a in answers],'task':TASKS[task],'source_image':row['image'],
                         'source_split':'test','source_repo':'echo840/Monkey_Data',
                         'source_question_id':row.get('question_id',n)})
    # Idempotent: keep parquet-origin records, then add all distinct original DUE records.
    data={s:[json.loads(l) for l in (OUT/f'{s}.jsonl').read_text().splitlines() if l] for s in ['train','val','test']}
    for s in data:data[s]=[r for r in data[s] if r.get('source_repo')!='echo840/Monkey_Data']
    data['test']+=test
    heldout={r['document_id'] for s in ['val','test'] for r in data[s]}
    removed=sum(r['document_id'] in heldout for r in added)
    data['train'] += [r for r in added if r['document_id'] not in heldout]
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest.update(due_source='echo840/Monkey_Data revision dbff559fd451ae5e46e108cc4797fe0a86a6a05f',
                    due_duplicate_policy='Same source image + question grouped into all distinct reference answers',
                    due_training_questions_dropped_for_heldout_overlap=removed)
    manifest['splits']={s:{'questions':len(rows),'documents':len({r['document_id'] for r in rows}),
                           'tasks':dict(collections.Counter(r['task'] for r in rows))} for s,rows in data.items()}
    for s,rows in data.items():
        with (OUT/f'{s}.jsonl').open('w') as handle:
            for r in rows:handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'FULL_CORPUS_READY').write_text('Complete public splits + original DUE converted\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
