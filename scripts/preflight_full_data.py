"""Validate complete corpus and sequence lengths before releasing accelerator training."""
import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/full'

def main():
    cfg=json.loads((ROOT/'configs/full.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['language_model'],local_files_only=True)
    data={s:[json.loads(l) for l in (OUT/f'{s}.jsonl').read_text().splitlines() if l] for s in ('train','val','test')}
    invalid=[]
    for split in ('train','val'):
        valid=[]
        for row in data[split]:
            answers=[a for a in row['answers'] if isinstance(a,str) and a.strip()]
            if not row['question'].strip() or not answers:
                invalid.append({'id':row['id'],'task':row['task'],'split':split,'reason':'Empty question or no nonempty source answer'})
                continue
            valid.append({**row,'answers':answers})
        data[split]=valid
    (OUT/'invalid_supervision_exclusions.json').write_text(json.dumps(invalid,indent=2))
    ids=[r['id'] for rr in data.values() for r in rr]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate question identifiers')
    docs={r['document_id']:r['image'] for rr in data.values() for r in rr}
    tokens={}; decoded={}; failed=[]
    def inspect_image(item):
        doc,path=item
        source=ROOT/path
        blob=source.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=doc:raise ValueError(f'Image hash mismatch {path}')
        with Image.open(io.BytesIO(blob)) as image:
            image=image.convert('RGB')
            h,w=smart_resize(image.height,image.width,factor=28,min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
            digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        return doc,h*w//196,digest
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,(doc,count,digest) in enumerate(pool.map(inspect_image,sorted(docs.items()))):
            tokens[doc]=count;decoded[doc]=digest
            if i%1000==0:print(json.dumps({'phase':'preflight_images','done':i+1,'total':len(docs)}),flush=True)
    # Images with identical decoded pixels but different encodings also belong to heldout.
    heldout={decoded[r['document_id']] for s in ('val','test') for r in data[s]}
    before=len(data['train'])
    data['train']=[r for r in data['train'] if decoded[r['document_id']] not in heldout]
    maxima={}
    left='<|im_start|>user\n<|vision_start|>'
    for split,rows in data.items():
        lengths=[];answers=[]
        for base in range(0,len(rows),512):
            batch=rows[base:base+512]
            qs=tokenizer([left+'<|vision_end|>'+r['question']+'<|im_end|>\n<|im_start|>assistant\n' for r in batch],add_special_tokens=False)['input_ids']
            aa=tokenizer([r['answers'][0]+tokenizer.eos_token for r in batch],add_special_tokens=False)['input_ids']
            for row,q,a in zip(batch,qs,aa):
                n=len(q)+tokens[row['document_id']]+len(a)
                lengths.append(n);answers.append(len(a))
                if len(a)>cfg['max_answer_tokens'] or n>cfg['max_sequence_length']:
                    failed.append({'id':row['id'],'split':split,'answer_tokens':len(a),'sequence_tokens':n})
        maxima[split]={'max_sequence_tokens':max(lengths),'max_answer_tokens':max(answers),'questions':len(rows)}
    result={'images_verified':len(docs),'training_records_removed_for_decoded_pixel_overlap':before-len(data['train']),
            'invalid_supervision_exclusions':dict(collections.Counter(f"{r['split']}:{r['task']}" for r in invalid)),
            'splits':maxima,'violations':failed,'visual_tokens':{'min':min(tokens.values()),'max':max(tokens.values()),'mean':sum(tokens.values())/len(tokens)}}
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2))
    if failed:raise ValueError(f'{len(failed)} length violations; inspect data/full/preflight.json')
    for split in ('train','val'):
        (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data[split]))
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest['preflight']=result
    manifest['splits']={s:{'questions':len(rr),'documents':len({r['document_id'] for r in rr}),'tasks':dict(collections.Counter(r['task'] for r in rr))} for s,rr in data.items()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'ALL_TASKS_READY').write_text('All sources prepared; hashes, images, splits and sequence lengths verified\n')
    print(json.dumps(result),flush=True)
    import subprocess,sys
    subprocess.run([sys.executable,str(ROOT/'scripts/report_full.py')],check=True)

if __name__=='__main__':main()
