"""Freeze a decoder-independent protocol and acquire disjoint OCR training labels."""
import ast, hashlib, json
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def parse_ocr(s):
    outer=ast.literal_eval(s)
    data=json.loads(outer[0]) if isinstance(outer,list) else outer
    words=data.get('WORD',[])
    line_map={}
    for i,line in enumerate(data.get('LINE',[])):
        for rel in line.get('Relationships',[]):
            for key in rel.get('Ids',[]): line_map[key]=i
    output=[]
    for w in words:
        box=w['Geometry']['BoundingBox']
        if w.get('Confidence',100)<70: continue
        output.append({'text':w['Text'],'box':[box['Left'],box['Top'],box['Width'],box['Height']], 'line':line_map.get(w['Id'],-1)})
    return output

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/token_proxy/random_v1_20x'
    heldout=set()
    for split in ['val','test']:
        for r in map(json.loads,(ROOT/f'data/full/{split}.jsonl').read_text().splitlines()): heldout.add(r['document_id'])
    allowed={r['document_id']:r for r in map(json.loads,(ROOT/'data/full/train.jsonl').read_text().splitlines()) if r['task']=='InfoVQA'}
    docs={}
    paths=sorted((ROOT/'data/full/raw/lmms-lab/DocVQA/InfographicVQA').glob('train*.parquet'))
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16,columns=['image','ocr']):
            for r in batch.to_pylist():
                blob=r['image']['bytes']; doc=hashlib.sha256(blob).hexdigest()
                if doc in docs or doc in heldout or doc not in allowed: continue
                try: words=parse_ocr(r['ocr'])
                except (ValueError,KeyError,TypeError,json.JSONDecodeError): continue
                if len(words)<12: continue
                docs[doc]={'document_id':doc,'image':allowed[doc]['image'],'words':words}
        if len(docs)>=960: break
    # Image-separated split, deterministic before any features/loss/outcomes are read.
    rows=sorted(docs.values(),key=lambda r:hashlib.sha256(('evidence-head-v1'+r['document_id']).encode()).hexdigest())[:960]
    assert len(rows)>=720,len(rows)
    for i,r in enumerate(rows):r['split']='train' if i<720 else 'holdout'
    (OUT/'probe_documents.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    protocol={'source':'random_v1_20x','model':'MonkeyOCRv2-B-Und','input':'raw encoder X[*,768], C=P X; never merger/projector or Qwen representations',
      'samples_sha256':hashlib.sha256((source/'samples.jsonl').read_bytes()).hexdigest(),
      'generator_sha256':hashlib.sha256((ROOT/'vqa/token_proxy.py').read_bytes()).hexdigest(),
      'ocr_train_images':sum(r['split']=='train' for r in rows),'ocr_holdout_images':sum(r['split']=='holdout' for r in rows),
      'probe_labels':'InfoVQA training AWS WORD/LINE OCR annotations, confidence >=70; no VQA answers',
      'heldout_vqa_image_overlap':len(set(docs)&heldout),
      'projection_dim':64,'coverage_teacher_anchors':256,'sinkhorn_grid':[32,16],'sinkhorn_epsilon':0.05,'sinkhorn_iterations':150,
      'position_cost_weight':0.25,'question_uniform_floor':0.3,'subspace_dim':768,'subspace_ridge_relative':0.001,
      'evidence':'non-autoregressive OCR character/bigram/type probe; spatially addressed readout; text-number binding relations',
      'analysis':'all individual random trials; main: same question/ratio/family paired seeds, image cluster bootstrap; no seed score averaging',
      'selection':'fixed formula variants, no VQA-based fitting or model/weight selection; existing evaluation images are exploratory development replay',
      'abandoned':['projector-space ResPrune/CDPruner variants','Qwen question embedding bridge','LLM attention/logit/answer-supervised evidence head'],
      'probe_doc_sha256':hashlib.sha256((OUT/'probe_documents.jsonl').read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(protocol,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
