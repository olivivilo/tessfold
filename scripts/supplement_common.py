"""CPU-only protocol, hashing, source freeze and cohort utilities."""
from __future__ import annotations
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/compression_supplement_20261111'
REG=BASE/'registration.json'
SOURCES=[
 'vqa/supplement_generators.py','vqa/supplement_s2prune.py',
 'scripts/supplement_common.py','scripts/compression_supplement_math.py',
 'scripts/train_compression_supplement.py','scripts/eval_compression_supplement.py',
 'scripts/smoke_compression_supplement.py','scripts/benchmark_compression_supplement.py',
 'scripts/run_compression_supplement_stage.py',
 'vqa/compression_generators_v2.py','vqa/compression_generators.py','vqa/meta_compress.py',
 'vqa/official_und.py','vqa/decoder_dependent_loss.py','vqa/benchmark_metrics.py','vqa/metrics.py',
 'scripts/train_compression_lab.py','scripts/train_compression_lab_v2.py',
 'scripts/train_meta_compress.py','scripts/train_meta_compress_strict.py',
 'scripts/eval_token_proxy.py','scripts/compression_lab_v2_io.py','scripts/compression_lab_v2_io_retry.py',
 'models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2.py',
 'models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2_vision.py',
 'models/MonkeyOCRv2-B-Und/config.json',
 'artifacts/compression_supplement_20261111/sources/S2Prune/s2prune/allocation.py',
]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,default=None):
    try:return json.loads(Path(p).read_text())
    except FileNotFoundError:return {} if default is None else default
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def lines(p):
    p=Path(p)
    if not p.exists():return []
    result=[];text=p.read_text().splitlines()
    for i,s in enumerate(text):
        if not s.strip():continue
        try:result.append(json.loads(s))
        except json.JSONDecodeError:
            if i!=len(text)-1:raise
    return result
def save_lines(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))

def benchmark_rows():
    return lines(ROOT/'artifacts/compression_lab_v2/column_tree_post_full_20261108/formal_benchmark.jsonl')

def verify():
    reg=load(REG);assert reg
    for p,d in reg['source_sha256'].items():assert sha(ROOT/p)==d,f'Changed source: {p}'
    for p,d in reg['data_sha256'].items():assert sha(ROOT/p)==d,f'Changed data: {p}'
    for r in reg['runs']:assert sha(ROOT/r['config'])==r['config_sha256']
    return reg

def legacy(kind,ratio,position='pre-merger'):
    return f'artifacts/compression_lab_v2/full/full_{kind}_{position}_r{ratio:g}'

def summary(rows):
    tasks=collections.defaultdict(list)
    for r in rows:tasks[r['task']].append(r['student_score'])
    return dict(rows=len(rows),images=len({r['document_id'] for r in rows}),
                by_task={t:100*sum(v)/len(v) for t,v in sorted(tasks.items())},
                macro=sum(100*sum(v)/len(v) for v in tasks.values())/len(tasks),tasks=len(tasks))

def teacher_records():
    result={}
    for f in sorted((ROOT/legacy('quadtree_question',.25)/'benchmark').glob('predictions.rank*.jsonl')):
        with f.open() as h:
            for line in h:
                r=json.loads(line)
                if r['variant']=='trained':result[r['id']]=r
    assert len(result)==23565
    return result
