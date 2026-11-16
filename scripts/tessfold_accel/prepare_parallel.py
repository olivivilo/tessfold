"""Preserve stopped formal work, redistribute committed records without edits.

Run ONLY after oldjob has exited. Oldrawfiles/manifests stay in an immutable
archive. Every complete JSON row is retained with the exact same content;
partial EOF remains preserved in archive and is not counted as committed.
"""
import argparse,collections,json,subprocess
from pathlib import Path
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
from scripts.tessfold_accel.live_io import live_lines
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['bidirectional_formal','qwen_formal'],required=True)
    parser.add_argument('--old-job',required=True);parser.add_argument('--world',type=int,required=True);parser.add_argument('--pool-world',type=int,required=True);args=parser.parse_args()
    queue=subprocess.check_output(['squeue','-h','-j',args.old_job,'-o','%i'],text=True).strip();assert not queue,'Oldwriter still running'
    base=BASE/args.stage;assert not (base/'complete.json').exists()
    manifests=[load(p) for p in base.glob('manifest.rank*.json')];assert manifests
    for m in manifests:
        for path,digest in m.get('source_sha256',m.get('sources',{})).items():assert sha(ROOT/path)==digest
    formal=lines(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108/formal_benchmark.jsonl');assert len(formal)==23565
    rows=[r for p in sorted(base.glob('rows.rank*.jsonl')) for r in live_lines(p)]
    key=lambda r:(r['id'],r['model']);assert len(rows)==len({key(r) for r in rows})
    assignment={}
    if args.stage=='qwen_formal':assignment={r['id']:i%args.world for i,r in enumerate(formal)}
    else:
        pages=collections.defaultdict(list)
        for r in formal:pages[r['document_id']].append(r)
        counts=[0]*args.world
        for page,rr in sorted(pages.items(),key=lambda kv:(-len(kv[1]),kv[0])):
            rank=min(range(args.world),key=lambda k:counts[k]);counts[rank]+=len(rr)
            for r in rr:assignment[r['id']]=rank
    for r in rows:assert r['id'] in assignment
    archive=base.with_name(base.name+'_archive_job'+args.old_job);assert not archive.exists()
    hashes={p.name:sha(p) for p in base.iterdir() if p.is_file()}
    base.rename(archive);base.mkdir();grouped=collections.defaultdict(list)
    for r in rows:grouped[assignment[r['id']]].append(r)
    for rank in range(args.world):
        (base/f'rows.rank{rank}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in grouped[rank]))
    reread=[r for p in sorted(base.glob('rows.rank*.jsonl')) for r in lines(p)]
    original={key(r):r for r in rows};assert len(reread)==len(rows) and all(original[key(r)]==r for r in reread)
    evaluator='scripts/tessfold_accel/'+('bidirectional_formal.py' if args.stage=='bidirectional_formal' else 'qwen_formal.py')
    registration=dict(time=now(),stage=args.stage,old_job=args.old_job,archive=str(archive.relative_to(ROOT)),archived_file_sha256=hashes,
        committed_records_retained=len(rows),all_committed_record_contents_equal=True,logical_world=args.world,pool_world=args.pool_world,
        frozen_evaluator=evaluator,frozen_evaluator_sha256=sha(ROOT/evaluator),driver_sha256=sha(ROOT/'scripts/tessfold_accel/inference_parallel.py'),
        scientific_change=False,assignment='Exactsamefrozenquestion/pageassignmentalgorithm, onlylargerlogicalworld; archive+CPUrecordredistribution; freshindependentdecoderperQA.')
    save(base/'parallel_registration.json',registration);print(json.dumps(registration))

if __name__=='__main__':main()
