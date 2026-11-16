"""Move committed independent QA records between ranks; never select by scores.

Run only after stopping the controller and this budget's benchmark. Both old
manifests and prediction files are archived before changing the partition.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from scripts.compression_lab_v2_jobs import live_jobs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/compression_lab_v2'


def partition(records, ids, world):
    index = {key:i for i,key in enumerate(ids)}
    assert len(index)==len(ids) and world>0
    seen=set();parts=[[] for _ in range(world)]
    for r in records:
        key=(r['id'],r['variant'])
        assert key not in seen and r['id'] in index
        seen.add(key);parts[index[r['id']]%world].append(r)
    return parts


def read(path):
    lines=path.read_text().splitlines();result=[]
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
    return result


def digest(records):
    h=hashlib.sha256()
    for r in sorted(records,key=lambda r:(r['id'],r['variant'])):
        h.update(json.dumps(r,sort_keys=True,ensure_ascii=False).encode())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--world',type=int,required=True);a=p.parse_args()
    with (BASE/'advance.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((BASE/'stages.json').read_text());cfg=json.loads((ROOT/a.config).read_text())
        record=state['ratios'][f'{cfg["ratio"]:g}'];jobs=live_jobs()
        assert 'v2-controller' not in jobs.values(), 'Stop controller first'
        assert str(record.get('benchmark_job')) not in jobs, 'Stop benchmark first'
        out=ROOT/cfg['output']/'benchmark';list(out.iterdir())
        assert not (out/'complete.json').exists(), 'Completed evaluations are immutable'
        manifests=sorted(out.glob('manifest.rank*.json'))
        signatures=[json.loads(f.read_text()) for f in manifests]
        assert signatures and all(s==signatures[0] for s in signatures)
        oldworld=signatures[0]['world'];assert oldworld!=a.world
        assert hashlib.sha256((out.parent/'epoch4.pt').read_bytes()).hexdigest()==signatures[0]['checkpoint_sha256']
        rows=[r for r in read(ROOT/'data/full/val.jsonl') if r['task'] in ['DocVQA','InfoVQA']]+read(ROOT/'data/full/test.jsonl')
        assert len(rows)==23565
        files=sorted(out.glob('predictions.rank*.jsonl'))
        records=[r for f in files for r in read(f)]
        parts=partition(records,[r['id'] for r in rows],a.world)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive=BASE/'infrastructure'/f'benchmark_repartition_{cfg["ratio"]:g}_{stamp}'
        archive.mkdir(parents=True)
        for f in files+manifests:shutil.copy2(f,archive/f.name)
        before=digest(records)
        for rank,items in enumerate(parts):
            path=out/f'predictions.rank{rank}.jsonl';tmp=path.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items));tmp.replace(path)
        for rank in range(a.world,oldworld):(out/f'predictions.rank{rank}.jsonl').unlink(missing_ok=True)
        after=[r for f in out.glob('predictions.rank*.jsonl') for r in read(f)]
        assert digest(after)==before and len(after)==len(records)
        for f in manifests:f.unlink()
        (out.parent/'benchmark_hardware.json').write_text(json.dumps({'world':a.world})+'\n')
        audit={'old_world':oldworld,'new_world':a.world,'records_preserved':len(records),
               'canonical_records_sha256':before,'archive':str(archive.relative_to(ROOT)),
               'reason':'Hardware throughput only; no scores inspected to choose models or partitions.',
               'checkpoint_fixed':'epoch4.pt','cap':256,'source_math_unchanged':True}
        (archive/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
