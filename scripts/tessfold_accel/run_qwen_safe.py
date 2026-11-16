"""Versioned shared-cache IO recovery; original training source/signatures stay.

Accept EEXIST only for bit-identical immutable feature/teacher/logp payloads;
bounded read retries preserve mmap paths. No optimizer/model changes.
"""
import argparse,json,os,sys,time
from pathlib import Path
import torch
import torch_npu
from scripts.tessfold_repair.common import ROOT,save,sha,now
from scripts.compression_lab_v2_io import identical

ORIGINAL_LOAD=torch.load

def visible_load(file,*args,**kwargs):
    path=Path(file) if isinstance(file,(str,os.PathLike)) else None
    immutable=path is not None and path.parent.name in ['features','teachers','logp'] and path.parent.parent.name=='qwen_cache'
    for attempt in range(12):
        try:return ORIGINAL_LOAD(file,*args,**kwargs)
        except FileNotFoundError:
            if not immutable or attempt==11:raise
            list(path.parent.iterdir());time.sleep(min(2.,.2*(attempt+1)))
            print(json.dumps(dict(cache_read_visibility_retry=str(path),attempt=attempt+1)),flush=True)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--stable-implementation',action='store_true');args,rest=parser.parse_known_args()
    if args.stable_implementation:from scripts.tessfold_accel import qwen_training_stable as implementation
    else:from scripts.tessfold_accel import qwen_training as implementation
    original=implementation.atomic_save;torch.load=visible_load
    original_runtime=implementation.init_runtime
    def runtime(*args,**kwargs):
        result=original_runtime(*args,**kwargs)
        # Initialize communicator while ranks are synchronized, rather than
        # lazily after a long, uneven independent cache-priming phase.
        if result[1]>1:
            import torch.distributed as dist
            probe=torch.ones(1,device=result[3]);dist.all_reduce(probe);del probe
        return result
    implementation.init_runtime=runtime
    def guarded(path,obj):
        try:return original(path,obj)
        except FileExistsError:
            path=Path(path)
            if path.parent.name not in ['features','teachers','logp'] or path.parent.parent.name!='qwen_cache':raise
            winner=visible_load(path,map_location='cpu',weights_only=True)
            if not identical(obj,winner):raise RuntimeError(f'Concurrent cache payload differs: {path}')
            temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.unlink(missing_ok=True)
            print(json.dumps(dict(cache_eexist_identical=str(path),rank=os.environ.get('RANK'))),flush=True)
    implementation.atomic_save=guarded
    base=ROOT/'artifacts/tessfold_acceleration_20261115/io_recovery';base.mkdir(exist_ok=True)
    save(base/f'job{os.environ.get("SLURM_JOB_ID")}.rank{os.environ.get("RANK")}.json',dict(time=now(),source_sha256=sha(__file__),
        training_source_sha256=sha(implementation.__file__),scope='only shared immutablecache rename/read visibility; optimizer/model math unchanged',job=os.environ.get('SLURM_JOB_ID')))
    sys.argv=[sys.argv[0],*rest];implementation.main()

if __name__=='__main__':main()
