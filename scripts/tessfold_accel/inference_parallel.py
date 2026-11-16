"""Operational worker pools for frozen formal evaluators, unchanged model math.

Two independent Slurm allocations each keep their actual HCCL/Gloo groups and
NPU local ranks. Frozen evaluator sees a larger logical rank/world for disjoint
request/page assignment only. Atomic filesystem rendezvous replaces its two
metadata barriers. Original evaluator source and model methods stay unchanged.
"""
import argparse,importlib,json,os,sys,time
from pathlib import Path
import torch.distributed as dist
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stage',choices=['bidirectional_formal','qwen_formal'],required=True)
    parser.add_argument('--offset',type=int,required=True);parser.add_argument('--logical-world',type=int,required=True)
    args,rest=parser.parse_known_args();base=BASE/args.stage
    registration=load(base/'parallel_registration.json');assert registration['logical_world']==args.logical_world
    assert registration['driver_sha256']==sha(__file__)
    if args.stage=='bidirectional_formal':
        implementation=importlib.import_module('scripts.tessfold_accel.bidirectional_formal');entry=implementation.main
    else:
        implementation=importlib.import_module('scripts.tessfold_accel.qwen_formal')
        entry=importlib.import_module('scripts.tessfold_accel.qwen_eval_safe').main
    assert sha(implementation.__file__)==registration['frozen_evaluator_sha256']
    original_runtime=implementation.init_runtime;original_barrier=dist.barrier;calls=[0];logical_rank=[None]
    def runtime(*a,**kw):
        rank,world,local,device=original_runtime(*a,**kw)
        assert world==registration['pool_world']
        logical_rank[0]=args.offset+rank;assert 0<=logical_rank[0]<args.logical_world
        save(base/f'parallel.rank{logical_rank[0]}.json',dict(time=now(),logical_rank=logical_rank[0],logical_world=args.logical_world,
            actual_rank=rank,actual_world=world,local_rank=local,job=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,
            driver_sha256=sha(__file__),frozen_evaluator_sha256=sha(implementation.__file__),registration_sha256=sha(base/'parallel_registration.json'),
            scientific_change=False,scope='Only disjointassignmentandmetadataIO, samefrozenmain/forward/greedy/config/checkpoint. Archivedcommittedoutputsredistributedbyfixedlogicalassignment.'))
        return logical_rank[0],args.logical_world,local,device
    def barrier(*a,**kw):
        rank=logical_rank[0];assert rank is not None;calls[0]+=1
        phase=calls[0];marker=base/f'parallel.barrier{phase}.rank{rank}.json'
        save(marker,dict(rank=rank,phase=phase,time=now(),driver_sha256=sha(__file__)))
        start=time.monotonic()
        while True:
            ready=all((base/f'parallel.barrier{phase}.rank{k}.json').exists() for k in range(args.logical_world))
            if phase==2:ready=ready and (base/'complete.json').exists()
            if ready:break
            if time.monotonic()-start>7200:raise TimeoutError('Independentpoolrendezvous incomplete; no completionclaim')
            time.sleep(2)
        return original_barrier(*a,**kw)
    implementation.init_runtime=runtime;dist.barrier=barrier
    sys.argv=[sys.argv[0],*rest]
    try:entry()
    finally:dist.barrier=original_barrier

if __name__=='__main__':main()
