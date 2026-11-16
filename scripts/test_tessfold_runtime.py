"""Independent original-reference tests; not self-confirming shared partitions."""
import json,time
from pathlib import Path
import numpy as np
import torch
from vqa.compression_generators_v2 import SpatialGenerator,adaptive_rectangles
from vqa.tessfold_runtime import *

def main():
 torch.set_num_threads(2);rng=np.random.default_rng(20261114);torch.manual_seed(20261114)
 results=[];counterexamples=[]
 for h,w in [(1,8),(8,1),(2,2),(4,6),(16,20),(48,36),(8,8)]:
  for mode in ['col','tree']:
   for r in [.1,.25,.5,.75,1.]:
    for case in range(5):
     a=(np.ones((h,w)) if case==0 else rng.uniform(.01,5,(h,w))).astype(np.float32);m=budget(h*w,r)
     ref=adaptive_rectangles(torch.from_numpy(a),h,w,m,mode);cand=partition_integral_candidate(a,m,mode)
     equal=np.array_equal(ref,cand)
     results.append(dict(h=h,w=w,mode=mode,ratio=r,equal=equal))
     if not equal and len(counterexamples)<12:counterexamples.append(dict(h=h,w=w,mode=mode,ratio=r,case=case,different_positions=int((ref!=cand).sum())))
 comparisons=[]
 for kind in ['col_adaptive','quadtree_question']:
  module=SpatialGenerator(kind).eval();h,w=8,10;n=h*w
  raw=torch.randn(n,768).bfloat16();q=torch.randn(8,2048);grid=torch.tensor([[1,h,w]])
  for r in [.1,.25,.5,.75]:
   with torch.no_grad():
    _,p=module(torch.zeros(n,2048),grid,r,raw=raw,question=q)
    ref=(p@raw.float()).to(raw.dtype);fast,detail=compress(module,raw,grid,q,r)
    assert torch.equal(ref,fast) and torch.equal(p,detail['weights'])
    fp32,_=compress(module,raw,grid,q,r,'segmented_fp32');bf16,_=compress(module,raw,grid,q,r,'segmented_bf16')
   comparisons.append(dict(kind=kind,ratio=r,strict_bit_exact=True,segmented_fp32_equal=torch.equal(ref,fp32),segmented_bf16_equal=torch.equal(ref,bf16),
    segmented_fp32_max_error=float((ref.float()-fp32.float()).abs().max()),segmented_bf16_max_error=float((ref.float()-bf16.float()).abs().max())))
 out=Path('artifacts/tessfold_system_20261114');out.mkdir(exist_ok=True)
 save=dict(partition_cases=len(results),integral_candidate_mismatches=sum(not r['equal'] for r in results),counterexamples=counterexamples,feature_tests=comparisons,
  strict_tests_passed=True,notes='New proposal compared to original float32 reference, not shared candidate partition. No speed or accuracy guarantee inferred.')
 (out/'cpu_equivalence.json').write_text(json.dumps(save,indent=2)+'\n');print(json.dumps(save,indent=2))
if __name__=='__main__':main()
