"""Independent NumPy reference vs compiled FP32 partition; include near ties."""
import json,time
from pathlib import Path
import numpy as np
import torch
from vqa.compression_generators_v2 import adaptive_rectangles
from vqa.tessfold_native_partition import partition,library

def main():
    torch.set_num_threads(2);rng=np.random.default_rng(20261115);library()
    cases=[];errors=[];times=[]
    for h,w in [(1,8),(8,1),(2,2),(4,6),(8,10),(48,36),(72,70),(128,130)]:
      for mode in ['tree','col']:
       for ratio in [.1,.25,.5,.75,1.]:
        for kind in ['uniform','near_uniform','random','lognormal']:
         a={'uniform':lambda:np.ones((h,w)), 'near_uniform':lambda:1+rng.uniform(-1e-6,1e-6,(h,w)), 'random':lambda:rng.uniform(.01,5,(h,w)), 'lognormal':lambda:rng.lognormal(0,3,(h,w))}[kind]().astype(np.float32)
         m=max(1,round(h*w*ratio));ts=time.perf_counter();ref=adaptive_rectangles(torch.from_numpy(a),h,w,m,mode);old=(time.perf_counter()-ts)*1e3
         ts=time.perf_counter();new=partition(a,m,mode);fast=(time.perf_counter()-ts)*1e3
         row=dict(h=h,w=w,mode=mode,ratio=ratio,kind=kind,equal=bool(np.array_equal(ref,new)),reference_ms=old,native_ms=fast)
         cases.append(row)
         if not row['equal']:
          errors.append(row)
          if len(errors)==1:np.save('artifacts/tessfold_acceleration_20261115/native/first_counterexample.npy',a)
    result=dict(cases=len(cases),mismatches=len(errors),errors=errors[:20],rows=cases)
    p=Path('artifacts/tessfold_acceleration_20261115/native/cpu_partition_tests.json');p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),mismatches=len(errors),errors=errors[:8])))
    for row in cases:
      if row['h']==72 and row['w']==70 and row['ratio']==.25 and row['kind']=='random':print(row)

if __name__=='__main__':main()
