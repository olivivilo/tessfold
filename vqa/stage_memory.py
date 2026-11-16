"""Allocator peak plus time-weighted sampled average for sequential NPU stages.

Peak is the backend allocator high-water mark; the average is a 1ms sampling
estimate, including CPU gaps within a stage. Sampling does not synchronize NPU.
All boundaries synchronize to assign queued kernels to the correct stage.
"""
from __future__ import annotations
import threading,time
import numpy as np


def time_average(samples):
 assert len(samples)>=2
 t=np.asarray([x[0] for x in samples],dtype=np.float64)
 assert (np.diff(t)>=0).all() and t[-1]>t[0]
 v=np.asarray([[x[1],x[2]] for x in samples],dtype=np.float64)
 return ((np.diff(t)[:,None]*(v[1:]+v[:-1])/2).sum(0)/(t[-1]-t[0])).tolist()


class StageMemory:
 def __init__(self,device,interval=.001):
  import torch
  self.api=torch.npu;self.device=device;self.interval=interval
  self.active=None;self.rows=[];self.phases={};self.lock=threading.Lock();self.quit=threading.Event();self.ready=threading.Event();self.error=None
  self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();self.ready.wait()
  if self.error:raise self.error
 def _sample(self):return (time.perf_counter(),int(self.api.memory_allocated(self.device)),int(self.api.memory_reserved(self.device)))
 def _loop(self):
  try:
   self.api.set_device(self.device);self.ready.set()
   while not self.quit.wait(self.interval):
    with self.lock:
     if self.active is not None:self.rows.append(self._sample())
  except BaseException as e:self.error=e;self.ready.set()
 def begin(self,name):
  assert self.active is None and name not in self.phases
  self.api.synchronize(self.device);self.api.reset_peak_memory_stats(self.device)
  with self.lock:self.rows=[self._sample()];self.active=name
 def end(self):
  assert self.active is not None
  self.api.synchronize(self.device)
  with self.lock:
   self.rows.append(self._sample());rows=self.rows;name=self.active;self.active=None
  if self.error:raise self.error
  avg=time_average(rows);times=[x[0] for x in rows]
  values=dict(start_allocated_bytes=rows[0][1],end_allocated_bytes=rows[-1][1],
   start_reserved_bytes=rows[0][2],end_reserved_bytes=rows[-1][2],
   peak_allocated_bytes=int(self.api.max_memory_allocated(self.device)),peak_reserved_bytes=int(self.api.max_memory_reserved(self.device)),
   time_average_allocated_bytes=avg[0],time_average_reserved_bytes=avg[1],
   samples=len(rows),seconds=times[-1]-times[0],max_sample_gap_seconds=max(b-a for a,b in zip(times,times[1:])))
  values['incremental_peak_allocated_bytes']=max(0,values['peak_allocated_bytes']-values['start_allocated_bytes'])
  self.phases[name]=(values,rows)
  return values
 def reset(self):assert self.active is None;self.phases={}
 def export(self,path):
  arrays={}
  for name,(_,rows) in self.phases.items():
   a=np.asarray(rows,dtype=np.float64);a[:,0]-=a[0,0];arrays[name]=a
  np.savez_compressed(path,**arrays)
  return {k:v[0] for k,v in self.phases.items()}
 def close(self):
  assert self.active is None
  self.quit.set();self.thread.join(timeout=5)
  if self.error:raise self.error


def cache_nbytes(cache):
 """Count unique KV storage tensors, not references or model weights."""
 if cache is None:return 0
 seen=set();total=0
 if hasattr(cache,'layers'):
  tensors=[v for layer in cache.layers for v in [getattr(layer,'keys',None),getattr(layer,'values',None)]]
 elif hasattr(cache,'key_cache'):tensors=list(cache.key_cache)+list(cache.value_cache)
 else:tensors=[v for layer in cache for v in layer]
 for v in tensors:
  if v is None:continue
  key=(str(v.device),v.untyped_storage().data_ptr())
  if key not in seen:seen.add(key);total+=v.untyped_storage().nbytes()
 return int(total)
