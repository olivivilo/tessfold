"""No-dense-P CPU SIMD readout; counts all actual transfers in steady state."""
import ctypes,hashlib,os,subprocess,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from vqa.tessfold_fast_runtime import ScoreGraph,geometry
from vqa.tessfold_native_partition import partition
ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def library():
    source=ROOT/'vqa/tessfold_host_pool.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory=ROOT/'artifacts/tessfold_acceleration_20261115/native';directory.mkdir(exist_ok=True)
    target=directory/f'host_pool_{digest}.so'
    if not target.exists():
        temp=target.with_name(target.name+f'.{os.getpid()}.tmp')
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(source),'-o',str(temp)],check=True);temp.replace(target)
    lib=ctypes.CDLL(str(target));a=lambda dtype:np.ctypeslib.ndpointer(dtype=dtype,flags='C_CONTIGUOUS')
    lib.tessfold_host_pool.argtypes=[a(np.uint16),a(np.float32),a(np.int64),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,a(np.uint16),a(np.int64)]
    lib.tessfold_host_pool.restype=ctypes.c_int;return lib

class HostPool:
    @torch.no_grad()
    def __init__(self,module,raw,question,shape,ratio):
        assert raw.dtype==torch.bfloat16
        self.module=module;self.shape=shape;self.n=len(raw);self.m=max(1,min(self.n,round(self.n*ratio)));self.lib=library()
        before=int(torch.npu.memory_allocated());torch.npu.synchronize();ts=time.perf_counter()
        self.score=ScoreGraph(module,raw,question,*shape)
        self.host_raw=torch.empty(raw.shape,dtype=raw.dtype,pin_memory=True)
        self.host_score=torch.empty(self.n,dtype=torch.float32,pin_memory=True)
        self.host_energy=torch.empty(self.n,dtype=torch.float32,pin_memory=True)
        self.host_output=torch.empty((self.m,raw.shape[1]),dtype=raw.dtype,pin_memory=True)
        self.output=self.host_output.to(raw.device);self.representatives_cpu=np.empty(self.m,np.int64)
        self.temperature=float((.25+F.softplus(module.temperature)).cpu())
        self.raw_array=self.host_raw.view(torch.uint16).numpy();self.out_array=self.host_output.view(torch.uint16).numpy()
        self.score_array=self.host_score.numpy();self.energy_array=self.host_energy.numpy();self.done=torch.npu.Event()
        self.setup_ms=(time.perf_counter()-ts)*1e3;self.persistent_allocated_bytes=int(torch.npu.memory_allocated())-before

    @torch.no_grad()
    def __call__(self,raw,question):
        score,energy=self.score(raw,question)
        self.host_raw.copy_(raw,non_blocking=True);self.host_score.copy_(score,non_blocking=True);self.host_energy.copy_(energy,non_blocking=True)
        self.done.record();self.done.synchronize()
        h,w=self.shape;block,inverse,_,_=geometry(h,w,str(raw.device))
        raster=partition(self.energy_array[inverse].reshape(h,w),self.m,'col' if self.module.kind=='col_adaptive' else 'tree')
        groups=np.ascontiguousarray(raster[block])
        result=self.lib.tessfold_host_pool(self.raw_array,self.score_array,groups,self.n,raw.shape[1],self.m,self.temperature,self.out_array,self.representatives_cpu)
        assert result==self.m
        self.output.copy_(self.host_output,non_blocking=True)
        return self.output,dict(representatives_cpu=self.representatives_cpu,n=self.n,m=self.m)
