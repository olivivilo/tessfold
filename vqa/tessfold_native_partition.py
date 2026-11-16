"""Compiled reference arithmetic; builds a versioned .so without pip installs."""
import ctypes,hashlib,os,subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vqa/tessfold_partition_native.cpp'

@lru_cache(maxsize=1)
def library():
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    base=ROOT/'artifacts/tessfold_acceleration_20261115/native';base.mkdir(parents=True,exist_ok=True)
    target=base/f'partition_{digest}.so'
    if not target.exists():
        temp=target.with_name(target.name+f'.{os.getpid()}.tmp')
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(SOURCE),'-o',str(temp)],check=True)
        temp.replace(target)
    lib=ctypes.CDLL(str(target))
    lib.tessfold_partition.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')]
    lib.tessfold_partition.restype=ctypes.c_int
    lib.tessfold_support.argtypes=[np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')]
    lib.tessfold_support.restype=ctypes.c_int
    return lib

def partition(image,m,mode='tree'):
    image=np.ascontiguousarray(image,dtype=np.float32);h,w=image.shape
    result=np.empty(h*w,dtype=np.int64)
    status=library().tessfold_partition(image,h,w,m,int(mode=='col'),result)
    if status!=m:raise RuntimeError(f'Native partition returned {status}')
    return result

def support(groups,m):
    groups=np.ascontiguousarray(groups,dtype=np.int64)
    permutation=np.empty(len(groups),dtype=np.int64);offsets=np.empty(m+1,dtype=np.int64)
    status=library().tessfold_support(groups,len(groups),m,permutation,offsets)
    if status!=len(groups):raise RuntimeError(f'Native support returned {status}')
    return permutation,offsets
