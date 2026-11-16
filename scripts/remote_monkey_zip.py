"""Read original multipart Monkey ZIP using HTTP ranges; extract only document tasks."""
import argparse
import bisect
import hashlib
import io
import os
import json
import time
import zipfile
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]
REVISION='dbff559fd451ae5e46e108cc4797fe0a86a6a05f'
BASE=f'https://hf-mirror.com/datasets/echo840/Monkey_Data/resolve/{REVISION}/'
BLOCK=8*1024*1024

class MultipartRangeFile(io.RawIOBase):
    def __init__(self,test=False):
        self.parts=[('Monkey_Test_data.zip',13441778691)] if test else [(f'Monkey_Train_data.zip.{i:02d}',10485760000 if i<9 else 776011448) for i in range(1,10)]
        self.ends=[]
        for _,size in self.parts:self.ends.append((self.ends[-1] if self.ends else 0)+size)
        self.position=0
        self.cache=ROOT/'data/full/range_cache'/('test' if test else 'train');self.cache.mkdir(parents=True,exist_ok=True)
        self.session=requests.Session();self.session.headers.update({'User-Agent':'curl/8.0'})
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.ends[-1]+offset
        if self.position<0:raise ValueError('Negative seek')
        return self.position
    def block(self,part,index):
        path=self.cache/f'{part}_{index}.block'
        start=index*BLOCK;end=min(start+BLOCK,self.parts[part][1])-1
        expected=end-start+1
        if path.exists() and path.stat().st_size==expected:return path.read_bytes()
        for attempt in range(6):
            try:
                response=self.session.get(BASE+self.parts[part][0],headers={'Range':f'bytes={start}-{end}'},timeout=180,stream=True)
                if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    response.close();raise RuntimeError('Server did not honor exact byte range')
                blob=response.content;response.close()
                if len(blob)!=expected:raise RuntimeError('Incomplete HTTP range')
                tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_bytes(blob);tmp.replace(path)
                return blob
            except Exception:
                if attempt==5:raise
                time.sleep(min(2**attempt,30))
    def read(self,size=-1):
        if size<0:size=self.ends[-1]-self.position
        size=min(size,self.ends[-1]-self.position)
        chunks=[]
        while size>0:
            part=bisect.bisect_right(self.ends,self.position)
            start=self.ends[part-1] if part else 0
            offset=self.position-start
            index,within=divmod(offset,BLOCK)
            blob=self.block(part,index)
            count=min(size,len(blob)-within)
            chunks.append(blob[within:within+count]);self.position+=count;size-=count
        return b''.join(chunks)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--test',action='store_true')
    p.add_argument('--index-only',action='store_true')
    p.add_argument('--annotations-only',action='store_true')
    p.add_argument('--tasks',default='DeepForm,KleisterCharity,WikiTableQuestions')
    args=p.parse_args()
    label='test' if args.test else 'train'
    dest=ROOT/'data/full/monkey_original'/label;dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(MultipartRangeFile(args.test)) as archive:
        entries=[{'name':x.filename,'size':x.file_size,'compressed_size':x.compress_size,'offset':x.header_offset} for x in archive.infolist()]
        (dest/'zip_index.json').write_text(json.dumps(entries))
        print(f'INDEX {label}: {len(entries)} entries',flush=True)
        if args.index_only:return
        tasks=args.tasks.split(',')
        selected=[x for x in archive.infolist() if any(t in x.filename for t in tasks) and not x.is_dir()]
        if args.annotations_only:
            selected=[x for x in selected if x.filename.endswith(('.json','.jsonl'))]
        print(f'SELECTED {len(selected)} files, {sum(x.compress_size for x in selected)/1e9:.2f} compressed GB',flush=True)
        for i,entry in enumerate(sorted(selected,key=lambda x:x.header_offset)):
            path=dest/entry.filename
            if not path.resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe ZIP member')
            if path.exists() and path.stat().st_size==entry.file_size:continue
            blob=archive.read(entry) # zipfile verifies CRC32 on decompression
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
            if i%100==0:print(f'EXTRACT {label} {i+1}/{len(selected)} {entry.filename}',flush=True)
        if not args.annotations_only:
            (dest/'EXTRACT_COMPLETE').write_text(json.dumps({'tasks':tasks,'files':len(selected),'revision':REVISION}))

if __name__=='__main__':main()
