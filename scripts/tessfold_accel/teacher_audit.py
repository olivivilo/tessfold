"""CPU audit of actual native8B distillation prefixes; no gold relabeling."""
import collections,concurrent.futures,hashlib,json
import numpy as np
import torch
from scripts.tessfold_repair.common import ROOT,load,lines,save,now,sha
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'
SOURCE='Qwen3-VL-8B-Instruct_0c351dd01ed87e9c1b53cbc748cba10e6187ff3b_nativeBF16_short64_pixels1003520_v1'


def inspect(row,eos):
    path=BASE/'qwen_cache/teachers'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt')
    teacher=torch.load(path,map_location='cpu',weights_only=True)
    assert teacher['source']==SOURCE and teacher['id']==row['id']
    response=teacher['response'];assert isinstance(response,list) and 1<=len(response)<=64
    assert not any(t in eos for t in response[:-1])
    assert teacher['prompt'].ndim==2 and teacher['prompt'].shape[0]==1
    return dict(task=row['task'],length=len(response),stopped_eos=response[-1] in eos,truncated=len(response)==64 and response[-1] not in eos)


def main():
    torch.set_num_threads(1);eos=set(load(ROOT/'models/Qwen3-VL-8B-Instruct/generation_config.json')['eos_token_id'])
    summary={}
    for split,path in [('train','artifacts/compression_lab_v2/full_train.jsonl'),('dev','artifacts/compression_lab_v2/validation.jsonl')]:
        rows=lines(ROOT/path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rr=list(pool.map(lambda row:inspect(row,eos),rows))
        by={}
        for task in sorted({r['task'] for r in rr}):
            ss=[r for r in rr if r['task']==task]
            by[task]=dict(questions=len(ss),truncated=sum(r['truncated'] for r in ss),eos_terminated=sum(r['stopped_eos'] for r in ss))
        summary[split]=dict(questions=len(rr),length_quantiles=np.quantile([r['length'] for r in rr],[0,.5,.9,.99,1]).tolist(),
            truncated=sum(r['truncated'] for r in rr),by_task=by,data_sha256=sha(ROOT/path))
    result=dict(time=now(),source=SOURCE,scope='Actual frozen greedy short64distillation prefixes, no gold scores and no formal selection. EndingEOSatstep64 distinguished from truncation.',splits=summary)
    save(BASE/'teacher_prefix_audit.json',result);print(json.dumps(result))

if __name__=='__main__':main()
