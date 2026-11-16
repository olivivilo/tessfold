"""Direct installed PFA window semantics, independent of any model weights."""
import json,math,os
import numpy as np
import torch,torch_npu
from scripts.tessfold_repair.common import ROOT,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False;torch.manual_seed(20261115);rows=[]
    for dtype in [torch.bfloat16,torch.float16]:
      for n in [128,257,513,1024]:
        q,k,v=[torch.randn(1,12,n,64,device='npu',dtype=dtype) for _ in range(3)]
        mask=torch.zeros((1,n,n),dtype=torch.bool,device='npu');scale=64**-.5
        logits=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(64)
        full=torch.matmul(logits.softmax(-1,dtype=torch.float32).to(dtype),v)
        causal=torch.matmul(logits.masked_fill(torch.ones(n,n,device='npu',dtype=torch.bool).triu(1),-torch.inf).softmax(-1,dtype=torch.float32).to(dtype),v)
        variants=dict(default_mask=torch_npu.npu_prompt_flash_attention(q,k,v,atten_mask=mask,num_heads=12,input_layout='BNSD',scale_value=scale),
            explicit_bidirectional=torch_npu.npu_prompt_flash_attention(q,k,v,atten_mask=mask,num_heads=12,input_layout='BNSD',scale_value=scale,pre_tokens=2147483647,next_tokens=2147483647),
            no_mask=torch_npu.npu_prompt_flash_attention(q,k,v,num_heads=12,input_layout='BNSD',scale_value=scale))
        record=dict(dtype=str(dtype),length=n,variants={})
        for name,out in variants.items():
            record['variants'][name]=dict(full_max_abs=float((out.float()-full.float()).abs().max()),full_rms=float((out.float()-full.float()).square().mean().sqrt()),
                causal_max_abs=float((out.float()-causal.float()).abs().max()),causal_rms=float((out.float()-causal.float()).square().mean().sqrt()))
        positions=torch.arange(n,device='npu')
        for block in [64,128,256]:
            blocked=positions[None,:]//block>positions[:,None]//block
            target=torch.matmul(logits.masked_fill(blocked,-torch.inf).softmax(-1,dtype=torch.float32).to(dtype),v)
            for name,out in variants.items():record['variants'][name][f'block{block}_causal_rms']=float((out.float()-target.float()).square().mean().sqrt())
        rows.append(record)
    result=dict(time=now(),rows=rows,source_sha256=sha(__file__),no_model_weights=True,
        test='IdenticalQKV, defaultmask vs explicitfullfuture vs nomask againstmanualfullandcausaleager, BF16/FP16,2lengths. Onlytestsinstalledoperator semantics, not bit-equivalence.')
    result['previous_failed_hypothesis']='Job165927 strictdefaultcausalassertionfailed. Expandedlengthsandblockedreferences, rawvaluesnowalwayssaved; noforcedcausalclaim.'
    save(BASE/'attention_semantics.json',result);print(json.dumps(result))

if __name__=='__main__':main()
