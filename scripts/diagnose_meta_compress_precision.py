"""Directional finite differences: tiny SGD updates vs BF16 input rounding.

FP32 mode casts the SAME checkpoint values; it does not recover full precision
pretraining weights. Teacher target remains the same cached BF16 official target.
"""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import json
import math
import torch
from scripts.diagnose_meta_compress import OUT,CACHE,DiagnosticModule,VARIANTS
from scripts.train_meta_compress import prepare_example,teacher_logits_for,atomic_json,sha
from scripts.eval_token_proxy import shortened_inputs
from vqa.official_und import OfficialUND
from vqa.meta_compress import meta_losses
from vqa.decoder_dependent_loss import common_prefix_logits


def main():
    torch.set_num_threads(16);torch.manual_seed(20261026)
    out=OUT/'precision';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    assert not (out/'directions.jsonl').exists(),'Explicit restart needed'
    atomic_json(out/'protocol.json',{'code_sha256':sha(__file__),'job':os.environ.get('SLURM_JOB_ID'),
        'steps':[1e-5,1e-4,1e-3,1e-2],'target':'same cached BF16 official teacher in both modes',
        'direction':'unit negative KL gradient at fixed original generator initialization, recomputed per precision',
        'fp32_meaning':'cast same BF16 checkpoint values to FP32 for student forward; same BF16 cached source, not new full-precision weights'})
    engine=OfficialUND(device='cpu');engine.model.eval().requires_grad_(False)
    row=next(json.loads(x) for x in (OUT/'samples.jsonl').read_text().splitlines() if json.loads(x)['role']=='fit')
    prompt,raw,grid,x,response=prepare_example(engine,row,CACHE,64)
    tl=teacher_logits_for(engine,row,prompt,x,response,CACHE)
    records=[]
    original_buffers={n:str(b.dtype) for n,b in engine.model.named_buffers()}
    for precision,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
        # Native HF loading keeps RoPE inv_freq buffers in FP32. Casting the
        # entire model to BF16 changes those buffers and is NOT the baseline.
        # Leave the BF16 engine untouched; only promote for the FP32 control.
        if precision=='fp32':engine.model.float()
        module=DiagnosticModule(VARIANTS[0],x,grid);params=list(module.parameters())
        initial=[p.detach().clone() for p in params]
        si=shortened_inputs({'input_ids':prompt},module.generator.output_length(len(x),.1),engine.model.config.image_token_id)
        def forward():
            c,p=module(x,grid);sp=c.to(dtype)
            logits=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
            return meta_losses(p,tl,logits)['prediction'],sp
        loss,sp=forward();initial_loss=float(loss.detach());initial_sp=sp.detach().clone()
        loss.backward();norm=math.sqrt(sum(float(p.grad.square().sum()) for p in params))
        direction=[-p.grad.detach()/norm for p in params]
        with torch.no_grad():
            repeated,_=forward()
            records.append({'precision':precision,'step_norm':0.,'kl':float(repeated),
                'baseline_kl':initial_loss,'repeat_delta':float(repeated)-initial_loss,'gradient_norm':norm})
            for h in [1e-5,1e-4,1e-3,1e-2]:
                for p,a,d in zip(params,initial,direction):p.copy_(a+h*d)
                value,s=forward()
                item={'precision':precision,'step_norm':h,'kl':float(value),'baseline_kl':initial_loss,
                    'actual_delta':float(value)-initial_loss,'first_order_predicted_delta':-h*norm,
                    'compressed_elements_changed_fraction':float((s!=initial_sp).float().mean()),
                    'compressed_relative_l2':float((s.float()-initial_sp.float()).norm()/initial_sp.float().norm()),
                    'gradient_norm':norm}
                records.append(item)
                with (out/'directions.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
        del module,params,initial,direction,sp,loss
    atomic_json(out/'complete.json',{'records':records,'fit_id':row['id'],
        'native_buffer_dtypes':original_buffers,
        'fp32_buffer_dtypes':{n:str(b.dtype) for n,b in engine.model.named_buffers()}})


if __name__=='__main__':main()
