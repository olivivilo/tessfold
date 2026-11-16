"""One real native-resolution VLM loss, compared across CPU FP32 and NPU BF16."""
import argparse,json
from pathlib import Path
import torch
from vqa.model import VLM,resolve
from vqa.full_train import amp,set_runtime
from vqa.checkpoint import load_tensor_state

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','npu'],required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/npu_ddp_probe.json').read_text())
    cfg.update(device=args.device,threads=4,mixed_precision='bfloat16' if args.device=='npu' else None,
               language_model='artifacts/npu_ddp_probe_eager/stage2/language_model')
    set_runtime(cfg)
    model=VLM(cfg)
    model.projector.load_state_dict(load_tensor_state(root/'artifacts/npu_ddp_probe_eager/stage2/projector.pt'))
    model.eval()
    sample=json.loads((root/'data/full_probe/train.jsonl').read_text().splitlines()[0])
    with torch.no_grad(),amp(cfg):loss,tokens=model.loss(sample)
    result={'device':args.device,'compute_precision':cfg['mixed_precision'] or 'float32','document_id':sample['document_id'],
            'visual_tokens':model.feature(sample).shape[0],'answer_tokens':tokens,'answer_nll':float(loss),
            'checkpoint':'Native-resolution engineering probe weights; numerical backend check only'}
    from vqa.full_train import atomic_json
    atomic_json(root/f'artifacts/lm_reference_{args.device}.json',result)
    print(json.dumps(result),flush=True)
    if args.device=='npu':
        from vqa.full_train import cpu_tree
        model.lm.train()
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW(model.projector.parameters(),lr=1e-5,foreach=False)
        with amp(cfg):probe_loss,_=model.loss(sample)
        probe_loss.backward();optimizer.step()
        directory=root/'artifacts/checkpoint_portability_probe';directory.mkdir(exist_ok=True)
        torch.save(cpu_tree({'projector':model.projector.state_dict(),'optimizer':optimizer.state_dict()}),directory/'state.pt')
        portable=torch.load(directory/'state.pt',map_location='cpu',weights_only=True)
        assert all(v.device.type=='cpu' for v in portable['projector'].values())
        restored=torch.optim.AdamW(model.projector.parameters(),lr=1e-5,foreach=False)
        restored.load_state_dict(portable['optimizer'])
        for parameter in model.projector.parameters():
            assert torch.equal(optimizer.state[parameter]['exp_avg'],restored.state[parameter]['exp_avg'])
            assert torch.equal(optimizer.state[parameter]['exp_avg_sq'],restored.state[parameter]['exp_avg_sq'])
        restored.zero_grad(set_to_none=True)
        with amp(cfg):next_loss,_=model.loss(sample)
        next_loss.backward()
        restored.step()
        assert all(torch.isfinite(p).all() for p in model.projector.parameters())
        atomic_json(directory/'verification.json',{'passed':True,'checks':['CPU-only tensor serialization','AdamW NPU optimizer state restored exactly','Restored optimizer executes the next update']})
    if (root/'artifacts/lm_reference_cpu.json').exists() and (root/'artifacts/lm_reference_npu.json').exists():
        cpu=json.loads((root/'artifacts/lm_reference_cpu.json').read_text())
        npu=json.loads((root/'artifacts/lm_reference_npu.json').read_text())
        gap=abs(cpu['answer_nll']-npu['answer_nll'])
        summary={'cpu_fp32':cpu,'npu_bf16':npu,'absolute_nll_difference':gap,'passed':gap<.25}
        atomic_json(root/'artifacts/lm_backend_verification.json',summary)
        print(json.dumps(summary),flush=True)
        if not summary['passed']:raise ValueError('NPU language model differs substantially from CPU reference')
    if args.device=='npu':
        # Exercise the saved-model/image entry before the full run.
        import subprocess,sys,gc
        del model
        gc.collect();torch.npu.empty_cache()
        subprocess.run([sys.executable,'-m','scripts.verify_full','--config','configs/npu_ddp_probe.json','--run-infer'],
                       cwd=root,check=True)
if __name__=='__main__':main()
