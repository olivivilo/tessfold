"""Independent saved-dev/log/optimizer audit on required checkpoint backend.

Oldtorch_npu checkpoint tensor rebuild needs an actualNPU even map_locationCPU;
no model training or checkpoint rewriting occurs here.
"""
import math,json,torch,torch_npu
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now
from scripts.tessfold_accel.qwen_control_stats import scores
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    torch.set_num_threads(1);torch.npu.set_device(0)
    registry=load(BASE/'qwen_selected_runs.json');assert len(registry['runs'])==4;results=[]
    for meta in registry['runs']:
        p=ROOT/meta['full_run'];m=load(p/'manifest.json');rr=lines(p/'training.jsonl');total=math.ceil(m['train_questions']/m['batch'])*m['epochs']
        assert len(rr)==total and [r['step'] for r in rr]==list(range(1,total+1))
        assert m['backbone_trainable']==0
        for source,digest in m['source_sha256'].items():assert sha(ROOT/source)==digest
        warm=round(total*.1)
        for r in rr:
            step=r['step']-1;factor=(step+1)/warm if step<warm else .1+.9*(1+math.cos(math.pi*(step-warm)/(total-warm-1)))/2
            assert abs(r['lr']-m['lr']*factor)<1e-12
            assert abs(r['clip_multiplier']-min(1.,5./(r['gradient_norm']+1e-6)))<1e-9
            assert math.isfinite(r['kl']) and math.isfinite(r['gradient_norm'])
        dev=[]
        for stage in ['initial']+[f'epoch{i}' for i in range(1,5)]:
            preds=lines(p/f'{stage}.predictions.jsonl');summary=load(p/f'{stage}.summary.json');assert len(preds)==56
            metric=scores(preds);assert abs(metric['native_macro']-summary['macro'])<1e-9
            assert abs(sum(r['kl'] for r in preds)/56-summary['kl'])<1e-9
            dev.append(dict(stage=stage,macro=metric['native_macro'],kl=summary['kl']))
        latest=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
        assert latest['step']==total and latest['optimizer']['state']
        state_steps=[int(v['step']) for v in latest['optimizer']['state'].values()];assert all(s==total for s in state_steps)
        assert sha(ROOT/meta['checkpoint'])==meta['checkpoint_sha256']
        results.append(dict(model=meta['id'],optimizer_steps=total,actual_latest_optimizer_saved=True,optimizer_states=len(state_steps),all_optimizer_state_steps_equal_total=True,
            dev_predictions_recomputed=5*56,selected_stage=meta['stage'],dev=dev,
            clip_fraction=sum(r['clip_multiplier']<1 for r in rr)/len(rr),source_sha256_verified=True))
        del latest
    result=dict(time=now(),all_passed=True,runs=results,formal_scores_not_read_for_selection=True,source_sha256=sha(__file__),
        backend_note='CPUonlyattemptfailedbecauseoldtorch_npu pickle rebuildneedsactualNPU. Read-onlyaudit moved to allocatedNPU node, no reserialization or modelupdates.',
        note='Recomputed5stagesx56QAperrun devnativeF1/QAmetrics+KL from saved predictions; exact optimizer-step/LR/clip arithmetic; actualsavedoptimizer/latestcheckpointstate; source signatures.')
    save(BASE/'full_training_audit.json',result)
    print(json.dumps(dict(all_passed=True,optimizer_steps=sum(r['optimizer_steps'] for r in results),dev_records_recomputed=sum(r['dev_predictions_recomputed'] for r in results))))

if __name__=='__main__':main()
