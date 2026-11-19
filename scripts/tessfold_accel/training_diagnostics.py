"""Observed optimization diagnostics; does not modify or choose experiments."""
import collections,json
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,save,now
from scripts.tessfold_accel.live_io import live_lines
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    out=[]
    for label in ['qwen_full_pass1','qwen_full_tree25','qwen_full_stable50']:
      for run in sorted((BASE/label).glob('*_r*')):
        manifest=load(run/'manifest.json');initial=load(run/'initial.summary.json')
        if not manifest or not initial:continue
        logs=live_lines(run/'training.jsonl');epoch_groups=collections.defaultdict(list)
        for r in logs:epoch_groups[r['epoch']].append(r)
        epochs=[]
        for ep,rows in sorted(epoch_groups.items()):
            dev=load(run/f'epoch{ep}.summary.json')
            q=np.quantile([r['gradient_norm'] for r in rows],[.5,.9,.99,1]).tolist()
            epochs.append(dict(epoch=ep,steps=len(rows),epoch_dev_complete=bool(dev),train_kl=float(np.mean([r['kl'] for r in rows])),dev_kl=dev.get('kl'),dev_macro=dev.get('macro'),
                gradient_q50_q90_q99_max=q,clip_fraction=float(np.mean([r['clip_multiplier']<1 for r in rows])),
                clip_multiplier_median=float(np.median([r['clip_multiplier'] for r in rows])),lr_first=rows[0]['lr'],lr_last=rows[-1]['lr']))
        out.append(dict(run=label+'/'+run.name,initial_dev_kl=initial['kl'],initial_dev_macro=initial['macro'],epochs=epochs,complete=bool(load(run/'complete.json'))))
    save(BASE/'training_diagnostics.json',dict(time=now(),runs=out,scope='Observed unfinished logs may include partialepoch. Does not change hyperparameters or choose checkpoints.'))
    doc=['# Native 8B Compressor Training: Loss, Gradients, and Generation Diagnostics','',f'UTC: {now()}. Only committed logs are summarized; the final epoch may be incomplete.','',
        'All settings are fixed: frozen backbone, AdamW, weight_decay=0, clip=5, 10% warmup, and cosine decay to 0.1 LR. Per-step pre-clipping norms and actual multipliers come from training logs. A gradient norm >5 on one step does not establish failure; interpret its frequency and clipping magnitude together with development KL and generated-answer scores.','',
        '|Run|epoch|Complete-epoch development evaluation|train KL|dev KL|Development generation macro|norm p50|norm p99|Clipping frequency|Median clipping multiplier|','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'Incomplete' if v is None else f'{v:.4f}'
    for run in out:
        for r in run['epochs']:
            doc.append(f'|{run["run"]}|{r["epoch"]}|{r["epoch_dev_complete"]}|{r["train_kl"]:.4f}|{fmt(r["dev_kl"])}|{fmt(r["dev_macro"])}|{r["gradient_q50_q90_q99_max"][0]:.3f}|{r["gradient_q50_q90_q99_max"][2]:.3f}|{r["clip_fraction"]:.1%}|{r["clip_multiplier_median"]:.3f}|')
    doc+=['','KL and generation scores can move in different directions. KL measures average full-vocabulary discrepancy under teacher response prefixes, whereas generation depends on the greedy trajectory and discrete task scoring. Partitioning includes detached-score discrete decisions; small input or parameter changes can change a split. Training is not fully differentiable partition learning. StableColumn50% explicitly fixes detail-based partition allocation to address this instability; any benefit still requires independent evaluation.','',
        'Evaluate KL and generation macro each epoch, select by prespecified eligibility/development score, and retain all epochs. A 56-question development set (8/task) cannot predict full 23,565-question trends reliably. Fixed 512-image/2,404-QA initial controls and full tests assess generalization, not reselection.','',
        'This is observational diagnosis: optimizer, scheduler, batch, and clip are unchanged during training, and tests do not tune parameters.']
    (ROOT/'docs/TESSFOLD_8B_TRAINING_DIAGNOSTICS.md').write_text('\n'.join(doc)+'\n')
    print(json.dumps(dict(runs=len(out),report='docs/TESSFOLD_8B_TRAINING_DIAGNOSTICS.md')))

if __name__=='__main__':main()
