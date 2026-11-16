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
    doc=['# 原生8B压缩器全量训练：loss、梯度与生成分数诊断','',f'UTC：{now()}。仅统计已经写入的日志；最后一轮可能尚未结束。','',
        '所有设置固定：基模全冻结、AdamW、weight_decay0、clip5、10%warmup后cosine到0.1LR。每步裁剪前gradient norm及实际clip multiplier来自训练日志。不能因为某步gradient norm>5就说训练已失败；应结合发生频率、裁剪比例、dev KL和实际生成分数判断。','',
        '|运行|epoch|完整epoch dev|train KL|dev KL|dev生成macro|norm p50|norm p99|裁剪频率|clip倍率中位|','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'未完成' if v is None else f'{v:.4f}'
    for run in out:
        for r in run['epochs']:
            doc.append(f'|{run["run"]}|{r["epoch"]}|{r["epoch_dev_complete"]}|{r["train_kl"]:.4f}|{fmt(r["dev_kl"])}|{fmt(r["dev_macro"])}|{r["gradient_q50_q90_q99_max"][0]:.3f}|{r["gradient_q50_q90_q99_max"][2]:.3f}|{r["clip_fraction"]:.1%}|{r["clip_multiplier_median"]:.3f}|')
    doc+=['','KL和生成分数可能方向不一致：KL度量teacher回答前缀上完整词表的平均分布差异，生成指标取决于greedy路径和任务离散计分。分区又包含detached score的离散决策，输入或参数轻微变化会改变切分，训练不等于完整可微分区学习。StableColumn50%固定detail分区正是针对这类不稳定性的显式变体；其收益只能由独立评测确认。','',
        '因此每轮同时评估KL和生成宏分，按预设合格门槛及dev分数选择，保留所有epoch。56题dev规模很小，单个任务只有8题；其波动无法直接代表23565题正式趋势。固定512图/2404题初始对照与全量正式表用于独立验证泛化，不用于追选模型。','',
        '此文档是观测诊断，没有中途修改optimizer、scheduler、batch或clip，也没有依据测试分数调参。']
    (ROOT/'docs/TESSFOLD_8B_TRAINING_DIAGNOSTICS.md').write_text('\n'.join(doc)+'\n')
    print(json.dumps(dict(runs=len(out),report='docs/TESSFOLD_8B_TRAINING_DIAGNOSTICS.md')))

if __name__=='__main__':main()
