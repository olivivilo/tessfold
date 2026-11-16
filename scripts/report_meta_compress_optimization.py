"""Read recorded updates to explain clipping, optimizers, schedule and budget."""
import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/meta_compress_search'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    search = {r['name']: r for r in json.loads((BASE / 'summary.search.json').read_text())}
    results = {}
    for name, r in search.items():
        if not name.startswith(('r1_', 'r2_', 'r3_')):
            continue
        log = read(BASE / 'runs' / name / 'training.jsonl')
        assert len({x['step'] for x in log}) == len(log)
        cfg = r['config']

        def summarize(rows):
            return {
                'steps': len(rows),
                'pre_clip_gradient_mean': stats.mean(x['grad_norm'] for x in rows),
                'pre_clip_gradient_median': stats.median(x['grad_norm'] for x in rows),
                'nominal_post_clip_gradient_median': stats.median(min(x['grad_norm'], cfg['clip']) for x in rows),
                'clipped_fraction': stats.mean(x['clip_multiplier'] < 1 for x in rows),
                'clip_multiplier_median': stats.median(x['clip_multiplier'] for x in rows),
                'update_l2_mean': stats.mean(x['update_l2'] for x in rows),
                'update_l2_median': stats.median(x['update_l2'] for x in rows),
                'train_KL': stats.mean(x['prediction'] for x in rows),
                'first_lr': rows[0]['lr'], 'last_lr': rows[-1]['lr'],
                'sum_lr': sum(x['lr'] for x in rows),
                'update_path_length': sum(x['update_l2'] for x in rows),
            }
        results[name] = {
            'config': cfg, 'training_complete': r['training_complete'],
            'early_stop': r.get('early_stop'), 'overall': summarize(log),
            'epochs': {str(ep): summarize([x for x in log if x['epoch'] == ep])
                       for ep in sorted({x['epoch'] for x in log})},
            'search_stages': {k: v for k, v in r['stages'].items() if v['complete']},
        }
    (BASE / 'optimization_audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    lines = [
        '# MetaCompress：梯度、optimizer、scheduler 与 token 预算实测说明', '',
        '本报告只读取三轮已完成日志；入口：', '',
        '```bash', '.venv/bin/python -m scripts.report_meta_compress_optimization', '```', '',
        '## 1. 梯度范数现在是什么情况', '',
        '需要区分三个量：`grad_norm` 是 DDP 平均后、裁剪前全部生成器参数梯度的 L2 norm；'
        '`clip_multiplier=min(1, threshold/(norm+1e-6))` 是裁剪缩放系数；'
        '`update_l2` 则直接测 optimizer.step 前后全部参数的 L2 变化。'
        '最后一项才直接说明模型每步实际移动了多少。'
        '裁剪保持总梯度方向，不修复 loss 分项的方向冲突。', '',
        '下面统一取本轮 pilot 日志，避免将旧 20k/batch36 与新 pilot/batch8 的统计混在一张表里。'
        '每行是已记录全部更新的统计；10/25/50 是 token 保留率。', '',
        '| 候选 | 设置摘要 | 裁剪前 norm 中位数 | clip 阈值 | 裁剪比例 | 缩放系数中位数 | 平均实际 update L2 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    selected = [
        ('r1_sgd_original10', 'dense10 / 全 loss / SGD'),
        ('r1_sgd_cos_clip1_10', 'dense10 / 全 loss / SGD+cosine'),
        ('r1_adamw_cos10', 'dense10 / 全 loss / AdamW+cosine'),
        ('r2_dense_kl_sgdm25', 'dense25 / KL-only / momentum SGD'),
        ('r2_local_sgdm50', 'local50 / 全 loss / momentum SGD'),
        ('r3_local_kl_adamw50', 'local50 / KL-only / AdamW'),
        ('r3_local_kl75', 'local75 / KL-only / AdamW'),
    ]
    for name, label in selected:
        r = results[name]; z = r['overall']
        lines.append(f"| {name} | {label} | {z['pre_clip_gradient_median']:.3f} | {r['config']['clip']:g} | "
                     f"{z['clipped_fraction']:.1%} | {z['clip_multiplier_median']:.6f} | {z['update_l2_mean']:.6g} |")
    old = results['r1_sgd_original10']['overall']
    new = results['r3_local_kl_adamw50']['overall']
    lines += [
        '', '原配置的无 momentum SGD 更新为 `Δθ=-LR*clip(g)`，因此 clip=.01、LR=.001 时，'
        '每步名义上限为 1e-5；实际 FP32 参数差有很小的舍入偏差。'
        f"本轮原配置中位裁剪乘数 {old['clip_multiplier_median']:.7f}，约将梯度缩小 "
        f"{1/old['clip_multiplier_median']:.0f} 倍。"
        '这不是 norm 非有限或已经证明存在梯度爆炸，而是阈值远小于当前目标的梯度尺度。', '',
        f"最新 local50/KL-only/AdamW 的中位 norm 为 {new['pre_clip_gradient_median']:.3f}、"
        f"中位乘数 {new['clip_multiplier_median']:.3f}，平均实际参数更新约为原配置的 "
        f"{new['update_l2_mean']/old['update_l2_mean']:.0f} 倍。"
        '**仍然经常裁剪，不等于仍处于原先更新几乎被压死的状态。**'
        '不能把梯度 norm 的降低全部归功于 optimizer：预算、结构、loss 和训练轨迹也改变了。', '',
        '旧 20k 正式训练中位乘数约 0.0001765，与此处 pilot 的 0.0002030 不同，'
        '是数据/batch/训练时长不同的实测，不能互相覆盖。旧实验的结论仍是强裁剪和极小 SGD 更新。', '',
        '最新 KL-only 候选 entropy/collapse 权重均为 0，因此它的更新方向没有这两项正则贡献；'
        '原完整 loss 的梯度冲突不能直接套在它身上。'
        '原分项梯度审计仍仅覆盖那八张诊断图像，未重测为全部 pilot 数据的分项统计。', '',
        '## 2. optimizer 的变化与实际影响', '',
        '无 momentum SGD 的更新量与裁剪后梯度、LR 直接成比例。'
        'momentum SGD 累积多个步骤的方向，因此单步更新不再受 LR×clip 的同一个上界限制。'
        'AdamW 用梯度一阶/二阶历史做逐参数缩放并施加 weight decay，'
        '不能用 LR×clip 估计其参数更新上界，也不能只比较不同 optimizer 的 LR 数字。', '',
        '保持 dense、完整 loss、10% 预算时：', '',
        '| 配置 | 实际平均 update L2 | 初始分数 | epoch 1 | epoch 2 | epoch 3 | epoch 4 | 最终 KL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ['r1_sgd_original10', 'r1_sgd_cos_clip1_10', 'r1_adamw_cos10']:
        r = results[name]; st = r['search_stages']
        scores = ' | '.join(f"{st[f'epoch{e}/trained']['score']:.3f}" for e in range(1, 5))
        lines.append(f"| {name} | {r['overall']['update_l2_mean']:.6g} | {st['initial/trained']['score']:.3f} | "
                     f"{scores} | {st['epoch4/trained']['prediction']:.5f} |")
    lines += [
        '', '这些是配置组合对照：SGD 放宽 clip 时还加入 cosine；AdamW 的 LR 也不同，'
        '不能将全部差异归因于 optimizer 的名称。实测表明新配置让参数移动明显增大，'
        '但原 dense 结构没有因此恢复任务表现。AdamW10 第 2 epoch 的短暂高分也没有稳定保持。', '',
        '局部结构下 momentum SGD 和 AdamW 均能学习。代表性组合：', '',
        '| 候选 | 训练图数 | 目标 | search 初始→所选 epoch | confirm 初始→所选 epoch | confirm KL 降幅 |',
        '|---|---:|---|---:|---:|---:|',
        '| r2_local_sgdm50 | 224 | 完整 loss | 37.192→40.621（epoch 3） | 35.399→33.966 | 19.7% |',
        '| r3_local_kl_adamw50 | 672 | KL-only | 37.192→44.745（epoch 4） | 35.399→35.913 | 20.4% |', '',
        '训练图数、loss、LR 等也不同，因此这张表不证明 AdamW 比 momentum SGD 更优；'
        '它说明两种 optimizer 都已能降低独立验证 KL，但均未通过独立答案收益门槛。', '',
        '## 3. scheduler 做了什么，有没有证实改善', '',
        'constant 始终使用指定 LR。warmup cosine 先线性升到 peak LR，'
        '再按余弦降至 peak 的 10%。112-step pilot 的 warmup 是 11 步；'
        '336-step 扩展 pilot 是 34 步。最新 AdamW50 首步约 2.94e-6，升到 1e-4，最后 1e-5。', '',
        '条件匹配的 scheduler 对照是 dense25、完整 loss、相同 224 图、AdamW、peak LR=3e-5、'
        'clip=1、相同初始化/顺序，仅 scheduler 不同：', '',
        '| scheduler | 平均 update L2 | 初始分数 | epoch 1 | epoch 2 | epoch 3 | epoch 4 | 最终 KL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ['r1_adamw_constant25', 'r1_adamw_cos25']:
        r = results[name]; st = r['search_stages']
        scores = ' | '.join(f"{st[f'epoch{e}/trained']['score']:.3f}" for e in range(1, 5))
        lines.append(f"| {r['config']['scheduler']} | {r['overall']['update_l2_mean']:.6g} | "
                     f"{st['initial/trained']['score']:.3f} | {scores} | {st['epoch4/trained']['prediction']:.5f} |")
    lines += [
        '', '**两组均失败，没有证据表明 cosine 修复了原方案。**'
        '相同 peak 下 cosine 的 LR 总和更小（0.0018465 vs constant 0.00336），'
        '所以它既改变启动/后期步长，也改变总优化强度，不是保持相同 LR 总量的形状对照。', '',
        '最新局部 AdamW50 的每 epoch 实测：', '',
        '| epoch | LR 首步→末步 | 梯度 norm 中位数 | 裁剪比例 | 平均 update L2 | train KL | search KL |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    r = results['r3_local_kl_adamw50']
    for ep, z in r['epochs'].items():
        v = r['search_stages'][f'epoch{ep}/trained']
        lines.append(f"| {ep} | {z['first_lr']:.3g}→{z['last_lr']:.3g} | {z['pre_clip_gradient_median']:.3f} | "
                     f"{z['clipped_fraction']:.1%} | {z['update_l2_mean']:.6f} | {z['train_KL']:.5f} | {v['prediction']:.5f} |")
    lines += [
        '', '后期实际更新量随 LR 衰减明显减小，KL 仍下降。这说明优化在起作用，'
        '不能证明同结构/同数据下它优于 constant；这组尚无只改变 scheduler 的配对对照。', '',
        '## 4. 不同 token 预算下的表现', '',
        '下表为同一 56 张不同图像、七任务各 8 图的 search；teacher 宏均分固定为 58.134。'
        '比例均指保留率，10% 即减少约 90%；分数百分制，KL 越低越好。'
        '这是未训练基线，便于观察预算影响，不冒充训练提升。', '',
        '| 保留率 | 固定局部 mean 分数 | mean KL | dense 生成器初始分数 | local 生成器初始分数 |',
        '|---|---:|---:|---:|---:|',
    ]
    mapping = {0.1: ('r1_sgd_original10', None),
               0.25: ('r1_adamw_cos25', 'r2_local_balanced25'),
               0.5: ('r1_adamw_cos50', 'r2_local_balanced50'),
               0.75: (None, 'r3_local_kl75')}
    for ratio, (dense, local) in mapping.items():
        rows = [r for f in (BASE / f'baselines/ratio{ratio:g}/search').glob('eval.*.rank*.jsonl') for r in read(f)]
        assert len(rows) == len({r['id'] for r in rows}) == 56
        score = 100 * stats.mean(r['student_score'] for r in rows)
        kl = stats.mean(r['prediction'] for r in rows)
        fmt = lambda n: f"{results[n]['search_stages']['initial/trained']['score']:.3f}" if n else '未测'
        lines.append(f'| {ratio:.0%} | {score:.3f} | {kl:.5f} | {fmt(dense)} | {fmt(local)} |')
    lines += [
        '', '固定 mean 中只有预算在变，因此预算增加的收益明确。'
        '原 dense 初始化在 10/25/50% 均很差，说明多给 token 不能自动修复全局混合。'
        '75% local 的初始高分主要来自较宽预算/局部保留，不能算作学习收益。', '',
        '训练后采用同配置的 25/50% 对照：相同 224 图、AdamW 3e-5、clip=1、cosine、'
        'local bias=20、entropy/collapse 权重均 .01，仅保留率不同。固定报告第 4 epoch：', '',
        '| 保留率 | 分数：初始→epoch 4 | 分数增量 | KL：初始→epoch 4 |',
        '|---|---:|---:|---:|',
    ]
    for name in ['r2_local_balanced25', 'r2_local_balanced50']:
        r = results[name]; st = r['search_stages']; a, b = st['initial/trained'], st['epoch4/trained']
        lines.append(f"| {r['config']['ratio']:.0%} | {a['score']:.3f}→{b['score']:.3f} | "
                     f"{b['score']-a['score']:+.3f} | {a['prediction']:.5f}→{b['prediction']:.5f} |")
    lines += [
        '', '50% 的基础质量明显更好，训练增量也更大，但这一对照没有分别做独立 confirm；'
        '不能把 search 改善直接视为可靠泛化。', '',
        '更高预算的 r3_local_kl75：672 图、AdamW 3e-5、KL-only、cosine、clip=1。'
        '分数 55.587→55.030，KL 0.15613→0.10190；虽然绝对质量接近本小集 teacher，'
        '训练没有优于同预算初始化。与 r3_local_kl_adamw50 的 LR 也不同，'
        '二者不能作为只改变预算的训练对照。10% 没有通过多图验证的候选。', '',
        '## 5. 结论边界', '',
        '强裁剪造成极小 SGD 更新的问题已获得直接证据；新配置大幅增加实际更新量，'
        '最新局部方案也确实降低 train/search/confirm KL。'
        '但 optimizer 和 scheduler 尚未带来经过独立确认的稳定问答收益；'
        '预算扩大能改善基础保留能力，仍不能冒充训练成功。'
        '原 20k 重训未启动，以上是小规模多图实验，不是新的全量 benchmark。', '',
        '[全部 20 组优化统计 JSON](../artifacts/meta_compress_search/optimization_audit.json) · '
        '[逐阶段 search](../artifacts/meta_compress_search/RESULTS.search.md) · '
        '[独立 confirm](../artifacts/meta_compress_search/RESULTS.confirm.md) · '
        '[综合诊断](METACOMPRESS_CURRENT_DIAGNOSIS.md)。',
    ]
    dest = ROOT / 'docs/METACOMPRESS_OPTIMIZATION_AND_BUDGET.md'
    dest.write_text('\n'.join(lines) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
