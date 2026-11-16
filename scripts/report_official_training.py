"""Export real Swift losses and paired DocVQA predictions after official SFT."""
import collections
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/official_und'


def predictions(label, directory=OUT):
    rows = {}
    for path in sorted(directory.glob(f'{label}.rank*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['task'] == 'DocVQA':
                assert row['id'] not in rows
                rows[row['id']] = row
    return rows


def export_review():
    baseline, tuned = predictions('baseline_full'), predictions('finetuned_full')
    old = predictions('final_val', ROOT / 'artifacts/full')
    images = {p.stem: '../full/docvqa_images/' + p.name for p in (ROOT / 'artifacts/full/docvqa_images').glob('*')}
    assert len(baseline) == 5349 and baseline.keys() == tuned.keys()
    records = []
    for key, base in baseline.items():
        ft, previous = tuned[key], old.get(key, {})
        records.append({'id': key, 'image': base['image'], 'question': base['question'],
                        'gold': base['answers'], 'baseline': base['prediction'], 'finetuned': ft['prediction'],
                        'baseline_anls': base['score'], 'finetuned_anls': ft['score'],
                        'delta': ft['score'] - base['score'], 'old_prediction': previous.get('prediction', ''),
                        'old_anls': previous.get('score'), 'document_id': base['document_id'],
                        'review_image': images.get(base['document_id'], '../../' + base['image'])})
    records.sort(key=lambda r: (r['delta'], r['baseline_anls'], r['id']))
    with (OUT / 'docvqa_comparison.jsonl').open('w') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (OUT / 'docvqa_comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({**row, 'gold': json.dumps(row['gold'], ensure_ascii=False)})
    summary = {'samples': len(records), 'improved': sum(r['delta'] > 1e-9 for r in records),
               'regressed': sum(r['delta'] < -1e-9 for r in records),
               'unchanged_score': sum(abs(r['delta']) <= 1e-9 for r in records)}
    for label in ('baseline', 'finetuned'):
        summary[label] = {'anls_percent': 100 * sum(r[f'{label}_anls'] for r in records) / len(records),
                          'top_predictions': collections.Counter(r[label] for r in records).most_common(10),
                          'shank_predictions': sum(r[label] == 'Dr. Robert E. Shank' for r in records),
                          'incorrect_shank_predictions': sum(r[label] == 'Dr. Robert E. Shank' and r[f'{label}_anls'] < 1 for r in records)}
    shank = [r for r in records if r['old_prediction'] == 'Dr. Robert E. Shank']
    summary['old_shank_group'] = {'samples': len(shank),
        **({f'{label}_anls_percent': 100 * sum(r[f'{label}_anls'] for r in shank) / len(shank)
           for label in ('baseline', 'finetuned')} if shank else {})}
    (OUT / 'docvqa_comparison_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    payload = json.dumps(records, ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>官方 Und DocVQA 作答对比</title>
<style>body{font:16px system-ui;margin:24px;background:#f7f8fa;color:#18222f}button,input,select{font:inherit;padding:7px;margin:4px}main{display:grid;grid-template-columns:48% 1fr;gap:24px}img{width:100%;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px;border-radius:8px}label{display:block;font-weight:bold}small{color:#566}h1{font-size:24px}@media(max-width:800px){main{display:block}}</style>
<h1>DocVQA：gold／官方基线／微调／旧自建模型</h1>
<p>完整 validation 5,349 题。分数为 0–1 ANLS；默认从退化最多的题开始。图片路径相对项目根目录。</p>
<input id="search" placeholder="搜索问题、gold、作答、ID" size="42">
<select id="filter"><option value="all">全部</option><option value="regressed">微调退化</option><option value="improved">微调提升</option><option value="wrong">官方基线未全对</option><option value="shank">旧模型固定 Shank 组</option></select>
<button id="prev">上一题</button><button id="next">下一题</button><span id="count"></span>
<main><div><a id="imageLink" target="_blank"><img id="image"></a></div><div id="detail"></div></main>
<script>const rows=__DATA__;let filtered=rows,index=0;
function render(){document.getElementById('count').textContent=filtered.length?`${index+1} / ${filtered.length}`:'无匹配';const detail=document.getElementById('detail');detail.replaceChildren();if(!filtered.length)return;const r=filtered[index];document.getElementById('image').src=r.review_image;document.getElementById('imageLink').href=r.review_image;
for(const [label,value] of [['ID',r.id],['问题',r.question],['Gold answers',JSON.stringify(r.gold,null,2)],[`官方基线 · ANLS ${r.baseline_anls.toFixed(4)}`,r.baseline],[`微调 · ANLS ${r.finetuned_anls.toFixed(4)} · Δ ${r.delta.toFixed(4)}`,r.finetuned],['旧自建模型',r.old_prediction]]){const l=document.createElement('label');l.textContent=label;const p=document.createElement('pre');p.textContent=value;detail.append(l,p)}}
function apply(){const q=document.getElementById('search').value.toLowerCase(),f=document.getElementById('filter').value;filtered=rows.filter(r=>(f==='all'||f==='regressed'&&r.delta<0||f==='improved'&&r.delta>0||f==='wrong'&&r.baseline_anls<1||f==='shank'&&r.old_prediction==='Dr. Robert E. Shank')&&JSON.stringify(r).toLowerCase().includes(q));index=0;render()}
document.getElementById('search').oninput=apply;document.getElementById('filter').onchange=apply;document.getElementById('prev').onclick=()=>{if(filtered.length)index=(index+filtered.length-1)%filtered.length;render()};document.getElementById('next').onclick=()=>{if(filtered.length)index=(index+1)%filtered.length;render()};render();</script></html>'''
    (OUT / 'docvqa_comparison.html').write_text(html.replace('__DATA__', payload))
    return summary


def export_losses():
    states = list((OUT / 'sft').glob('checkpoint-*/trainer_state.json'))
    state = max((json.loads(p.read_text()) for p in states), key=lambda r: r['global_step'])
    history = state['log_history']
    rows = [{'step': r['step'], 'epoch': r.get('epoch'), 'train_loss': r.get('loss'),
             'eval_loss': r.get('eval_loss'), 'learning_rate': r.get('learning_rate'),
             'grad_norm': r.get('grad_norm')} for r in history if 'loss' in r or 'eval_loss' in r]
    destination = OUT / 'loss_audit'
    destination.mkdir(exist_ok=True)
    with (destination / 'loss_history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train = [r for r in rows if r['train_loss'] is not None]
    evaluation = [r for r in rows if r['eval_loss'] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    for ax, x, xlabel in zip(axes, ('step', 'epoch'), ('Optimizer step', 'Epoch')):
        ax.plot([r[x] for r in train], [r['train_loss'] for r in train], alpha=.35, color='#2671b8', label='Train (batch loss)')
        smooth = [sum(r['train_loss'] for r in train[max(0, i-9):i+1]) / len(train[max(0, i-9):i+1]) for i in range(len(train))]
        ax.plot([r[x] for r in train], smooth, color='#2671b8', label='Train (10-step mean)')
        ax.plot([r[x] for r in evaluation], [r['eval_loss'] for r in evaluation], 'o-', color='#d05b32', label='Heldout validation NLL')
        ax.set(xlabel=xlabel, ylabel='Answer-token cross entropy', title='Official Und continued LoRA SFT')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(destination / f'training_loss.{extension}', dpi=180)
    plt.close(fig)
    return {'global_step': state['global_step'], 'epoch': state['epoch'],
            'best_model_checkpoint': state['best_model_checkpoint'], 'best_eval_loss': state['best_metric'],
            'initial_eval_loss': evaluation[0]['eval_loss'], 'last_eval_loss': evaluation[-1]['eval_loss'],
            'first_20_steps_mean_train_loss': sum(r['train_loss'] for r in train[:20]) / len(train[:20]),
            'last_20_steps_mean_train_loss': sum(r['train_loss'] for r in train[-20:]) / len(train[-20:]),
            'eval_points': evaluation, 'note': 'One real epoch; within-epoch validation is plotted. Batch train loss is not a fixed training-set evaluation.'}


def main():
    losses = export_losses()
    review = export_review()
    selection = json.loads((OUT / 'model_selection.json').read_text())
    adapter = Path(json.loads((OUT / 'sft_selection.json').read_text())['adapter'])
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in adapter.glob('adapter*') if p.is_file()}
    def header(path):
        with path.open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            return {k: v for k, v in json.loads(handle.read(size)).items() if k != '__metadata__'}
    base_header = header(ROOT / 'models/MonkeyOCRv2-B-Und/model.safetensors')
    adapter_header = header(adapter / 'adapter_model.safetensors')
    assert all('lora_' in key for key in adapter_header), 'Unexpected full-weight update in adapter'
    assert all('vision_tower' not in key or '.merger.' in key for key in adapter_header), 'Encoder must stay frozen'
    parameters = {'encoder_parameters': sum(math.prod(v['shape']) for k, v in base_header.items()
                                             if 'vision_tower' in k and '.merger.' not in k),
                  'adapter_parameters': sum(math.prod(v['shape']) for v in adapter_header.values()),
                  'encoder_adapter_parameters': 0,
                  'vision_merger_adapter_parameters': sum(math.prod(v['shape']) for k, v in adapter_header.items() if '.merger.' in k),
                  'adapter_tensors': len(adapter_header)}
    summary = {'losses': losses, 'docvqa': review, 'selection': selection, 'adapter_sha256': files,
               'parameters': parameters,
               'reload_evidence': 'finetuned_validation/full use fresh native model instances with PeftModel.from_pretrained; no in-memory trainer reused'}
    interim = OUT / 'sft100_validation_metrics.json'
    if interim.exists():
        summary['intermediate_step100_generation'] = json.loads(interim.read_text())
    (OUT / 'training_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ['# 官方 Und 微调：训练与作答检查', '',
             '入口：`bash scripts/run_official_und.sh`；总成绩：[OFFICIAL_UND_RESULTS.md](OFFICIAL_UND_RESULTS.md)。', '',
             f"实际完成 **{losses['global_step']} optimizer steps / {losses['epoch']:.4f} epoch**；"
             '只冻结官方 vision encoder，LLM 与 MLP 使用 LoRA。', '',
             f"权重头和 adapter 检查：encoder **{parameters['encoder_parameters']:,}** 参数，"
             f"LoRA **{parameters['adapter_parameters']:,}** 参数；encoder 内无 LoRA 张量，"
             '视觉侧仅 MLP merger 有 adapter。', '',
             f"初始 validation NLL：**{losses['initial_eval_loss']:.6f}**；最低：**{losses['best_eval_loss']:.6f}**；"
             f"最后：**{losses['last_eval_loss']:.6f}**。", '',
             f"训练前/后 20 steps 的 batch loss 均值：{losses['first_20_steps_mean_train_loss']:.6f} → "
             f"{losses['last_20_steps_mean_train_loss']:.6f}。不同 batch 的训练 loss 存在波动，不能要求每步严格下降。", '',
             '![真实 train/eval loss](../artifacts/official_und/loss_audit/training_loss.png)', '',
             '只有一个实际 epoch，图展示 epoch 内的多次验证，没有补造逐 epoch 数据。'
             '训练线为变化中的 batch 平均，验证线来自固定 256 题，不能把两者的差值直接作为严格泛化差距。', '',
             '- [Loss CSV](../artifacts/official_und/loss_audit/loss_history.csv)',
             '- [Loss PDF](../artifacts/official_und/loss_audit/training_loss.pdf)',
             '- [完整机器可读训练汇总](../artifacts/official_und/training_summary.json)', '',
             '## DocVQA 逐题作答', '',
             '- **[交互式图片／gold／官方基线／微调／旧模型对比](../artifacts/official_und/docvqa_comparison.html)**',
             '- [CSV](../artifacts/official_und/docvqa_comparison.csv) · [JSONL](../artifacts/official_und/docvqa_comparison.jsonl)', '',
             f"同一 5,349 题：微调后 {review['improved']} 题 ANLS 提升，{review['regressed']} 题下降，"
             f"{review['unchanged_score']} 题分数不变。", '',
             f"完整 DocVQA ANLS：官方基线 {review['baseline']['anls_percent']:.4f}；微调 {review['finetuned']['anls_percent']:.4f}。", '',
             f"精确回答 `Dr. Robert E. Shank` 的题数：官方基线 {review['baseline']['shank_predictions']}，"
             f"微调 {review['finetuned']['shank_predictions']}；其中非满分回答分别为 "
             f"{review['baseline']['incorrect_shank_predictions']} / {review['finetuned']['incorrect_shank_predictions']}。", '',
             '旧模型该回答组的变化见 `docvqa_comparison_summary.json` 的 `old_shank_group`；'
             'HTML 可筛选“旧模型固定 Shank 组”。', '',
             '## 模型选择与重载', '',
             f"按最低 validation NLL 选择候选：`{adapter.relative_to(ROOT)}`。"
             f"按固定 256 题生成分数选择最终模型：**{selection['selected']}**。", '',
             '微调评测为另起进程，从官方原权重＋磁盘 adapter 重新加载后的真实生成。'
             'adapter 哈希保存在 training_summary.json。测试分数不参与候选检查点与最终模型选择。', '',
             '本实验说明官方成品模型在本地协议下的表现及继续微调效果，不能单凭它反推出此前从零对齐失败的唯一原因。']
    if interim.exists():
        values = json.loads(interim.read_text())
        lines.extend(['', '## 中途生成检查', '',
                      f"第 100 步在同一固定 256 题上的四任务生成均分为 {values['task_macro_mean_percent']:.4f}，"
                      '原始官方模型为 59.2318。DocVQA/InfoVQA 提升而 ChartQA 下降；'
                      '这说明较低 teacher-forced loss 不保证更高生成分数。'
                      '该检查未改变既定训练参数、轮数和最终检查点选择规则。', '',
                      '逐题原始结果：`artifacts/official_und/sft100_validation.rank*.jsonl`。'])
    (ROOT / 'docs/OFFICIAL_UND_TRAINING.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
