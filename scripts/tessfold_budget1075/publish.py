"""Post-audit publication only: all budgets, paired errors and updated ZIP."""
import collections
import csv
import hashlib
import json
import os
import zipfile
from scripts.tessfold_budget1075.common import ROOT, BASE, PARENT, load, save, sha, now, verify
from scripts.tessfold_budget1075.report import strict


def main():
    verify()
    publication = load(BASE / 'publication_registration.json')
    assert publication['main_registration_sha256'] == sha(BASE / 'registration.json')
    for source, digest in publication['source_sha256'].items():
        assert sha(ROOT / source) == digest
    assert load(BASE / 'all_stages_complete.json')['all_complete']
    assert load(BASE / 'final_audit.json')['all_checks_passed']
    old = load(PARENT / 'qwen_formal/complete.json')
    new = load(BASE / 'qwen_formal/complete.json')
    models = {r['id']: r for base in [PARENT, BASE] for r in load(base / 'qwen_selected_runs.json')['runs']}
    results = {**old['native_metric_results'], **new['native_metric_results']}
    tasks = ['DocVQA', 'InfoVQA', 'ChartQA', 'OCRBench', 'DeepForm', 'KLC', 'WTQ', 'DT-VQA']
    order = ['REF'] + [f'{kind}_r{ratio:g}' for ratio in [.1, .25, .5, .75] for kind in ['Column', 'Tree']]
    label = lambda identity: identity.replace('Column_', 'StableColumn_') if models.get(identity, {}).get('allocation') else identity
    doc = ['# 原生8B：四个token预算的最终完整对照', '', f'更新时间UTC：{now()}。10%/75%扩展已通过完整审计。', '',
           '[新增预算完整训练、曲线及结果](TESSFOLD_8B_10_75_RESULTS.md) · [25%/50%原报告](TESSFOLD_FINAL_RESULTS.md)', '',
           '相同原生Qwen3-VL-8B-Instruct，基模全部冻结，仅训练P。20k训练QA×4epochs，固定56QA dev选模，23,565QA正式测试。每配置单seed。DeepForm/KLC为DUE corpus F1，其他任务为原生指标。StableColumn使用固定图像detail划分，是明确的结构变体，不能将全部差异归因于预算。', '',
           '|模型|保留率|DocVQA|InfoVQA|ChartQA|OCRBench|DeepForm F1|KLC F1|WTQ|DT-VQA|macro|与teacher差pp|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    latex = ['% Native DUE corpus F1 for KLC/DeepForm; all others task native metrics.',
             r'\begin{tabular}{lrrrrrrrrrr}', r'\toprule',
             r'Model & Keep & DocVQA & InfoVQA & ChartQA & OCRBench & DeepForm & KLC & WTQ & DT-VQA & Macro \\', r'\midrule']
    for identity in order:
        r = results[identity]
        ratio = models.get(identity, {}).get('ratio', 1.)
        values = [r['by_task'][t] for t in tasks] + [r['macro']]
        doc.append(f'|{label(identity)}|{ratio:.0%}|' + '|'.join(f'{v:.4f}' for v in values) + f'|{r["macro"]-results["REF"]["macro"]:+.4f}|')
        escaped = label(identity).replace('_', r'\_')
        latex.append(f'{escaped} & {ratio*100:.0f}\\% & ' + ' & '.join(f'{v:.2f}' for v in values) + r' \\')
    latex += [r'\bottomrule', r'\end{tabular}']
    save(BASE / 'all_budget_quality.json', dict(time=now(), results=results, models=models, order=order))
    (BASE / 'paper').mkdir(exist_ok=True)
    (BASE / 'paper/all_budget_quality.tex').write_text('\n'.join(latex)+'\n')
    doc += ['', '## 实际冷请求部署效率', '',
            '每个预算使用所在实验同期REF作同题配对：90题、3次重复。两个批次的绝对硬件耗时可能不同，加速比均相对各自REF，不将跨批次时差归因于预算。以下natural EOS；fixed64及分层结果见原始statistics.json。', '',
            '|模型|保留率|E2E加速|95%页面CI|TTFT加速|95%页面CI|', '|---|---:|---:|---|---:|---|']
    timing = {}
    memory = {}
    for directory in [PARENT, BASE]:
        for v in load(directory / 'qwen_timing/statistics.json')['comparisons']:
            if v['mode'] == 'natural_eos' and v['stratum'] == 'all':
                timing[v['model']] = v
        for identity, v in load(directory / 'qwen_memory/summary.json')['results'].items():
            if identity != 'REF':
                memory[identity] = v
    for identity in order[1:]:
        v = timing[identity]
        ci, ti = v['e2e_bootstrap_95_ci'], v['ttft_bootstrap_95_ci']
        doc.append(f'|{label(identity)}|{models[identity]["ratio"]:.0%}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
    doc += ['', '## 独立内存测量', '',
            '独立1ms内存采样，含常驻模型权重；不与无采样计时混算，生成token已审计一致。视觉KV为逻辑子段，不能推导整卡显存或最大并发按比例降低。', '',
            '|模型|全流程峰值GiB|下游峰值GiB|视觉KV中位MiB|', '|---|---:|---:|---:|']
    for identity in order[1:]:
        v = memory[identity]
        doc.append(f'|{label(identity)}|{v["full_max_gib"]:.4f}|{v["downstream_max_gib"]:.4f}|{v["visual_kv_median_mib"]:.3f}|')
    # Match teacher and student on each question; these are selected error cases,
    # never a sample used for model selection or an estimate of full performance.
    rows = strict(BASE / 'qwen_formal')
    teacher = {r['id']: r for r in rows if r['model'] == 'REF'}
    errors = collections.defaultdict(list)
    for r in rows:
        if r['role'] != 'primary' or r['model'] == 'REF':
            continue
        t = teacher[r['id']]
        errors[(r['model'], r['task'])].append(dict(model=label(r['model']), task=r['task'], id=r['id'],
            document_id=r['document_id'], image=r['image'], question=r['question'],
            gold_answers=json.dumps(r.get('answers', r.get('answer', r.get('gold_answers'))), ensure_ascii=False),
            teacher_prediction=t['prediction'], student_prediction=r['prediction'],
            teacher_score=t['score'], student_score=r['score'], drop=t['score']-r['score'],
            original_visual_tokens=r['original_visual_tokens'], visual_tokens=r['visual_tokens'],
            checkpoint=r['checkpoint']))
    error_rows = [r for key in sorted(errors) for r in sorted(errors[key], key=lambda r:(-r['drop'], r['id']))[:32]]
    (BASE / 'error_analysis').mkdir(exist_ok=True)
    with (BASE / 'error_analysis/largest_losses.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(error_rows[0]));writer.writeheader();writer.writerows(error_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for kind, marker in [('Column', 'o'), ('Tree', 's')]:
        ax.plot([10, 25, 50, 75], [results[f'{kind}_r{r:g}']['macro'] for r in [.1,.25,.5,.75]], marker=marker, label=kind)
    ax.axhline(results['REF']['macro'], color='gray', linestyle='--', label='Teacher')
    ax.set(xlabel='Retained visual tokens (%)', ylabel='Native eight-task macro (%)', xticks=[10,25,50,75])
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    (BASE / 'figures').mkdir(exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(BASE / f'figures/all_budget_quality.{extension}', dpi=180)
    plt.close(fig)
    doc += ['', '图中Column曲线包含标为StableColumn的结构变体，应结合上表解释；不能视为严格控制结构的预算消融。', '',
            '![四预算完整分数](../artifacts/tessfold_acceleration_20261115/budget1075/figures/all_budget_quality.png)', '',
            '[论文LaTeX表](../artifacts/tessfold_acceleration_20261115/budget1075/paper/all_budget_quality.tex) · [新增预算gold/teacher/student掉分案例CSV](../artifacts/tessfold_acceleration_20261115/budget1075/error_analysis/largest_losses.csv)', '',
            '案例CSV每模型每任务取掉分最大32题，非随机总体样本，不用于选模型。完整逐题预测见新增实验qwen_formal/rows.rank*.jsonl。']
    (ROOT / 'docs/TESSFOLD_8B_ALL_BUDGETS.md').write_text('\n'.join(doc)+'\n')
    for path in [ROOT / 'README.md', ROOT / 'docs/CURRENT_PROGRESS.md']:
        content = path.read_text()
        if path.name == 'README.md':
            content = content.replace('原生8B的10%／75%保留率完整训练评测进行中。', '原生8B的10%／75%保留率完整训练评测及审计已完成。')
        else:
            content = content.replace('**本扩展尚未完成**', '**本扩展全部训练、评测及最终审计已完成**')
        link = 'docs/TESSFOLD_8B_ALL_BUDGETS.md' if path.name == 'README.md' else 'TESSFOLD_8B_ALL_BUDGETS.md'
        if f']({link})' not in content:
            at = content.find('\n\n')
            content = content[:at] + f'\n\n[原生8B四预算最终对照]({link})' + content[at:]
        path.write_text(content)
    save(BASE / 'publication.json', dict(time=now(), source_sha256=sha(__file__), error_cases=len(error_rows),
        final_audit_sha256=sha(BASE / 'final_audit.json'), report='docs/TESSFOLD_8B_ALL_BUDGETS.md'))
    # Preserve the previous archive until the new ZIP has passed its CRC check.
    extensions={'.json','.jsonl','.csv','.tsv','.md','.pdf','.png','.svg','.tex','.log','.txt'}
    files=set(ROOT.glob('docs/TESSFOLD*.md'))|{ROOT/'tessfold.md'}
    files.update(p for p in PARENT.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    tmp=ROOT/'.tessfold.zip.tmp'
    manifest=[]
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        z.writestr('README.md', '# TessFold 实验结果\n\n四预算总入口：docs/TESSFOLD_8B_ALL_BUDGETS.md\n新增预算训练详情：docs/TESSFOLD_8B_10_75_RESULTS.md\n此前系统实验：docs/TESSFOLD_FINAL_RESULTS.md\n\n包含结果、逐题预测、图表、日志及SHA-256清单；不含权重、数据集或特征缓存。保留原目录结构。\n')
        for path in sorted(files):
            data=path.read_bytes();name=path.relative_to(ROOT).as_posix();z.writestr(name,data)
            manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('MANIFEST.json',json.dumps(dict(file_count=len(manifest),files=manifest),ensure_ascii=False,indent=2))
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
    os.replace(tmp,ROOT/'tessfold.zip')
    save(BASE/'publication_complete.json',dict(time=now(),archive='tessfold.zip',bytes=(ROOT/'tessfold.zip').stat().st_size,
         archive_sha256=sha(ROOT/'tessfold.zip'),publication_source_sha256=sha(__file__)))


if __name__ == '__main__':
    main()
