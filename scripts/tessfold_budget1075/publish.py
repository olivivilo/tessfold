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
    doc = ['# Native 8B: Final Full Comparison Across Four Token Budgets', '', f'Updated (UTC):{now()}. The 10%/75% extension passed the complete audit.', '',
           '[New-budget full training, curves, and results](TESSFOLD_8B_10_75_RESULTS.md) · [Original 25%/50% report](TESSFOLD_FINAL_RESULTS.md)', '',
           'Same native Qwen3-VL-8B-Instruct, fully frozen except P. Train on 20k QA × 4 epochs, select on 56 fixed development QA, and formally test on 23,565 QA. One seed/configuration. DeepForm/KLC use DUE corpus F1; others use native metrics. StableColumn\'s fixed image-detail partitioning is an explicit architectural variant, so not all differences are attributable to budget.', '',
           '|Model|Retention fraction|DocVQA|InfoVQA|ChartQA|OCRBench|DeepForm F1|KLC F1|WTQ|DT-VQA|macro|Difference from teacher (pp)|',
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
    doc += ['', '## Measured Cold-Request Deployment Efficiency', '',
            'Each budget pairs with its contemporaneous REF on 90 questions/3 repeats. Absolute timings can differ between batches; speedups use each batch’s own REF. Do not attribute cross-batch timing differences to budget. Natural EOS below; fixed64/strata in statistics.json.', '',
            '|Model|Retention fraction|E2E speedup|95% page-level CI|TTFT speedup|95% page-level CI|', '|---|---:|---:|---|---:|---|']
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
    doc += ['', '## Independent Memory Measurements', '',
            'Independent 1 ms memory sampling includes resident weights and remains separate from unsampled timing; generated tokens match. Logical visual KV does not imply proportional device-memory/concurrency reduction.', '',
            '|Model|Full-pipeline peak (GiB)|Downstream peak (GiB)|Median visual KV (MiB)|', '|---|---:|---:|---:|']
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
    doc += ['', 'Column curves include a labeled StableColumn structural variant. Interpret with the table; this is not an architecture-controlled budget ablation.', '',
            '![Full four-budget scores](../artifacts/tessfold_acceleration_20261115/budget1075/figures/all_budget_quality.png)', '',
            '[Manuscript LaTeX tables](../artifacts/tessfold_acceleration_20261115/budget1075/paper/all_budget_quality.tex) · [New-budget gold/teacher/student largest-drop examples CSV](../artifacts/tessfold_acceleration_20261115/budget1075/error_analysis/largest_losses.csv)', '',
            'Example CSV selects 32 largest drops/model/task, not a random population sample or selection set. Full predictions: new qwen_formal/rows.rank*.jsonl.']
    (ROOT / 'docs/TESSFOLD_8B_ALL_BUDGETS.md').write_text('\n'.join(doc)+'\n')
    for path in [ROOT / 'README.md', ROOT / 'docs/CURRENT_PROGRESS.md']:
        content = path.read_text()
        if path.name == 'README.md':
            content = content.replace('Native8B 10%/75% full training/evaluation in progress.', 'Native8B 10%/75% full training, evaluation, and audits completed.')
        else:
            content = content.replace('**This extension is not yet complete**', '**All extension training, evaluation, and final audits are complete**')
        link = 'docs/TESSFOLD_8B_ALL_BUDGETS.md' if path.name == 'README.md' else 'TESSFOLD_8B_ALL_BUDGETS.md'
        if f']({link})' not in content:
            at = content.find('\n\n')
            content = content[:at] + f'\n\n[Final native 8B four-budget comparison]({link})' + content[at:]
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
        z.writestr('README.md', '# TessFold Experimental Results\n\nFour-budget entry: docs/TESSFOLD_8B_ALL_BUDGETS.md\nNew-budget training details: docs/TESSFOLD_8B_10_75_RESULTS.md\nPrevious system experiments: docs/TESSFOLD_FINAL_RESULTS.md\n\nContains results, predictions, figures, logs, and SHA-256 manifest, excluding weights, datasets, and feature caches. Original layout retained.\n')
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
