"""Trial-level correlation analysis; random downstream outcomes are never averaged away."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['DocVQA', 'InfoVQA', 'ChartQA', 'DT-VQA', 'WTQ', 'OCRBench', 'DeepForm', 'KLC']


def coefficient(x, y, method='spearman'):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = getattr(stats, {'spearman': 'spearmanr', 'pearson': 'pearsonr', 'kendall': 'kendalltau'}[method])(x, y).statistic
    return float(v) if np.isfinite(v) else None


def cluster_ci(rows, loss='total', repetitions=1000):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['document_id']].append(i)
    blocks = [np.array(v) for v in groups.values()]
    if len(blocks) < 8:
        return None
    x = np.array([r['loss'][loss] for r in rows])
    y = np.array([r['score_drop'] for r in rows])
    rng = np.random.default_rng(20261016)
    values = []
    for _ in range(repetitions):
        ix = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        v = coefficient(x[ix], y[ix])
        if v is not None:
            values.append(v)
    return list(map(float, np.quantile(values, [.025, .975]))) if len(values) > 50 else None


def describe(rows, bootstrap=False, loss='total'):
    x = np.array([r['loss'][loss] for r in rows])
    y = np.array([r['score_drop'] for r in rows])
    return {'trials': len(rows), 'images': len({r['document_id'] for r in rows}),
            'spearman': coefficient(x, y), 'pearson': coefficient(x, y, 'pearson'),
            'kendall': coefficient(x, y, 'kendall'),
            'spearman_image_cluster_ci95': cluster_ci(rows, loss) if bootstrap else None,
            'loss_min': float(x.min()) if len(x) else None, 'loss_max': float(x.max()) if len(x) else None,
            'teacher_positive_trials': sum(r['teacher_score'] > 0 for r in rows),
            'student_zero_trials': sum(r['student_score'] == 0 for r in rows),
            'drop_positive_trials': int((y > 0).sum()), 'drop_zero_trials': int((y == 0).sum()),
            'drop_negative_trials': int((y < 0).sum())}


def residualize(values, keys):
    groups = collections.defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    result = np.array(values, dtype=float).copy()
    for indices in groups.values():
        result[indices] -= result[indices].mean()
    return result


def within(rows, keys, loss='total', bootstrap=False):
    # Rank within each question/stratum, then correlate centered ranks.
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[tuple(r[k] for k in keys)].append(i)
    xs, ys, owners = [], [], []
    informative = 0
    for ix in groups.values():
        x = np.array([rows[i]['loss'][loss] for i in ix])
        y = np.array([rows[i]['score_drop'] for i in ix])
        if np.ptp(y) > 0:
            informative += 1
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        xs.extend((rx-rx.mean())/max(1, len(ix)))
        ys.extend((ry-ry.mean())/max(1, len(ix)))
        owners.extend(rows[i]['document_id'] for i in ix)
    ci = None
    if bootstrap:
        by_image = collections.defaultdict(list)
        for i, owner in enumerate(owners):
            by_image[owner].append(i)
        blocks = [np.array(v) for v in by_image.values()]
        if len(blocks) >= 8:
            rng, values = np.random.default_rng(20261016), []
            xs, ys = np.asarray(xs), np.asarray(ys)
            for _ in range(1000):
                ii = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
                val = coefficient(xs[ii], ys[ii], 'pearson')
                if val is not None:
                    values.append(val)
            if len(values) > 50:
                ci = list(map(float, np.quantile(values, [.025, .975])))
    return {'rank_residual_correlation': coefficient(xs, ys, 'pearson'), 'cluster_ci95': ci, 'strata': len(groups),
            'strata_with_outcome_variation': informative, 'trials': len(rows), 'keys': keys}


def partial_rank(rows):
    # Complete balanced question x configuration design: exact two-way demeaning.
    x = stats.rankdata([r['loss']['total'] for r in rows])
    y = stats.rankdata([r['score_drop'] for r in rows])
    q = [r['sample_id'] for r in rows]
    c = [(r['ratio'], r['family']) for r in rows]
    def transform(v):
        return residualize(v, q) + residualize(v, c)-v+v.mean()
    return coefficient(transform(x), transform(y), 'pearson')


def paired_seed_ordering(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['sample_id'], r['ratio'], r['family'])].append(r)
    concordant = discordant = outcome_ties = loss_ties = 0
    examples = []
    image_counts = collections.defaultdict(lambda: np.zeros(2))
    for rr in groups.values():
        if len(rr) != 2:
            continue
        a, b = sorted(rr, key=lambda r: r['loss']['total'])
        if a['score_drop'] == b['score_drop']:
            outcome_ties += 1
        elif a['loss']['total'] == b['loss']['total']:
            loss_ties += 1
        elif a['score_drop'] < b['score_drop']:
            concordant += 1
            image_counts[a['document_id']][0] += 1
        else:
            discordant += 1
            image_counts[a['document_id']][1] += 1
            examples.append({'low_loss_trial': a['id'], 'low_loss': a['loss']['total'],
                             'low_loss_drop': a['score_drop'], 'high_loss_trial': b['id'],
                             'high_loss': b['loss']['total'], 'high_loss_drop': b['score_drop']})
    ci = None
    if len(image_counts) >= 8:
        counts = np.stack(list(image_counts.values()))
        rng = np.random.default_rng(20261016)
        draw = counts[rng.integers(len(counts), size=(1000, len(counts)))].sum(1)
        ci = list(map(float, np.quantile(draw[:, 0]/draw.sum(1), [.025, .975])))
    return {'pairs': concordant+discordant+outcome_ties+loss_ties, 'concordant': concordant,
            'discordant': discordant, 'outcome_ties': outcome_ties, 'loss_ties': loss_ties,
            'ordering_accuracy_on_nonties': concordant/(concordant+discordant) if concordant+discordant else None,
            'ordering_accuracy_cluster_ci95': ci,
            'discordant_examples': examples[:12]}


def write_review(out, rows, filename='review.html'):
    if filename == 'review.html' and len(rows) > 20000:
        # Keep the full trial data; split the browser view so a 20x experiment
        # does not require parsing a single ~100 MB inline JSON document.
        by_task = collections.defaultdict(list)
        for r in rows:
            by_task[r['task']].append(r)
        links = []
        for task, task_rows in by_task.items():
            task_filename = f'review_{task}.html'
            write_review(out, task_rows, task_filename)
            links.append(f'<li><a href="{task_filename}">{task}</a>: {len(task_rows):,}  independent compressions</li>')
        (out/filename).write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                                 '<title>Random-compression answer comparison</title><style>body{font:17px system-ui;margin:40px;line-height:1.9}</style>'
                                 f'<h1>Per-trial random-token-compression answers</h1><p>Total {len(rows):,}  compressions. Pages are split by task; filter retention/family/trial type and sort loss/drop, then click a row for image/full answers.</p>'
                                 '<ul>'+''.join(links)+'</ul><p><a href="trials.csv">Complete per-trial CSV</a> · <a href="RESULTS.md">Full experimental report</a></p></html>')
        return
    data = [{k: r[k] for k in ('id', 'sample_id', 'task', 'image', 'question', 'answers', 'kind', 'family', 'ratio',
                               'replicate', 'matrix_seed', 'teacher_prediction', 'teacher_score',
                               'student_prediction', 'student_score', 'score_drop', 'loss')} for r in rows]
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Token compression trial review</title>
<style>body{font:15px system-ui;margin:24px;color:#182535}select,input,button{margin:6px;padding:7px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}tr:hover{background:#eaf3fa;cursor:pointer}#detail{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}img{max-width:100%;max-height:680px}pre{white-space:pre-wrap;overflow-wrap:anywhere}th{position:sticky;top:0;background:white}.hint{color:#516274}</style>
<h1>Per-trial random-token-compression answers</h1><p>Each row is an independent compression; scores in [0,1], drop=teacher−student. Random-trial answers are not averaged.</p>
<div id="controls"></div><label><input type="checkbox" id="positive">Teacher score &gt; 0 only</label><label>Ordering <select id="sort"><option value="loss">Loss: high to low</option><option value="drop">Score drop: high to low</option><option value="lossasc">Loss: low to high</option></select></label><p id="count"></p>
<div id="detail"><div id="text"></div><img id="image"></div><button id="prev">Previous page</button><button id="next">Next page</button>
<table><thead><tr><th>Task / question ID</th><th>Retention / family / random seed</th><th>V1 loss</th><th>teacher</th><th>student</th><th>Drop</th><th>Student answer</th></tr></thead><tbody id="body"></tbody></table>
<script>const rows=__DATA__;let page=0;const filters={};const $=id=>document.getElementById(id);for(const [key,name] of [['task','Task'],['ratio','Retention'],['family','Family'],['kind','Type']]){const label=document.createElement('label');label.textContent=name;const s=document.createElement('select');s.add(new Option('All',''));[...new Set(rows.map(r=>r[key]))].sort().forEach(v=>s.add(new Option(v,v)));s.onchange=()=>{filters[key]=s.value;page=0;render()};label.append(s);$('controls').append(label)}
function detail(r){$('text').textContent='';for(const [name,value] of [['Question ID',r.sample_id],['question',r.question],['Gold',r.answers.join(' | ')],['Teacher',r.teacher_prediction+' [score '+r.teacher_score+']'],['Student',r.student_prediction+' [score '+r.student_score+']'],['Loss',JSON.stringify(r.loss,null,2)],['Random matrix',JSON.stringify({family:r.family,ratio:r.ratio,seed:r.matrix_seed,replicate:r.replicate},null,2)]]){const b=document.createElement('b');b.textContent=name;const p=document.createElement('pre');p.textContent=value;$('text').append(b,p)}$('image').src='../../../'+r.image}
function render(){let selected=rows.filter(r=>Object.entries(filters).every(([k,v])=>v===''||String(r[k])===v)&&(!$('positive').checked||r.teacher_score>0));const sort=$('sort').value;selected.sort((a,b)=>sort==='drop'?b.score_drop-a.score_drop:sort==='lossasc'?a.loss.total-b.loss.total:b.loss.total-a.loss.total);page=Math.max(0,Math.min(page,Math.ceil(selected.length/100)-1));$('count').textContent=selected.length+'  independent compressions; page  '+(page+1)+' ; at most  100  rows. Click for the image, gold and complete answers.';$('body').textContent='';for(const r of selected.slice(page*100,page*100+100)){const tr=document.createElement('tr');for(const value of [r.task+' / '+r.sample_id,r.ratio+' / '+r.family+' / '+r.matrix_seed,r.loss.total.toFixed(5),r.teacher_score.toFixed(4),r.student_score.toFixed(4),r.score_drop.toFixed(4),r.student_prediction]){const td=document.createElement('td');td.textContent=value;tr.append(td)}tr.onclick=()=>detail(r);$('body').append(tr)}}$('positive').onchange=$('sort').onchange=()=>{page=0;render()};$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();detail(rows[0]);</script></html>'''
    (out/filename).write_text(template.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')))


def fmt(v):
    return 'NA' if v is None else f'{v:.3f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='random_v1')
    a = p.parse_args()
    out = ROOT/'artifacts/token_proxy'/a.label
    protocol = json.loads((out/'protocol.json').read_text())
    rows, teachers = [], []
    for path in sorted(out.glob('trials.rank*.jsonl')):
        rows.extend(map(json.loads, path.read_text().splitlines()))
    for path in sorted(out.glob('teachers.rank*.jsonl')):
        teachers.extend(map(json.loads, path.read_text().splitlines()))
    assert len(rows) == protocol['random_trials']+protocol['control_trials'], 'Not complete'
    assert len({r['id'] for r in rows}) == len(rows), 'Duplicate trial IDs'
    assert len(teachers) == protocol['questions']
    random_rows = [r for r in rows if r['kind'] == 'random']
    positive = [r for r in random_rows if r['teacher_score'] > 0]
    perfect = [r for r in random_rows if r['teacher_score'] == 1]
    summary = {'protocol': protocol, 'all_random': describe(random_rows, True),
               'teacher_positive': describe(positive, True), 'teacher_perfect': describe(perfect, True),
               'per_task': {t: describe([r for r in random_rows if r['task'] == t], True) for t in TASKS},
               'per_task_teacher_positive': {t: describe([r for r in positive if r['task'] == t], True) for t in TASKS},
               'per_ratio': {str(v): describe([r for r in random_rows if r['ratio'] == v]) for v in sorted({r['ratio'] for r in random_rows})},
               'per_family': {v: describe([r for r in random_rows if r['family'] == v]) for v in ('local', 'global')},
               'per_ratio_family': {f'{v}/{f}': describe([r for r in random_rows if r['ratio'] == v and r['family'] == f])
                                    for v in sorted({r['ratio'] for r in random_rows}) for f in ('local', 'global')},
               'within_question': within(random_rows, ['sample_id'], bootstrap=True),
               'within_question_ratio': within(random_rows, ['sample_id', 'ratio'], bootstrap=True),
               'within_question_ratio_family': within(random_rows, ['sample_id', 'ratio', 'family'], bootstrap=True),
               'partial_rank_question_and_ratio_family': partial_rank(random_rows),
               'paired_random_seed_ordering': paired_seed_ordering(random_rows),
               'component_ablation': {k: describe(random_rows, loss=k) for k in ('A_rec', 'B_rec_rel', 'C_rec_op', 'D_rec_rel_op', 'total', 'rec', 'rel', 'op', 'reg')},
               'component_within_question': {k: within(random_rows, ['sample_id'], loss=k) for k in ('A_rec', 'B_rec_rel', 'C_rec_op', 'D_rec_rel_op', 'total', 'rec', 'rel', 'op', 'reg')},
               'interface_checks': [r['interface_checks'] for r in teachers if 'interface_checks' in r],
               'prior_baseline_matches': sum(r['prior_baseline_match'] for r in teachers),
               'teacher_attention_entropy_fraction_range': [min(r['teacher_attention_entropy_fraction'] for r in teachers), max(r['teacher_attention_entropy_fraction'] for r in teachers)],
               'teacher_truncations': sum(r['runtime']['hit_token_limit'] for r in teachers),
               'student_truncations': sum(r['runtime']['hit_token_limit'] for r in rows),
               'matrix_column_relative_min': min(r['matrix_diagnostics']['column_mass_min_relative'] for r in rows),
               'matrix_column_relative_max': max(r['matrix_diagnostics']['column_mass_max_relative'] for r in rows),
               'matrix_max_non_singleton_weight': max(r['matrix_diagnostics']['non_singleton_max_entry'] for r in rows),
               'total_trial_seconds': sum(r['trial_seconds'] for r in rows)}
    summary['control_results'] = {str(rate): describe([r for r in rows if r['kind'] == 'mean_pool_control' and r['ratio'] == rate])
                                  for rate in (.25, .5)}
    temperature_path = out/'temperature_sensitivity.json'
    if temperature_path.exists():
        summary['exploratory_temperature_sensitivity'] = json.loads(temperature_path.read_text())
    fields = ['id', 'sample_id', 'task', 'image', 'question', 'answers', 'kind', 'family', 'ratio', 'replicate', 'matrix_seed',
              'teacher_prediction', 'teacher_score', 'student_prediction', 'student_score', 'score_drop']
    lossfields = ['total', 'rec', 'rel', 'op', 'reg', 'op_cos', 'op_l2', 'div', 'loc', 'cov', 'A_rec', 'B_rec_rel', 'C_rec_op', 'D_rec_rel_op']
    with (out/'trials.csv').open('w') as h:
        writer = csv.DictWriter(h, fieldnames=fields+['loss_'+k for k in lossfields])
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in fields}, **{'loss_'+k: r['loss'][k] for k in lossfields}})
    (out/'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    write_review(out, rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for ax, task in zip(axes.flat, TASKS):
        rr = [r for r in random_rows if r['task'] == task]
        for family, color in [('local', '#1576ad'), ('global', '#dc793a')]:
            subset = [r for r in rr if r['family'] == family]
            ax.scatter([r['loss']['total'] for r in subset], [r['score_drop'] for r in subset], s=12, alpha=.28, c=color, label=family, rasterized=True)
        ax.set_title(task+' | rho='+fmt(summary['per_task'][task]['spearman']))
        ax.set_xlabel('V1 proxy loss')
        ax.axhline(0, color='gray', lw=.5)
    axes[0, 0].set_ylabel('Teacher score - student score')
    axes[1, 0].set_ylabel('Teacher score - student score')
    axes[0, 0].legend()
    fig.suptitle('Each point is one random compression; no averaging over random trials')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'loss_vs_drop.{extension}', dpi=170)
    plt.close(fig)
    ratios = sorted({r['ratio'] for r in random_rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for family in ('local', 'global'):
        axes[0].plot(ratios, [summary['per_ratio_family'][f'{r}/{family}']['spearman'] for r in ratios], 'o-', label=family)
    axes[0].axhline(.7, ls='--', color='gray', label='predefined strong threshold')
    axes[0].set(xlabel='Retained token fraction', ylabel='Spearman(loss, score drop)', ylim=(-1, 1))
    axes[0].legend()
    keys = ['rec', 'rel', 'op', 'A_rec', 'B_rec_rel', 'C_rec_op', 'D_rec_rel_op', 'total']
    axes[1].bar(range(len(keys)), [summary['component_ablation'][k]['spearman'] or 0 for k in keys])
    axes[1].set_xticks(range(len(keys)), keys, rotation=35, ha='right')
    axes[1].set_ylabel('Spearman(loss, score drop)')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'stratified_and_ablation.{extension}', dpi=170)
    plt.close(fig)
    ordering_accuracy = summary['paired_random_seed_ordering']['ordering_accuracy_on_nonties']
    ordering_text = 'NA' if ordering_accuracy is None else f'{100*ordering_accuracy:.1f}%'
    main_rho = summary['all_random']['spearman']
    verdict = ('**This round does not establish strong correlation for V1.**' if main_rho is None or main_rho < .7
               else '**Pooled correlation meets the prespecified threshold; conditional analyses below remain necessary.**')
    protocol_doc = 'TOKEN_PROXY_20X.md' if a.label == 'random_v1_20x' else 'TOKEN_PROXY.md'
    lines = ['# Random Token Compression: Decoder-Free Loss-Proxy Validation', '',
             f"Completed **{len(teachers)}  questions /  {len({r['document_id'] for r in teachers})}  images, {len(random_rows)}  random compressions, {len(rows)-len(random_rows)}  mean-pooling controls**. Each compression retains its own answer/loss; downstream trials are never averaged before correlation.", '',
             f"Primary Spearman(loss, teacher score − student score) = **{fmt(summary['all_random']['spearman'])}**, image-cluster bootstrap 95% CI **{[round(v,3) for v in summary['all_random']['spearman_image_cluster_ci95']]}**. The prespecified strong-correlation threshold is 0.7.", '',
             f"{verdict} Within-question cross-configuration rank correlation:  {fmt(summary['within_question']['rank_residual_correlation'])}; controlling question and rate×family gives  {fmt(summary['partial_rank_question_and_ratio_family'])}; non-tied ordering accuracy between two random matrices at fixed question/retention/family:  {ordering_text}. Correlation findings apply to this random-matrix distribution; they prove neither inevitable success nor inevitable failure of training with the loss.", '',
             'Entry point: [Design and commands]('+protocol_doc+'); [Filterable per-trial answers](../artifacts/token_proxy/'+a.label+'/review.html); [Per-trial CSV](../artifacts/token_proxy/'+a.label+'/trials.csv); [Complete statistics JSON](../artifacts/token_proxy/'+a.label+'/analysis.json). ', '',
             '| Task | Images | Random trials | Spearman | Image-bootstrap 95% CI | Teacher>0 subset Spearman |',
             '|---|---:|---:|---:|---|---:|']
    for task in TASKS:
        s, ps = summary['per_task'][task], summary['per_task_teacher_positive'][task]
        ci = s['spearman_image_cluster_ci95']
        lines.append(f"| {task} | {s['images']} | {s['trials']} | {fmt(s['spearman'])} | {str([round(v,3) for v in ci]) if ci else 'NA'} | {fmt(ps['spearman'])} |")
    lines += ['', '## Controlling for Retention and Question Difficulty', '',
              f"- Teacher>0 subset: ρ={fmt(summary['teacher_positive']['spearman'])}; teacher=1 subset: ρ={fmt(summary['teacher_perfect']['spearman'])}. ",
              f"- Within-question centered-rank correlation: {fmt(summary['within_question']['rank_residual_correlation'])}. ",
              f"- Same question and retention: {fmt(summary['within_question_ratio']['rank_residual_correlation'])}. ",
              f"- Same question, retention, and family, comparing two random draws only: {fmt(summary['within_question_ratio_family']['rank_residual_correlation'])}; {summary['within_question_ratio_family']['strata_with_outcome_variation']}/{summary['within_question_ratio_family']['strata']}  paired groups change downstream score.",
              f"- Two-factor rank-residual correlation controlling for question and retention×family: {fmt(summary['partial_rank_question_and_ratio_family'])}. ", '',
              f"- Paired seeds for the same question, retention, and family: {summary['paired_random_seed_ordering']['concordant']}  pairs agree that higher loss means larger drop, {summary['paired_random_seed_ordering']['discordant']}  pairs disagree, {summary['paired_random_seed_ordering']['outcome_ties']}  downstream-score ties; non-tied ordering accuracy  {fmt(summary['paired_random_seed_ordering']['ordering_accuracy_on_nonties'])}. ", '',
              f"- Image-cluster bootstrap 95% CI for non-tied ordering accuracy: {summary['paired_random_seed_ordering']['ordering_accuracy_cluster_ci95']}; conditional-rank CIs: analysis.json within_question* fields.", '',
              '| Retention M/N | local ρ | global ρ | Pooled ρ |', '|---:|---:|---:|---:|']
    for r in ratios:
        lines.append(f"| {r} | {fmt(summary['per_ratio_family'][f'{r}/local']['spearman'])} | {fmt(summary['per_ratio_family'][f'{r}/global']['spearman'])} | {fmt(summary['per_ratio'][str(r)]['spearman'])} |")
    lines += ['', '## Loss-Component Ablations (Recombining the Same Independent Trials; No Retraining)', '',
              '| Component | pooled Spearman | Within-question rank correlation |', '|---|---:|---:|']
    for k, s in summary['component_ablation'].items():
        lines.append(f"| {k} | {fmt(s['spearman'])} | {fmt(summary['component_within_question'][k]['rank_residual_correlation'])} |")
    if 'exploratory_temperature_sensitivity' in summary:
        lines += ['', '## Exploratory Temperature Sensitivity (Prespecified Primary Results Unchanged)', '',
                  'Initial inspection found that dividing unit queries by sqrt(128) makes reads at τop=1 nearly uniform. Additional fixed temperatures τop∈{1,0.1,0.03} therefore replay the operator term on the same X/P samples, without running the decoder or choosing the best temperature as the formal loss. This was absent from the original preregistration and is exploratory diagnosis.', '',
                  '| τop | pooled Spearman | teacher>0 Spearman | Within-question rank correlation | Within-question, retention, and family rank correlation | Teacher entropy range |', '|---:|---:|---:|---:|---:|---|']
        for tau, s in summary['exploratory_temperature_sensitivity']['per_tau'].items():
            lines.append(f"| {tau} | {fmt(s['all_random']['spearman'])} | {fmt(s['teacher_positive']['spearman'])} | {fmt(s['within_question']['rank_residual_correlation'])} | {fmt(s['within_question_ratio_family']['rank_residual_correlation'])} | {[round(v,4) for v in s['entropy_range']]} |")
    lines += ['', '## Interface Checks and Scope', '',
              f"- {len(summary['interface_checks'])}  interface checks: native/cached prefill logits are elementwise equal; generated tokens match; identity-projector outputs and generations match.",
              f"- This round’s teacher matches the existing full official baseline: {summary['prior_baseline_matches']}/{len(teachers)}. ",
              f"- Truncated teacher responses:  {summary['teacher_truncations']}; truncated student responses  {summary['student_truncations']}(cap=256).",
              f"- Relative mean column-mass range [{summary['matrix_column_relative_min']:.4f}, {summary['matrix_column_relative_max']:.4f}]; largest weight in a nonsingleton row:  {summary['matrix_max_non_singleton_weight']:.4f}. ",
              f"- Normalized entropy range of operator teacher attention:  {summary['teacher_attention_entropy_fraction_range']}, checking whether random probes read almost uniformly.",
              '- Tests visual-only V1 from lossplus Section 12, excluding optional QA-aware terms requiring a trained text-query bridge. Compressor, projector, and LLM are not trained.',
              '- Per-question DeepForm/KLC correlation uses normalized exact match, not corpus document-set F1. Other tasks retain native per-question metrics.',
              f"- Images per task in this round:  {protocol['counts']}; correlation does not prove that training the loss will improve compression. ToMe, image downsampling, and QA-aware bridging were not tested.", '',
              '![Per-trial scatter](../artifacts/token_proxy/'+a.label+'/loss_vs_drop.png)', '',
              '![Stratification and ablations](../artifacts/token_proxy/'+a.label+'/stratified_and_ablation.png)', '']
    if not protocol['smoke']:
        lines += ['## Interpreting These Results', '',
                  'Prioritize same-question/retention/family ordering and CIs. Pooled correlation across budgets alone does not establish reliable answer preservation at low loss.', '']
        example_rows = sorted([r for r in random_rows if r['sample_id'] == 'ChartQA_test_1636' and r['family'] == 'local' and r['ratio'] == .5], key=lambda r: r['loss']['total'])
        if len(example_rows) == 2:
            lines += ['For example, local compression of ChartQA_test_1636 at 50%:', '',
                      '| Seed replicate | V1 loss | teacher / gold | student | Student score |', '|---:|---:|---|---|---:|']
            for r in example_rows:
                lines.append(f"| {r['replicate']} | {r['loss']['total']:.6f} | {r['teacher_prediction']} / {r['answers'][0]} | {r['student_prediction']} | {r['student_score']} |")
            lines += ['', 'Here lower-loss compression answers incorrectly. All seeds remain; the example does not change the statistical sample.', '']
        lines += [
                  'The operator probe at τ=1 is nearly uniform, a genuine numerical finding. Temperature replay shows that sharpening reads alone does not solve within-configuration ordering. V1 is still question-independent and cannot emphasize the evidence needed for a particular question. That mechanism requires independent follow-up; this round does not establish failure of the optional QA-aware variant.', '',
                  f"Additionally {summary['all_random']['student_zero_trials']}/{len(random_rows)}  random trials have zero student scores, and  {summary['paired_random_seed_ordering']['outcome_ties']}/{summary['paired_random_seed_ordering']['pairs']}  two-seed pairs tie, indicating saturation/ties. This matrix space intentionally excludes coverage collapse and does not represent all future learned compressors. Next test a question-conditioned query bridge and richer constrained random families on independent data before choosing an objective; those training tasks are not started.", '']
    artifact_text = '\n'.join(lines).replace('../artifacts/token_proxy/'+a.label+'/', '')
    artifact_text = artifact_text.replace('('+protocol_doc+')', '(../../../docs/'+protocol_doc+')')
    (out/'RESULTS.md').write_text(artifact_text)
    if not protocol['smoke']:
        docs_name = 'TOKEN_PROXY_20X_RESULTS.md' if a.label == 'random_v1_20x' else 'TOKEN_PROXY_RESULTS.md'
        (ROOT/'docs'/docs_name).write_text('\n'.join(lines))
    print(json.dumps({'all_random': summary['all_random'], 'within_question_ratio_family': summary['within_question_ratio_family'],
                      'report': str(out/'RESULTS.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
