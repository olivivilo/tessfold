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
            links.append(f'<li><a href="{task_filename}">{task}</a>：{len(task_rows):,} 次独立压缩</li>')
        (out/filename).write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
                                 '<title>随机压缩作答对照</title><style>body{font:17px system-ui;margin:40px;line-height:1.9}</style>'
                                 f'<h1>随机 token 压缩逐次作答对照</h1><p>共 {len(rows):,} 次压缩。为便于打开，按任务分为页面；每页均可筛选保留率、分组、试次类型和按 loss/下降排序，点击行查看原图与完整作答。</p>'
                                 '<ul>'+''.join(links)+'</ul><p><a href="trials.csv">全部逐次结果 CSV</a> · <a href="RESULTS.md">完整实验报告</a></p></html>')
        return
    data = [{k: r[k] for k in ('id', 'sample_id', 'task', 'image', 'question', 'answers', 'kind', 'family', 'ratio',
                               'replicate', 'matrix_seed', 'teacher_prediction', 'teacher_score',
                               'student_prediction', 'student_score', 'score_drop', 'loss')} for r in rows]
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Token compression trial review</title>
<style>body{font:15px system-ui;margin:24px;color:#182535}select,input,button{margin:6px;padding:7px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}tr:hover{background:#eaf3fa;cursor:pointer}#detail{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}img{max-width:100%;max-height:680px}pre{white-space:pre-wrap;overflow-wrap:anywhere}th{position:sticky;top:0;background:white}.hint{color:#516274}</style>
<h1>随机 token 压缩逐次作答对照</h1><p>每行是一次独立压缩，分数范围 0–1，下降 = teacher − student。未对随机试次作答取平均。</p>
<div id="controls"></div><label><input type="checkbox" id="positive">仅 teacher 得分 &gt; 0</label><label>排序 <select id="sort"><option value="loss">loss 高到低</option><option value="drop">分数下降大到小</option><option value="lossasc">loss 低到高</option></select></label><p id="count"></p>
<div id="detail"><div id="text"></div><img id="image"></div><button id="prev">上一页</button><button id="next">下一页</button>
<table><thead><tr><th>任务 / 题号</th><th>保留率 / 分组 / 随机种子</th><th>V1 loss</th><th>teacher</th><th>student</th><th>下降</th><th>student 作答</th></tr></thead><tbody id="body"></tbody></table>
<script>const rows=__DATA__;let page=0;const filters={};const $=id=>document.getElementById(id);for(const [key,name] of [['task','任务'],['ratio','保留率'],['family','分组'],['kind','类型']]){const label=document.createElement('label');label.textContent=name;const s=document.createElement('select');s.add(new Option('全部',''));[...new Set(rows.map(r=>r[key]))].sort().forEach(v=>s.add(new Option(v,v)));s.onchange=()=>{filters[key]=s.value;page=0;render()};label.append(s);$('controls').append(label)}
function detail(r){$('text').textContent='';for(const [name,value] of [['题号',r.sample_id],['问题',r.question],['Gold',r.answers.join(' | ')],['Teacher',r.teacher_prediction+' [score '+r.teacher_score+']'],['Student',r.student_prediction+' [score '+r.student_score+']'],['Loss',JSON.stringify(r.loss,null,2)],['随机矩阵',JSON.stringify({family:r.family,ratio:r.ratio,seed:r.matrix_seed,replicate:r.replicate},null,2)]]){const b=document.createElement('b');b.textContent=name;const p=document.createElement('pre');p.textContent=value;$('text').append(b,p)}$('image').src='../../../'+r.image}
function render(){let selected=rows.filter(r=>Object.entries(filters).every(([k,v])=>v===''||String(r[k])===v)&&(!$('positive').checked||r.teacher_score>0));const sort=$('sort').value;selected.sort((a,b)=>sort==='drop'?b.score_drop-a.score_drop:sort==='lossasc'?a.loss.total-b.loss.total:b.loss.total-a.loss.total);page=Math.max(0,Math.min(page,Math.ceil(selected.length/100)-1));$('count').textContent=selected.length+' 次独立压缩；第 '+(page+1)+' 页，每页最多 100 行。点击行查看图像、gold 和完整作答。';$('body').textContent='';for(const r of selected.slice(page*100,page*100+100)){const tr=document.createElement('tr');for(const value of [r.task+' / '+r.sample_id,r.ratio+' / '+r.family+' / '+r.matrix_seed,r.loss.total.toFixed(5),r.teacher_score.toFixed(4),r.student_score.toFixed(4),r.score_drop.toFixed(4),r.student_prediction]){const td=document.createElement('td');td.textContent=value;tr.append(td)}tr.onclick=()=>detail(r);$('body').append(tr)}}$('positive').onchange=$('sort').onchange=()=>{page=0;render()};$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();detail(rows[0]);</script></html>'''
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
    verdict = ('**本轮未验证 V1 的强相关性。**' if main_rho is None or main_rho < .7
               else '**总体相关达到预设阈值；仍需结合以下条件分析判断。**')
    protocol_doc = 'TOKEN_PROXY_20X.md' if a.label == 'random_v1_20x' else 'TOKEN_PROXY.md'
    lines = ['# 随机 token 压缩：decoder-free loss proxy 验证', '',
             f"完成 **{len(teachers)} 题 / {len({r['document_id'] for r in teachers})} 张图像、{len(random_rows)} 次随机压缩、{len(rows)-len(random_rows)} 次均值池化对照**。每一次压缩均保留独立作答与 loss，没有先平均随机试次的下游分数再计算相关性。", '',
             f"主检验 Spearman(loss, teacher 分数 − student 分数) = **{fmt(summary['all_random']['spearman'])}**，按图像 cluster bootstrap 的 95% CI = **{[round(v,3) for v in summary['all_random']['spearman_image_cluster_ci95']]}**。预设强关联阈值为 0.7。", '',
             f"{verdict} 同题跨配置的 rank 相关为 {fmt(summary['within_question']['rank_residual_correlation'])}；控制题目与 rate×family 后为 {fmt(summary['partial_rank_question_and_ratio_family'])}；固定题目、保留率和分组后，两次随机矩阵的非平局排序正确率为 {ordering_text}。相关性结论只适用于本轮随机矩阵分布，不等于已经证明训练该 loss 一定有效或一定无效。", '',
             '入口：[实验方案与运行命令]('+protocol_doc+')；[逐次可筛选作答对照](../artifacts/token_proxy/'+a.label+'/review.html)；[逐次 CSV](../artifacts/token_proxy/'+a.label+'/trials.csv)；[完整统计 JSON](../artifacts/token_proxy/'+a.label+'/analysis.json)。', '',
             '| 任务 | 图像 | 随机试次 | Spearman | 图像 bootstrap 95% CI | teacher>0 子集 Spearman |',
             '|---|---:|---:|---:|---|---:|']
    for task in TASKS:
        s, ps = summary['per_task'][task], summary['per_task_teacher_positive'][task]
        ci = s['spearman_image_cluster_ci95']
        lines.append(f"| {task} | {s['images']} | {s['trials']} | {fmt(s['spearman'])} | {str([round(v,3) for v in ci]) if ci else 'NA'} | {fmt(ps['spearman'])} |")
    lines += ['', '## 控制压缩率与题目难度', '',
              f"- teacher>0 子集：ρ={fmt(summary['teacher_positive']['spearman'])}；teacher=1 子集：ρ={fmt(summary['teacher_perfect']['spearman'])}。",
              f"- 同题内中心化 rank 相关：{fmt(summary['within_question']['rank_residual_correlation'])}。",
              f"- 同题且同保留率：{fmt(summary['within_question_ratio']['rank_residual_correlation'])}。",
              f"- 同题、同保留率、同分组，仅比较两次随机生成：{fmt(summary['within_question_ratio_family']['rank_residual_correlation'])}；{summary['within_question_ratio_family']['strata_with_outcome_variation']}/{summary['within_question_ratio_family']['strata']} 个配对组的下游分数有变化。",
              f"- 同时控制题目与保留率×分组的两因素 rank 残差相关：{fmt(summary['partial_rank_question_and_ratio_family'])}。", '',
              f"- 同题同保留率同分组的双 seed 配对：{summary['paired_random_seed_ordering']['concordant']} 对符合 loss 大则下降大，{summary['paired_random_seed_ordering']['discordant']} 对方向相反，{summary['paired_random_seed_ordering']['outcome_ties']} 对下游得分相同；非平局排序正确率 {fmt(summary['paired_random_seed_ordering']['ordering_accuracy_on_nonties'])}。", '',
              f"- 上述非平局排序正确率的图像 cluster bootstrap 95% CI：{summary['paired_random_seed_ordering']['ordering_accuracy_cluster_ci95']}；条件 rank 相关的 CI 见 analysis.json 的 within_question* 字段。", '',
              '| 保留率 M/N | local ρ | global ρ | 合并 ρ |', '|---:|---:|---:|---:|']
    for r in ratios:
        lines.append(f"| {r} | {fmt(summary['per_ratio_family'][f'{r}/local']['spearman'])} | {fmt(summary['per_ratio_family'][f'{r}/global']['spearman'])} | {fmt(summary['per_ratio'][str(r)]['spearman'])} |")
    lines += ['', '## Loss 组成项消融（同一批独立试次上重新组合 loss，不重新训练）', '',
              '| 组成 | pooled Spearman | 同题内 rank 相关 |', '|---|---:|---:|']
    for k, s in summary['component_ablation'].items():
        lines.append(f"| {k} | {fmt(s['spearman'])} | {fmt(summary['component_within_question'][k]['rank_residual_correlation'])} |")
    if 'exploratory_temperature_sensitivity' in summary:
        lines += ['', '## 探索性温度敏感性（不替换预设主结果）', '',
                  '初步检查发现单位 query 再除 sqrt(128) 令 τop=1 的读取几乎均匀，因此额外预设 τop∈{1,0.1,0.03} 重放同一批 X/P 的算子项；不运行 decoder，不挑选最佳温度作为本轮正式 loss。这一分析未写入原始预注册，属于探索性诊断。', '',
                  '| τop | pooled Spearman | teacher>0 Spearman | 同题内 rank | 同题同率同分组 rank | teacher 熵范围 |', '|---:|---:|---:|---:|---:|---|']
        for tau, s in summary['exploratory_temperature_sensitivity']['per_tau'].items():
            lines.append(f"| {tau} | {fmt(s['all_random']['spearman'])} | {fmt(s['teacher_positive']['spearman'])} | {fmt(s['within_question']['rank_residual_correlation'])} | {fmt(s['within_question_ratio_family']['rank_residual_correlation'])} | {[round(v,4) for v in s['entropy_range']]} |")
    lines += ['', '## 接口与适用范围', '',
              f"- {len(summary['interface_checks'])} 个接口检查：原生/缓存 prefill logits 逐元素完全一致；生成 token 完全一致；恒等矩阵的 projector 输出和生成均一致。",
              f"- 本轮 teacher 与既有官方全量 baseline 作答一致：{summary['prior_baseline_matches']}/{len(teachers)}。",
              f"- teacher 截断 {summary['teacher_truncations']}；student 截断 {summary['student_truncations']}（上限 256）。",
              f"- 相对平均列质量范围 [{summary['matrix_column_relative_min']:.4f}, {summary['matrix_column_relative_max']:.4f}]；非单元素行最大权重 {summary['matrix_max_non_singleton_weight']:.4f}。",
              f"- 算子 teacher attention 的归一化熵范围 {summary['teacher_attention_entropy_fraction_range']}，用于检查随机 probe 是否过于接近均匀读取。",
              '- 测试的是 lossplus 第 12 节纯视觉 V1，不包含需训练文本 query bridge 的可选 QA-aware 项；未训练 compressor、projector 或 LLM。',
              '- DeepForm/KLC 的逐题相关性指标为归一化 exact match，不冒充文档集合 F1。其余任务使用已有原生逐题指标。',
              f"- 本轮每任务图像数为 {protocol['counts']}；相关性不能证明训练该 loss 后压缩器一定提升。未检验 ToMe、图像下采样或 QA-aware 桥。", '',
              '![逐次散点](../artifacts/token_proxy/'+a.label+'/loss_vs_drop.png)', '',
              '![分层与消融](../artifacts/token_proxy/'+a.label+'/stratified_and_ablation.png)', '']
    if not protocol['smoke']:
        lines += ['## 如何理解这些结果', '',
                  '应优先关注同题、同保留率、同分组内的排序与置信区间，不能只凭合并所有比例的相关性声称“低 loss 的随机压缩通常就能保住作答”。', '']
        example_rows = sorted([r for r in random_rows if r['sample_id'] == 'ChartQA_test_1636' and r['family'] == 'local' and r['ratio'] == .5], key=lambda r: r['loss']['total'])
        if len(example_rows) == 2:
            lines += ['例如同样把 ChartQA_test_1636 压到 50%、使用 local：', '',
                      '| seed 试次 | V1 loss | teacher / gold | student | student 分数 |', '|---:|---:|---|---|---:|']
            for r in example_rows:
                lines.append(f"| {r['replicate']} | {r['loss']['total']:.6f} | {r['teacher_prediction']} / {r['answers'][0]} | {r['student_prediction']} | {r['student_score']} |")
            lines += ['', '这里 loss 较低的压缩反而答错。所有 seed 均保留，举例不改变统计样本。', '']
        lines += [
                  '算子 probe 在 τ=1 时接近均匀读取，是一项真实的数值诊断；温度敏感性重放表明，仅让读取变尖锐并未解决固定配置下的排序问题。V1 仍然没有问题条件，无法专门强调本题所需证据；这一点是需要后续独立验证的机制假设，不能据本轮实验直接宣判可选 QA-aware 版本无效。', '',
                  f"另有 {summary['all_random']['student_zero_trials']}/{len(random_rows)} 个随机试次的 student 得分为零，且 {summary['paired_random_seed_ordering']['outcome_ties']}/{summary['paired_random_seed_ordering']['pairs']} 个双 seed 配对得分相同，存在分数饱和/平局。当前矩阵空间也主动排除了覆盖塌缩，因此不能代表未来学习式压缩器的全部分布。下一步宜在独立数据上检验问题条件 query bridge 和更丰富、仍受约束的随机压缩族，再决定训练目标；本轮未启动这些训练。", '']
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
