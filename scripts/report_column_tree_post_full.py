"""Audit and merge all eight fixed column/tree experiments, including live progress."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path

from scripts.column_tree_post_full_suite import ROOT, BASE, SUITE, REGISTRATION, STATE, load, save, sha, rows, now, verify, benchmark_rows
from vqa.benchmark_metrics import native_score

VARIANTS = ['initial', 'trained', 'mean', 'selection', 'cell_mean', 'cell_selection']
LABEL = {'col_adaptive': '自适应列划分', 'quadtree_question': '问题条件空间树'}
REPORT = ROOT / 'docs/COLUMN_TREE_POST_MERGER_FULL_COMPARISON.md'


def jsonlines(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    result = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return result


def training_info(record):
    out = ROOT / record['output']
    rr = jsonlines(out / 'training.jsonl')
    steps = [r['step'] for r in rr]
    assert steps == list(range(1, len(steps) + 1)), f'Duplicate or skipped training updates: {record["name"]}'
    complete = (out / 'complete.json').exists()
    by_epoch = collections.defaultdict(list)
    for r in rr:
        assert all(math.isfinite(r[k]) for k in ['kl', 'total', 'grad_norm', 'update_l2'])
        assert r['update_l2'] > 0 and r['epoch'] == (r['step'] - 1) // 625 + 1
        by_epoch[r['epoch']].append(r['kl'])
    epochs = []
    for e in range(5):
        stage = 'initial' if e == 0 else f'epoch{e}'
        val = load(out / f'validation.{stage}.summary.json')
        audit = load(out / f'train_audit.{stage}.summary.json')
        if val:
            assert val['rows'] == 56
        if audit:
            assert audit['rows'] == 14
        values = by_epoch[e]
        epochs.append(dict(epoch=e, updates=len(values), train_batch_kl=sum(values) / len(values) if values else None,
                           validation_kl=val.get('kl'), audit_kl=audit.get('kl'), validation_score=val.get('student_macro'),
                           teacher_validation_score=val.get('teacher_macro')))
    if complete:
        done = load(out / 'complete.json')
        assert len(steps) == 2500 and done['steps'] == 2500 and done['epochs'] == 4
        assert done['trainable_backbone'] == 0 and done['padded_samples_per_epoch'] == 0
        assert all(len(by_epoch[e]) == 625 for e in range(1, 5))
        assert all(x['validation_kl'] is not None and x['audit_kl'] is not None for x in epochs)
        assert (out / 'epoch4.pt').exists()
    # Count only committed newline records, not an unfinished final JSON line.
    parts = sorted((out / 'benchmark').glob('predictions.rank*.jsonl'))
    count = 0
    for p in parts:
        with p.open('rb') as f:
            count += sum(line.endswith(b'\n') for line in f)
    return dict(logged_updates=len(steps), complete=complete, epochs=epochs, benchmark_records=count)


def audit_benchmark(record, expected, teacher_reference):
    out = ROOT / record['output']
    bench = out / 'benchmark'
    if not (bench / 'complete.json').exists():
        return None
    parts = sorted(bench.glob('predictions.rank*.jsonl'))
    manifest_paths = sorted(bench.glob('manifest.rank*.json'))
    sig = dict(checkpoint_sha256=sha(out / 'epoch4.pt'), initial_sha256=sha(out / 'initial.pt'),
               summary_sha256=sha(bench / 'summary.json'),
               manifest_sha256={p.name: sha(p) for p in manifest_paths},
               prediction_stats={p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in parts})
    cached_path = SUITE / 'audits' / f'{record["name"]}.json'
    cached = load(cached_path)
    if cached.get('signature') == sig and cached.get('passed'):
        return load(bench / 'summary.json')
    cfg = load(ROOT / record['config'])
    summary = load(bench / 'summary.json')
    assert summary['config'] == cfg and summary['questions'] == 23565 and summary['cap'] == 256
    assert summary['benchmark_used_for_selection'] is False
    manifests = [load(p) for p in manifest_paths]
    assert manifests and all(m == manifests[0] for m in manifests)
    assert len(manifests) == manifests[0]['world'] == len(parts)
    assert manifests[0]['checkpoint_sha256'] == sig['checkpoint_sha256']
    assert manifests[0]['initial_sha256'] == sig['initial_sha256']
    for source, digest in manifests[0]['source_sha256'].items():
        assert sha(ROOT / source) == digest
    train_manifest = load(out / 'manifest.json')
    assert train_manifest['config'] == cfg and train_manifest['trainable_backbone'] == 0
    reg = load(REGISTRATION)
    for k, digest in train_manifest['data_sha256'].items():
        assert digest == reg['dataset'][k]['sha256']
    seen = set()
    scores = collections.defaultdict(list)
    teachers = {}
    limit = collections.Counter()
    max_error = 0.
    for p in parts:
        for r in jsonlines(p):
            key = (r['id'], r['variant'])
            assert key not in seen and r['variant'] in VARIANTS
            seen.add(key)
            source = expected[r['id']]
            for field in ['document_id', 'task', 'question', 'answers']:
                assert r[field] == source[field]
            assert r['ratio'] == record['ratio']
            assert r['compressed_tokens'] == max(1, round(record['ratio'] * r['tokens']))
            assert len(r['student_ids']) <= 256 and 0 <= r['row_sum_error'] < 1e-4
            max_error = max(max_error, r['row_sum_error'])
            assert abs(native_score(r['student_prediction'], source) - r['student_score']) < 1e-12
            teacher = (r['teacher_prediction'], r['teacher_score'])
            assert teacher == teacher_reference[r['id']]
            assert abs(native_score(r['teacher_prediction'], source) - r['teacher_score']) < 1e-12
            teachers[r['id']] = teacher[1]
            scores[(r['task'], r['variant'])].append(r['student_score'])
            limit[r['variant']] += bool(r['hit_token_limit'])
    assert seen == {(key, v) for key in expected for v in VARIANTS}
    for task, metrics in summary['tasks'].items():
        for variant in VARIANTS:
            values = scores[(task, variant)]
            assert abs(100 * sum(values) / len(values) - metrics[variant]) < 1e-8
        tt = [teachers[k] for k, r in expected.items() if r['task'] == task]
        assert abs(100 * sum(tt) / len(tt) - metrics['teacher']) < 1e-8
    assert set(summary['tasks']) == {r['task'] for r in expected.values()}
    for variant in ['teacher', *VARIANTS]:
        assert abs(sum(t[variant] for t in summary['tasks'].values()) / 8 - summary['macro'][variant]) < 1e-8
    assert abs(summary['macro']['trained'] - summary['macro']['initial'] - summary['paired_trained_vs_initial']['gain_points']) < 1e-8
    save(cached_path, dict(passed=True, audited_utc=now(), signature=sig, questions=23565, records=len(seen),
                          teacher_matches_legacy10=True, scores_recomputed=True, unique_coverage=True,
                          question_gold_identity=True, budget_and_generation_cap=True, max_row_sum_error=max_error,
                          source_and_checkpoint_hashes=True, token_limit_counts=dict(limit)))
    return summary


def browser(record):
    out = ROOT / record['output'] / 'benchmark'
    if (out / 'review.html').exists():
        return
    from urllib.parse import quote
    from scripts.review_compression_lab_v2_benchmark import TEMPLATE
    source = {r['id']: r for r in benchmark_rows()}
    initial, trained = {}, {}
    for p in sorted(out.glob('predictions.rank*.jsonl')):
        for r in jsonlines(p):
            if r['variant'] == 'initial':
                initial[r['id']] = r
            elif r['variant'] == 'trained':
                trained[r['id']] = r
    assert len(initial) == len(trained) == 23565 and initial.keys() == trained.keys()
    payload = []
    for key, r in trained.items():
        b = initial[key]
        payload.append(dict(id=key, task=r['task'], question=r['question'], answers=r['answers'],
                            teacher=r['teacher_prediction'], initial=b['student_prediction'], trained=r['student_prediction'],
                            teacher_score=r['teacher_score'], initial_score=b['student_score'], trained_score=r['student_score'],
                            gain=r['student_score'] - b['student_score'], tokens=r['tokens'], compressed_tokens=r['compressed_tokens'],
                            hit_token_limit=r['hit_token_limit'], image=quote(os.path.relpath(ROOT / source[key]['image'], out))))
    html = TEMPLATE.replace('BUDGET', f'{record["ratio"]:.0%} ({record["kind"]})').replace('PAYLOAD', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    tmp = out / f'review.{os.getpid()}.tmp'
    tmp.write_text(html)
    tmp.replace(out / 'review.html')


def plot(record, info):
    signature = [{k: v for k, v in e.items() if k != 'train_batch_kl'} | {'complete_train_kl': e['train_batch_kl'] if e['updates'] == 625 else None} for e in info['epochs']]
    out = SUITE / 'plots'
    out.mkdir(exist_ok=True)
    stamp = out / f'{record["name"]}.json'
    if load(stamp, []) == signature:
        return
    if not any(e['validation_kl'] is not None for e in info['epochs']):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    e = info['epochs']
    for key, label in [('audit_kl', 'historical audit KL'), ('validation_kl', 'validation KL')]:
        vv = [x for x in e if x[key] is not None]
        axes[0].plot([x['epoch'] for x in vv], [x[key] for x in vv], 'o-', label=label)
    vv = [x for x in e if x['updates'] == 625]
    axes[0].plot([x['epoch'] for x in vv], [x['train_batch_kl'] for x in vv], 's-', label='full train epoch mean KL')
    vv = [x for x in e if x['validation_score'] is not None]
    axes[1].plot([x['epoch'] for x in vv], [x['validation_score'] for x in vv], 'o-', label='student validation macro')
    if vv:
        axes[1].axhline(vv[0]['teacher_validation_score'], ls=':', label='uncompressed teacher')
    for ax in axes:
        ax.set_xlabel('epoch'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    axes[0].set_ylabel('teacher-prefix KL'); axes[1].set_ylabel('7-task validation macro (%)')
    fig.suptitle(f'{record["kind"]}, post-merger, keep {record["ratio"]:.0%}, seed20261105')
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'{record["name"]}.{ext}', dpi=150)
    plt.close(fig)
    save(stamp, signature)


def smoke_report(state):
    done = load(SUITE / 'smoke/complete.json')
    doc = ['## 启动与技术检查过程', '', f"当前控制阶段：`{state.get('suite_stage', 'registered')}`。", '',
           'Merger前八组完整评测及审计结束后才提交本轮NPU任务。每个配置先在独立进程中检查P非负、行和为1、压缩数与预算一致、输入Qwen特征严格等于PX、梯度非零且有限、主干冻结和参数实际更新；检查权重不用于全量初始化。', '',
           '两道训练题与两道固定验证题上的技术检查loss/生成分仅用于记录，不是多图泛化结论，也不用于改超参数或选择配置。此前144组筛查已包含这两种结构的post-merger 10%/25%/50%实验。', '']
    if done:
        doc += [f"八配置技术检查通过，作业 {done['job']}。", '', '|生成器|保留率|训练KL前→后|2题验证KL前→后|2题验证分前→后|梯度范数|', '|---|---:|---:|---:|---:|---:|']
        for r in done['results']:
            doc.append(f"|{LABEL[r['kind']]}|{r['ratio']:.0%}|{r['train_before']['kl']:.5f}→{r['train_after']['kl']:.5f}|{r['validation_before']['kl']:.5f}→{r['validation_after']['kl']:.5f}|{r['validation_before']['student_macro']:.2f}→{r['validation_after']['student_macro']:.2f}|{r['gradient_norm']:.5f}|")
        doc.append('')
    else:
        doc += ['技术检查尚未完成；未报告为已通过。', '']
    return doc


def position_comparison(reg, results):
    pre = load(BASE / 'column_tree_full_20261108/merged_results.json').get('results', {})
    doc = ['## 与merger前的配对位置对照', '',
           '两位置使用同样的数据、seed、训练步数和固定epoch4。前位置生成器的X槽为0；后位置允许读取原始H及merger后的X，因此差异包含插入位置和其可用输入信息的共同影响，不称为仅线性算子顺序的严格消融。官方vision构造器实际向PatchMerger传spatial_merge_size=1（config.json中的2用于空间顺序），逐token做768→2048映射，两个位置token数一致；预算为max(1, round(ratio×N))。', '',
           '|保留率|生成器|前：八任务均分|后：八任务均分|后−前|前：DocVQA|后：DocVQA|', '|---|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: '—' if x is None else f'{x:.2f}'
    for record in sorted(reg['runs'], key=lambda r: (r['ratio'], r['kind'])):
        a = pre.get(record['name'].replace('_post-merger_', '_pre-merger_'))
        b = results.get(record['name'])
        av = a['macro']['trained'] if a else None
        bv = b['macro']['trained'] if b else None
        doc.append(f"|{record['ratio']:.0%}|{LABEL[record['kind']]}|{fmt(av)}|{fmt(bv)}|{fmt(bv-av if a and b else None)}|{fmt(a['tasks']['DocVQA']['trained'] if a else None)}|{fmt(b['tasks']['DocVQA']['trained'] if b else None)}|")
    return doc + ['']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    reg = verify()
    state = load(STATE)
    expected = {r['id']: r for r in benchmark_rows()}
    assert len(expected) == 23565
    # Historical teacher is the fixed reference, not a newly trained model.
    teacher_reference = {}
    reference = BASE / 'full/full_quadtree_question_pre-merger_r0.1/benchmark'
    for p in reference.glob('predictions.rank*.jsonl'):
        for r in jsonlines(p):
            if r['variant'] == 'trained':
                teacher_reference[r['id']] = (r['teacher_prediction'], r['teacher_score'])
    assert teacher_reference.keys() == expected.keys()
    results, progress = {}, {}
    for record in reg['runs']:
        info = training_info(record)
        progress[record['name']] = info
        summary = audit_benchmark(record, expected, teacher_reference)
        if summary:
            assert info['complete']
            results[record['name']] = summary
            browser(record)
        plot(record, info)
    fmt = lambda x, n=2: '—' if x is None else f'{x:.{n}f}'
    doc = ['# Merger后压缩：自适应列划分与问题条件空间树全量对照', '',
           f'更新UTC：{now()}。正式结果已完成且审计通过 {len(results)}/8 组；全部8组本轮新训练。', '',
           '**入口**：状态/更新报告 `.venv/bin/python -m scripts.column_tree_post_full_suite status`；中断恢复 `.venv/bin/python -m scripts.column_tree_post_full_suite resume`；最终核验 `.venv/bin/python -m scripts.report_column_tree_post_full --require-complete`。', '',
           '## 实验设计与纠正', '',
           '本轮在token保留率10%、25%、50%、75%比较col_adaptive与quadtree_question，全部固定post-merger：encoder H → 原始merger X → PX → Qwen。生成器输入为归一化的H(768维)与X(2048维)，不是merger(PH)。列方案不使用问题计算P；问题条件树使用当前问题前64个token的冻结Qwen输入词embedding。代码里的树是矩形递归二分，非标准四子节点quadtree。', '',
           '本轮按用户要求，在merger前八组评测完成并通过审计后启动；两结构×四预算均从同一seed新初始化，不使用任何merger前生成器权重。原merger前结果与作答文件保持独立。', '',
           '本轮为用户指定的固定位置对照，不按正式分数选择架构、seed或checkpoint；先进行八配置真实UND前向/反向技术检查，再完整训练。技术检查不筛选较好方案、不复用其权重。正式集合此前已被查看，本轮不称为新的盲测。', '',
           '## 实际生成器与压缩结构', '',
           '共同骨架：H与X分别做无仿射LayerNorm，拼接2816→96线性+GELU，再接96通道3×3 depthwise/pointwise残差卷积；重要性MLP为96→96→1，末层权重std=.01。区域内使用温度τ=.25+softplus(t)的masked softmax生成P，初始τ约.943；P非负、每行和为1，输出只能是PX，latent不能直接替代视觉值。', '',
           '自适应列划分：以encoder局部高频细节乘以学习的重要性构造分区能量，优先纵向二分，高或宽耗尽时切换方向，得到恰好M个非空、不相交矩形；每个区域内部学习凸组合权重。它不使用问题计算P。', '',
           '问题条件空间树：在同一骨架上增加问题词embedding的2048→96映射及视觉word_key，取归一化视觉/词特征最大相似度加入重要性；按区域长轴递归二分得到M个矩形，并学习各区域内权重。虽代码名quadtree，实际为二叉矩形空间树。问题来自当前题目，teacher答案、gold和其他问答上下文都不输入P生成器。', '',
           '两方案的离散分区决策均detach，不宣称整棵树可微；梯度通过区域内softmax和重要性网络。二维操作遵循官方block顺序，输出按P行最大权重所对应的原token序号排序。详细算法与文献来源沿用[第二轮结构文档](COMPRESSION_GENERATOR_V2_DESIGNS.md)；本轮不新增架构。', '',
           '## 数据、训练与评测配置', '',
           '- 原20,000QA / 16,731图像，保持文件、样本顺序和逐epoch shuffle规则；不重新抽样、不做任务平衡。',
           '- 原固定验证56QA/56图像、历史审计14QA/7图像。历史审计仅2图属于20k训练集，保持原集不变；全训练KL另按每epoch全部batch汇总。',
           '- 正式评测严格复用23,565QA：DocVQA/InfoVQA取原val的对应任务，加原test的全部任务；不把未使用的原val任务混入。训练/验证/正式集按document_id核对无交集。',
           '- 官方MonkeyOCRv2-B-Und encoder、原merger、Qwen3-1.7B全冻结，只训练P生成器；每组fresh初始化，seed20261105，无小规模权重warm start。',
           '- 每组4epochs、625更新/epoch、共2500更新；AdamW LR3e-4、wd0、batch32、clip1；250步warmup，cosine至峰值LR的10%。8NPU，每rank累积4题，无padding。',
           '- KL-only、T=1、teacher||student；教师自身greedy回答作共同前缀。训练教师与小验证生成cap64；正式greedy生成cap256。无行熵/列质量正则；不是decoder-free loss。',
           '- 每25更新提交latest与optimizer；每epoch固定保存并验证，最终固定epoch4。正式逐题保留initial/trained/mean/selection/cell_mean/cell_selection六项，每组141,390条，不跨seed或压缩采样取平均。',
           '- CPU使用.venv，NPU使用.venv-und；沿用已验证不可变cache读写修复，生成器、loss、原模型推理和scorer源码不变。正式评测8个独立worker，与旧14worker分片不同，但问题/生成设置一致。', '',
           '正式分数为八任务原生指标等权宏平均（百分制）；DeepForm/KLC为字段exact，DT-VQA沿用既有scorer，不能当作论文官方文档级F1或统一榜单。单seed完整训练；图像bootstrap不表示训练seed方差。', '',
           '[不可变实验登记、配置哈希与数据审计](../artifacts/compression_lab_v2/column_tree_post_full_20261108/registration.json) · [实时机器状态](../artifacts/compression_lab_v2/column_tree_post_full_20261108/state.json) · [逐组结果审计](../artifacts/compression_lab_v2/column_tree_post_full_20261108/audits/)。', '',
           '## 八组训练与正式结果总表', '',
           '|保留率|生成器|来源/阶段|训练更新|正式记录|teacher|initial|trained|teacher分数保留率|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for record in sorted(reg['runs'], key=lambda x: (x['ratio'], x['kind'])):
        name = record['name']; info = progress[name]; result = results.get(name); mm = result['macro'] if result else {}
        stage = '历史复用，已审计' if record['reuse'] else state.get('runs', {}).get(name, {}).get('stage', 'registered')
        if result and not record['reuse']:
            stage = '正式完成，已审计'
        retained = 100 * mm['trained'] / mm['teacher'] if mm else None
        doc.append(f'|{record["ratio"]:.0%}|{LABEL[record["kind"]]}|{stage}|{info["logged_updates"]}/2500|{info["benchmark_records"]}/141390|{fmt(mm.get("teacher"))}|{fmt(mm.get("initial"))}|{fmt(mm.get("trained"))}|{fmt(retained)}|')
    legacy = load(BASE / 'full/full_global_topk_pre-merger_r0.25/benchmark/summary.json')
    doc += ['', f'历史25% global_topk单列参考：teacher {legacy["macro"]["teacher"]:.2f}，initial {legacy["macro"]["initial"]:.2f}，trained {legacy["macro"]["trained"]:.2f}。它不进入问题条件树的预算曲线。', '',
            '“—”表示尚无完成且审计通过的正式结果，不能用56QA小验证或筛选分替代。', '', '## 各预算逐任务正式对比', '']
    tasks = sorted({r['task'] for r in expected.values()})
    for ratio in [.1, .25, .5, .75]:
        cc = results.get(f'full_col_adaptive_post-merger_r{ratio:g}')
        tt = results.get(f'full_quadtree_question_post-merger_r{ratio:g}')
        doc += [f'### 保留{ratio:.0%}', '', '|任务|teacher|列initial|列trained|树initial|树trained|列trained−树trained|', '|---|---:|---:|---:|---:|---:|---:|']
        for task in ['宏平均', *tasks]:
            c = (cc['macro'] if task == '宏平均' else cc['tasks'][task]) if cc else {}
            t = (tt['macro'] if task == '宏平均' else tt['tasks'][task]) if tt else {}
            teacher = (c or t).get('teacher')
            delta = c['trained'] - t['trained'] if c and t else None
            doc.append(f'|{task}|{fmt(teacher)}|{fmt(c.get("initial"))}|{fmt(c.get("trained"))}|{fmt(t.get("initial"))}|{fmt(t.get("trained"))}|{fmt(delta)}|')
        doc.append('')
    doc += position_comparison(reg, results)
    doc += smoke_report(state)
    doc += ['## 逐组配置、逐epoch loss与生成分', '']
    for record in reg['runs']:
        name = record['name']; info = progress[name]; cfg = load(ROOT / record['config'])
        doc += [f'### {LABEL[record["kind"]]} / 保留{record["ratio"]:.0%}', '',
                f'[完整配置](../{record["config"]})；seed {cfg["seed"]}，post-merger，固定epoch4。', '',
                '|epoch|已记录更新|全训练batch平均KL|历史审计KL|验证KL|小验证生成宏平均|', '|---|---:|---:|---:|---:|---:|']
        for e in info['epochs']:
            doc.append(f'|{e["epoch"]}|{e["updates"]}/625|{fmt(e["train_batch_kl"],5)}|{fmt(e["audit_kl"],5)}|{fmt(e["validation_kl"],5)}|{fmt(e["validation_score"])}|')
        doc += ['', 'epoch0为初始化，不包含训练更新；未满625更新的均值是当前部分epoch。不同集合的KL绝对数值不可直接比较；小验证cap64不等于正式cap256。', '']
        curve = SUITE / 'plots' / f'{name}.pdf'
        if curve.exists():
            doc += [f'[训练/审计/验证KL与验证生成分曲线](../{curve.relative_to(ROOT)})。', '']
        out = ROOT / record['output']
        if name in results:
            mm = results[name]['macro']; pair = results[name]['paired_trained_vs_initial']; ci = pair['cluster_bootstrap_95ci_points']
            doc += [f'正式训练增益 {pair["gain_points"]:+.2f} 点，图像聚类95%CI [{ci[0]:.2f}, {ci[1]:.2f}]；相对teacher {mm["trained"]-mm["teacher"]:+.2f} 点。', '',
                    f'[逐题CSV](../{record["output"]}/benchmark/answers.csv) · [图像/gold/teacher/initial/trained作答对比](../{record["output"]}/benchmark/review.html) · [机器结果](../{record["output"]}/benchmark/summary.json) · [epoch4权重](../{record["output"]}/epoch4.pt)。', '']
    doc += ['## 恢复与完成规则', '',
            'CPU控制器独立于交互会话，最多同时保持2个NPU阶段作业，按固定顺序训练并自动进入全量评测，随后进行逐题覆盖、gold、teacher、评分复算、配置/代码/权重哈希审计，生成CSV、作答HTML与曲线并更新本页。', '',
            '每阶段最多3次基础设施恢复，保留原seed、8NPU分片与checkpoint，不放宽科学协议；超过次数保留失败状态与日志，等待诊断。完成标志、逐题覆盖与审计全部通过后才写正式分数。全部merger前状态文件、权重、正式作答均不覆盖。', '',
            '任务未完成时可用resume重新启动控制器，已有作业会保留。控制器及所有阶段日志位于artifacts/compression_lab_v2/column_tree_post_full_20261108/。']
    tmp = REPORT.with_name(REPORT.name + f'.{os.getpid()}.tmp')
    tmp.write_text('\n'.join(doc) + '\n')
    tmp.replace(REPORT)
    save(SUITE / 'merged_results.json', dict(updated_utc=now(), completed=len(results), planned=8, results=results, progress=progress))
    if args.require_complete:
        assert len(results) == 8 and all(p['complete'] for p in progress.values()), 'Some registered runs are not yet fully trained/evaluated/audited'
    print(json.dumps(dict(report=str(REPORT), formal_completed=len(results), planned=8,
                          logged_updates={k: v['logged_updates'] for k, v in progress.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
