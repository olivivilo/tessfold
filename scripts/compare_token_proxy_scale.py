"""Scale comparison plus direct rerun agreement on the original fixed samples."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_records(out, prefix):
    result = {}
    for path in out.glob(prefix+'.rank*.jsonl'):
        with path.open() as handle:
            for r in map(json.loads, handle):
                assert r['id'] not in result
                result[r['id']] = r
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='random_v1_20x')
    a = p.parse_args()
    base = ROOT/'artifacts/token_proxy'
    old, new = base/'random_v1', base/a.label
    pa, pb = (json.loads((d/'protocol.json').read_text()) for d in (old,new))
    scientific_keys = ['sample_seed','tasks','trial_configs','smoke','model','adapter','max_new_tokens','max_pixels',
                       'do_sample','dtype','matrix_uniform_mix','matrix_global_floor','loss_version','loss_constants','analysis','lossplus_sha256']
    assert all(pa[k] == pb[k] for k in scientific_keys)
    original_code = json.loads((old/'run.rank0.json').read_text())['code_sha256']
    new_manifests = [json.loads(path.read_text()) for path in new.glob('run.rank*.json')]
    assert new_manifests and all(m['code_sha256'] == original_code for m in new_manifests)
    protocol_audit = {'scientific_settings_identical': True, 'compared_keys': scientific_keys,
                      'core_code_sha256_identical': True, 'core_code_sha256': original_code,
                      'new_inference_ranks': len(new_manifests),
                      'old_protocol_sha256': hashlib.sha256((old/'protocol.json').read_bytes()).hexdigest(),
                      'new_protocol_sha256': hashlib.sha256((new/'protocol.json').read_bytes()).hexdigest(),
                      'old_counts': pa['counts'], 'new_counts': pb['counts'],
                      'old_questions': pa['questions'], 'new_questions': pb['questions'],
                      'actual_scale_factor': pb['questions']/pa['questions']}
    old_teachers, new_teachers = read_records(old, 'teachers'), read_records(new, 'teachers')
    assert old_teachers.keys() <= new_teachers.keys()
    matches = {k: 0 for k in ['prediction','score','feature_sha256']}
    for rid, r in old_teachers.items():
        for key in matches:
            matches[key] += r[key] == new_teachers[rid][key]
    old_trials, new_trials = read_records(old, 'trials'), read_records(new, 'trials')
    assert old_trials.keys() <= new_trials.keys()
    trial_matches = {k: 0 for k in ['matrix_seed','feature_sha256','compressed_feature_sha256','teacher_prediction','student_prediction','teacher_score','student_score','score_drop']}
    max_loss_diff = 0.
    mismatches = []
    for rid, r in old_trials.items():
        s = new_trials[rid]
        for key in trial_matches:
            trial_matches[key] += r[key] == s[key]
        diff = max(abs(r['loss'][key]-s['loss'][key]) for key in r['loss'])
        max_loss_diff = max(max_loss_diff, diff)
        if r['student_prediction'] != s['student_prediction']:
            mismatches.append({'id': rid, 'old': r['student_prediction'], 'new': s['student_prediction']})
    sa, sb = (json.loads((d/'analysis.json').read_text()) for d in (old, new))
    rows = []
    for title, keys in [('Overall Spearman',['all_random','spearman']),
                        ('Teacher>0 Spearman',['teacher_positive','spearman']),
                        ('Teacher=1 Spearman',['teacher_perfect','spearman']),
                        ('Within question rank correlation',['within_question','rank_residual_correlation']),
                        ('Within question/rate/family',['within_question_ratio_family','rank_residual_correlation']),
                        ('Partial rank question/rate/family',['partial_rank_question_and_ratio_family']),
                        ('Paired seed ordering accuracy',['paired_random_seed_ordering','ordering_accuracy_on_nonties'])]:
        def value(source):
            for key in keys:
                source = source[key]
            return source
        rows.append({'metric': title, 'original': value(sa), 'expanded': value(sb)})
    result = {'protocol_audit': protocol_audit, 'original_teachers_rerun': len(old_teachers),
              'teacher_matches': matches, 'original_trials_rerun': len(old_trials),
              'trial_matches': trial_matches, 'max_loss_difference': max_loss_diff,
              'prediction_mismatches': mismatches, 'summary_comparison': rows,
              'per_task_comparison': {task: {'original_spearman': sa['per_task'][task]['spearman'],
                                            'expanded_spearman': sb['per_task'][task]['spearman'],
                                            'expanded_ci95': sb['per_task'][task]['spearman_image_cluster_ci95']}
                                      for task in sa['per_task']}}
    (new/'scale_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = ['# 扩大图像规模的结果对比', '',
             f"原实验 {pa['questions']} 张；扩大后 {pb['questions']} 张，实际 {pb['questions']/pa['questions']:.4f} 倍。每任务目标 480 张，DeepForm/WTQ 的评测图像不足，使用全部 300/421 张。", '',
             '所有科学参数、矩阵边界、随机种子规则、loss、推理和统计方法保持一致。每图仍选一题，每个随机压缩独立保留；扩大后的样本包含原先全部图像，重叠部分也重新运行，没有复用旧预测。', '',
             '| 指标 | 原规模 | 扩大规模 |', '|---|---:|---:|']
    for r in rows:
        fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
        lines.append(f"| {r['metric']} | {fmt(r['original'])} | {fmt(r['expanded'])} |")
    lines += ['', '| 任务 | 原 Spearman | 扩大后 Spearman | 扩大后图像 bootstrap 95% CI |', '|---|---:|---:|---|']
    for task, r in result['per_task_comparison'].items():
        lines.append(f"| {task} | {r['original_spearman']:.4f} | {r['expanded_spearman']:.4f} | {[round(v,4) for v in r['expanded_ci95']]} |")
    lines += ['', '## 重叠样本的独立重跑一致性', '',
              f"- 原先 {len(old_teachers)} 个 teacher 均重新生成：作答相同 {matches['prediction']}，视觉特征 hash 相同 {matches['feature_sha256']}。",
              f"- 原先 {len(old_trials)} 个压缩试次均重新运行：student 作答相同 {trial_matches['student_prediction']}，压缩特征 hash 相同 {trial_matches['compressed_feature_sha256']}。",
              f"- 全部 loss 组成项最大差异：{max_loss_diff:.9g}。",
              '- 每任务图像数受到数据量上限影响，合并统计的任务权重与原先略有不同，应结合分任务结果解读。', '',
              '[扩大规模完整报告](TOKEN_PROXY_20X_RESULTS.md)；[逐次作答](../artifacts/token_proxy/'+a.label+'/review.html)；[原规模报告](TOKEN_PROXY_RESULTS.md)。', '']
    (ROOT/'docs/TOKEN_PROXY_SCALE_COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('prediction_mismatches','per_task_comparison')},indent=2))


if __name__ == '__main__':
    main()
