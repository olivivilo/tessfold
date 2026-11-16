"""Validate completed fixed-endpoint benchmarks and write the concise final report."""
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path
from vqa.benchmark_metrics import native_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/compression_lab_v2'
VARIANTS=['initial','trained','mean','selection','cell_mean','cell_selection']


def read(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def main():
    state=json.loads((BASE/'stages.json').read_text())['ratios']
    budgets=[r for r in ['0.1','0.25','0.5'] if state.get(r,{}).get('full_config')]
    rows=[r for r in read(ROOT/'data/full/val.jsonl') if r['task'] in ['DocVQA','InfoVQA']]+list(read(ROOT/'data/full/test.jsonl'))
    expected={r['id']:r for r in rows};assert len(expected)==23565
    teachers={};results={};audits={}
    for ratio in budgets:
        cfg=json.loads((ROOT/state[ratio]['full_config']).read_text());out=ROOT/cfg['output']
        bench=out/'benchmark'
        list(out.iterdir())
        try:list(bench.iterdir())
        except FileNotFoundError:continue
        if not (bench/'complete.json').exists():continue
        summary=json.loads((bench/'summary.json').read_text())
        assert summary['questions']==23565 and summary['cap']==256 and not summary['benchmark_used_for_selection']
        assert (out/'complete.json').exists() and (out/'epoch4.pt').exists()
        assert summary['config']==cfg
        manifests=[json.loads(p.read_text()) for p in bench.glob('manifest.rank*.json')]
        assert manifests and all(m==manifests[0] for m in manifests)
        assert len(manifests)==manifests[0]['world']
        assert manifests[0]['checkpoint_sha256']==hashlib.sha256((out/'epoch4.pt').read_bytes()).hexdigest()
        assert manifests[0]['initial_sha256']==hashlib.sha256((out/'initial.pt').read_bytes()).hexdigest()
        for source, expected_hash in manifests[0]['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected_hash
        seen=set();scores=collections.defaultdict(list);limit=collections.Counter();teacher_scores={};max_error=0.
        for p in sorted(bench.glob('predictions.rank*.jsonl')):
            for r in read(p):
                key=(r['id'],r['variant']);assert key not in seen and r['variant'] in VARIANTS
                seen.add(key);source=expected[r['id']]
                for field in ['document_id','task','question','answers']:assert r[field]==source[field]
                assert r['ratio']==float(ratio)
                assert r['compressed_tokens']==max(1,round(float(ratio)*r['tokens']))
                assert len(r['student_ids'])<=256 and r['row_sum_error']<1e-4
                max_error=max(max_error,r['row_sum_error'])
                assert 0<=r['student_score']<=1 and 0<=r['teacher_score']<=1
                assert abs(native_score(r['student_prediction'],source)-r['student_score'])<1e-12
                teacher=(r['teacher_prediction'],r['teacher_score'])
                if r['id'] in teachers:assert teachers[r['id']]==teacher
                else:assert abs(native_score(r['teacher_prediction'],source)-r['teacher_score'])<1e-12
                teachers[r['id']]=teacher;teacher_scores[r['id']]=r['teacher_score']
                scores[(r['task'],r['variant'])].append(r['student_score'])
                limit[r['variant']]+=bool(r['hit_token_limit'])
        assert seen=={(key,v) for key in expected for v in VARIANTS}
        for task,metrics in summary['tasks'].items():
            for v in VARIANTS:
                values=scores[(task,v)]
                assert abs(100*sum(values)/len(values)-metrics[v])<1e-8
            values=[teacher_scores[r['id']] for r in rows if r['task']==task]
            assert abs(100*sum(values)/len(values)-metrics['teacher'])<1e-8
        for v in ['teacher',*VARIANTS]:
            assert abs(sum(t[v] for t in summary['tasks'].values())/len(summary['tasks'])-summary['macro'][v])<1e-8
        audits[ratio]={'questions':len(expected),'variant_records':len(seen),'variants':VARIANTS,
                       'final_benchmark_world':manifests[0]['world'],
                       'fixed_epoch4_sha256':hashlib.sha256((out/'epoch4.pt').read_bytes()).hexdigest(),
                       'coverage_unique':True,'source_question_gold_match':True,'cap_and_budget_match':True,
                       'scores_recomputed_from_predictions':True,'scientific_source_hashes_match':True,
                       'macro_recomputed':True,'max_row_sum_error':max_error,'token_limit_counts':dict(limit)}
        results[ratio]=(cfg,summary)
    if '--require-complete' in sys.argv:
        assert set(results)==set(budgets), 'All registered formal benchmark artifacts must be visible and valid'
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    audit={'updated_utc':timestamp,'budgets':audits,'teacher_consistent_across_variants_and_completed_budgets':True}
    (BASE/'infrastructure/recovery_20261106/final_artifact_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    doc=['# 新压缩生成器：固定终点完整训练与正式评测','',f'更新 UTC：{timestamp}。已完成正式评测预算：{len(results)}/{len(budgets)}。',
         '', '**入口**：`bash scripts/run_compression_lab_v2.sh status`；恢复：`bash scripts/run_compression_lab_v2.sh resume`；重新核对并生成本报告：`bash scripts/run_compression_lab_v2.sh final-report`。',
         '', '官方 MonkeyOCRv2-B-Und encoder、merger、Qwen3-1.7B 全冻结，只训练P生成器。10%与50%使用问题条件四叉树，25%使用全局top-k，均在merger前压缩768维encoder输出，学生输入为merger(PH)。仅优化共同teacher前缀上的全词表KL；不加行熵或列塌缩正则。',
         '', '各预算的完整训练设置均为：原20,000 QA / 16,731图像，从头训练4epochs、2,500updates；AdamW LR3e-4、batch32、clip1、10%warmup及cosine。固定epoch4，不按中间验证或正式分数追选。正式集合为原有23,565QA、八任务，独立greedy生成上限256；每题保留六种变体结果，没有跨随机压缩或seed取均值。下表仅填写完成标志和完整性审计通过的正式结果。',
         '', '下表为各任务原生分数的等权宏平均（百分制），不是官方统一榜单。DeepForm/KLC为逐字段exact，DT-VQA沿用既有scorer，不能将这些分数冒充论文的官方文档级F1。',
         '', '|保留率|teacher|initial|trained|mean|selection|cell mean|cell selection|',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ratio in budgets:
        if ratio not in results:
            doc.append(f'|{float(ratio):.0%}|—|—|未完成正式评测|—|—|—|—|');continue
        _,s=results[ratio];doc.append(f'|{float(ratio):.0%}|'+ '|'.join(f'{s["macro"][v]:.2f}' for v in ['teacher',*VARIANTS])+'|')
    if {'0.1','0.25'}<=set(results):
        doc+=['', '10%与25%均相对各自初始化和本位置的固定控制提高，但仍明显低于未压缩teacher，不能视为已恢复原模型性能。10%与25%采用不同生成器，二者分差同时包含预算和结构差异，不能归因于token预算单一因素。',
              '', '25%全训练KL在epoch1之后上升，小验证集生成分在epoch2后回落；10%验证KL下降，但逐epoch生成分不单调。固定epoch4照常全量评测，没有根据这些波动或测试分数改选较早权重。']
    if '0.5' in budgets:
        doc+=['', '50%为用户明确授权的后续扩展：原112图/209QA独立确认中，seed20261105增益CI跨零，原“两seed都通过”门槛仍记为失败。本次沿用已经选定的问题条件四叉树、原第一seed20261105，从头训练完整20k，不从较好seed或小规模checkpoint继续训练。登记见[50%扩展协议](COMPRESSION_GENERATOR_V2_KEEP50_FULL.md)及[原始授权与配置哈希](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/registration.json)。']
    for ratio,(cfg,s) in results.items():
        m=s['macro'];ci=s['paired_trained_vs_initial']['cluster_bootstrap_95ci_points']
        best=max(m[v] for v in VARIANTS[2:]);directory='../'+cfg['output'];rel=directory+'/benchmark'
        doc+=['',f'## 保留{float(ratio):.0%} token','',
              f'trained相对initial提高{s["paired_trained_vs_initial"]["gain_points"]:.2f}点，按图像聚类bootstrap95%CI [{ci[0]:.2f}, {ci[1]:.2f}]；相对本位置最强固定控制为{m["trained"]-best:+.2f}点，相对teacher为{m["trained"]-m["teacher"]:+.2f}点，分数保留率{100*m["trained"]/m["teacher"]:.2f}%。后者不是速度保留率。',
              '', '|epoch|全训练batch平均KL|固定训练审计KL|验证KL|小筛选集生成分|',
              '|---|---:|---:|---:|---:|']
        by_epoch=collections.defaultdict(list);run=ROOT/cfg['output'];steps=[]
        for r in read(run/'training.jsonl'):
            by_epoch[r['epoch']].append(r['kl']);steps.append(r['step'])
        assert sorted(steps)==list(range(1,2501))
        for epoch in range(1,5):
            values=by_epoch[epoch];assert len(values)==625
            val=json.loads((run/f'validation.epoch{epoch}.summary.json').read_text())
            train=json.loads((run/f'train_audit.epoch{epoch}.summary.json').read_text())
            doc.append(f'|{epoch}|{sum(values)/len(values):.5f}|{train["kl"]:.5f}|{val["kl"]:.5f}|{val["student_macro"]:.2f}|')
        doc+=['', '全训练均值覆盖该epoch全部20,000QA；固定审计/验证为另列的小集合，绝对数值和加权口径不同。小筛选集为56QA、生成上限64 token；正式评测为23,565QA、上限256 token，样本与生成设置不同，不能把两者直接当作训练前后对照。小筛选集不用于追选epoch。',
              '', '|任务|QA|teacher|initial|trained|mean|selection|cell mean|cell selection|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        counts=collections.Counter(r['task'] for r in rows)
        for task,values in sorted(s['tasks'].items()):doc.append(f'|{task}|{counts[task]}|'+ '|'.join(f'{values[v]:.2f}' for v in ['teacher',*VARIANTS])+'|')
        doc+=['',f'[逐题CSV]({rel}/answers.csv) · [图像/gold/teacher/initial/trained浏览器]({rel}/review.html) · [机器统计]({rel}/summary.json) · [固定epoch4权重]({directory}/epoch4.pt)。',
              '',f'[完整训练/固定训练审计/验证 KL 曲线](../artifacts/compression_lab_v2/plots/loss_v2_full_20k_pre-merger_keep{float(ratio):g}.pdf) · [逐epoch生成分数曲线](../artifacts/compression_lab_v2/plots/score_v2_full_20k_pre-merger_keep{float(ratio):g}.pdf)。全训练batch均值与固定审计/验证集合不同，分别观察趋势。',
              '',f'达到256-token上限的题数：`{json.dumps(s["student_token_limit_counts"],ensure_ascii=False)}`；teacher为{s["teacher_token_limit_count"]}题，全部计入评分。']
    doc+=['','## 恢复与复核','',
          '过期Slurm作业号通过查询当前用户队列处理，阶段转换先落盘；完整训练/评测最多自动恢复3次。缓存ENOENT最多12次刷新/退避，先导入Ascend再安装torch.load补丁，避免后端覆盖；并发写入EEXIST只接受逐tensor及metadata相同的已提交缓存。真实缺失和内容不一致仍报错。',
          '', '10%旧作业463步没有提交checkpoint，保留旧日志后按原seed/样本顺序重放。增加每25步保存latest.pt；一次补丁部署从step150恢复成功，初始权重逐tensor一致。首个重放更新与原日志一致，后续NPU数值轨迹非逐位一致，未宣称完全确定性。25%完成权重未重训。',
          '', '评测并行度只影响独立问题的分配。若重分片，旧manifest及逐题记录先归档，所有已提交作答原样迁移并核对canonical记录hash；不根据作答分数重跑或选模型。具体硬件记录见恢复目录中的计划与重分片audit。',
          '', '[完整实验协议、逐epoch曲线与确认结果](COMPRESSION_GENERATOR_V2_EXPERIMENTS.md) · [生成器结构](COMPRESSION_GENERATOR_V2_DESIGNS.md) · [恢复记录](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/recovery.json) · [最终逐题完整性审计](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/final_artifact_audit.json)。']
    target=ROOT/'docs/COMPRESSION_GENERATOR_V2_FINAL_RESULTS.md';tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
    print(json.dumps({'completed_budgets':list(results),'audit':audit},ensure_ascii=False))


if __name__=='__main__':main()
