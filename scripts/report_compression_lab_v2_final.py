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
    doc=['# New Compression Generators: Fixed-Endpoint Full Training and Formal Evaluation','',f'Updated (UTC): {timestamp}. Completed formal-evaluation budgets: {len(results)}/{len(budgets)}. ',
         '', '**Entry point**: `bash scripts/run_compression_lab_v2.sh status`; resume: `bash scripts/run_compression_lab_v2.sh resume`; recheck/rebuild report: `bash scripts/run_compression_lab_v2.sh final-report`. ',
         '', 'The official MonkeyOCRv2-B-Und encoder, merger, and Qwen3-1.7B are frozen; only P is trained. The 10%/50% variants use the question-conditioned tree and 25% uses global top-k, all compressing 768-dimensional encoder outputs before the merger, with student input merger(PH). Optimize only shared-teacher-prefix full-vocabulary KL, without row entropy or column-collapse regularization.',
         '', 'Full training at each budget uses the original 20,000 QA / 16,731 images, fresh initialization, 4 epochs / 2,500 updates, AdamW LR=3e-4, batch=32, clip=1, and 10% warmup followed by cosine decay. Epoch 4 is fixed; intermediate development or formal scores do not retrospectively select models. Formal evaluation uses the original eight-task 23,565-QA set with independent greedy generation capped at 256, retaining six variants per question without averaging compression trials or seeds. Only completed, coverage-audited results appear below.',
         '', 'Scores below are equally weighted native-task macros (percentages), not a unified official leaderboard. DeepForm/KLC use field exact match and DT-VQA its retained scorer; they are not the paper’s document-level F1.',
         '', '|Retention fraction|teacher|initial|trained|mean|selection|cell mean|cell selection|',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ratio in budgets:
        if ratio not in results:
            doc.append(f'|{float(ratio):.0%}|—|—|Formal evaluation incomplete|—|—|—|—|');continue
        _,s=results[ratio];doc.append(f'|{float(ratio):.0%}|'+ '|'.join(f'{s["macro"][v]:.2f}' for v in ['teacher',*VARIANTS])+'|')
    if {'0.1','0.25'}<=set(results):
        doc+=['', 'Both 10% and 25% improve over their initial models and same-position controls but remain well below teacher, so original performance is not recovered. Their generators differ as well as budgets; the gap is not a token-budget-only effect.',
              '', 'At 25%, full-training KL rises after epoch 1 and small-set generation falls after epoch 2. At 10%, validation KL falls but generation is nonmonotonic. Full evaluation still uses fixed epoch 4; neither these fluctuations nor tests select an earlier checkpoint.']
    if '0.5' in budgets:
        doc+=['', 'The 50% run is a separately authorized extension: on the original independent 112-image/209-QA confirmation set, the gain CI for seed 20261105 crosses zero, so the original requirement that both seeds pass remains failed. This extension retains the selected question-conditioned tree family and original first seed 20261105, with fresh full-20k training rather than continuing the better seed or a small-run checkpoint. See the [50% extension protocol](COMPRESSION_GENERATOR_V2_KEEP50_FULL.md)and [Original authorization and configuration hashes](../artifacts/compression_lab_v2/infrastructure/extension_keep50_20261106/registration.json). ']
    for ratio,(cfg,s) in results.items():
        m=s['macro'];ci=s['paired_trained_vs_initial']['cluster_bootstrap_95ci_points']
        best=max(m[v] for v in VARIANTS[2:]);directory='../'+cfg['output'];rel=directory+'/benchmark'
        doc+=['',f'## Retained Components{float(ratio):.0%} token','',
              f'Trained gain over initial: {s["paired_trained_vs_initial"]["gain_points"]:.2f} points; image-cluster bootstrap 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]; relative to the strongest same-position control: {m["trained"]-best:+.2f} points; relative to teacher {m["trained"]-m["teacher"]:+.2f} points; score retention {100*m["trained"]/m["teacher"]:.2f}%. The latter is not a speed-retention fraction.',
              '', '|epoch|Full-training batch-mean KL|Fixed training-audit KL|Development KL|Small selection-set generation score|',
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
        doc+=['', 'Full-training means cover all 20,000 QA in each epoch. Separate fixed audit/validation subsets have different sizes and aggregation weights; absolute values are not directly comparable. Screening uses 56 QA with a 64-token cap; formal evaluation uses 23,565 QA with cap=256. These are not a direct before/after comparison. The small screening set is not used to retrospectively select an epoch.',
              '', '|Task|QA|teacher|initial|trained|mean|selection|cell mean|cell selection|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        counts=collections.Counter(r['task'] for r in rows)
        for task,values in sorted(s['tasks'].items()):doc.append(f'|{task}|{counts[task]}|'+ '|'.join(f'{values[v]:.2f}' for v in ['teacher',*VARIANTS])+'|')
        doc+=['',f'[Per-question CSV]({rel}/answers.csv) · [Image/gold/teacher/initial/trained viewer]({rel}/review.html) · [Machine-readable statistics]({rel}/summary.json) · [Fixed epoch-4 weights]({directory}/epoch4.pt). ',
              '',f'[Full-training / fixed-audit / validation KL curves](../artifacts/compression_lab_v2/plots/loss_v2_full_20k_pre-merger_keep{float(ratio):g}.pdf) · [Per-epoch generation curves](../artifacts/compression_lab_v2/plots/score_v2_full_20k_pre-merger_keep{float(ratio):g}.pdf). Full-training batch means and fixed audit/validation sets differ; inspect their trends separately.',
              '',f'Questions reaching cap=256: `{json.dumps(s["student_token_limit_counts"],ensure_ascii=False)}`; teacher: {s["teacher_token_limit_count"]} questions, all included in scoring.']
    doc+=['','## Recovery and Verification','',
          'Resolve stale Slurm job IDs against the current user\'s live queue; persist stage transitions first. Full training/evaluation allow at most 3 automatic recoveries. Cache ENOENT permits at most 12 refresh/backoff attempts. Install torch.load patches after importing Ascend so the backend cannot replace them; concurrent EEXIST accepts only committed caches with identical tensors and metadata. Genuine missing files or mismatched content still fail.',
          '', 'The earlier 10% job logged 463 steps without a checkpoint. After preserving its logs, replay uses the original seed and sample order, with latest.pt saved every 25 steps. Recovery from step 150 succeeded; initial tensors match exactly. The first replayed update matches the original log, but subsequent NPU trajectories are not bit-exact; complete determinism is not claimed. Completed 25% weights were not retrained.',
          '', 'Evaluation parallelism only allocates independent questions. Before resharding, archive old manifests/records; migrate committed answers unchanged and verify canonical hashes. No score-based reruns or selection. Hardware plans/audits remain in recovery directories.',
          '', '[Complete protocol, per-epoch curves, and confirmation](COMPRESSION_GENERATOR_V2_EXPERIMENTS.md) · [Generator architecture](COMPRESSION_GENERATOR_V2_DESIGNS.md) · [Recovery records](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/recovery.json) · [Final per-question completeness audit](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/final_artifact_audit.json). ']
    target=ROOT/'docs/COMPRESSION_GENERATOR_V2_FINAL_RESULTS.md';tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
    print(json.dumps({'completed_budgets':list(results),'audit':audit},ensure_ascii=False))


if __name__=='__main__':main()
