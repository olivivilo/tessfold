"""Final Markdown and CSV; refuses incomplete full training/benchmark artifacts."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import statistics
from scripts.report_meta_compress_search import ROOT,BASE,read,collect,paired,average


def main():
    state=json.loads((BASE/'controller.json').read_text());cfgpath=ROOT/state['full_config'];cfg=json.loads(cfgpath.read_text())
    run=ROOT/cfg['output'];bench=run/'benchmark'
    assert (run/'complete.json').exists(),'Training is incomplete'
    completion=json.loads((run/'complete.json').read_text());assert completion['rows']==20000 and completion['steps']==1112
    for rank in range(32):
        assert (bench/f'complete.rank{rank}.json').exists()
        assert (bench/f'teacher_check.rank{rank}.json').exists()
    records=[r for f in bench.glob('predictions.rank*.jsonl') for r in read(f)]
    expected={r['id'] for r in read(bench/'samples.jsonl')}
    keys=[(r['id'],r['variant']) for r in records]
    assert len(keys)==len(set(keys)) and set(keys)=={(i,v) for i in expected for v in ['initial','trained','mean']}
    assert len(expected)==512
    fresh_teacher={}
    for r in records:
        value=(r['teacher_prediction'],r['teacher_generated_ids'],r['teacher_score'],r['original_teacher_ids_equal'])
        if r['id'] in fresh_teacher:assert fresh_teacher[r['id']]==value
        fresh_teacher[r['id']]=value
    teacher_ids_equal=sum(v[3] for v in fresh_teacher.values())
    variants={v:[r for r in records if r['variant']==v] for v in ['initial','trained','mean']}
    tasks=sorted({r['task'] for r in records});table={}
    for t in tasks:
        table[t]={}
        for v,rr in variants.items():
            rows=[r for r in rr if r['task']==t]
            table[t][v]={'n':len(rows),'score':100*average([r['student_score'] for r in rows]),
                'teacher':100*average([r['teacher_score'] for r in rows]),
                'KL':average([r['prediction'] for r in rows]),'C_fraction':average([r['column_mass_fraction'] for r in rows])}
    macro={v:average([table[t][v]['score'] for t in tasks]) for v in variants}
    teacher=average([table[t]['trained']['teacher'] for t in tasks])
    changes={v:paired(variants[v],variants['trained']) for v in ['initial','mean']}
    old_records=[r for f in (ROOT/'artifacts/meta_compress/eval').glob('predictions.benchmark.rank*.jsonl') for r in read(f) if r['variant']=='trained']
    old_macro=average([100*average([r['student_score'] for r in old_records if r['task']==t]) for t in tasks])
    pilot=state['selection']['candidate'];confirmation=state['selection']['confirmation'];confmean=state['selection']['matched_mean']
    selected=collect(BASE/f"runs/{pilot['name']}");search=selected['stages'][f"epoch{pilot['epoch']}/trained"]
    pilot_training_rows=read(ROOT/selected['config']['train_rows'])
    pilot_training_images=len({r['document_id'] for r in pilot_training_rows})
    search_images=len({r['document_id'] for r in read(BASE/'search.jsonl')})
    confirm_images=len({r['document_id'] for r in read(BASE/'confirm.jsonl')})
    train=read(run/'training.jsonl');assert len(train)==1112 and {r['step'] for r in train}==set(range(1,1113))
    epochs=[]
    for ep in [1,2]:
        rows=[r for r in train if r['epoch']==ep]
        epochs.append({'epoch':ep,'steps':len(rows),**{k:average([r[k] for r in rows]) for k in ['prediction','entropy','collapse','objective','grad_norm','clip_multiplier','update_l2']},
            'clipped_fraction':average([float(r['clip_multiplier']<1) for r in rows])})
    summary={'complete':True,'full_config':str(cfgpath.relative_to(ROOT)),'pilot':pilot,'confirmation':confirmation,
        'benchmark_samples':512,'benchmark_records':len(records),'teacher_macro':teacher,'student_macros':macro,
        'old_10pct_trained_macro':old_macro,'matched_paired_changes':changes,'tasks':table,'epochs':epochs,
        'teacher_retention':macro['trained']/teacher,'fresh_teacher_questions':len(fresh_teacher),'teacher_old_exact_ids_equal':teacher_ids_equal}
    (BASE/'FINAL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    fields=['task','id','document_id','variant','question','answers','teacher_prediction','student_prediction','teacher_score','student_score','prediction','entropy','collapse','objective','tokens','compressed_tokens']
    with (BASE/'final_answers.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in records:writer.writerow({**row,'answers':json.dumps(row['answers'],ensure_ascii=False)})
    structure=('Retain the original dense-generator architecture.' if not cfg.get('local_bias') else
        f"**Adds a fixed local logit bias absent from the paper’s generator (a substantial structural difference)**: let  `g(j)=floor(j*m/n)`, in original attention logits, for  `g(j) != i` positions minus  {cfg['local_bias']:g}, then row softmax. Constraints use contiguous groups in official token order; each row mainly combines n/m tokens while retaining nonzero weights for every source. This adds a fixed prior without trainable parameters; gains are not optimizer/clipping-only effects.")
    weightnote=('All three weights remain 1 as in the paper.' if cfg['entropy_weight']==cfg['collapse_weight']==1 else
        f"**Loss weights differ from the paper**: entropy={cfg['entropy_weight']:g}, collapse={cfg['collapse_weight']:g}; all weights are 1 in the paper.")
    text=f'''# MetaCompress: Multi-Image Search, Independent Confirmation, and Full Original-20k Retraining

**Real pilot, independent confirmation, 2-epoch retraining on the original 20,000 training QA, and fixed 512-question/eight-task evaluation are complete.** Final token retention: **{cfg['ratio']:.0%}**; teacher / matched-architecture initialization / mean / trained evaluation macros: **{teacher:.2f} / {macro['initial']:.2f} / {macro['mean']:.2f} / {macro['trained']:.2f}**. Paired trained-versus-same-budget-mean change: **{changes['mean']['gain_points']:+.2f}** percentage points; 95% bootstrap CI: **{changes['mean']['paired_bootstrap_95ci_points']}**. These are observations, not attainment of the paper’s compression performance.

Entry point: [Experimental design](METACOMPRESS_TRAINING_SEARCH.md); [All candidates by stage](../artifacts/meta_compress_search/RESULTS.search.md); [Independent confirmation results](../artifacts/meta_compress_search/RESULTS.confirm.md); [Per-question teacher/student/gold CSV](../artifacts/meta_compress_search/final_answers.csv); [Machine-readable final results](../artifacts/meta_compress_search/FINAL_SUMMARY.json).

## Final Method and Differences from the Paper

Use official  `MonkeyOCRv2-B-Und` , freezing encoder, official merger, and Qwen3-1.7B. Compress 2,048-dimensional post-merger visual tokens. Original generator: learned positional encoding → fractional max query pooling (kernel=3) → 64-dimensional Q/K and channel weights → row softmax →  `P @ X`; 458,816 trainable parameters. Main-text average pooling differs from supplementary fractional max; this implementation uses the latter. Learnable row/column positional tables are supplied without claiming identity to undisclosed author code.

{structure}

Actual training objective: `KL(teacher || student) + {cfg['entropy_weight']:g} * mean_row_entropy(P) + {cfg['collapse_weight']:g} * max_column_mass(P)`. {weightnote} Full-vocabulary KL at T=1 uses shared teacher-generated prefixes, never gold. Students independently generate per question for validation. P has unit row sums and compressed tokens are convex combinations.

Final optimizer=`{cfg['optimizer']}`, peak LR=`{cfg['lr']}`, clipping norm=`{cfg['clip']}`, scheduler=`{cfg['scheduler']}`, warmup fraction=`{cfg.get('warmup_fraction')}`, minimum LR fraction=`{cfg.get('min_lr_fraction')}`, momentum=`{cfg.get('momentum',0)}`. AdamW betas=(0.9,0.999), weight decay={cfg.get('weight_decay',.01)}. Full configuration: [config](../{state['full_config']}).

## Hyperparameter Search and Independent Confirmation

Selected candidate’s pilot uses  {pilot_training_images}  training images /  {len(pilot_training_rows)}  questions; search uses  {search_images}  distinct images; confirmation uses  {confirm_images}  distinct images across seven tasks. Rounds 1/2 use 32 images/task, 224 total; round 3, if reached, expands from original 20k to at most 96/task, with actual counts in its protocol. Search/confirmation images are outside original 20k and formal val/test; teacher correctness is not a filter. Search selects candidates; final benchmarks do not. All failed rounds remain, including unfavorable results. Confirmation only gates progression and receives no gradients. The actual number of confirmed candidates is  {len(state.get('confirmation_attempts',[]))}; repeated confirmation reuse has adaptive-selection limitations.

Selected pilot: `{pilot['name']}`, epoch {pilot['epoch']}.

| Cohort | Trained score | Change from same-architecture initialization | Change relative to mean | KL reduction from initial  | Paired 95% CI versus initial |
|---|---:|---:|---:|---:|---|
| 56-image search | {search['score']:.3f} | {search['paired_vs_initial']['gain_points']:+.3f} | {search['score']-selected['matched_mean']['score']:+.3f} | {search['kl_relative_reduction']:.1%} | {search['paired_vs_initial']['paired_bootstrap_95ci_points']} |
| 112-image confirmation | {confirmation['score']:.3f} | {confirmation['paired_vs_initial']['gain_points']:+.3f} | {confirmation['score']-confmean['score']:+.3f} | {confirmation['kl_relative_reduction']:.1%} | {confirmation['paired_vs_initial']['paired_bootstrap_95ci_points']} |

Original screening requires ≥2-point gain from initial, ≥5% KL reduction, and mean maximum column mass/m ≤0.25. Before formal updates, an additional ≥2-point advantage over same-budget mean pooling was registered. After the first epoch-1 results, before any candidate passed or confirmation was inspected, an additional practicality criterion requiring at least 50% teacher-score retention was registered; see  `suitability_addendum.json`. This was not in the original protocol; all observations remain, and the original  `screen_pass` and final  `practical_pass` Report separately. These eligibility criteria are not significance tests.

## Full-Training Trajectory and Curves

Restart from **original generator initialization**, resetting optimizer state rather than continuing pilot weights. Original 20k samples/order remain; global batch=36, 36-rank CPU/Gloo, threads per rank={json.loads((run/'manifest.json').read_text())['threads']} , 2 epochs / 1,112 updates. Each short final batch is padded from that epoch's shuffled sequence, giving 20,016 sample presentations/epoch. Backbone is frozen BF16 with original FP32 RoPE buffers; generator/optimizer are FP32. Initial, each-epoch, latest checkpoints, and source snapshots are stored in [run](../{cfg['output']}).

| epoch | updates | train KL | entropy | collapse | weighted objective | grad norm | Clipping frequency | Mean clipping multiplier | Parameter-update L2 norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for ep in epochs:text+=f"| {ep['epoch']} | {ep['steps']} | {ep['prediction']:.5f} | {ep['entropy']:.4f} | {ep['collapse']:.4f} | {ep['objective']:.5f} | {ep['grad_norm']:.4f} | {ep['clipped_fraction']:.1%} | {ep['clip_multiplier']:.5f} | {ep['update_l2']:.7f} |\n"
    text+=f'''
[Training and validation curves PDF](../{cfg['output']}/curves.pdf). KL is teacher-prefix distribution distillation, **not gold-answer cross-entropy**. Validation also reports free-generation task scores rather than inferring effectiveness from lower NLL/KL alone.

![Full training and development curves](../{cfg['output']}/curves.png)

Full training fixes epoch 2 for final evaluation, without benchmark-based checkpoint selection.

## Fixed Eight-Task Evaluation

Use the same 512 images/questions as the failed run, 64 per task. DocVQA/InfoVQA use public validation; other tasks retain test splits. max_new_tokens=256; the teacher is uncompressed, and trained, matched-architecture initial, and fixed-mean models share a token budget. **All 512 uncompressed teacher answers are independently regenerated** in the current CPU BF16 / 8-thread environment and provide the common reference. Exact full-token-ID agreement with the earlier evaluation is  {teacher_ids_equal}/512. Old teachers are used only for consistency audits, not as substitutes for new answers. All compressed paths independently generate and recompute KL under the same teacher prefix.

| Task | n | teacher | Same-architecture initialization | matched mean | Trained | Trained − mean |
|---|---:|---:|---:|---:|---:|---:|
'''
    for t in tasks:
        d=table[t];text+=f"| {t} | {d['trained']['n']} | {d['trained']['teacher']:.2f} | {d['initial']['score']:.2f} | {d['mean']['score']:.2f} | {d['trained']['score']:.2f} | {d['trained']['score']-d['mean']['score']:+.2f} |\n"
    text+=f"| Eight-task native-metric macro | 512 | {teacher:.2f} | {macro['initial']:.2f} | {macro['mean']:.2f} | {macro['trained']:.2f} | {macro['trained']-macro['mean']:+.2f} |\n"
    text+=f'''
Final teacher-score retention:  {macro['trained']/teacher:.1%}. Paired change from initialization:  {changes['initial']['gain_points']:+.3f}  points; 95% CI={changes['initial']['paired_bootstrap_95ci_points']}; change relative to mean  {changes['mean']['gain_points']:+.3f}  points; 95% CI={changes['mean']['paired_bootstrap_95ci_points']}. Bootstrap is task-stratified with images as resampling units, 10,000 repetitions; intervals describe sampling uncertainty on this fixed set only.

Earlier training results:  {old_macro:.2f}, original budget 10%; new budget  {cfg['ratio']:.0%} . Gains from a different budget cannot be attributed to training settings; matched-budget initialization and mean pooling are the primary controls here. This evaluation uses 512 fixed questions rather than all 23,565. DeepForm/KLC remain field-level exact match, not paper-level corpus F1. This small evaluation alone cannot establish reproduction of formal paper scores.

Retain every answer, gold, loss, token count, truncation, and failed candidate. Report the preselected epoch unchanged even when full results disagree with small-validation trends.
'''
    destination=ROOT/'docs/METACOMPRESS_RETRAIN_RESULTS.md';destination.write_text(text)
    readme=ROOT/'README.md';s=readme.read_text()
    start='<!-- META_RETRAIN_FINAL_START -->';end='<!-- META_RETRAIN_FINAL_END -->'
    block=(f'{start}\n**MetaCompress search and 20k retraining are complete: token retention  {cfg["ratio"]:.0%}, '
           f'Fixed 512-question teacher / trained macro {teacher:.2f} / {macro["trained"]:.2f}; '
           f'[Final measured report](docs/METACOMPRESS_RETRAIN_RESULTS.md), [Per-question answers](artifacts/meta_compress_search/final_answers.csv). **\n{end}')
    if start in s:s=s[:s.index(start)]+block+s[s.index(end)+len(end):]
    else:
        first,rest=s.split('\n',1);s=first+'\n\n'+block+'\n'+rest
    readme.write_text(s)
    print(destination)


if __name__=='__main__':main()
