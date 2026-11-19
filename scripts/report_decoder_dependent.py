"""Audit and report per-intervention decoder KL / entropy correlations."""
import argparse,collections,csv,hashlib,json,re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.report_encoder_candidates import analyze, rho
from vqa.benchmark_metrics import native_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/token_proxy/random_v1_20x'
OUT=ROOT/'artifacts/decoder_dependent_loss'
NAMES=['prediction_kl','entropy','total','first_token_kl','entropy_normalized']


def fmt(v):return 'Not estimable (constant)' if v is None else f'{v:.4f}'
def ci(v):return f'[{v[0]:.4f}, {v[1]:.4f}]'


def paired_metric_comparison(x_kl,x_total,y,draws=1000):
    """Paired image bootstrap for adding entropy; preserve outcome ties."""
    a=np.diff(x_kl.reshape(len(y),12,2),axis=2)[:,:,0]
    b=np.diff(x_total.reshape(len(y),12,2),axis=2)[:,:,0]
    d=np.diff(y.reshape(len(y),12,2),axis=2)[:,:,0]
    eligible=(a!=0)&(b!=0)&(d!=0)
    ac=((a*d>0)&eligible).sum(1);bc=((b*d>0)&eligible).sum(1);n=eligible.sum(1)
    rng=np.random.default_rng(20261025);ix=rng.integers(len(y),size=(draws,len(y)))
    boot=(bc[ix].sum(1)-ac[ix].sum(1))/np.maximum(1,n[ix].sum(1))
    observed=float((bc.sum()-ac.sum())/max(1,n.sum()))
    # For pooled rho use the same image draws for both metrics and re-rank each.
    dr=[]
    for idx in ix:
        left=rho(x_kl[idx].ravel(),y[idx].ravel());right=rho(x_total[idx].ravel(),y[idx].ravel())
        if left is not None and right is not None:dr.append(right-left)
    return {'seed_order_total_minus_kl':observed,'seed_order_delta_ci95':list(map(float,np.quantile(boot,[.025,.975]))),
            'seed_joint_eligible_pairs':int(n.sum()),'pooled_rho_total_minus_kl':rho(x_total.ravel(),y.ravel())-rho(x_kl.ravel(),y.ravel()),
            'pooled_rho_delta_ci95':list(map(float,np.quantile(dr,[.025,.975])))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=5);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();out=a.output
    samples=[json.loads(l) for l in (BASE/'samples.jsonl').open()]
    samples_by_id={r['id']:r for r in samples}
    complete=[r for r in samples if (out/'losses'/f"{r['document_id']}.json").exists()]
    if not a.allow_partial:assert len(complete)==len(samples),'Missing image losses; results are not complete'
    wanted={r['id'] for r in complete};losses={};audits=[];audit_by_id={};manifests=set();seconds=[]
    for r in complete:
        d=json.loads((out/'losses'/f"{r['document_id']}.json").read_text())
        assert len(d['trials'])==26 and d['sample_id']==r['id']
        audits.append(d['audit']);seconds.append(d['elapsed_seconds']);manifests.add(json.dumps(d['code_sha256'],sort_keys=True))
        audit_by_id[d['sample_id']]=d['audit']
        for trial in d['trials']:
            assert trial['id'] not in losses
            losses[trial['id']]=trial
    manifest_list=[json.loads(m) for m in sorted(manifests)]
    # Scheduling runner was upgraded from fixed shards to atomic image claims.
    # Accept that change only when the numerical loss/model/generator hashes match.
    for m in manifest_list:
        assert all(m[f]==manifest_list[0][f] for f in m if f!='scripts/eval_decoder_dependent.py'), 'Numerical implementation changed within cohort'
    rows=[]
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for line in path.open():
            r=json.loads(line)
            if r['sample_id'] not in wanted:continue
            loss=losses[r['id']]
            assert loss['matrix_sha256']==r['matrix_file_sha256'] and loss['compressed_feature_sha256']==r['compressed_feature_sha256']
            for name in ['teacher_score','student_score','score_drop']:assert loss[name]==r[name]
            # Original trial JSON omits OCRBench dataset_name. Recover sample
            # metadata so HME100k uses its whitespace-insensitive native metric.
            score_row={**samples_by_id[r['sample_id']],**r}
            assert native_score(r['teacher_prediction'],score_row)==r['teacher_score']
            assert native_score(r['student_prediction'],score_row)==r['student_score']
            assert abs(loss['prediction_kl']+loss['entropy']-loss['total'])<1e-9
            r['new_loss']={k:loss[k] for k in NAMES};rows.append(r)
    assert len(rows)==26*len(complete)
    # Reuse saved CPU generation audits only when EOS proves the response is
    # complete. A 16-token truncated prefix is never scored as a full answer.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/MonkeyOCRv2-B-Und'),local_files_only=True,trust_remote_code=True)
    lowest={r['sample_id']:r for r in rows if r['kind']=='random' and r['ratio']==.125 and r['family']=='local' and r['replicate']==0}
    generation_checks=[]
    for sid,ad in audit_by_id.items():
        ti,si=ad.get('teacher_regenerated_ids',[]),ad.get('student_regenerated_ids',[])
        if 151645 not in ti or 151645 not in si:continue
        r=lowest[sid];score_row={**samples_by_id[sid],**r}
        tp=tokenizer.decode(ti,skip_special_tokens=True);sp=tokenizer.decode(si,skip_special_tokens=True)
        ts,ss=native_score(tp,score_row),native_score(sp,score_row)
        generation_checks.append({'sample_id':sid,'task':r['task'],'question':r['question'],'answers':r['answers'],
            'teacher_cpu_prediction':tp,'student_cpu_prediction':sp,'teacher_npu_prediction':r['teacher_prediction'],
            'student_npu_prediction':r['student_prediction'],'cpu_teacher_score':ts,'cpu_student_score':ss,
            'npu_teacher_score':r['teacher_score'],'npu_student_score':r['student_score'],
            'cpu_score_drop':ts-ss,'npu_score_drop':r['score_drop'],**r['new_loss']})
    (out/'cpu_generation_checks.json').write_text(json.dumps(generation_checks,ensure_ascii=False,indent=2)+'\n')
    fields=['id','sample_id','document_id','task','question','image','answers','kind','family','ratio','replicate',
            'matrix_seed','matrix_file','teacher_prediction','teacher_score','student_prediction','student_score','score_drop']
    with (out/'trials.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields+NAMES);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in fields},**r['new_loss']})
    rr=sorted([r for r in rows if r['kind']=='random'],key=lambda r:(r['document_id'],r['ratio'],r['family'],r['replicate']))
    y=np.array([r['score_drop'] for r in rr]).reshape(-1,24)
    teacher=np.array([r['teacher_score'] for r in rr])[::24];tasks=np.array([r['task'] for r in rr])[::24]
    ratios=[r['ratio'] for r in rr[:24]];families=[r['family'] for r in rr[:24]]
    items=[(name,np.array([r['new_loss'][name] for r in rr]).reshape(-1,24),y,teacher,tasks,ratios,families) for name in NAMES]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:metrics=dict(pool.map(analyze,items))
    for name,x,_,_,_,_,_ in items:
        dx=np.diff(x.reshape(len(x),12,2),axis=2).ravel()
        dy=np.diff(y.reshape(len(y),12,2),axis=2).ravel()
        nx=int((dx!=0).sum());ny=int((dy!=0).sum());overlap=int(((dx!=0)&(dy!=0)).sum())
        metrics[name]['within_config_tie_ceiling']=float(overlap/np.sqrt(nx*ny)) if nx and ny else None
        metrics[name]['seed_loss_nontie_pairs']=nx
    for name in NAMES:
        for rate in [.25,.5]:
            cr=[r for r in rows if r['kind']=='mean_pool_control' and r['ratio']==rate]
            metrics[name]['controls'][str(rate)]=rho([r['new_loss'][name] for r in cr],[r['score_drop'] for r in cr])
        for label,condition in [
            ('teacher_first_token_match',lambda d:d['teacher_first_argmax']==d['teacher_response_first_id']),
            ('teacher_full_forced_argmax_match',lambda d:d['teacher_forced_greedy_agreement']==1.0)]:
            subset=[r for r in rr if condition(audit_by_id[r['sample_id']])]
            metrics[name][label]={'images':len(subset)//24,'pooled_spearman':rho([r['new_loss'][name] for r in subset],[r['score_drop'] for r in subset])}
    per_task={}
    for t in sorted(set(tasks)):
        tr=[r for r in rr if r['task']==t];mask=tasks==t
        per_task[t]={'images':int(mask.sum()),'teacher_score':float(teacher[mask].mean()),
                     'student_score':float(np.mean([r['student_score'] for r in tr])),
                     'score_drop':float(np.mean([r['score_drop'] for r in tr])),
                     'student_zero_fraction':float(np.mean([r['student_score']==0 for r in tr]))}
    examples={}
    for name in NAMES[:3]:
        good=[];bad=[]
        for i in range(0,len(rr),2):
            left,right=sorted(rr[i:i+2],key=lambda r:r['new_loss'][name])
            if right['new_loss'][name]==left['new_loss'][name]:continue
            pair={'low_loss':left,'high_loss':right}
            if left['score_drop']<right['score_drop']:good.append(pair)
            elif left['score_drop']>right['score_drop']:bad.append(pair)
        key=lambda p:abs(p['low_loss']['score_drop']-p['high_loss']['score_drop'])
        examples[name]={'concordant':sorted(good,key=key,reverse=True)[:3],
                        'counterexamples':sorted(bad,key=key,reverse=True)[:3]}
    (out/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    audit={'complete':len(complete)==len(samples),'images':len(complete),'expected_images':len(samples),
           'random_trials':len(rr),'controls':len(rows)-len(rr),'unique_matrices':len(losses),
           'all_feature_matrix_compressed_hashes_verified':True,'all_cached_scores_recomputed':True,
           'code_sha256_versions':manifest_list,
           'identity_checks':[r for r in audits if 'identity_KL_max_abs' in r],
           'teacher_forced_first_token_matches':sum(r['teacher_first_argmax']==r['teacher_response_first_id'] for r in audits),
           'teacher_forced_agreement_mean':float(np.mean([r['teacher_forced_greedy_agreement'] for r in audits])),
           'compute_seconds_sum':sum(seconds),'compute_seconds_per_image_median':float(np.median(seconds))}
    regenerated=[r for r in audits if 'teacher_generation_prefix_reproduced' in r]
    audit['regenerated_prefix_checks']={'images':len(regenerated),
        'teacher_match':sum(r['teacher_generation_prefix_reproduced'] for r in regenerated),
        'student_match':sum(r['student_generation_prefix_reproduced'] for r in regenerated),
        'both_match':sum(r['teacher_generation_prefix_reproduced'] and r['student_generation_prefix_reproduced'] for r in regenerated)}
    audit['completed_cpu_generation_checks']={'images':len(generation_checks),
        'teacher_score_equal':sum(r['cpu_teacher_score']==r['npu_teacher_score'] for r in generation_checks),
        'student_score_equal':sum(r['cpu_student_score']==r['npu_student_score'] for r in generation_checks),
        'score_drop_equal':sum(r['cpu_score_drop']==r['npu_score_drop'] for r in generation_checks),
        'cpu_vs_npu_score_drop_spearman':rho([r['cpu_score_drop'] for r in generation_checks],[r['npu_score_drop'] for r in generation_checks]),
        'loss_vs_cpu_drop_spearman':{n:rho([r[n] for r in generation_checks],[r['cpu_score_drop'] for r in generation_checks]) for n in NAMES[:3]},
        'loss_vs_npu_drop_spearman':{n:rho([r[n] for r in generation_checks],[r['npu_score_drop'] for r in generation_checks]) for n in NAMES[:3]}}
    audit['analysis_code_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in [
        'scripts/report_decoder_dependent.py','scripts/report_encoder_candidates.py','vqa/benchmark_metrics.py']}
    audit['source_generation_audit_sha256']=hashlib.sha256((BASE/'audit.json').read_bytes()).hexdigest()
    audit['source_generation_audit']=json.loads((BASE/'audit.json').read_text())
    if (OUT/'model_audit.json').exists():audit['model_audit']=json.loads((OUT/'model_audit.json').read_text())
    summary={'audit':audit,'metrics':metrics,'per_task':per_task,'bootstrap_repeats':1000,
             'student_zero_fraction':float(np.mean([r['student_score']==0 for r in rr])),
             'teacher_positive_images':int((teacher>0).sum()),'teacher_perfect_images':int((teacher==1).sum()),
             'lambda_entropy':1.0,'protocol':json.loads((OUT/'protocol.json').read_text())}
    summary['adding_entropy_paired_bootstrap']=paired_metric_comparison(items[0][1],items[2][1],y)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plot(out,rr,summary)
    sections=[]
    sections.append('# Decoder-Dependent Compression Loss: Measured Results\n')
    sections.append(f"Completed **{len(complete):,}  distinct images, {len(rr):,}  independent random compressions, {len(rows)-len(rr):,}  mean-pooling controls**."+('All planned samples completed.' if audit['complete'] else '**These are partial-sample results; the planned full scale is not complete.**'))
    sections.append('Downstream answers/scores reuse independent generations from the same official baseline, original features, and per-trial matrices. New decoder forwards compute full-vocabulary prediction KL; earlier encoder-only losses are not relabeled. Every seed’s score enters statistics independently.')
    sections.append('Score drop Δ=teacher_score−student_score. Positive correlation means higher loss predicts greater degradation. Primary analysis includes random compressions only; mean pooling is separate. Scores lie in [0,1].')
    sections.append('## Three Primary Metrics\n')
    sections.append('| Loss | Pooled Spearman ρ (image-bootstrap 95% CI) | Within-question rank correlation | Fixed question×retention×family rank correlation (95% CI) | Two-seed non-tied ordering accuracy (95% CI) |\n|---|---|---|---|---|')
    for name in NAMES[:3]:
        m=metrics[name];pair=m['paired_seed_ordering']
        sections.append(f"| {name} | {fmt(m['pooled_spearman'])} {ci(m['pooled_image_bootstrap_ci95'])} | {fmt(m['within_question'])} | {fmt(m['within_question_ratio_family'])} {ci(m['within_question_ratio_family_ci95'])} | {100*pair['accuracy']:.2f}% {ci([100*v for v in pair['image_bootstrap_ci95']])} |")
    sections.append('95% CIs use 1,000 image-cluster resamples, reranking pooled Spearman each time. Within-question correlation centers ranks within groups. Fixed question×retention×family groups have only two seeds, so interpret with ordering accuracy and tie counts; this is not another pooled Spearman.')
    sections.append('Many downstream ties limit within-two-seed rank correlation. Holding observed loss/score tie locations fixed, even perfectly ordering every non-tied pair yields a maximum of only  `non-tied intersection count / sqrt(loss loss-non-tied pair count × score-non-tied pair count)`; do not mechanically require .7 for this conditional metric. Interpret correlation together with non-tied ordering accuracy.')
    add=summary['adding_entropy_paired_bootstrap']
    sections.append(f"Paired image bootstrap for entropy addition: pooled-ρ change, total versus KL={add['pooled_rho_total_minus_kl']:.4f}, 95% CI {ci(add['pooled_rho_delta_ci95'])}; accuracy change on shared non-tied seed pairs={100*add['seed_order_total_minus_kl']:.2f}  percentage points, 95% CI  {ci([100*v for v in add['seed_order_delta_ci95']])}. This directly compares two metrics on identical image/matrix pairs. A small point-estimate difference does not establish an entropy benefit.")
    sections.append('| Loss | Pooled ρ on teacher-score>0 subset | Pooled ρ on teacher-score=1 subset | Global rank-residual correlation controlling question/configuration | Non-tied seeds at the same retention and family | Downstream ties | Loss ties with non-tied downstream scores |\n|---|---|---|---|---|---|---|')
    for name in NAMES[:3]:
        m=metrics[name];pair=m['paired_seed_ordering']
        sections.append(f"| {name} | {fmt(m['teacher_positive_spearman'])} | {fmt(m['teacher_perfect_spearman'])} | {fmt(m['partial_global_rank_question_config'])} | {pair['concordant']+pair['discordant']:,} | {pair['outcome_ties']:,} | {pair['loss_ties_outcome_nontie']:,} |")
    sections.append('## Per-Task Results\n')
    sections.append('| Dataset | Images | Teacher mean score | Student mean score | Mean drop | KL ρ | Entropy ρ | Total loss ρ |\n|---|---|---|---|---|---|---|---|')
    for t,d in per_task.items():
        sections.append(f"| {t} | {d['images']} | {100*d['teacher_score']:.2f} | {100*d['student_score']:.2f} | {100*d['score_drop']:.2f} | "+' | '.join(fmt(metrics[n]['per_task'][t]['pooled_spearman']) for n in NAMES[:3])+' |')
    sections.append('Mean task scores describe difficulty only. Correlation/ordering always use individual compression scores, never seed/answer averages. Retain native_score and interpret pooled cross-task correlation alongside per-task findings.')
    sections.append('| Dataset | KL seed-ordering accuracy / non-tied pair count | Entropy seed-ordering accuracy / non-tied pairs | Total-loss seed-ordering accuracy / non-tied pairs |\n|---|---|---|---|')
    for t in per_task:
        values=[]
        for n in NAMES[:3]:
            d=metrics[n]['per_task'][t]
            values.append(f"{100*d['seed_order_accuracy']:.2f}% / {d['nontie_pairs']:,}" if d['nontie_pairs'] else 'No non-tied pairs')
        sections.append('| '+t+' | '+' | '.join(values)+' |')
    sections.append('Per-task seed-ordering tables are point estimates. Each pair fixes question/retention/family and varies only seed; do not mix with cross-question pooled correlation.')
    sections.append('## Fixed Retention Fractions and Matrix Families\n')
    sections.append('| Retention / family | KL pooled ρ | Entropy pooled ρ | Total-loss pooled ρ |\n|---|---|---|---|')
    for key in metrics['total']['per_ratio_family']:
        sections.append('| '+key+' | '+' | '.join(fmt(metrics[n]['per_ratio_family'][key]['pooled_spearman']) for n in NAMES[:3])+' |')
    sections.append('Pooled values per retention/family still cross questions; stricter same-question/seed checks appear in the primary table.')
    sections.append('## Controls and Diagnostics\n')
    for name in NAMES:
        m=metrics[name]
        sections.append(f"- `{name}` range / median: {m['loss_quantiles'][0]:.6g} / {m['loss_quantiles'][3]:.6g} / {m['loss_quantiles'][-1]:.6g}(min/median/max); mean-pooling 25% / 50% ρ: {fmt(m['controls']['0.25'])} / {fmt(m['controls']['0.5'])}. ")
        if name in NAMES[:3]:sections.append(f"  Conditional rank-correlation ceiling under observed ties: {fmt(m['within_config_tie_ceiling'])}. ")
    sections.append(f"- Fraction of random trials with zero student score:  {100*summary['student_zero_fraction']:.2f}%; images with teacher score >0:  {summary['teacher_positive_images']:,}  images; teacher-perfect images  {summary['teacher_perfect_images']:,}  images.")
    sections.append(f"- First shared-teacher-prefix argmax matches the historical teacher’s first token: {audit['teacher_forced_first_token_matches']:,}/{len(complete):,}. Mean greedy agreement over complete prefixes: {100*audit['teacher_forced_agreement_mean']:.4f}%. Hardware/parallel reductions and stepwise KV generation can introduce BF16 differences; checks remain in audit.json.")
    rg=audit['regenerated_prefix_checks']
    sections.append(f"- CPU/NPU free-generation-prefix cross-check: {rg['images']}  images; teacher agreement  {rg['teacher_match']}, student agreement  {rg['student_match']}, both identical {rg['both_match']}. Prefix cap=16; one matching pilot does not establish equivalence everywhere. Scores remain original NPU free generations, while loss uses CPU BF16 shared-prefix forwards, with cross-device numerical/trajectory limitations.")
    cg=audit['completed_cpu_generation_checks']
    sections.append(f"- Using saved regeneration records, among cases where teacher and student both generate EOS within 16 tokens: **{cg['images']} ** images receive full CPU scoring: teacher agreement  {cg['teacher_score_equal']}, student-score agreement  {cg['student_score_equal']}, matching score drops  {cg['score_drop_equal']}; Spearman of CPU/NPU score drops={fmt(cg['cpu_vs_npu_score_drop_spearman'])}. This EOS-completed short-answer diagnostic subset checks only 12.5% local seed 0, not all 93,626 interventions. Complete records:  `cpu_generation_checks.json`. ")
    sections.append('| Loss on EOS-completed subset | ρ with CPU score drop | ρ with NPU score drop on the same subset |\n|---|---|---|')
    for n in NAMES[:3]:sections.append(f"| {n} | {fmt(cg['loss_vs_cpu_drop_spearman'][n])} | {fmt(cg['loss_vs_npu_drop_spearman'][n])} |")
    sections.append('First-token KL has no answer prefix; H/log N diagnoses length scaling. Neither replaces the paper objective. Evaluation does not tune λ or choose the best variant.')
    sections.append('| Cross-device consistency diagnostic subset | Images | KL ρ | Entropy ρ | Total loss ρ |\n|---|---|---|---|---|')
    for label in ['teacher_first_token_match','teacher_full_forced_argmax_match']:
        sections.append('| '+label+' | '+str(metrics['total'][label]['images'])+' | '+' | '.join(fmt(metrics[n][label]['pooled_spearman']) for n in NAMES[:3])+' |')
    sections.append('## Independent Answer Examples\n')
    for typ in ['concordant','counterexamples']:
        sections.append('### '+('KL agrees with downstream ordering' if typ=='concordant' else 'Counterexample to KL ordering')+'\n')
        for pair in examples['prediction_kl'][typ][:2]:
            left,right=pair['low_loss'],pair['high_loss']
            sections.append(f"- `{left['sample_id']}`, {left['task']}, retention  {left['ratio']}, {left['family']}. Question: {left['question']}; gold: {json.dumps(left['answers'],ensure_ascii=False)}; teacher: {left['teacher_prediction']!r}, score  {left['teacher_score']:.4f}. ")
            for label,r in [('Low KL',left),('High KL',right)]:
                sections.append(f"  - {label}, seed index {r['replicate']}: KL={r['new_loss']['prediction_kl']:.6f}, entropy={r['new_loss']['entropy']:.6f}, total={r['new_loss']['total']:.6f}; student={r['student_prediction']!r}, score={r['student_score']:.4f}, drop={r['score_drop']:.4f}. ")
    sections.append('## Result Files and Audits\n')
    sections.append('- `trials.csv`: every answer, gold, both component losses, total, score, and drop.\n- `losses/<document_id>.json`: per-response-position KL, teacher token IDs, and matrix/feature summaries.\n- `analysis.json`: full correlations, task/configuration breakdowns, and CIs.\n- `examples.json`: concordant-order examples and counterexamples for three losses.\n- `audit.json`: coverage, score recomputation, provenance, and interface checks.\n- `correlation_scatter.pdf/png`: per-compression scatter plots.')
    markdown='\n\n'.join(sections)+'\n'
    markdown=re.sub(r'(\|)\n\n(?=\|)',r'\1\n',markdown)
    (out/'RESULTS.md').write_text(markdown)
    print(json.dumps({'complete':audit['complete'],'images':len(complete),'random_trials':len(rr),'metrics':{n:metrics[n]['pooled_spearman'] for n in NAMES[:3]}},ensure_ascii=False),flush=True)


def plot(out,rows,summary):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));rng=np.random.default_rng(25)
    chosen=rng.choice(len(rows),size=min(len(rows),18000),replace=False)
    for ax,name in zip(axes,NAMES[:3]):
        for family,color in [('local','#2471a3'),('global','#ca6f1e')]:
            rr=[rows[i] for i in chosen if rows[i]['family']==family]
            ax.scatter([r['new_loss'][name] for r in rr],[r['score_drop'] for r in rr],s=3,alpha=.18,color=color,label=family,rasterized=True)
        m=summary['metrics'][name];ax.set_title(f"{name}\npooled rho={m['pooled_spearman']:.3f}; within config={m['within_question_ratio_family']:.3f}")
        ax.set_xlabel('Loss (nats)');ax.set_ylabel('Teacher score - student score')
    axes[0].legend();fig.tight_layout();fig.savefig(out/'correlation_scatter.pdf');fig.savefig(out/'correlation_scatter.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    import multiprocessing;multiprocessing.set_start_method('spawn');main()
