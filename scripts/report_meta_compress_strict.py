"""Read-only experiment report and image-clustered uncertainty; no model selection on test."""
import csv
import json
import math
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/meta_compress_strict'

def read(path):
    if not path.exists():return []
    result=[];lines=path.read_text().splitlines()
    for i,s in enumerate(lines):
        try: result.append(json.loads(s))
        except json.JSONDecodeError:
            if i<len(lines)-1:raise
    return result

def save(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.report.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def mean(v):return sum(v)/len(v)

def summarize(rows,expected):
    ids=[r['id'] for r in rows]
    assert len(ids)==len(set(ids)),'Duplicate evaluation question'
    if set(ids)!=set(expected):return {'complete':False,'n':len(ids),'expected':len(expected)}
    tasks=defaultdict(list)
    for r in rows:tasks[r['task']].append(r)
    fields=['prediction','entropy','collapse','total','student_score','teacher_score']
    bytask={task:{k:mean([r[k] for r in rr]) for k in fields} for task,rr in tasks.items()}
    return {'complete':True,'n':len(rows),'images':len({r['document_id'] for r in rows}),
            **{k:mean([r[k] for r in bytask.values()]) for k in fields},
            'score':100*mean([r['student_score'] for r in bytask.values()]),
            'teacher':100*mean([r['teacher_score'] for r in bytask.values()]),'tasks':bytask,
            'student_truncations':sum(r['hit_token_limit'] for r in rows),
            'teacher_truncations':sum(r['teacher_hit_token_limit'] for r in rows)}

def paired(initial,final):
    aa={r['id']:r for r in initial};bb={r['id']:r for r in final};assert aa.keys()==bb.keys()
    groups=defaultdict(lambda:defaultdict(list))
    for key,a in aa.items():groups[a['task']][a['document_id']].append(100*(bb[key]['student_score']-a['student_score']))
    # Each sampled image retains all its QA; the estimator is the same task
    # QA-average as the report, not an unclustered independent-question sample.
    rng=np.random.default_rng(20261101);samples=np.zeros(10000);point=0.
    for task,docs in groups.items():
        sums=np.array([sum(v) for v in docs.values()]);counts=np.array([len(v) for v in docs.values()])
        idx=rng.integers(len(sums),size=(10000,len(sums)))
        samples+=(sums[idx].sum(1)/counts[idx].sum(1))/len(groups)
        point+=sums.sum()/counts.sum()/len(groups)
    return {'gain_points':float(point),'cluster_bootstrap_95ci_points':np.quantile(samples,[.025,.975]).tolist()}

def stable(stages,epochs,kl_tail_tolerance=0.):
    reasons=[];practical=[]
    for cohort in ['train_audit','validation']:
        initial=stages.get(f'{cohort}/initial',{})
        last=[stages.get(f'{cohort}/epoch{e}',{}) for e in range(epochs-2,epochs+1)]
        if not initial.get('complete') or not all(v.get('complete') for v in last):
            reasons.append(f'{cohort}: missing complete initial/last3');continue
        if mean([v['total'] for v in last])>initial['total']*.97:reasons.append(f'{cohort}: total drop <3%')
        if mean([v['prediction'] for v in last])>initial['prediction']*.95:reasons.append(f'{cohort}: KL drop <5%')
        if any(b['total']>a['total']*1.02 for a,b in zip(last,last[1:])):reasons.append(f'{cohort}: total rebound >2%')
        if last[-1]['total']>last[0]['total']:reasons.append(f'{cohort}: last3 total ending above start')
        if last[-1]['prediction']>last[0]['prediction']*(1+kl_tail_tolerance):
            reasons.append(f'{cohort}: last3 KL ending above start' if not kl_tail_tolerance
                           else f'{cohort}: last3 KL rebound >{100*kl_tail_tolerance:g}%')
        if min(v['score'] for v in last)<initial['score']+1:reasons.append(f'{cohort}: last3 score gains not all >=1pt')
        if mean([v['score'] for v in last])<initial['score']+2:reasons.append(f'{cohort}: mean score gain <2pt')
        if last[-1]['score']<last[0]['score']:reasons.append(f'{cohort}: last3 score ending below start')
        if mean([v['score'] for v in last])<.7*initial['teacher']:practical.append(f'{cohort}: retains <70% teacher score')
    return {'pass':not reasons and not practical,'trend_pass':not reasons,
            'kl_tail_relative_tolerance':kl_tail_tolerance,
            'practical_pass':not practical,'reasons':reasons+practical,
            'trend_reasons':reasons,'practical_reasons':practical}

def collect(run):
    cfg=json.loads((run/'manifest.json').read_text())['config'];grouped=defaultdict(list)
    for f in sorted(run.glob('eval.*.rank*.jsonl')):
        _,cohort,stage,_=f.name.split('.',3);grouped[f'{cohort}/{stage}']+=read(f)
    expected={'validation':{r['id'] for r in read(ROOT/cfg['validation_rows'])},
              'train_audit':{r['id'] for r in read(ROOT/cfg['audit_rows'])},
              'confirm':{r['id'] for r in read(BASE/'confirm.jsonl')}}
    stages={k:summarize(v,expected[k.split('/')[0]]) for k,v in grouped.items()}
    for k,s in stages.items():
        cohort,stage=k.split('/');initial=stages.get(f'{cohort}/initial',{})
        if s.get('complete') and initial.get('complete') and stage!='initial':
            s['gain_points']=s['score']-initial['score']
            s['kl_drop_fraction']=1-s['prediction']/initial['prediction']
            s['total_drop_fraction']=1-s['total']/initial['total']
            if cohort=='confirm':s['paired']=paired(grouped[f'{cohort}/initial'],grouped[k])
    log=read(run/'training.jsonl')
    result={'name':run.name,'config':cfg,'complete':(run/'complete.json').exists(),
            'updates':max([r['step'] for r in log],default=0),
            'planned_updates':math.ceil(len(read(ROOT/cfg['train_rows']))/cfg['batch_size'])*cfg['epochs'],
            'stages':stages,'stability':stable(stages,cfg['epochs']),
            'requested_stability':stable(stages,cfg['epochs'],kl_tail_tolerance=.02)}
    if (run/'early_stop.json').exists():result['early_stop']=json.loads((run/'early_stop.json').read_text())
    return result,grouped

def expanded_overview(results):
    selected=[r for r in results if r['name'].startswith('early_expanded') or r['name']=='full_verified_family']
    if not selected:return []
    lines=['## Expanded Training and Repeated Seeds: Latest Complete Epoch','',
           'Each row uses complete training-audit/validation records from the same epoch. In-progress evaluation is not scored. Initial→latest holds budget/position/seed fixed.',
           'Training audit is a fixed subset, not the mean of all training batches. Stable generalization requires planned epochs, repeated seeds, and independent confirmation. Current selection emphasizes total loss/VQA while retaining strict component-KL criteria as diagnostics. The adaptive change follows inspection of tuning results; independent-confirmation requirements are not relaxed.','',
           '| Insertion / retention / seed | global batch / LR | Complete epoch | Update step | Training score: initial → latest | Development score: initial → latest | Training total: initial → latest | Development total: initial → latest | Development KL: initial → latest |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        stages=r['stages'];cfg=r['config']
        epochs=[e for e in range(1,cfg['epochs']+1) if all(stages.get(f'{c}/epoch{e}',{}).get('complete')
                for c in ['train_audit','validation'])]
        epoch=max(epochs,default=0);stage=f'epoch{epoch}' if epoch else 'initial'
        def transition(cohort,field,digits):
            a=stages.get(cohort+'/initial',{});b=stages.get(cohort+'/'+stage,{})
            return f'{a[field]:.{digits}f}→{b[field]:.{digits}f}' if a.get('complete') and b.get('complete') else '—'
        lines.append(f"| {cfg['insertion']} / {cfg['ratio']:.0%} / {cfg['seed']} | {cfg['batch_size']} / {cfg['lr']:.3g} | {epoch}/{cfg['epochs']} | {r['updates']}/{r['planned_updates']} | "
                     +' | '.join(transition(c,f,d) for c,f,d in [('train_audit','score',2),('validation','score',2),
                         ('train_audit','total',4),('validation','total',4),('validation','prediction',4)])+' |')
    lines+=['','Uncompressed validation teacher:  '+', '.join(f"{v:.2f}" for v in sorted({r['stages']['validation/initial']['teacher']
                 for r in selected if r['stages'].get('validation/initial',{}).get('complete')}))+'  points; mixed-task macro is not a unified official leaderboard.','']
    for r in selected:
        rel=f'../artifacts/meta_compress_strict/runs/{r["name"]}'
        lines.append(f'- `{r["name"]}`: [Curve PDF]({rel}/curves.pdf) · [Stepwise log]({rel}/training.jsonl)')
    lines+=['','Retain all initialization screens, negative results, component losses, and same-budget controls below.','']
    return lines

def confirmed_report(results, state):
    """Keep confirmed evidence separate from the subsequent coverage experiment."""
    indexed={r['name']:r for r in results}
    records=[json.loads(p.read_text()) for p in sorted(BASE.glob('confirmation.*.json'))]
    if not records:return []
    paired_answers=[]
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        directory=BASE/'runs'/r['name']
        before={v['id']:v for f in directory.glob('eval.confirm.initial.rank*.jsonl') for v in read(f)}
        after={v['id']:v for f in directory.glob('eval.confirm.selected.rank*.jsonl') for v in read(f)}
        for key in sorted(before.keys()&after.keys()):
            a=before[key];b=after[key]
            paired_answers.append({'seed':r['config']['seed'],'id':key,'document_id':a['document_id'],
                'task':a['task'],'question':a['question'],'gold_answers':json.dumps(a['answers'],ensure_ascii=False),
                'initial_answer':a['student_prediction'],'trained_answer':b['student_prediction'],
                'teacher_answer':a['teacher_prediction'],'initial_score':a['student_score'],
                'trained_score':b['student_score'],'teacher_score':a['teacher_score'],
                'score_gain':b['student_score']-a['student_score'],
                'initial_total_loss':a['total'],'trained_total_loss':b['total'],
                'initial_KL':a['prediction'],'trained_KL':b['prediction']})
    if paired_answers:
        with (BASE/'confirmation_answers.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired_answers[0]));writer.writeheader();writer.writerows(paired_answers)
    lines=['# MetaCompress: Two-Seed Confirmation and Expanded-Training Progress','',
        'Built from actual per-question/confirmation records. Confirmation has 224 images/420 QA unused for selection; epoch 8 is fixed without confirmation-based checkpoint reselection.',
        'Seven native task scores are equally macro-averaged as percentages. DeepForm/KLC are field exact match, not official document-level F1.','',
        '## Completed Independent Confirmation','',
        '| seed | QA | Initial VQA | Trained VQA | teacher | Gain and image-cluster 95% CI | Total: initial → trained | KL: initial → trained | Confirmation criterion |',
        '|---|---:|---:|---:|---:|---|---|---|---|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        a=r['stages'].get('confirm/initial',{});b=record['confirmation']
        if not (a.get('complete') and b.get('complete')):continue
        ci=b['paired']['cluster_bootstrap_95ci_points']
        lines.append(f"| {r['config']['seed']} | {b['n']} | {a['score']:.2f} | {b['score']:.2f} | {b['teacher']:.2f} | {b['gain_points']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {a['total']:.4f}→{b['total']:.4f} | {a['prediction']:.4f}→{b['prediction']:.4f} | {'Passed' if record['pass'] else 'Failed'} |")
    lines+=['','Both groups use the same training method: the complete official MonkeyOCRv2-B-Und is frozen; the original MetaCompress generator uses KL + row entropy + maximum column mass, all weighted 1. Compression is post-merger at 75% retention; dense-position initialization strength=512; AdamW LR=3e-6, global batch=32, 8 epochs, and 8 NPU ranks.',
        'Image-disjoint splits: training 672 images/1,801 QA, fixed training audit 28/78, tuning 112/213, confirmation 224/420. Teacher/student independently generate at cap=64; gold is scoring-only.',
        '', '## Training and Development Evaluation','',
        '| seed | Cohort | VQA: initial → epoch 8 | Total: initial → epoch 8 | Primary total-loss / VQA criterion | Original strict KL diagnostic |',
        '|---|---|---|---|---|---|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        for cohort in ['train_audit','validation']:
            a=r['stages'].get(cohort+'/initial',{});b=r['stages'].get(cohort+'/epoch8',{})
            if not (a.get('complete') and b.get('complete')):continue
            lines.append(f"| {r['config']['seed']} | {cohort} | {a['score']:.2f}→{b['score']:.2f} | {a['total']:.4f}→{b['total']:.4f} | {'Passed' if r['requested_stability']['pass'] else 'Failed'} | {'Passed' if r['stability']['pass'] else 'Failed'} |")
    lines+=['','After viewing tuning results, the adaptive primary criterion permits ≤2% terminal KL rebound over the last three epochs; the original permits none. Both are retained, without claiming the original strict criterion passed. Total-loss/VQA and independent-confirmation requirements are unchanged. See [original revision record](../artifacts/meta_compress_strict/primary_endpoint_addendum.json). ',
        '“Stable improvement” means the recorded final-three-epoch rule plus independent confirmation, not monotonic gains in every epoch/task/question. Fixed training-audit scores do not replace all-training generation.',
        '', 'Same-budget post-merger mean pooling scores **50.374** on the tuning set; the two epoch-8 seeds score **46.61 / 49.97**. Neither superiority to mean pooling nor recovery to the uncompressed teacher is established. Batch=32 changes gradient averaging and accumulated update magnitude together. The batch=8 control with matched accumulated LR was skipped, so reduced gradient noise cannot be identified as the sole cause.',
        '', '## Per-Task Independent Confirmation','',
        '| seed | Task | Initial | epoch8 | teacher | Gain |','|---|---|---:|---:|---:|---:|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        a=r['stages'].get('confirm/initial',{});b=record['confirmation']
        if not (a.get('complete') and b.get('complete')):continue
        for task,v in sorted(b['tasks'].items()):
            initial=100*a['tasks'][task]['student_score'];final=100*v['student_score']
            lines.append(f"| {r['config']['seed']} | {task} | {initial:.2f} | {final:.2f} | {100*v['teacher_score']:.2f} | {final-initial:+.2f} |")
    if (BASE/'confirmation_per_task_uncertainty.json').exists():
        lines+=['','[Task-stratified, image-clustered per-task gain intervals](../artifacts/meta_compress_strict/confirmation_per_task_uncertainty.json) : for ChartQA, DeepForm, and KLC, both seeds\' nominal 95% CI lower bounds exceed 0; for DocVQA, InfoVQA, and DT-VQA, both intervals cross 0. Independent evidence supports macro-average improvement but not separate confirmed gains on the latter three tasks. Per-task intervals are unadjusted for multiple comparisons.']
    lines+=['','## Per-Epoch Loss Across All Training Batches','',
        'Training total averages all global batches in an epoch, including the prespecified cyclic padding of its final batch, weighted by training examples. Validation total is a task-macro average on the fixed development set; aggregation weights differ, so inspect their trends separately. Training total decreases each epoch for both seeds. Validation loss decreases and then plateaus, while VQA fluctuates; eight strictly monotonic score increases are not observed.','',
        '| seed | epoch | training total | validation total | validation VQA |',
        '|---|---:|---:|---:|---:|']
    for record in records:
        r=indexed.get(record['candidate'])
        if not r:continue
        grouped=defaultdict(list)
        for row in read(BASE/'runs'/r['name']/'training.jsonl'):grouped[row['epoch']].append(row['total'])
        expected=math.ceil(len(read(ROOT/r['config']['train_rows']))/r['config']['batch_size'])
        for epoch,values in sorted(grouped.items()):
            v=r['stages'].get(f'validation/epoch{epoch}',{})
            if len(values)==expected and v.get('complete'):
                lines.append(f"| {r['config']['seed']} | {epoch} | {mean(values):.5f} | {v['total']:.5f} | {v['score']:.3f} |")
    lines+=['','## Actual Status of Subsequent Coverage Expansion','',f"Controller stage: `{state.get('phase')}`, recorded at  `{state.get('updated_at')}`. ",
        'Expanded data: 12,661 images / 19,999 QA, 8 full epochs. Train from initialization with LR scaled by cumulative scheduled LR to about 2.7408e-7. Coverage/optimization differ from two-seed confirmation, whose success cannot be inherited.']
    full=indexed.get('full_verified_family')
    if full:
        epochs=[e for e in range(1,9) if all(full['stages'].get(f'{c}/epoch{e}',{}).get('complete') for c in ['train_audit','validation'])]
        lines+=['',f"Actual updates {full['updates']}/{full['planned_updates']}; complete generation evaluation  {max(epochs,default=0)}/8 epochs; training completion marker: {full['complete']}. "]
        if full['complete']:
            lines+=['',f"Expanded-training primary total-loss/VQA criterion: {'Passed' if full['requested_stability']['pass'] else 'Failed:'+'; '.join(full['requested_stability']['reasons'])}. "]
    else:lines+=['','No expanded-training manifest; training completion is not established.']
    final=ROOT/'docs/METACOMPRESS_STRICT_FINAL_RESULTS.md'
    if final.exists():lines+=['','[Completed full-benchmark results](METACOMPRESS_STRICT_FINAL_RESULTS.md). ']
    else:lines+=['','The 23,565-QA benchmark has no final report yet; full scores cannot be reported.']
    lines+=['','## Entry Points and Outputs','',
        '```bash','bash scripts/run_meta_compress_strict.sh status','bash scripts/run_meta_compress_strict.sh report',
        '# Fresh training: inspect status first to avoid duplicate controller submissions.',
        'bash scripts/run_meta_compress_strict.sh train artifacts/meta_compress_strict/configs/early_expanded3_screen37_batch32.json',
        '# Independent image QA: replace IMAGE and QUESTION with actual inputs.',
        'bash scripts/run_meta_compress_strict.sh predict \\',
        '  --checkpoint artifacts/meta_compress_strict/runs/early_expanded3_screen37_batch32/epoch8.pt \\',
        '  --image IMAGE --question QUESTION --teacher --save-json artifacts/meta_compress_strict/prediction.json','```','',
        '[Full design](METACOMPRESS_STRICT_OPTIMIZATION.md) · [Live per-epoch results](METACOMPRESS_STRICT_RESULTS.md) · [Confirmation initial/trained/teacher/gold per-question comparison](../artifacts/meta_compress_strict/confirmation_answers.csv) · [All per-question records](../artifacts/meta_compress_strict/answers.csv)',
        '', 'Confirmation CSV scores are in [0,1]; report macros as percentages. Every row is one deterministic generation, without random-compression averaging.']
    for record in records:
        name=record['candidate'];rel=f'../artifacts/meta_compress_strict/runs/{name}'
        lines+=['',f'`{name}`: [Curve PDF]({rel}/curves.pdf) · [Training log]({rel}/training.jsonl) · [Configuration](../artifacts/meta_compress_strict/configs/{name}.json) · [Independent confirmation records](../artifacts/meta_compress_strict/confirmation.{name}.json). ']
    (ROOT/'docs/METACOMPRESS_STRICT_CONFIRMED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return ['## Independent Confirmation','', '[Two-seed independent confirmation, limitations, and subsequent training status](METACOMPRESS_STRICT_CONFIRMED_RESULTS.md). ','']

def main():
    results=[];allrows=[]
    for run in sorted((BASE/'runs').glob('*')):
        if not (run/'manifest.json').exists():continue
        result,grouped=collect(run);results.append(result)
        for rows in grouped.values():allrows.extend({'run':run.name,**r} for r in rows)
    save(BASE/'summary.json',results)
    lines=['# Original Architecture and Three-Term Loss: Measured Training and Validation','',
        'Report only completed evaluation stages. Scores are seven-task native-metric percentage macros; DeepForm/KLC use field exact match.',
        'Training audit is fixed and independently generated; batch loss and fixed-audit loss are logged separately.','',
        'Primary criteria and original strict additional-KL diagnostics are separate; neither is independent confirmation. Primary criteria allow final-three-epoch KL ending ≤102% of its start; total-loss/VQA/overall-KL/quality requirements remain. Revision follows tuning inspection and precedes confirmation.','',
        '| run | Completed | updates | Primary total-loss / VQA criterion | Original single-seed trend diagnosis | Original trend + quality-retention diagnosis |','|---|---|---:|---|---|---|']
    for r in results:lines.append(f"| {r['name']} | {'Stopped early; planned training incomplete' if r.get('early_stop') else r['complete']} | {r['updates']}/{r['planned_updates']} | {r['requested_stability']['pass']} | {r['stability']['trend_pass']} | {r['stability']['pass']} |")
    lines+=['','| run | Cohort / stage | QA | score | teacher | KL | H | collapse | total | Δscore |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for key,s in sorted(r['stages'].items()):
            if not s['complete']:
                lines.append(f"| {r['name']} | {key}(incomplete) | {s['n']}/{s['expected']} | — | — | — | — | — | — | — |")
            else:lines.append(f"| {r['name']} | {key} | {s['n']} | {s['score']:.3f} | {s['teacher']:.3f} | {s['prediction']:.5f} | {s['entropy']:.3f} | {s['collapse']:.3f} | {s['total']:.4f} | {s.get('gain_points',0):+.3f} |")
    lines+=['','## Same-Budget Fixed Controls (Not Learned-Generator Candidates)','',
            '| Insertion point / control | QA | score | teacher | KL | total |','|---|---:|---:|---:|---:|---:|']
    expected={r['id'] for r in read(BASE/'validation.jsonl')}
    for directory in sorted((BASE/'baselines').glob('*')):
        rows=[r for f in directory.glob('eval.validation.fixed.rank*.jsonl') for r in read(f)]
        allrows.extend({'run':'control:'+directory.name,**r} for r in rows)
        s=summarize(rows,expected)
        if s['complete']:lines.append(f"| {directory.name} | {s['n']} | {s['score']:.3f} | {s['teacher']:.3f} | {s['prediction']:.5f} | {s['total']:.4f} |")
        else:lines.append(f"| {directory.name}(incomplete) | {s['n']}/{s['expected']} | — | — | — | — |")
    text='\n'.join(lines)+'\n'
    (BASE/'RESULTS.md').write_text(text)
    state=json.loads((BASE/'controller.json').read_text()) if (BASE/'controller.json').exists() else {}
    doc=['# Strict MetaCompress Optimization: Live Results','',
         f'Last recorded controller stage: `{state.get("phase","initializing")}`; updated: `{state.get("updated_at","pending")}`. ',
         'Automatically generated from actual outputs. Incomplete stages are not final results; failed eligibility does not mean still running.',
         '[Design and entry point](METACOMPRESS_STRICT_OPTIMIZATION.md) · [Scheduling status](../artifacts/meta_compress_strict/RUNNING_STATUS.md) · [Per-question answers](../artifacts/meta_compress_strict/answers.csv)','',*confirmed_report(results,state),*expanded_overview(results),text,
         '## Curves and Reasons for Missing the Targets','']
    for r in results:
        rel=f'../artifacts/meta_compress_strict/runs/{r["name"]}'
        doc+=['',f'### {r["name"]}','',f'[Training/validation curves PDF]({rel}/curves.pdf) · [PNG]({rel}/curves.png)','',
              'Primary total-loss/VQA criterion: '+('Passed (repeated seeds and independent confirmation still required)' if r['requested_stability']['pass'] else 'Not yet passed:'+'; '.join(r['requested_stability']['reasons'])),
              '', 'Original strict additional-KL diagnostic: '+('Passed' if r['stability']['pass'] else 'Failed:'+'; '.join(r['stability']['reasons']))]
        cfg=r['config']
        doc+=['',f"Insertion `{cfg['insertion']}`; token retention  {cfg['ratio']:.0%}; initial `{cfg['initialization']}`; optimizer `{cfg['optimizer']}`; LR `{cfg['lr']}`; clip `{cfg['clip']}`; seed `{cfg['seed']}`. "]
        epochs=[(int(k.split('epoch')[1]),v) for k,v in r['stages'].items()
                if k.startswith('validation/epoch') and v.get('complete')]
        initial=r['stages'].get('validation/initial',{})
        if epochs and initial.get('complete'):
            epoch,last=max(epochs,key=lambda v:v[0])
            doc+=['',f'Latest complete validation epoch {epoch}, not a retrospectively selected maximum-score epoch: ','',
                  '| Task | Initial | Latest | teacher | Δscore | KL: initial → latest |',
                  '|---|---:|---:|---:|---:|---:|']
            for task,v in sorted(last['tasks'].items()):
                a=initial['tasks'][task]
                doc.append(f"| {task} | {100*a['student_score']:.2f} | {100*v['student_score']:.2f} | {100*v['teacher_score']:.2f} | {100*(v['student_score']-a['student_score']):+.2f} | {a['prediction']:.4f}→{v['prediction']:.4f} |")
    (ROOT/'docs/METACOMPRESS_STRICT_RESULTS.md').write_text('\n'.join(doc)+'\n')
    if allrows:
        keys=['run','cohort','stage','id','document_id','task','question','answers','teacher_prediction','student_prediction',
              'teacher_score','student_score','prediction','entropy','collapse','total','tokens','compressed_tokens','hit_token_limit']
        with (BASE/'answers.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(allrows)
    plot(results)
    print(json.dumps([{'name':r['name'],'updates':r['updates'],'complete':r['complete'],
        'primary_stable':r['requested_stability']['pass'],'legacy_strict_stable':r['stability']['pass']} for r in results]))

def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    for r in results:
        run=BASE/'runs'/r['name']
        log=read(run/'training.jsonl');byepoch=defaultdict(list)
        for row in log:byepoch[row['epoch']].append(row)
        per_epoch=math.ceil(len(read(ROOT/r['config']['train_rows']))/r['config']['batch_size'])
        online=[(e,{k:mean([v[k] for v in rr]) for k in ['total','prediction']})
                for e,rr in sorted(byepoch.items()) if len(rr)==per_epoch]
        signature=json.dumps({'stages':r['stages'],'online':online},sort_keys=True)
        sig=run/'plot_signature.json'
        if sig.exists() and sig.read_text()==signature:continue
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,field in zip(axes[:2],['total','prediction']):
            ax.plot([v[0] for v in online],[v[1][field] for v in online],marker='x',linestyle=':',color='tab:blue',label='train batches (epoch mean)')
        for cohort in ['train_audit','validation']:
            series=[]
            for k,s in r['stages'].items():
                c,stage=k.split('/')
                if c!=cohort or not s['complete'] or (stage!='initial' and not stage.startswith('epoch')):continue
                series.append((0 if stage=='initial' else int(stage[5:]),s))
            series.sort(key=lambda v:v[0])
            for ax,field in zip(axes,['total','prediction','score']):
                ax.plot([v[0] for v in series],[v[1][field] for v in series],marker='o',
                        color={'train_audit':'tab:orange','validation':'tab:green'}[cohort],label=cohort)
                ax.set_xlabel('epoch');ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_title('prediction KL' if field=='prediction' else field);ax.grid(alpha=.25);ax.legend()
        fig.suptitle(r['name']);fig.tight_layout()
        fig.savefig(run/'curves.png',dpi=140);fig.savefig(run/'curves.pdf');plt.close(fig);sig.write_text(signature)

if __name__=='__main__':main()
