"""Read-only progress/selection statistics; refuses to treat partial eval as final."""
from pathlib import Path
import argparse
import json
import math
from collections import defaultdict
import csv

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/meta_compress_search'


def read(path):
    # Monitoring may race the final write of a JSONL record. Ignore only an
    # unfinished final line; never hide corrupt committed/middle records.
    text=path.read_text();lines=text.splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not text.endswith('\n'):break
            raise
    return rows


def average(values):return sum(values)/len(values) if values else None


def summarize(rows,expected):
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate evaluation IDs')
    bytask=defaultdict(list)
    for row in rows:bytask[row['task']].append(row)
    tasks={t:{'n':len(rr),'score':100*average([r['student_score'] for r in rr]),
              'teacher':100*average([r['teacher_score'] for r in rr])} for t,rr in bytask.items()}
    strata={}
    for name,predicate in [('teacher_positive',lambda r:r['teacher_score']>0),('teacher_full',lambda r:r['teacher_score']>=1-1e-8)]:
        selected=[r for r in rows if predicate(r)]
        strata[name]={'n':len(selected),'student_micro_score':100*average([r['student_score'] for r in selected]) if selected else None,
            'teacher_micro_score':100*average([r['teacher_score'] for r in selected]) if selected else None,
            'note':'Descriptive subset only; not the unfiltered task macro or model-selection metric'}
    normalize=lambda s:' '.join(str(s).lower().split())
    exact=lambda pred,answers:float(any(normalize(pred)==normalize(a) for a in answers))
    exacts={t:{'student':100*average([exact(r['student_prediction'],r['answers']) for r in rr]),
               'teacher':100*average([exact(r['teacher_prediction'],r['answers']) for r in rr])} for t,rr in bytask.items()}
    return {'n':len(rows),'complete':set(ids)==set(expected),'score':average([r['score'] for r in tasks.values()]),
            'teacher':average([r['teacher'] for r in tasks.values()]),'tasks':tasks,
            'teacher_strata':strata,'secondary_case_whitespace_exact':{'student_macro':average([r['student'] for r in exacts.values()]),
                'teacher_macro':average([r['teacher'] for r in exacts.values()]),'tasks':exacts,'used_for_selection':False},
            **{k:average([r[k] for r in rows]) for k in ['prediction','entropy','collapse','objective','column_mass_fraction','entropy_fraction','row_max_mean']}}


def paired(a,b):
    """Image-level paired bootstrap, stratified by task; same image, same budget."""
    import numpy as np
    aa={r['id']:r for r in a};bb={r['id']:r for r in b}
    assert aa.keys()==bb.keys()
    rng=np.random.default_rng(20261028);sample=np.zeros(10000)
    tasks=sorted({r['task'] for r in a});delta=[]
    for t in tasks:
        dd=np.array([bb[i]['student_score']-aa[i]['student_score'] for i in aa if aa[i]['task']==t])*100
        delta.extend(dd.tolist());sample+=dd[rng.integers(len(dd),size=(10000,len(dd)))].mean(1)/len(tasks)
    return {'gain_points':float(np.mean(delta)),'paired_bootstrap_95ci_points':np.quantile(sample,[.025,.975]).tolist(),
            'improved':sum(x>0 for x in delta),'worsened':sum(x<0 for x in delta),'unchanged':sum(x==0 for x in delta)}


def collect(run,cohort='search'):
    directory=run if cohort=='search' else run/cohort
    expected={r['id'] for r in read(BASE/f'{cohort}.jsonl')}
    grouped=defaultdict(list)
    for f in directory.glob('eval.*.rank*.jsonl'):
        stage,variant=f.name.split('.')[1:3];grouped[f'{stage}/{variant}']+=read(f)
    stages={k:summarize(v,expected) for k,v in grouped.items() if v}
    cfg=json.loads((run/'manifest.json').read_text())['config']
    result={'name':run.name,'config':cfg,'training_complete':(run/'complete.json').exists(),'stages':stages}
    if (run/'early_stop.json').exists():result['early_stop']=json.loads((run/'early_stop.json').read_text())
    meanpath=BASE/f"baselines/ratio{cfg['ratio']:g}/{cohort}"
    meanrows=[r for f in meanpath.glob('eval.*.rank*.jsonl') for r in read(f)]
    baseline=summarize(meanrows,expected) if meanrows else None
    result['matched_mean']=baseline
    initial_key='initial/trained' if cohort=='search' else f'{cohort}/initial'
    if initial_key in stages and stages[initial_key]['complete']:
        initial=stages[initial_key];options=[]
        suitability=BASE/'suitability_addendum.json'
        minimum=json.loads(suitability.read_text()).get('minimum_teacher_retention',0.) if suitability.exists() else 0.
        for key,s in stages.items():
            if key==initial_key or not s['complete']:continue
            s['paired_vs_initial']=paired(grouped[initial_key],grouped[key])
            s['kl_relative_reduction']=1-s['prediction']/initial['prediction']
            s['screen_pass']=(s['score']>=initial['score']+2 and s['kl_relative_reduction']>=.05 and s['column_mass_fraction']<=.25)
            s['teacher_retention']=s['score']/s['teacher'] if s['teacher'] else None
            s['practical_pass']=bool(s['screen_pass'] and baseline and baseline['complete'] and
                s['score']>=baseline['score']+2 and s['score']>=minimum*s['teacher'])
            if s['practical_pass']:options.append(key)
        if options:result['best_passing_stage']=max(options,key=lambda k:(stages[k]['score'],-stages[k]['prediction']))
    if (run/'training.jsonl').exists():
        log=read(run/'training.jsonl');result['updates_completed']=max([r['step'] for r in log],default=0)
        result['last_update']=log[-1] if log else None
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',default='search');a=p.parse_args()
    runs=[]
    for run in sorted((BASE/'runs').glob('*')):
        if run.name=='smoke' or not (run/'manifest.json').exists():continue
        runs.append(collect(run,a.cohort))
    (BASE/f'summary.{a.cohort}.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# 多图调参逐阶段实测','','score 为七任务原生分数宏平均（百分制）；某个 epoch 的评测完整不代表原定全部训练完成。','',
           '| run | 实际更新数 | 原定全部训练完成 | 提前停止 |','|---|---:|---|---|']
    for r in runs:lines.append(f"| {r['name']} | {r.get('updates_completed',0)} | {r['training_complete']} | {'是，见 early_stop.json' if r.get('early_stop') else '否'} |")
    lines+=['','## 逐阶段验证','',
           '| run | stage | n | 完整 | score | teacher | KL | H | C/m | Δscore | 95% CI | 初始改善 / 胜过 mean |',
           '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in runs:
        for key,s in sorted(r['stages'].items()):
            pair=s.get('paired_vs_initial',{});ci=pair.get('paired_bootstrap_95ci_points')
            lines.append(f"| {r['name']} | {key} | {s['n']} | {s['complete']} | {s['score']:.3f} | {s['teacher']:.3f} | {s['prediction']:.5f} | {s['entropy']:.3f} | {s['column_mass_fraction']:.4f} | {pair.get('gain_points',0):+.3f} | {str([round(x,3) for x in ci]) if ci else '—'} | {s.get('screen_pass','—')} / {s.get('practical_pass','—')} |")
        if r.get('matched_mean'):
            s=r['matched_mean'];lines.append(f"| {r['name']} | matched mean | {s['n']} | {s['complete']} | {s['score']:.3f} | {s['teacher']:.3f} | {s['prediction']:.5f} | {s['entropy']:.3f} | {s['column_mass_fraction']:.4f} | — | — | — |")
    lines+=['','## 补充答案正确性核查','',
        '严格 exact 仅忽略大小写与空白，跨任务不等同于各自的正式 scorer；只作审查，不用于选择。teacher>0 / 满分分组是子集微平均，不能与总体任务宏平均混比。','',
        '| run | stage | 严格 exact 宏均分 | teacher>0 的 n / student 分数 | teacher满分的 n / student 分数 |',
        '|---|---|---:|---|---|']
    for r in runs:
        for key,s in sorted(r['stages'].items()):
            if not s['complete']:continue
            pos=s['teacher_strata']['teacher_positive'];full=s['teacher_strata']['teacher_full']
            posvalue=f"{pos['student_micro_score']:.3f}" if pos['n'] else '—'
            fullvalue=f"{full['student_micro_score']:.3f}" if full['n'] else '—'
            lines.append(f"| {r['name']} | {key} | {s['secondary_case_whitespace_exact']['student_macro']:.3f} | {pos['n']} / {posvalue} | {full['n']} / {fullvalue} |")
    (BASE/f'RESULTS.{a.cohort}.md').write_text('\n'.join(lines)+'\n')
    fields=['run','stage','variant','task','id','document_id','question','answers','teacher_prediction','student_prediction',
            'teacher_score','student_score','prediction','entropy','collapse','objective','ratio','tokens','compressed_tokens','column_mass_fraction']
    with (BASE/f'answers.{a.cohort}.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for r in runs:
            directory=BASE/'runs'/r['name']
            if a.cohort!='search':directory=directory/a.cohort
            for f in sorted(directory.glob('eval.*.rank*.jsonl')):
                for row in read(f):writer.writerow({**row,'run':r['name'],'answers':json.dumps(row['answers'],ensure_ascii=False)})
    for r in runs:
        print(r['name'],r.get('updates_completed',0),r.get('best_passing_stage','no complete passing stage'))
        if r.get('early_stop'):print('  EARLY STOP:',r['early_stop']['reason'])


if __name__=='__main__':main()
