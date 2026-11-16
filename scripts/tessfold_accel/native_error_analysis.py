"""Posthoc explanation only: teacher-conditioned QA retention and review CSV.

No model selection. KLC/DeepForm here use per-field EM for conditioning, with
full native corpus F1 separately preserved in the main formal report.
"""
import collections,csv,json
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,lines,save,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def compute():
    assert (BASE/'qwen_formal/complete.json').exists()
    rows=[r for p in (BASE/'qwen_formal').glob('rows.rank*.jsonl') for r in lines(p)]
    teacher={r['id']:r for r in rows if r['model']=='REF'};assert len(teacher)==23565
    registry=load(BASE/'qwen_selected_runs.json');result={};review=[]
    for meta in registry['runs']:
        rr=sorted([r for r in rows if r['model']==meta['id']],key=lambda r:r['id']);assert len(rr)==23565
        groups={}
        for group in ['all','teacher_gt0','teacher_perfect']:
            ss=[r for r in rr if group=='all' or (teacher[r['id']]['score']>0 if group=='teacher_gt0' else teacher[r['id']]['score']>=1.-1e-12)]
            tasks={}
            for task in sorted({r['task'] for r in ss}):
                qq=[r for r in ss if r['task']==task]
                tasks[task]=dict(questions=len(qq),teacher_mean_percent=100*np.mean([teacher[r['id']]['score'] for r in qq]),
                    student_mean_percent=100*np.mean([r['score'] for r in qq]),mean_loss_pp=100*np.mean([teacher[r['id']]['score']-r['score'] for r in qq]),
                    lower_questions=sum(r['score']<teacher[r['id']]['score']-1e-12 for r in qq),equal_questions=sum(abs(r['score']-teacher[r['id']]['score'])<=1e-12 for r in qq),
                    higher_questions=sum(r['score']>teacher[r['id']]['score']+1e-12 for r in qq))
            groups[group]=dict(questions=len(ss),images=len({r['document_id'] for r in ss}),by_task=tasks,
                shared_fieldEM_macro=float(np.mean([v['student_mean_percent'] for v in tasks.values()])),
                teacher_shared_fieldEM_macro=float(np.mean([v['teacher_mean_percent'] for v in tasks.values()])))
        result[meta['id']]=groups
        # Balanced review: largest compression loss per task, plus generated
        # output mismatches on teacher-perfect questions. No cherry-picked score.
        for task in sorted({r['task'] for r in rr}):
            qq=[r for r in rr if r['task']==task]
            chosen=sorted(qq,key=lambda r:(-(teacher[r['id']]['score']-r['score']),r['id']))[:32]
            for r in chosen:
                t=teacher[r['id']];review.append(dict(model=meta['id'],task=task,id=r['id'],document_id=r['document_id'],image=r['image'],question=r['question'],
                    gold_answers=json.dumps(r['answers'],ensure_ascii=False),teacher_prediction=t['prediction'],student_prediction=r['prediction'],
                    teacher_score=t['score'],student_score=r['score'],drop=t['score']-r['score'],original_visual_tokens=r['original_visual_tokens'],visual_tokens=r['visual_tokens'],
                    selection='Per-task32largestscoreloss,tiesbyfixedid; explanationonly,nocheckpointselection',metric='KLC/DeepForm per-fieldEM; other tasks nativeQA'))
    directory=BASE/'error_analysis';directory.mkdir(exist_ok=True)
    with (directory/'largest_losses.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(review[0]));writer.writeheader();writer.writerows(review)
    out=dict(time=now(),results=result,review_rows=len(review),scope='Posthocfulltestexplanation, nocheckpointselection. TeacherconditionusesperQAmetrics, KLC/DeepFormfieldEM; not corpusF1 andnotanunconditionalbenchmarkmacro.')
    save(directory/'teacher_conditioned.json',out);return out

if __name__=='__main__':print(json.dumps(compute(),ensure_ascii=False))
