"""Task-specific scorers, preserving original official scoring rules."""
from .metrics import distance, exact_match, relaxed_accuracy

def chartqa_score(prediction, answers):
    # Qwen-VL / lmms-eval official relaxed correctness, including percent scaling.
    def number(text):
        try:
            return float(text.rstrip('%')) / 100 if text.endswith('%') else float(text)
        except ValueError:
            return None
    p=number(prediction)
    for answer in answers:
        a=number(answer)
        if p is not None and a:
            if abs(p-a)/abs(a)<=.05:return 1.0
        elif prediction.lower()==answer.lower():return 1.0
    return 0.0

def dtvqa_scores(prediction, answers):
    anls=benchmark_anls(prediction,answers)
    accuracy=float(any(str(a).lower() in prediction.lower() for a in answers))
    return {'ANLS':anls,'accuracy':accuracy,'accanls':max(accuracy,anls)}

def benchmark_anls(prediction, answers):
    # Original DocVQA/InfoVQA/DT-VQA: lowercase, raw characters, threshold strictly < .5.
    scores=[]
    for a in answers:
        p,a=str(prediction).lower(),str(a).lower()
        length=max(len(p),len(a))
        d=distance(p,a)/length if length else 1
        scores.append(1-d if d<.5 else 0)
    return max(scores,default=0)

def wtq_score(prediction, answers):
    from due_evaluator.scorers.wtq_scorer import WtqScorer
    scorer=WtqScorer()
    def values(s):
        return [p.strip() for p in str(s).split('|')]
    return float(any(scorer.check_denotation(values(prediction),values(a)) for a in answers))

def due_field_f1(rows):
    from due_evaluator.scorers.fscorer import FScorer
    scorer=FScorer()
    documents={}
    for row in rows:documents.setdefault(row['document_id'],[]).append(row)
    def empty(s):return str(s).strip().lower() in ('none','null','not found','n/a','')
    for doc,records in documents.items():
        gold,pred=[],[]
        for row in records:
            key=row['question'].lower().strip()
            answers=[a.lower().strip() for a in row['answers'] if not empty(a)]
            if answers:
                gold.append({'key':key,'values':[{'value':answers[0],'value_variants':answers[1:]}]})
            if not empty(row['prediction']):
                pred.append({'key':key,'values':[{'value':row['prediction'].lower().strip()}]})
        scorer.add({'name':doc,'annotations':pred},{'name':doc,'annotations':gold})
    return {'score_percent':100*scorer.score(),'metric':'DUE corpus set F1',
            'precision':scorer.precision(),'recall':scorer.recall(),
            'reference_format':'Original Monkey single-page per-field QA converted to DU-schema; None means absent field'}

def ocrbench_score(prediction, row):
    if row.get('dataset_name')=='HME100k':
        p=''.join(prediction.split())
        return float(any(''.join(a.split()) in p for a in row['answers']))
    p=prediction.lower().strip().replace('\n',' ')
    return float(any(a.lower().strip().replace('\n',' ') in p for a in row['answers']))

def native_score(prediction,row):
    task=row['task']
    if task in ('DocVQA','InfoVQA'):return benchmark_anls(prediction,row['answers'])
    if task=='DT-VQA':return dtvqa_scores(prediction,row['answers'])['accanls']
    if task=='ChartQA':return chartqa_score(prediction,row['answers'])
    if task=='WTQ':return wtq_score(prediction,row['answers'])
    if task=='OCRBench':return ocrbench_score(prediction,row)
    return exact_match(prediction,row['answers'])
