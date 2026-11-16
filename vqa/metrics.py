"""Metrics in [0,1]; aggregate reporting multiplies by 100."""
import re
import unicodedata

def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).lower().split())

def distance(a, b):
    previous = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(ca != cb)))
        previous = current
    return previous[-1]

def anls(prediction, answers):
    p = normalize(prediction)
    scores = []
    for answer in answers:
        a = normalize(answer)
        d = distance(p, a) / max(len(p), len(a), 1)
        scores.append(1-d if d < 0.5 else 0.0)
    return max(scores, default=0.0)

def exact_match(prediction, answers):
    return float(any(normalize(prediction) == normalize(a) for a in answers))

def relaxed_accuracy(prediction, answers):
    p = normalize(prediction)
    for answer in answers:
        a = normalize(answer)
        try:
            # Allow commas and percent symbols as used in ChartQA numeric scoring.
            pv = float(p.replace(',', '').rstrip('%'))
            av = float(a.replace(',', '').rstrip('%'))
            if abs(pv-av) <= 0.05 * abs(av):
                return 1.0
        except ValueError:
            if p == a:
                return 1.0
    return 0.0
