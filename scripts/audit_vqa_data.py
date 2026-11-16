"""Reproduce dataset/sequence evidence for the DocVQA context audit without loading Qwen."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/full/context_audit'
TARGET = 'Dr. Robert E. Shank'


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', action='store_true', help='Also hash the cached tensors for all 171 affected documents')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    prior = {}
    for split in ('train', 'val'):
        counts = collections.Counter()
        matches, near = [], []
        with (ROOT / f'data/full/{split}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['task'] != 'DocVQA':
                    continue
                counts[row['answers'][0].strip().lower()] += 1
                if any(a.strip().lower() == TARGET.lower() for a in row['answers']):
                    matches.append(row)
                if any('shank' in a.lower() for a in row['answers']):
                    near.append(row)
        prior[split] = {'questions': sum(counts.values()), 'exact_target_first_answer_count': counts[TARGET.lower()],
                        'exact_target_any_gold_count': len(matches),
                        'exact_target_any_gold_documents': len({r['document_id'] for r in matches}),
                        'shank_any_gold_questions': len(near), 'top20_first_answers': counts.most_common(20)}
        (OUT / f'{split}_shank_gold.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in near))
    dump('training_prior.json', prior)
    sequence = {'per_rank': []}
    rows = []
    for path in sorted((ROOT / 'artifacts/full').glob('final_val.rank*.jsonl')):
        rr = [json.loads(line) for line in path.read_text().splitlines()]
        rows.extend(r for r in rr if r['task'] == 'DocVQA')
        indices = [i for i,r in enumerate(rr) if r['task'] == 'DocVQA' and r['prediction'] == TARGET]
        if indices:
            i = indices[0]
            sequence['per_rank'].append({'file': path.name, 'target_count': len(indices), 'first_target_line': i+1,
                                        'first_target': rr[i], 'preceding_requests': rr[max(0,i-2):i],
                                        'target_after_same_target_prediction': sum(i>0 and rr[i-1]['prediction'] == TARGET for i in indices)})
    target_rows = [r for r in rows if r['prediction'] == TARGET]
    sequence.update(target_question_first_words=dict(collections.Counter(r['question'].strip().lower().split()[0] for r in target_rows)),
                    target_questions_with_who=sum('who' in r['question'].lower() for r in target_rows),
                    all_docvqa_questions_with_who=sum('who' in r['question'].lower() for r in rows),
                    question_includes_chat_special_tokens=sum(any(t in r['question'] for t in ('<|im_start|>','<|im_end|>','<|vision_start|>','<|vision_end|>')) for r in rows))
    dump('sequence_statistics.json', sequence)
    if args.features:
        os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        import torch
        torch.set_num_threads(4)
        docs = {r['document_id']:r for r in target_rows}
        results = []
        for doc,row in sorted(docs.items()):
            cached = torch.load(ROOT / f'artifacts/full/features/{doc}.pt', map_location='cpu', weights_only=True)
            tensor = cached['features']
            digest = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            assert hashlib.sha256((ROOT / row['image']).read_bytes()).hexdigest() == doc
            assert torch.isfinite(tensor).all()
            results.append({'document_id':doc,'feature_sha256':digest,'shape':list(tensor.shape),
                            'image_sha256_matches':True,'signature':cached['signature']})
        assert len({r['feature_sha256'] for r in results}) == len(results)
        dump('shank_feature_audit.json', {'documents':len(results),'distinct_feature_hashes':len(results),
                                        'all_original_image_hashes_match':True,'all_features_finite':True,'details':results})
    print(json.dumps({'training_prior':prior,'target_predictions':len(target_rows),
                      'target_documents':len({r['document_id'] for r in target_rows})},ensure_ascii=False))


if __name__ == '__main__':
    main()
