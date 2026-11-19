"""Read-only answer-position audit of strict MetaCompress checkpoints.

Uses the unchanged full-vocabulary KL on the existing validation cohort. It
decomposes measurements; it does not reweight the training objective or use
gold answers in model forwards. Free generations come from committed evals.
"""
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
import torch.distributed as dist

from scripts.train_meta_compress_strict import ROOT, BASE, Examples, read
from scripts.train_meta_compress import init_runtime, barrier, atomic_json, sha
from scripts.benchmark_meta_compress_strict import build, project
from scripts.eval_token_proxy import shortened_inputs
from vqa.decoder_dependent_loss import common_prefix_logits
from vqa.official_und import OfficialUND


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['stage'], row['task'])].append(row)
    result = {}
    for (stage, task), rows in groups.items():
        fields = ['mean_kl', 'first_kl', 'forced_label_accuracy',
                  'teacher_label_agreement', 'first_label_margin',
                  'student_score', 'teacher_score', 'free_prefix_fraction']
        values = {k: sum(r[k] for r in rows) / len(rows) for k in fields}
        values['questions'] = len(rows)
        for kind in ['first', 'later', 'digits', 'eos']:
            positions = [v for r in rows for v in r['positions'] if kind in v['groups']]
            values[kind] = {'positions': len(positions)}
            if positions:
                values[kind].update({k: sum(v[k] for v in positions) / len(positions)
                                    for k in ['kl', 'label_probability', 'label_margin', 'label_rank']})
        result.setdefault(stage, {})[task] = values
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', action='append', required=True,
                   help='Committed evaluation stage=checkpoint, e.g. epoch8=.../epoch8.pt')
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    cfg = json.loads((ROOT/a.config).read_text())
    checkpoints = dict(v.split('=', 1) for v in a.checkpoint)
    assert all(s == 'initial' or re.fullmatch(r'epoch\d+', s) for s in checkpoints)
    run = ROOT/cfg['output']
    out = run/'token_audit'/('__'.join(checkpoints))
    out.mkdir(parents=True, exist_ok=True)
    rows = read(ROOT/cfg['validation_rows'])
    rank, world, _, device = init_runtime('npu', a.threads)
    sources = ['scripts/audit_meta_compress_strict_tokens.py', 'scripts/benchmark_meta_compress_strict.py',
               'scripts/train_meta_compress_strict.py', 'vqa/meta_compress.py',
               'vqa/decoder_dependent_loss.py', 'scripts/eval_token_proxy.py']
    signature = {'config': cfg, 'checkpoints': {k: sha(ROOT/v) for k, v in checkpoints.items()},
                 'validation_sha256': sha(ROOT/cfg['validation_rows']),
                 'world': world, 'source_sha256': {s: sha(ROOT/s) for s in sources},
                 'purpose': 'Read-only diagnostic; unchanged losses; validation only, no confirmation/test.'}
    manifest = out/f'manifest.rank{rank}.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature
    atomic_json(manifest, signature)
    generations = {}
    for stage in checkpoints:
        rr = [r for f in run.glob(f'eval.validation.{stage}.rank*.jsonl') for r in read(f)]
        assert len(rr) == len(rows) and {r['id'] for r in rr} == {r['id'] for r in rows}
        generations[stage] = {r['id']: r for r in rr}
    engine = OfficialUND(device=str(device))
    engine.model.eval().requires_grad_(False)
    examples = Examples(engine, 'npu')
    modules = {}
    for stage, checkpoint in checkpoints.items():
        module = build(cfg)
        module.load_state_dict(torch.load(ROOT/checkpoint, map_location='cpu', weights_only=False)['generator'])
        modules[stage] = module.to(device).eval()
    output = out/f'positions.rank{rank}.jsonl'
    committed = read(output)
    if output.exists():
        output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in committed))
    done = {(r['id'], r['stage']) for r in committed}
    eos = engine.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with torch.no_grad(), output.open('a') as f:
        for index, row in enumerate(rows[rank::world]):
            if all((row['id'], stage) in done for stage in modules):
                continue
            prompt, raw, grid, x, response, teacher, _ = examples.get(row)
            labels = torch.tensor(response, device=device)
            t = teacher[0].float()
            tlogp = t.log_softmax(-1)
            ttop = t.argmax(-1)
            for stage, module in modules.items():
                if (row['id'], stage) in done:
                    continue
                sp, _ = project(engine, module, raw, grid, x, cfg)
                si = shortened_inputs({'input_ids': prompt}, len(sp), engine.model.config.image_token_id)
                s = common_prefix_logits(engine.model, si['input_ids'], sp[None], response)[0].float()
                slogp = s.log_softmax(-1)
                kl = (tlogp.exp() * (tlogp-slogp)).sum(-1)
                assert torch.isfinite(kl).all()
                label_logits = s.gather(-1, labels[:, None]).squeeze(-1)
                alternatives = s.clone()
                alternatives.scatter_(-1, labels[:, None], float('-inf'))
                margin = label_logits-alternatives.max(-1).values
                ranks = (s > label_logits[:, None]).sum(-1)+1
                probability = slogp.gather(-1, labels[:, None]).squeeze(-1).exp()
                positions = []
                for i, token_id in enumerate(response):
                    token = engine.processor.decode([token_id], skip_special_tokens=False)
                    tags = ['first' if i == 0 else 'later']
                    if re.search(r'\d', token):
                        tags.append('digits')
                    if token_id in eos:
                        tags.append('eos')
                    positions.append({'index': i, 'token_id': token_id, 'token': token, 'groups': tags,
                                      'kl': float(kl[i]), 'label_probability': float(probability[i]),
                                      'label_margin': float(margin[i]), 'label_rank': int(ranks[i])})
                gen = generations[stage][row['id']]
                generated = gen['student_generated_ids']
                prefix = 0
                for aa, bb in zip(response, generated):
                    if aa != bb:
                        break
                    prefix += 1
                item = {'id': row['id'], 'document_id': row['document_id'], 'task': row['task'],
                        'stage': stage, 'mean_kl': float(kl.mean()), 'first_kl': float(kl[0]),
                        'forced_label_accuracy': float((s.argmax(-1) == labels).float().mean()),
                        'teacher_label_agreement': float((ttop == labels).float().mean()),
                        'first_label_margin': float(margin[0]),
                        'student_score': gen['student_score'], 'teacher_score': gen['teacher_score'],
                        'free_prefix_fraction': prefix/len(response), 'free_common_prefix_tokens': prefix,
                        'teacher_tokens': len(response), 'positions': positions}
                f.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
                f.flush()
            if index % 25 == 0:
                print({'rank': rank, 'questions': index+1, 'total': len(rows[rank::world])}, flush=True)
    barrier(world)
    if rank == 0:
        records = [r for ff in out.glob('positions.rank*.jsonl') for r in read(ff)]
        assert len(records) == len(rows)*len(checkpoints)
        assert len({(r['id'], r['stage']) for r in records}) == len(records)
        stats = summarize(records)
        atomic_json(out/'summary.json', stats)
        lines = ['# MetaCompress: Answer-Position Audit on the Validation Set', '',
                 'Decompose the original KL without changing loss, trajectory, or generation. First-position/digit/EOS groups may overlap. Digit groups detect digits in individual teacher-token decoding, not complete numerical entities.', '',
                 '| stage | Task | QA | KL | First-token KL | First-token teacher-label margin | Teacher-prefix label agreement | VQA | teacher |',
                 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for stage, tasks in stats.items():
            for task, v in sorted(tasks.items()):
                lines.append(f"| {stage} | {task} | {v['questions']} | {v['mean_kl']:.5f} | {v['first_kl']:.5f} | {v['first_label_margin']:.4f} | {v['forced_label_accuracy']:.3f} | {100*v['student_score']:.2f} | {100*v['teacher_score']:.2f} |")
        lines += ['', 'Margin is the student\'s logit for the teacher response token minus its strongest alternative logit. Negative values mean that the teacher token is not the student\'s argmax. Teacher self-label agreement is retained to expose BF16 differences between parallel forward evaluation and stepwise generation.', '',
                  'Original position/question records and additional group means are in local JSONL/summary.json. Validation diagnosis is not independent confirmation.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        atomic_json(out/'complete.json', {'questions': len(rows), 'stages': list(checkpoints), 'records': len(records)})
    barrier(world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
