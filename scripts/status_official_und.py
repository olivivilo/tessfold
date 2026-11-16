"""Read-only progress for the official baseline / SFT / final chain."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/official_und'


def main():
    queue = json.loads((OUT / 'queue.json').read_text())
    jobs = {k: queue[k] for k in ('baseline_full', 'sft', 'final') if k in queue}
    job_ids = ','.join(map(str, jobs.values()))
    states = subprocess.check_output(['sacct', '-n', '-X', '-j', job_ids,
                                     '--format=JobID,State,Elapsed', '--parsable2'], text=True).strip()
    result = {'jobs': jobs, 'slurm': states.splitlines()}
    path = OUT / 'sft/logging.jsonl'
    if path.exists():
        history = []
        for line in path.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Last line can be in the middle of a logging write.
        if history:
            latest = next((r for r in reversed(history) if 'global_step/max_steps' in r), history[-1])
            result['training'] = {k: latest[k] for k in ('global_step/max_steps', 'epoch', 'loss', 'remaining_time') if k in latest}
            result['validation_loss'] = [{k: row[k] for k in ('global_step/max_steps', 'eval_loss')}
                                         for row in history if 'eval_loss' in row]
    for label in ('baseline_full', 'finetuned_validation', 'finetuned_full'):
        path = OUT / f'{label}_metrics.json'
        if path.exists():
            scores = json.loads(path.read_text())
            result[label] = {'samples': scores['samples'], 'task_macro_mean_percent': scores['task_macro_mean_percent']}
        elif label == 'finetuned_full':
            shards = list(OUT.glob(f'{label}.rank*.jsonl'))
            if shards:
                result['finetuned_full_generated'] = sum(p.read_bytes().count(b'\n') for p in shards)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
