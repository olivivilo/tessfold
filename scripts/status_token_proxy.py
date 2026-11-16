"""Read-only status, including interrupted/incomplete final log lines."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='random_v1')
    a = p.parse_args()
    out = Path(__file__).resolve().parents[1]/'artifacts/token_proxy'/a.label
    protocol = json.loads((out/'protocol.json').read_text())
    def count(pattern):
        return sum(sum(line.endswith(b'\n') for line in path.open('rb')) for path in out.glob(pattern))
    print(json.dumps({'label': a.label, 'teachers_done': count('teachers.rank*.jsonl'),
                      'teachers_expected': protocol['questions'], 'trials_done': count('trials.rank*.jsonl'),
                      'trials_expected': protocol['random_trials']+protocol['control_trials'],
                      'complete_ranks': len(list(out.glob('rank*.complete.json')))}, indent=2))


if __name__ == '__main__':
    main()
