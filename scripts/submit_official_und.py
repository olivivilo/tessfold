"""Submit an idempotent baseline -> official SFT -> validation/final-eval chain."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/official_und'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queue = OUT / 'queue.json'
    jobs = json.loads(queue.read_text()) if queue.exists() else {}
    def submit(key, script, args=(), dependencies=()):
        if key in jobs:
            print(f'{key}: existing job {jobs[key]} (inspect sacct before retrying failures)')
            return jobs[key]
        command = ['sbatch', '--parsable']
        pending = []
        for dependency in dependencies:
            state = subprocess.check_output(['squeue', '-h', '-j', str(dependency), '-o', '%T'],
                                            text=True).strip().splitlines()
            if not state:
                state = subprocess.check_output(['sacct', '-n', '-X', '-j', str(dependency),
                                                 '--format=State', '--parsable2'], text=True).strip().splitlines()
            assert len(state) == 1, f'Cannot establish dependency state: {dependency}: {state}'
            if state[0].strip('| ') == 'COMPLETED':
                continue  # Slurm forgets completed job IDs after MinJobAge.
            assert state[0].strip('| ') in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'), (dependency, state)
            pending.append(dependency)
        if pending:
            command.append('--dependency=afterok:' + ':'.join(map(str, pending)))
        command.extend([script, *args])
        job = int(subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0])
        jobs[key] = job
        queue.write_text(json.dumps(jobs, indent=2))
        print(f'{key}: submitted {job}')
        return job
    validation = submit('baseline_validation', 'scripts/slurm_official_eval.sh', ['--label', 'baseline_validation', '--mode', 'validation'])
    baseline = submit('baseline_full', 'scripts/slurm_official_eval.sh', ['--label', 'baseline_full'])
    # A real fixed-validation baseline precedes training; the full benchmark can
    # finish concurrently without affecting training or checkpoint selection.
    dependencies = [validation]
    if 'sft_probe' in jobs:
        dependencies.append(jobs['sft_probe'])
    sft = submit('sft', 'scripts/slurm_official_sft.sh', dependencies=dependencies)
    submit('final', 'scripts/slurm_official_final.sh', dependencies=[sft, baseline])


if __name__ == '__main__':
    main()
