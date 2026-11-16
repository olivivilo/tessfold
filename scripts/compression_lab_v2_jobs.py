"""Read the user's live queue without querying expired individual job IDs."""
import getpass
import subprocess


def live_jobs():
    result = subprocess.run(
        ['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j'],
        text=True, capture_output=True, check=True)
    return dict(line.strip().split('|', 1) for line in result.stdout.splitlines()
                if line.strip())


def active(job):
    return bool(job) and str(job) in live_jobs()


def named(name):
    return name in live_jobs().values()
