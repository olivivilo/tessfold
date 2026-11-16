"""Recover append-only prediction logs after an interrupted write."""
import json


def recover_predictions(path):
    if not path.exists():
        return {}
    records = {}
    # Only an incomplete final line can result from an interrupted append.
    # A malformed complete/interior line is data corruption and must be reported.
    with path.open('rb+') as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                handle.truncate(start)
                break
            row = json.loads(line)
            records[row['id']] = row
    return records
