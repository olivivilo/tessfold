"""Read-only snapshots of growing logs; never used to certify completeness."""
import json,time
from pathlib import Path


def live_lines(path):
    path=Path(path)
    for attempt in range(4):
        try:content=path.read_bytes()
        except FileNotFoundError:return []
        # Only newline-terminated records are committed by the writers. A UTF8
        # codepoint or large JSON write may be incomplete at the current EOF.
        committed=content[:content.rfind(b'\n')+1]
        try:return [json.loads(line) for line in committed.splitlines() if line.strip()]
        except (json.JSONDecodeError,UnicodeDecodeError):
            if attempt==3:raise
            time.sleep(.1)
