#!/usr/bin/env python3
"""Guard one profiler command without capturing or rewriting its output."""
import sys
from pathlib import Path
from guard import run_checked
if len(sys.argv)<4 or sys.argv[2]!='--':
    raise SystemExit('usage: guard_command.py TELEMETRY.jsonl -- COMMAND [ARG ...]')
path=Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True)
with path.open('a') as telemetry:
    run_checked(sys.argv[3:],sys.stdout,sys.stderr,telemetry)
