#!/usr/bin/env python3
"""Resume only an interrupted last case, checking prior evidence and identity."""
import argparse,datetime,hashlib,json,shlex
from pathlib import Path
from guard import run_checked
p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('out',type=Path);p.add_argument('--allow-heavy',action='store_true');a=p.parse_args();
if not a.allow_heavy:raise SystemExit('Refusing automatic retry of heavy 16-layer Racecheck; desktop responsiveness was affected. Use lightweight report path.')
a.build=a.build.resolve();a.out=a.out.resolve()
status=a.out/'validation-status.json';old=json.loads(status.read_text());assert old['status']=='running'
cases=['decoder','gemv-rms','attention-cold','attention-bf16-cold','envelope-f32','envelope-f16','envelope-bf16','envelope-large-f16']
for tool in ['memcheck','racecheck']:
    for case in cases:
        if (tool,case)==('racecheck','envelope-large-f16'):continue
        txt=(a.out/f'{case}-{tool}.log').read_text()
        assert ('ERROR SUMMARY: 0 errors' if tool=='memcheck' else '0 errors, 0 warnings') in txt,(tool,case)
for name in ['bench_decode','bench_operators']:
    binary=a.build/name
    with binary.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    history=[json.loads(line) for line in (a.out/'validation-telemetry.jsonl').read_text().splitlines()]
    stored={row.get('artifacts_sha256',{}).get(str(binary)) for row in history};stored.discard(None)
    if not stored:
        if name=='bench_decode':
            rows=[json.loads(line) for line in (a.out/'decode-telemetry.jsonl').read_text().splitlines()]
            stored={row.get('artifacts_sha256',{}).get(str(binary)) for row in rows};stored.discard(None)
        else:stored={json.loads((a.out/'initial-measurement-state.json').read_text())['binary_sha256'][name]}
    assert stored=={digest},(name,stored,digest)
last='envelope-large-f16-racecheck';stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive=a.out/('interrupted-last-case-'+stamp);archive.mkdir()
for suffix in ['.log','.stdout','-validation.log']:
    file=a.out/(last+suffix)
    if file.exists():file.rename(archive/file.name)
command=['compute-sanitizer','--tool','racecheck','--error-exitcode','99','--log-file',str(a.out/(last+'.log')),str(a.build/'bench_decode'),'--audit-decode',str(a.build/'fixtures/audit16.bin'),'f16','5']
(a.out/'validation-resume.json').write_text(json.dumps(dict(prior_status=old,reason='Last-case log had no final summary and its process no longer existed; retained incomplete logs and resumed same binary only.',command=shlex.join(command),archive=str(archive)),indent=2)+'\n')
with (a.out/(last+'.stdout')).open('w') as out,(a.out/(last+'-validation.log')).open('w') as err,(a.out/'validation-telemetry.jsonl').open('a') as telemetry:
    run_checked(command,out,err,telemetry)
assert '0 errors, 0 warnings' in (a.out/(last+'.log')).read_text()
assert 'PASS audit decoder' in (a.out/(last+'-validation.log')).read_text()
status.write_text(json.dumps(dict(status='complete',error=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),resumed_last_case=True),indent=2)+'\n')
print('PASS all 16 sanitizer cases; last case resumed with identical binary')
