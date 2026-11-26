#!/usr/bin/env python3
"""Run original validation and new flush/greedy/dynamic-metadata cases serially."""
import argparse,json,shlex
from pathlib import Path
from guard import run_checked,Phase
p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('out',type=Path);p.add_argument('--full-large-racecheck',action='store_true',help='Optional expensive instrumented 16-layer loop; disabled by default on desktop GPU');a=p.parse_args();a.build=a.build.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
commands=[]
def run(name,args):
    print(name,flush=True);commands.append(shlex.join(map(str,args)));(a.out/'validation-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (a.out/(name+'.stdout')).open('w') as out,(a.out/(name+'-validation.log')).open('w') as err,(a.out/'validation-telemetry.jsonl').open('a') as telemetry:
        run_checked(args,out,err,telemetry)
with Phase(a.out,'validation'):
    run('ctest',['ctest','--test-dir',a.build,'--output-on-failure','-j1'])
    for tool in ['memcheck','racecheck']:
        for name,exe,args in [('decoder','bench_decode',['--test-all',a.build/'fixtures']),('gemv-rms','bench_operators',['--test']),('attention-cold','bench_decode',['--audit-attention','f16',32768,2,'cold',5,-1]),('attention-bf16-cold','bench_decode',['--audit-attention','bf16',32768,2,'cold',5,-1])]:
            run(f'{name}-{tool}',['compute-sanitizer','--tool',tool,'--error-exitcode','99','--log-file',a.out/f'{name}-{tool}.log',a.build/exe,*args])
        for dt in ['f32','f16','bf16']:
            run(f'envelope-{dt}-{tool}',['compute-sanitizer','--tool',tool,'--error-exitcode','99','--log-file',a.out/f'envelope-{dt}-{tool}.log',a.build/'bench_decode','--audit-decode',a.build/'fixtures/perf.bin',dt,5])

        if tool=='racecheck' and not a.full_large_racecheck:continue
        run(f'envelope-large-f16-{tool}',['compute-sanitizer','--tool',tool,'--error-exitcode','99','--log-file',a.out/f'envelope-large-f16-{tool}.log',a.build/'bench_decode','--audit-decode',a.build/'fixtures/audit16.bin','f16',5])

# The optional desktop-heavy case is explicitly outstanding, even when every
# default case succeeds. Do not label the entire 16-case matrix complete.
status_file=a.out/'validation-status.json'
status=json.loads(status_file.read_text())
status.update(completed_cases=16 if a.full_large_racecheck else 15,total_cases=16,
              remaining=[] if a.full_large_racecheck else ['envelope-large-f16-racecheck'])
if not a.full_large_racecheck:
    status['status']='partial'
    status['error']='Optional 16-layer full Racecheck disabled; default 15-case scope passed.'
status_file.write_text(json.dumps(status,indent=2)+'\n')
