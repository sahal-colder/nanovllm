#!/usr/bin/env python3
import argparse,json,shlex
from pathlib import Path
from guard import run_checked,Phase
p=argparse.ArgumentParser(description='Cold-L2 compute / full-CPU / full-GPU envelope benchmark')
p.add_argument('build',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.build=a.build.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
commands=[]
with Phase(a.out,'decode'):
    for name,model,dt in [('small-f16','perf.bin','f16'),('large-f16','audit16.bin','f16')]:
        cmd=[a.build/'bench_decode','--audit-decode',a.build/'fixtures'/model,dt,100];print(name,flush=True)
        commands.append(shlex.join(map(str,cmd)));(a.out/'decode-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        with (a.out/('decode-'+name+'.csv')).open('w') as out,(a.out/('decode-'+name+'.log')).open('w') as err,(a.out/'decode-telemetry.jsonl').open('a') as telemetry:
            run_checked(cmd,out,err,telemetry)
