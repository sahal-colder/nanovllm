#!/usr/bin/env python3
"""Reproduce cache audit; every process runs serially on the same GPU."""
import argparse,shlex,subprocess,json
from pathlib import Path
from guard import run_checked,Phase
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',type=Path,default=Path('build-cuda'))
p.add_argument('--out',type=Path,required=True)
p.add_argument('--profile',action='store_true')
a=p.parse_args();a.build=a.build.resolve();a.out=a.out.resolve()
a.out.mkdir(parents=True,exist_ok=False)
commands=[]
def run(name,args):
    commands.append(shlex.join(map(str,args)));(a.out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(name,flush=True)
    with (a.out/(name+'.csv')).open('w') as out,(a.out/(name+'.log')).open('w') as err,(a.out/'telemetry.jsonl').open('a') as telemetry:
        run_checked(args,out,err,telemetry)
with Phase(a.out,'performance'):
    for dtype,length,batch in [('f32',4096,1),('f16',32768,2),('bf16',32768,2)]:
        for cache in ['hot','cold']:
            run(f'attention-{dtype}-s{length}-b{batch}-{cache}',[a.build/'bench_decode','--audit-attention',dtype,length,batch,cache,100,-1])
    for cache in ['hot','cold']:
        run('operators-'+cache,[a.build/'bench_operators','--audit','--cache',cache,'--samples',20,'--inner',5])
    if a.profile:
        subprocess.run(['python3',str(Path(__file__).with_name('profile_audit.py')),str(a.build),'--out',str(a.out/'ncu')],check=True)
