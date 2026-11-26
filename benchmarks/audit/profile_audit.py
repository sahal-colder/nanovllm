#!/usr/bin/env python3
"""Application replay preserves the real cold partition -> warm scratch merge.
No profiler-injected per-kernel flush: the application evicts before the range.
Operator captures use cache-control all, as independent single-kernel samples.
"""
import argparse,subprocess,shlex,json,csv
from pathlib import Path
from guard import run_checked,Phase
p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.build=a.build.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
metrics=','.join(['dram__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes.sum.per_second','dram__bytes_read.sum','dram__bytes_write.sum','sm__warps_active.avg.pct_of_peak_sustained_active','lts__t_sector_hit_rate.pct','gpu__time_duration.sum'])
commands=[]
def capture(name,args):
    stem=a.out/name
    cmd=['ncu','--metrics',metrics,'--clock-control','none','--export',str(stem),*map(str,args)]
    commands.append(shlex.join(cmd));(a.out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(name,flush=True)
    with stem.with_suffix('.log').open('w') as log,(a.out/'telemetry.jsonl').open('a') as telemetry:
        log.write(shlex.join(cmd)+'\n');log.flush()
        run_checked(cmd,log,subprocess.STDOUT,telemetry)
    reports=list(a.out.glob(name+'.ncu-rep*'));assert len(reports)==1
    r=subprocess.run(['ncu','--import',str(reports[0]),'--csv','--page','raw'],capture_output=True,text=True,check=True)
    stem.with_suffix('.csv').write_text(r.stdout)
with Phase(a.out,'ncu'):
    for dtype,length,batch in [('f32',4096,1),('f16',32768,2),('bf16',32768,2)]:
        for cache in (['hot','cold'] if dtype=='f32' else ['cold']):
            for method in [0,1,2]:
                capture(f'attention-{dtype}-{cache}-m{method}',['--replay-mode','application','--cache-control','none','--profile-from-start','off','--kernel-name','regex:nv_attention_(partition|merge)_kernel',a.build/'bench_decode','--audit-attention',dtype,length,batch,cache,100,method,'--profile'])
    for op,prefix in [('gemv','mi_gemv'),('rms','mi_rms')]:
        for dtype in ['f32','f16','bf16']:
            for method,kernel in [(0,'scalar'),(1,'vec4')]:
                capture(f'{op}-{dtype}-m{method}',['--cache-control','all','--launch-count','1','--kernel-name',f'regex:{prefix}_{kernel}_kernel',a.build/'bench_operators','--profile',op,'--dtype',dtype])
    # Compact, machine readable metrics alongside full raw CSV/native reports.
    rows=[]
    for f in sorted(a.out.glob('*.csv')):
        data=list(csv.reader(f.open()));header=next(i for i,r in enumerate(data) if 'Kernel Name' in r)
        names=data[header];units=data[header+1]
        for r in data[header+2:]:
            if len(r)!=len(names):continue
            d=dict(zip(names,r));row={'case':f.stem,'kernel':d['Kernel Name']}
            for m in metrics.split(','):
                row[m]=d[m];row[m+'_unit']=units[names.index(m)]
            rows.append(row)
    with (a.out/'metrics.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
