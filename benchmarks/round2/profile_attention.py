#!/usr/bin/env python3
"""Capture both optimized attention kernels, retaining native NCU reports."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys

METRICS='dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second,sm__warps_active.avg.pct_of_peak_sustained_active'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path)
p.add_argument('--ncu',default=shutil.which('ncu'))
p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'operators/round2-results/ncu')
a=p.parse_args()
if not a.ncu:p.error('ncu unavailable')
a.out.mkdir(parents=True,exist_ok=True)
status=[]
for dtype in ['f32','f16','bf16']:
    stem=a.out.resolve()/('attention-'+dtype)
    if list(a.out.glob(stem.name+'.ncu-rep*')):p.error('choose a fresh output directory')
    command=[a.ncu,'--metrics',METRICS,'--kernel-name','regex:nv_attention_(partition|merge)_kernel',
             '--launch-count','2','--clock-control','none','--cache-control','all','--export',str(stem),
             str(a.binary.resolve()),'--profile-attention',dtype]
    r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    stem.with_suffix('.log').write_text('$ '+' '.join(command)+'\n'+r.stdout)
    reports=[f for f in [stem.with_suffix('.ncu-rep'),stem.with_suffix('.ncu-repz')] if f.exists()]
    reason=''
    if r.returncode or len(reports)!=1:reason='capture_failed'
    else:
        result=subprocess.run([a.ncu,'--import',str(reports[0]),'--csv','--page','raw'],capture_output=True,text=True)
        stem.with_suffix('.export.log').write_text(result.stderr)
        if result.returncode or any(k not in result.stdout for k in METRICS.split(',')):reason='export_failed'
        else:stem.with_suffix('.csv').write_text(result.stdout)
    status.append(dict(dtype=dtype,status='captured' if not reason else 'failed',reason=reason,exit_code=r.returncode,
                       report=reports[0].name if reports else '',csv=stem.with_suffix('.csv').name if not reason else ''))
    print(dtype,reason or 'captured',flush=True)
with (a.out/'status.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(status[0]));writer.writeheader();writer.writerows(status)
sys.exit(1 if any(x['reason'] for x in status) else 0)
