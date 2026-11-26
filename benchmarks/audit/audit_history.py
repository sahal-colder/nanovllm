#!/usr/bin/env python3
"""Read-only audit of archived CSV/logs/source hashes. Never launches CUDA work."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('benchmarks/audit/verification'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];old=root/'benchmarks/operators/round2-results';first=root/'benchmarks/operators/results'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
env=json.loads((old/'environment.json').read_text());first_env=json.loads((first/'environment.json').read_text())
source_checks={}
# environment.original_nano_commit identifies the pre-CUDA upstream, not the
# then-uncommitted CUDA files. Locate their actual blobs in local Git history.
matched_commits={}
for f in ['src/operators/attention.cu','src/operators/sgemv.cu','src/operators/rmsnorm.cu','src/cuda_runtime.c','benchmarks/round2/attention_bench.c','benchmarks/round2/decode_bench.c','benchmarks/operators/bench_operators.c']:
    recorded=env['source_sha256'][f]
    candidates=subprocess.check_output(['git','-C',str(root),'log','--format=%H','--all','--',f],text=True).splitlines()
    for commit in candidates:
        archived=subprocess.check_output(['git','-C',str(root),'show',commit+':'+f])
        if hashlib.sha256(archived).hexdigest()==recorded:
            matched_commits[f]=commit;break
    assert f in matched_commits,f
    source_checks[f]={'historical_source_commit':matched_commits[f],'historical_blob_matches_recorded_hash':True,'current_matches_historical':sha(root/f)==recorded}
for f in ['src/operators/sgemv.cu','src/operators/rmsnorm.cu','src/operators/vector.cuh']:
    assert first_env['source_sha256'][f]==env['source_sha256'][f]
checks={}
for name,needle in [('memcheck-final.log','ERROR SUMMARY: 0 errors'),('racecheck-final.log','0 errors, 0 warnings'),('gemv-rms-memcheck.log','ERROR SUMMARY: 0 errors'),('gemv-rms-racecheck.log','0 errors, 0 warnings'),('ctest.log','100% tests passed, 0 tests failed out of 8')]:
    assert needle in (old/name).read_text();checks[name]=needle
attn=read(old/'attention.csv');operators=read(first/'benchmark.csv');decode=read(old/'decode.csv');rows=[]
for dtype,batch,kv in [('f32',1,4),('f16',1,4),('bf16',1,4),('f16',4,16),('bf16',4,16)]:
    pair=[r for r in attn if r['dtype']==dtype and r['batch']==str(batch) and r['kv_heads']==str(kv) and r['seq_len']=='4096']
    base=next(r for r in pair if r['implementation']=='serial_online');opt=next(r for r in pair if r['implementation']=='split_online_vec4')
    size=4 if dtype=='f32' else 2;kv_bytes=size*2*sum(4096-3*b for b in range(batch))*kv*128
    rows.append(dict(dtype=dtype,batch=batch,heads=16,kv_heads=kv,head_dim=128,seq_len_max=4096,kv_MiB=kv_bytes/2**20,cache_policy='hot_no_explicit_flush',baseline_p50_us=float(base['p50_us']),optimized_p50_us=float(opt['p50_us']),speedup=float(base['p50_us'])/float(opt['p50_us']),baseline_effective_GiB_s=float(base['effective_GiB_s']),optimized_effective_GiB_s=float(opt['effective_GiB_s'])))
for group,data in [('attention',rows),('operators',[r for r in operators if (r['operator']=='sgemv' and r['m']=='8192' and r['k']=='4096' and r['batch']=='1') or (r['operator']=='rmsnorm' and r['m']=='128' and r['k']=='4096')]),('decode',decode)]:
    with (a.out/('historical-'+group+'.csv')).open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
raw=read(old/'ncu-metrics.csv');assert len(raw)==12
for folder,names in [('attention-ncu',['attention-f32','attention-f16','attention-bf16']),('operators-ncu',['gemv-f32','gemv-f16','gemv-bf16','rms-f32','rms-f16','rms-bf16'])]:
    for name in names:
        assert '--cache-control all' in (old/folder/(name+'.log')).read_text()
        assert len(list((old/folder).glob(name+'.ncu-rep*')))==1
        values=list(csv.reader((old/folder/(name+'.csv')).open()));header,units=values[:2]
        for line in values[2:]:
            d=dict(zip(header,line));kernel=d['Kernel Name']
            kind='attention_partition' if 'partition_kernel' in kernel else 'attention_merge' if 'merge_kernel' in kernel else 'sgemv' if 'gemv' in kernel else 'rmsnorm'
            expected=next(r for r in raw if r['operator']==kind and r['dtype']==name.split('-')[-1])
            rate=float(d['dram__bytes.sum.per_second'].replace(',',''));unit=units[header.index('dram__bytes.sum.per_second')]
            scale={'byte/s':1,'Kbyte/s':1e3,'Mbyte/s':1e6,'Gbyte/s':1e9,'Tbyte/s':1e12}[unit]
            assert abs(rate*scale/2**30-float(expected['dram_GiB_s']))<1e-6
            for m,target in [('dram__throughput.avg.pct_of_peak_sustained_elapsed','dram_pct'),('sm__warps_active.avg.pct_of_peak_sustained_active','achieved_occupancy_pct')]:
                assert abs(float(d[m])-float(expected[target]))<1e-6

nsys=json.loads((old/'nsys/timeline-summary.json').read_text());assert nsys['host_launches_per_step']=={'eager':51,'graph':1}
files=[old/'attention.csv',first/'benchmark.csv',old/'decode.csv',old/'ncu-metrics.csv',old/'correctness-final.log',old/'environment.json',first/'environment.json',old/'nsys/decode-graph.sqlite',old/'nsys/decode-graph.nsys-rep',old/'nsys/timeline-summary.json']
files+=list((old/'attention-ncu').glob('*.ncu-rep*'))+list((old/'operators-ncu').glob('*.ncu-rep*'))
result=dict(scope='ARCHIVED evidence only; no new GPU execution or fresh sanitizer claims',source_checks=source_checks,historical_validation=checks,historical_ncu_cache_control='all',historical_nsys_launches=nsys['host_launches_per_step'],evidence_sha256={str(f.relative_to(root)):sha(f) for f in files})
(a.out/'history-evidence.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS archived source hashes, baseline CSV, NCU cache-control, sanitizer logs and NSys launch counts')
