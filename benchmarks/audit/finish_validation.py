#!/usr/bin/env python3
"""Close the last Racecheck case with bounded correctness-only execution."""
import argparse,datetime,hashlib,json,os,shlex,shutil,time
from pathlib import Path
from guard import run_checked,Phase

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('build',type=Path);p.add_argument('out',type=Path)
p.add_argument('--model',type=Path,help='Original 16-layer checkpoint (default BUILD/fixtures/audit16.bin)')
a=p.parse_args();a.build=a.build.resolve();a.out=a.out.resolve()
root=Path(__file__).resolve().parents[2]
model=(a.model or a.build/'fixtures/audit16.bin').resolve()
binary=a.build/'bench_decode'
# Bound this process tree directly. Historical system Swap need not belong to
# this job, so use a 1 GiB system ceiling alongside zero job Swap and 6 GiB RAM.
os.environ['NV_BENCH_MAX_SWAP_MIB']='1024'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

status=a.out/'validation-status.json';old=json.loads(status.read_text())
assert old['status']=='partial' and old['remaining']==['envelope-large-f16-racecheck']
cases=['decoder','gemv-rms','attention-cold','attention-bf16-cold','envelope-f32','envelope-f16','envelope-bf16','envelope-large-f16']
for tool in ['memcheck','racecheck']:
    for case in cases:
        if (tool,case)==('racecheck','envelope-large-f16'):continue
        txt=(a.out/f'{case}-{tool}.log').read_text()
        assert ('ERROR SUMMARY: 0 errors' if tool=='memcheck' else '0 errors, 0 warnings') in txt,(tool,case)
        numeric=(a.out/f'{case}-{tool}-validation.log').read_text()
        assert 'FAIL ' not in numeric,(tool,case)
        if case.startswith('envelope-'):assert 'PASS audit decoder' in numeric,(tool,case)

# New validation identity; never rewrite the measured binary identity.
env=json.loads((a.out/'environment.json').read_text())
production={k:v for k,v in env['source_sha256'].items() if k.startswith(('src/','include/')) or k in ['benchmarks/audit/greedy.cu','benchmarks/audit/greedy.h']}
for name,expected in production.items():assert sha(root/name)==expected,name
recorded_model=json.loads((root/'benchmarks/audit/verification/audit16-model.json').read_text())
assert sha(model)==recorded_model['sha256']
resolution=a.out/'racecheck-resolution';resolution.mkdir()
archive=resolution/'previous-incomplete';archive.mkdir()
last='envelope-large-f16-racecheck'
for name in ['validation-status.json','validation-commands.json',last+'.log',last+'.stdout',last+'-validation.log']:
    file=a.out/name
    if file.exists():shutil.copy2(file,archive/name)
source_names=['benchmarks/round2/main.c','benchmarks/round2/decode_bench.c','benchmarks/round2/common.h','benchmarks/audit/finish_validation.py','benchmarks/audit/guard.py']
identity=dict(collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),binary=str(binary),binary_sha256=sha(binary),model=str(model),model_sha256=sha(model),source_sha256={name:sha(root/name) for name in source_names},unchanged_production_source_sha256=production,prior_measurement_binary_sha256=env['binary_sha256'],scope='Correctness only; no warmup, timing or eviction; no kernel filter; full 16-layer FP16 decoder')
identity['resource_limits']=dict(cgroup_MemoryMax_bytes=6*2**30,cgroup_MemorySwapMax_bytes=0,host_swap_ceiling_MiB=1024,min_available_host_MiB=3072,timeout_seconds=180,racecheck_workers=1)
(resolution/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
commands=[];durations={}
def run(name,command):
    command=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=6G','-p','MemorySwapMax=0','nice','-n','10',*command]
    print(name,flush=True);commands.append(dict(name=name,argv=list(map(str,command)),shell=shlex.join(map(str,command))))
    (resolution/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    begin=time.monotonic()
    with (resolution/(name+'.stdout')).open('w') as out,(resolution/(name+'-validation.log')).open('w') as err,(resolution/'telemetry.jsonl').open('a') as telemetry:
        run_checked(command,out,err,telemetry)
    durations[name]=time.monotonic()-begin
    txt=(resolution/(name+'-validation.log')).read_text()
    assert 'FAIL ' not in txt and 'PASS audit decoder dtype=f16 layers=16 dynamic_steps=5' in txt
    assert txt.count('PASS dynamic graph_eager_bitwise_equal=1')==5
    assert txt.count('PASS validation scope=')==3
    assert 'PASS validation_only=1 timing_samples=0 warmup_steps=0 flush_calls=0 kernel_filter=none dynamic_steps=5' in txt
    assert 'KV_slots=16777216' in txt
    assert not (resolution/(name+'.stdout')).read_text(),'Correctness mode must not emit timing CSV'

with Phase(resolution,'resolution'):
    args=[binary,'--validate-decode',model,'f16']
    run('ordinary',args)
    for tool in ['memcheck','racecheck']:
        command=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--log-file',resolution/(tool+'.log')]
        if tool=='racecheck':command+=['--racecheck-num-workers','1']
        run(tool,[*command,*args])
        txt=(resolution/(tool+'.log')).read_text()
        assert ('ERROR SUMMARY: 0 errors' if tool=='memcheck' else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)') in txt,tool

rows=[json.loads(line) for line in (resolution/'telemetry.jsonl').read_text().splitlines()]
summary=dict(status='complete',sanitizer_matrix_completed=16,sanitizer_matrix_total=16,scope=identity['scope'],dynamic_positions=[127,128,129,143,144],physical_KV_slots=16777216,validation_wall_seconds=durations,telemetry_snapshots=len(rows),min_MemAvailable_MiB=min(r['host_memory']['MemAvailable_KiB']/1024 for r in rows),max_swap_MiB=max((r['host_memory']['SwapTotal_KiB']-r['host_memory']['SwapFree_KiB'])/1024 for r in rows),max_process_RSS_MiB=max([int(v['VmRSS'].split()[0])/1024 for r in rows for v in r['process_memory'].values() if 'VmRSS' in v] or [0]),automatic_retries=0,performance_measurements_reused=True,BF16_tolerance_failure_unchanged=True)
(resolution/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for suffix in ['.log','.stdout','-validation.log']:shutil.copy2(resolution/('racecheck'+suffix),a.out/(last+suffix))
status.write_text(json.dumps(dict(status='complete',error=None,completed_cases=16,total_cases=16,remaining=[],updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),resolution=str(resolution.relative_to(root)),scope=identity['scope']),indent=2)+'\n')
print('PASS 16/16 sanitizer matrix; bounded full-model correctness-only Racecheck; old interruptions retained')
