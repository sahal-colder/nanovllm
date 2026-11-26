#!/usr/bin/env python3
"""CPU-only negative checks: incomplete/invalid evidence must never publish PASS."""
import argparse,json,shutil,subprocess,sys,tempfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('results',type=Path)
a=p.parse_args();source=a.results.resolve()
def edit_json(path,change):
    value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value))
def missing_summary(out):
    (out/'envelope-large-f16-racecheck.log').write_text('========= COMPUTE-SANITIZER\n')
def error_summary(out):
    path=out/'envelope-large-f16-racecheck.log'
    path.write_text(path.read_text().replace('0 errors','1 errors'))
def missing_numeric(out):
    path=out/'envelope-large-f16-racecheck-validation.log'
    path.write_text(path.read_text().replace('PASS audit decoder','REMOVED audit decoder'))
def missing_dynamic(out):
    path=out/'racecheck-resolution/racecheck-validation.log'
    value=path.read_text().replace('PASS dynamic graph_eager_bitwise_equal=1 pos=144','REMOVED dynamic pos=144')
    path.write_text(value);(out/'envelope-large-f16-racecheck-validation.log').write_text(value)
def filtered_kernels(out):
    edit_json(out/'racecheck-resolution/commands.json',lambda rows:rows[-1]['argv'].extend(['--kernel-name','kns=some_kernel']))
def changed_production(out):
    edit_json(out/'racecheck-resolution/identity.json',lambda obj:obj['unchanged_production_source_sha256'].__setitem__('src/operators/attention.cu','0'*64))
def swap_enabled(out):
    path=out/'racecheck-resolution/telemetry.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    changed=False
    for row in rows:
        for cg in row.get('process_cgroup',{}).values():
            if cg.get('memory.max')==str(6*2**30):cg['memory.swap.max']='1';changed=True;break
        if changed:break
    assert changed;path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')

for name,mutation in [('missing_summary',missing_summary),('error_summary',error_summary),('missing_numeric',missing_numeric),('missing_dynamic',missing_dynamic),('filtered_kernels',filtered_kernels),('changed_production',changed_production),('swap_enabled',swap_enabled)]:
    # Real copies isolate the evidence: the publisher writes summary outputs.
    with tempfile.TemporaryDirectory(prefix='.audit-negative-',dir=source.parent) as tmp:
        out=Path(tmp)/'data';shutil.copytree(source,out,ignore=shutil.ignore_patterns('tmp'))
        mutation(out)
        run=subprocess.run([sys.executable,str(root/'benchmarks/audit/summarize_audit.py'),str(out)],capture_output=True,text=True)
        assert run.returncode!=0,(name,'invalid evidence was accepted')
        assert 'AssertionError' in run.stderr,(name,run.stderr)
        print('PASS reject '+name)
