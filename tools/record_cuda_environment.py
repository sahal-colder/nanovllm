#!/usr/bin/env python3
"""Record the actual environment and source/binary hashes of a CUDA run."""
import argparse
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
from zoneinfo import ZoneInfo

def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=True)
fields=['name','memory.total','driver_version','pstate','clocks.sm','clocks.mem','temperature.gpu']
gpu=list(csv.reader(io.StringIO(run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits']))))
files=[root/'CMakeLists.txt']
for directory in ['include','src','tests','benchmarks/round2','benchmarks/operators','benchmarks/audit','tools']:
    files += [f for f in (root/directory).rglob('*') if f.is_file() and f.suffix in ['.h','.c','.cu','.cuh','.py','.sh']]
cublas_version=None
for log in sorted(a.out.glob('gemv-rms*validation.log')):
    found=re.search(r'cuBLAS=(\d+)',log.read_text())
    if found:cublas_version=int(found.group(1));break
assert cublas_version is not None, 'A completed operator validation log is required'
data=dict(collected_at=datetime.datetime.now(ZoneInfo('Australia/Perth')).isoformat(),
          gpu_post_run_snapshot=[dict(zip(fields,(v.strip() for v in row))) for row in gpu],
          nvcc=run(['nvcc','--version']),ncu=run(['ncu','--version']),nsys=run(['nsys','--version']),
          cublas_runtime_version_from_validation=cublas_version,
          cmake=run(['cmake','--version']),cc=run(['cc','--version']),
          original_nano_commit=run(['git','-C',str(root),'rev-parse','HEAD']),
          source_state='Uncommitted working tree; file SHA256 is the source identity',
          source_sha256={str(f.relative_to(root)):digest(f) for f in sorted(files)},
          binary_sha256={name:digest(a.build/name) for name in ['bench_decode','bench_operators','nanovllm','nanovllm_cuda']},
          perf_checkpoint_sha256=digest(a.build/'fixtures/perf.bin'),
          perf_config=list(struct.unpack('<7i',(a.build/'fixtures/perf.bin').read_bytes()[:28])))
(a.out/'environment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(a.out/'environment.json')
