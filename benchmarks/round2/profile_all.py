#!/usr/bin/env python3
"""Run only NCU workload captures, without changing driver configuration."""
import argparse
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('build',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2]
for script,binary,folder in [
    (root/'benchmarks/operators/profile_ncu.py',a.build/'bench_operators','operators-ncu'),
    (root/'benchmarks/round2/profile_attention.py',a.build/'bench_decode','attention-ncu')]:
    subprocess.run([sys.executable,str(script),str(binary.resolve()),'--ncu','/usr/local/cuda/bin/ncu',
                    '--out',str((a.out/folder).resolve())],check=True)
