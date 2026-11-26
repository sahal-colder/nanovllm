#!/usr/bin/env python3
"""Strict CPU/eager/Graph greedy token check using pinned pretrained weights."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('models',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
expected={'stories260K.bin':'b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696',
          'tok512.bin':'037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312'}
for name,digest in expected.items():assert hashlib.sha256((a.models/name).read_bytes()).hexdigest()==digest
base=[str((a.models/'stories260K.bin').resolve()),'-z',str((a.models/'tok512.bin').resolve()),'-n','32','-t','0','--ids','--ignore-eos','--batch','2','--budget','16']
prompts=['Once upon a time','A little girl','The cat','One day there was a boy']
for prompt in prompts:base+=['-i',prompt]
runs={}
for mode,exe,flags in [('cpu','nanovllm',[]),('eager','nanovllm_cuda',['--cuda']),('graph','nanovllm_cuda',['--cuda-graph'])]:
    cmd=[str((a.build/exe).resolve()),*base,*flags]
    r=subprocess.run(cmd,text=True,capture_output=True,check=True)
    (a.out/(mode+'.stdout')).write_text(r.stdout);(a.out/(mode+'.log')).write_text('$ '+' '.join(cmd)+'\n'+r.stderr);runs[mode]=r.stdout
assert runs['cpu']==runs['eager']==runs['graph'],'strict pretrained token mismatch'
data=dict(model='TinyStories 260K',dtype='f32',prompts=prompts,generated_tokens_per_prompt=32,total_generated_tokens=128,
          cpu_eager_graph_exact_token_identity=True,sha256=expected,
          config=list(struct.unpack('<7i',(a.models/'stories260K.bin').read_bytes()[:28])))
(a.out/'validation.json').write_text(json.dumps(data,indent=2)+'\n');print('PASS pretrained CPU/eager/Graph exact token identity: 4 prompts x 32 = 128 generated tokens')
