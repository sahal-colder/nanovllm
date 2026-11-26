#!/usr/bin/env python3
"""Deterministic random legacy-format model, not a trained checkpoint.
NumPy PCG64 seed=719; tensor streams are generated in bounded chunks.
"""
import argparse,math,struct,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path)
a=p.parse_args();a.path.parent.mkdir(parents=True,exist_ok=True)
d,f,l,h,kv,seq,v=1024,2816,16,16,4,2048,8192
rng=np.random.default_rng(719);kd=d//h*kv
shapes=[('embedding',v*d),('att_norm',l*d),('wq',l*d*d),('wk',l*kd*d),('wv',l*kd*d),('wo',l*d*d),('ffn_norm',l*d),('w1',l*f*d),('w2',l*d*f),('w3',l*f*d),('final_norm',d),('rope',seq*(d//h))]
with a.path.open('wb') as out:
    out.write(struct.pack('<7i',d,f,l,h,kv,v,seq))
    for name,n in shapes:
        for offset in range(0,n,1<<20):
            count=min(n-offset,1<<20)
            if name=='rope':x=np.zeros(count,dtype='<f4')
            elif 'norm' in name:x=rng.uniform(.95,1.05,count).astype('<f4')
            else:
                scale=.15 if name=='embedding' else .25/math.sqrt(f if name=='w2' else d)
                x=rng.uniform(-scale,scale,count).astype('<f4')
            out.write(x.tobytes())
digest=hashlib.file_digest(a.path.open('rb'),'sha256').hexdigest()
a.path.with_suffix('.json').write_text(json.dumps(dict(config=[d,f,l,h,kv,v,seq],seed=719,sha256=digest,bytes=a.path.stat().st_size,numpy=np.__version__,synthetic=True),indent=2)+'\n')
print(a.path,digest)
