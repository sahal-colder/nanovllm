"""Seeded, non-language fixtures for GPU math/Graph tests; no model downloads."""
import array
import math
from pathlib import Path
import random
import struct
import sys

def create(root,name,d,f,l,h,kv,seq,v=263):
    rng=random.Random(719+h+l)
    shapes=[('embedding',v*d),('att_norm',l*d),('wq',l*d*d),('wk',l*(d//h*kv)*d),
            ('wv',l*(d//h*kv)*d),('wo',l*d*d),('ffn_norm',l*d),
            ('w1',l*f*d),('w2',l*d*f),('w3',l*f*d),('final_norm',d),('rope',seq*(d//h))]
    path=root/(name+'.bin')
    with path.open('wb') as out:
        out.write(struct.pack('<7i',d,f,l,h,kv,v,seq))
        for kind,n in shapes:
            scale=.15 if kind=='embedding' else .25/math.sqrt(f if kind=='w2' else d)
            a=array.array('f',(1+rng.uniform(-.05,.05) if 'norm' in kind else 0. if kind=='rope' else rng.uniform(-scale,scale) for _ in range(n)))
            if sys.byteorder!='little':a.byteswap()
            a.tofile(out)
    print('Created',path)

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
create(root,'decode1',64,128,1,4,1,64)
create(root,'decode3',64,128,3,4,2,320)
create(root,'perf',512,1536,4,8,2,4096,512)
