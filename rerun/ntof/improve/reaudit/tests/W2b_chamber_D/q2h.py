import sys, numpy as np, pandas as pd
sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map')
import pm, hmap
d=pm.load()
for arm in 'ACD':
    g=d[(d.arm==arm)&d.gated.astype(bool)&(d.ms>=25)]
    A=pm.arm_arrays(g,arm); hx=hmap.holes(A['x'],A['y']); A['e_normal']=0.18
    tk=hmap.track_ok(A,hx)&(np.abs(A['y'])<=125)
    e=A['e'][tk]; rm=(A['shm']-A['e'])[tk]; rn=(A['sh']-A['e'])[tk]
    s=f'{arm}:'
    for lo,hi in ((-0.44,-0.38),(-0.38,-0.30),(-0.30,-0.24),(-0.24,-0.18),(0.18,0.24),(0.24,0.30),(0.30,0.38),(0.38,0.44)):
        b=(e>=lo)&(e<hi)
        c=pm.centre(rm[b]); c2=pm.centre(rn[b])
        s+=f' [{lo:+.2f},{hi:+.2f}] map {c[0]:+.4f}±{c[1]:.4f} nomap {c2[0]:+.4f} (n {b.sum()})'
    print(s)
