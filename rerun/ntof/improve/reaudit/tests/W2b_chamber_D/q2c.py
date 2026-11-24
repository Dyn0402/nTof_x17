import sys, numpy as np, pandas as pd
sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map'); sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/W2b_chamber_D')
import pm, hmap, q2lib
d=pm.load()
for arm in 'AC':
    g=d[(d.arm==arm)&d.gated.astype(bool)&(d.ms>=25)]
    A=pm.arm_arrays(g,arm); hx=hmap.holes(A['x'],A['y']); A['e_normal']=hmap.normal_halfwidth(A['sh'],A['y'])+0.06
    tk=hmap.track_ok(A,hx)&(np.abs(A['y'])<=125)
    o=pd.DataFrame(dict(e=A['e'],sh=A['sh'],shm=A['shm']))[tk]
    for emin in (0.18, A['e_normal']):
        print(arm,'emin',round(emin,3),'n',len(o),'k_nomap',[round(v,4) for v in q2lib.scale(o,'sh',emin=emin)],'k_map',[round(v,4) for v in q2lib.scale(o,'shm',emin=emin)])
