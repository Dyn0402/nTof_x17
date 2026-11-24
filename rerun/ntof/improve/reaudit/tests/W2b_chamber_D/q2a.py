import sys, numpy as np, pandas as pd
sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map')
import pm, hmap
d=pm.load()            # drops pre-access, D requires wall
d=d[(d.arm=='D')&d.gated.astype(bool)&(d.ms>=25)]
A=pm.arm_arrays(d,'D')
hx=hmap.holes(A['x'],A['y']); en=hmap.normal_halfwidth(A['sh'],A['y'])+0.06
A['e_normal']=en
tk=hmap.track_ok(A,hx)&(np.abs(A['y'])<=125)
print('n D late gated wall',len(d),'holes',hx,'e_normal',round(en,3),'kept',tk.sum())
# v_t per run from my cache
v=pd.read_parquet('db_all.parquet',columns=['run','arm','t_since_flash_ns','gated','v_t'],filters=[('arm','==','D')])
v=v[(v.t_since_flash_ns>=25e6)&v.gated]
vt=v.groupby('run').v_t.median()
out=pd.DataFrame(dict(run=d.run.values,rn=d.rn.values,e=A['e'],sh=A['sh'],shm=A['shm'],x=A['x'],y=A['y'],ok=tk,ms=A['ms']))
out['v_t']=out.run.map(vt)
out.to_parquet('q2_tracks.parquet',index=False)
print(out[out.ok].groupby('run').size().describe())
