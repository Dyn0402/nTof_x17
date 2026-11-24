import sys, numpy as np, pandas as pd
sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map'); sys.path.insert(0,'.')
import pm, q2lib
o=pd.read_parquet('q2_tracks.parquet')
runs=sorted(o.run.unique(),key=lambda r:int(r[4:]))
train=set(runs[0::2]); test=set(runs[1::2])
tr=o[o.run.isin(train)&o.ok]; te=o[o.run.isin(test)]
k,ke,a,chi=q2lib.scale(tr,'shm'); print('train k (mapped) %.4f +- %.4f offset %.4f'%(k,ke,a))
R=pd.read_csv('q2_scale_per_run.csv').set_index('run')
for lab,sel in (('reach region (ok)',te.ok),('all late gated wall, |y|<=125',np.abs(te.y)<=125)):
    s=te[sel]
    r0=s.shm-s.e; r1=s.shm/k-s.e
    # per-run scale (fitted on the run itself; not held out -> optimistic)
    kr=s.run.map(R.k_shm).fillna(k); r2=s.shm/kr-s.e
    c0=pm.centre(r0.to_numpy()); c1=pm.centre(r1.to_numpy()); c2=pm.centre(r2.to_numpy())
    print(lab,'n',len(s),'core sigma: as is %.4f, /k_D const %.4f, /k per run %.4f ; centre %.4f %.4f %.4f'%(c0[3],c1[3],c2[3],c0[0],c1[0],c2[0]))
    # in e bins: residual centres
    for lo,hi in ((-0.44,-0.3),(-0.3,-0.18),(0.18,0.3),(0.3,0.44)):
        b=(s.e>=lo)&(s.e<hi)
        print('   e %.2f..%.2f centre as-is %.4f  corrected %.4f  sigma %.4f -> %.4f'%(lo,hi,pm.centre(r0[b].to_numpy())[0],pm.centre(r1[b].to_numpy())[0],pm.centre(r0[b].to_numpy())[3],pm.centre(r1[b].to_numpy())[3]))
# dca-like: fraction within |r|<0.03
