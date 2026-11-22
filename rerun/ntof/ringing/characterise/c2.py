import numpy as np, pandas as pd
tt=pd.read_parquet('/media/dylan/data/x17/sept26_prelim/trigtime/trigtime_run_145_stat090_0000.parquet').set_index('eventId')
tt['dtprev']=tt.t_pred_ns.diff()/1e6; tt.loc[tt.bunch!=tt.bunch.shift(),'dtprev']=np.nan
tt['k']=tt.groupby('bunch').cumcount()
for feu in (3,8):
    z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; ev=z['ev']; nz=np.maximum(z['noise'],1)
    N,ns,_=A.shape
    B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
    zz=R/nz; nneg=(zz.min(axis=1)<-5).sum(axis=1)
    d=pd.DataFrame(dict(ms=ms,nneg=nneg,dt=tt.reindex(ev).dtprev.to_numpy(),k=tt.reindex(ev).k.to_numpy()))
    d=d[d.ms>0.9]
    print('FEU',feu)
    print(' n_neg5 median by ToF x dt_prev(ms):')
    d['tb']=pd.cut(d.ms,[0.9,2,3,5,8,12,20,30,80]); d['db']=pd.cut(d.dt,[0,0.1,0.2,0.3,0.5,1,2,5,100])
    print(d.pivot_table(index='tb',columns='db',values='nneg',aggfunc='median',observed=False).round(0).to_string())
    print(d.pivot_table(index='tb',columns='db',values='nneg',aggfunc='size',observed=False).to_string())
