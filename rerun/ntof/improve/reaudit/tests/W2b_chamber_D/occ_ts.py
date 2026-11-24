import glob, numpy as np, pandas as pd
from pathlib import Path
T7=Path('/media/ucla/work/claude-jobs/eb8200b3/T7_stability')
S=pd.read_parquet(T7/'stability.parquet',columns=['run','subrun','arm','t','rclass','n_tracks','n_late_gated'])
S=S[S.arm=='D'].sort_values('t')
E=np.linspace(-200,200,101); c=0.5*(E[1:]+E[:-1])
rows=[]
for _,r in S.iterrows():
    k=f"{r.run[4:]}_{r.subrun}"; z=np.load(T7/'cache/occ'/f'{k}.npz')
    d=dict(run=r.run,subrun=r.subrun,t=r.t,rclass=r.rclass)
    for arm in 'BD':
        for s in ('all','lg'):
            for p in ('x_local','y_local'):
                h=z[f'{arm}_{s}_{p}']; d[f'{arm}_{s}_{p}']=h
    rows.append(d)
pd.to_pickle(rows,'occ_rows.pkl')
