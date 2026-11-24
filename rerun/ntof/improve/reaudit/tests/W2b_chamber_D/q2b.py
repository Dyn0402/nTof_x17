import sys, numpy as np, pandas as pd
sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map')
import pm
o=pd.read_parquet('q2_tracks.parquet'); o=o[o.ok]
EB=np.r_[np.arange(-0.44,-0.179,0.04),np.arange(0.18,0.4401,0.04)]
def scale(g,col):
    xs,cs,es=[],[],[]
    for lo,hi in zip(EB[:-1],EB[1:]):
        if lo<0<hi: continue
        s=g[(g.e>=lo)&(g.e<hi)]
        if len(s)<200: continue
        c,e,n,sig=pm.centre((s[col]-s.e).to_numpy())
        if np.isfinite(c) and np.isfinite(e) and e>0:
            xs.append(s.e.mean()); cs.append(c); es.append(e)
    if len(xs)<4: return np.nan,np.nan,np.nan,np.nan
    xs,cs,es=map(np.array,(xs,cs,es)); w=1/es**2
    X=np.vstack([np.ones_like(xs),xs]).T
    cov=np.linalg.inv(X.T@(X*w[:,None])); beta=cov@X.T@(w*cs)
    chi=np.sum(w*(cs-X@beta)**2)/(len(xs)-2)
    err=np.sqrt(np.diag(cov))*np.sqrt(max(chi,1))
    return 1+beta[1],err[1],beta[0],chi
rows=[]
for run,g in o.groupby('run'):
    r=dict(run=run,rn=int(run[4:]),n=len(g),v_t=g.v_t.iloc[0])
    for col in ('sh','shm'):
        k,ke,a,chi=scale(g,col); r[f'k_{col}']=k; r[f'ke_{col}']=ke; r[f'a_{col}']=a; r[f'chi_{col}']=chi
    rows.append(r)
R=pd.DataFrame(rows).sort_values('rn')
R.to_csv('q2_scale_per_run.csv',index=False)
pd.set_option('display.width',200); print(R.round(4).to_string())
# pooled
for col in ('sh','shm'):
    k,ke,a,chi=scale(o,col); print('pooled',col,round(k,4),round(ke,4),'offset',round(a,4),'chi',round(chi,2))
