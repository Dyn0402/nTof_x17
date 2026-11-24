import sys, numpy as np
sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map')
import pm
def scale(g,col,emin=0.18,emax=0.44,w=0.04):
    EB=np.r_[np.arange(-emax,-emin+1e-9,w),np.arange(emin,emax+1e-9,w)]
    xs,cs,es=[],[],[]
    for lo,hi in zip(EB[:-1],EB[1:]):
        if lo<0<hi: continue
        s=g[(g.e>=lo)&(g.e<hi)]
        if len(s)<200: continue
        c,e,n,sig=pm.centre((s[col]-s.e).to_numpy())
        if np.isfinite(c) and np.isfinite(e) and e>0:
            xs.append(s.e.mean()); cs.append(c); es.append(e)
    if len(xs)<4: return np.nan,np.nan,np.nan,np.nan
    xs,cs,es=map(np.array,(xs,cs,es)); wt=1/es**2
    X=np.vstack([np.ones_like(xs),xs]).T
    cov=np.linalg.inv(X.T@(X*wt[:,None])); beta=cov@X.T@(wt*cs)
    chi=np.sum(wt*(cs-X@beta)**2)/(len(xs)-2)
    err=np.sqrt(np.diag(cov))*np.sqrt(max(chi,1))
    return 1+beta[1],err[1],beta[0],chi
