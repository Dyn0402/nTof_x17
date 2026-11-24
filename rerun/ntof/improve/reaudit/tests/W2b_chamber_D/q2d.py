import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=pd.read_csv('q2_scale_per_run.csv'); R=R[(R.n>=2000)&R.k_sh.notna()]
for col in ('sh','shm'):
    y=np.log(R[f'k_{col}']); x=np.log(R.v_t); w=(R[f'k_{col}']/R[f'ke_{col}'])**2
    X=np.vstack([np.ones_like(x),x-x.mean()]).T; cov=np.linalg.inv(X.T@(X*w.values[:,None])); b=cov@X.T@(w*y)
    chi=np.sum(w*(y-X@b)**2)/(len(y)-2)
    print(col,'d ln k / d ln v_t = %.3f +- %.3f (chi2/ndf %.2f, err scaled %.3f), corr %.2f; weighted mean k %.4f, rms of k %.4f, median err %.4f'%(b[1],np.sqrt(cov[1,1]),chi,np.sqrt(cov[1,1]*max(chi,1)),np.corrcoef(x,y)[0,1],np.average(R[f'k_{col}'],weights=w),R[f'k_{col}'].std(),R[f'ke_{col}'].median()))
# periods
R['per']=np.select([R.rn<=106,R.rn<=132,R.rn<=147],['84-106','107-132','135-147'],'150-162')
for col in ('sh','shm'):
    g=R.groupby('per').apply(lambda s: pd.Series(dict(k=np.average(s[f'k_{col}'],weights=1/s[f'ke_{col}']**2),err=1/np.sqrt((1/s[f'ke_{col}']**2).sum()),vt=s.v_t.mean(),n=s.n.sum())))
    print(col); print(g.round(4))
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for col,c,lab in (('sh','C0','unmapped'),('shm','C3','mapped')):
    ax[0].errorbar(R.rn,R[f'k_{col}'],R[f'ke_{col}'],fmt='o',ms=3,color=c,label=lab)
    ax[1].errorbar(R.v_t,R[f'k_{col}'],R[f'ke_{col}'],fmt='o',ms=3,color=c,label=lab)
ax[0].set_xlabel('run'); ax[0].set_ylabel('D angle scale k (measured / expected tan)'); ax[1].set_xlabel('D v_t (um/ns, late gated median)')
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig('fig_q2_scale.png',dpi=130)
