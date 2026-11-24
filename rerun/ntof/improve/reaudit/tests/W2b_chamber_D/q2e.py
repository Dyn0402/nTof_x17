import numpy as np, pandas as pd
d=pd.read_parquet('db_all.parquet',columns=['run','arm','t_since_flash_ns','gated','tanx','x_local','y_local','v_t','rn'],filters=[('arm','==','D')])
d=d[(d.t_since_flash_ns>=25e6)&d.gated]
def fit(df,lab,two=True):
    g=df.groupby('run').agg(rms=('tanx','std'),mean=('tanx','mean'),vt=('v_t','median'),n=('tanx','size'))
    f60=d.groupby('run').x_local.apply(lambda x:(x>60).mean()); g['f60']=f60
    g=g[g.n>=2000]
    x=np.log(g.vt); y=np.log(g.rms); sw=np.sqrt(g.n.values)
    X=np.vstack([np.ones(len(g)),x-x.mean()]).T
    b=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
    res=y-X@b; s=np.sqrt(np.sum(g.n*res**2)/g.n.sum()*len(g)/(len(g)-2)); err=s/np.sqrt(np.sum(g.n/g.n.mean()*(x-x.mean())**2))
    out=f'{lab}: runs {len(g)} dln rms/dln vt {b[1]:.3f} +- {err:.3f} corr {np.corrcoef(x,y)[0,1]:.2f}'
    if two:
        X=np.vstack([np.ones(len(g)),x-x.mean(),g.f60-g.f60.mean()]).T
        beta=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
        out+=f' | corr(rms, frac x>60) {np.corrcoef(g.f60,y)[0,1]:.2f}; joint fit: vt {beta[1]:.3f}, frac(x>60) {beta[2]:.3f}; corr(vt,frac x>60) {np.corrcoef(x,g.f60)[0,1]:.2f}'
    print(out)
fit(d,'all x')
fit(d[(d.x_local<-20)&(d.x_local>-150)],'x -150..-20 only',False)
fit(d[(d.x_local>60)&(d.x_local<150)],'x 60..150 only',False)
