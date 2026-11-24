import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17')
from rerun.ntof.improve import bd_map as BD
L=pd.read_parquet('/media/ucla/work/ntof_pass1/ana_r2/track_quality/bd_lines.parquet'); L['rn']=L.run.str[4:].astype(int)
v=pd.read_parquet('db_all.parquet',columns=['run','arm','v_t'],filters=[('arm','==','D')]).groupby('run').v_t.median()
L['v_t']=L.run.map(v)
L['per']=np.select([L.rn<=106,L.rn<=132,L.rn<=147],['80-106','107-132','133-147'],'149-159')
def k(g):
    c=BD.clean(g,'z'); 
    if len(c)<30: return np.nan,np.nan,len(c)
    X=np.vstack([np.ones(len(c)),c.j]).T
    b,res,_,_=np.linalg.lstsq(X,c.r,rcond=None); s=np.sqrt(np.sum((c.r-X@b)**2)/(len(c)-2))
    cov=s**2*np.linalg.inv(X.T@X); return 1+b[1],np.sqrt(cov[1,1]),len(c)
for per,g in L.groupby('per'):
    kk=k(g); print('B-D lines',per,'k = %.3f +- %.3f (n %d), v_t %.2f'%(kk[0],kk[1],kk[2],g.v_t.median()))
kk=k(L); print('all',kk)
# per-line: r vs v_t
c=BD.clean(L,'z'); c['v_t']=L.loc[c.index,'v_t']
X=np.vstack([np.ones(len(c)),c.j,c.j*np.log(c.v_t/c.v_t.mean())]).T
b=np.linalg.lstsq(X,c.r,rcond=None)[0]; s=np.sqrt(np.sum((c.r-X@b)**2)/(len(c)-3)); cov=s**2*np.linalg.inv(X.T@X)
print('cosmic lines: d k / d ln v_t = %.3f +- %.3f'%(b[2],np.sqrt(cov[2,2])), 'v_t range',c.v_t.min(),c.v_t.max())
