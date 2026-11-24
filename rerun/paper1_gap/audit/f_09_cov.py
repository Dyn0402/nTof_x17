import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit'); sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap')
from f_lib import *
import gapmap as GM, cellobs as CO
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
ex,ey=GM.GRIDS['4x4']
def line_cov(t,y,clip=3.5):
    ok=np.isfinite(t)&np.isfinite(y); t,y=t[ok],y[ok]
    keep=np.ones(len(t),bool)
    for _ in range(3):
        A=np.vstack([np.ones(keep.sum()),t[keep]]).T; c=np.linalg.lstsq(A,y[keep],rcond=None)[0]
        r=y-(c[0]+c[1]*t); rs=1.4826*np.median(np.abs(r[keep])); keep=np.abs(r)<clip*rs
    A=np.vstack([np.ones(keep.sum()),t[keep]]).T; c=np.linalg.lstsq(A,y[keep],rcond=None)[0]
    s2=np.sum((y[keep]-A@c)**2)/max(keep.sum()-2,1); cov=s2*np.linalg.inv(A.T@A)
    return c,cov,t[keep].mean()
cp={}; rho=[]
for run,R in GM.RUNS.items():
    tab=GM.load_table(GM.__dict__.get('WORK',WORK) if False else WORK,R['rel'])
    i,j=GM.cell_of(tab.xm.values,tab.ym.values,ex,ey)
    for a in range(4):
        for b in range(4):
            d=tab[(i==a)&(j==b)]
            for p in 'xy':
                tr=d[f'tan_ref_{p}'].values; s=np.abs(tr)<=GM.TAN_LADDER
                c,cov,tm=line_cov(tr[s],d[f'{p}_w'].values[s]*1e3)
                cp[(run,a,b,p)]=cov[0,1]; rho.append((run,a,b,p,tm,cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])))
rho=np.array([(r[4],r[5]) for r in rho])
print('mean tan_ref in ladder per cell: |mean| max %.3f median %.3f ; corr(w0,v): |rho| max %.2f median %.2f'%(abs(rho[:,0]).max(),np.median(abs(rho[:,0])),abs(rho[:,1]).max(),np.median(abs(rho[:,1]))))
cp2={(r,a,b,p):v for (r,a,b,p),v in cp.items()}
cpk={(r,a,b,p):v for (r,a,b,p),v in cp.items()}
L=[f'Per-cell corr(w0, v) from the line-fit covariance (64 cell-views): median |rho| {np.median(abs(rho[:,1])):.2f}, max |rho| {abs(rho[:,1]).max():.2f}; mean tan_ref of the ladder tracks: median |<tan>| {np.median(abs(rho[:,0])):.3f}, max {abs(rho[:,0]).max():.3f}\n','| arm | fit | chi2 | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta | mesh x@150 | cath x@150 | cath y@150 |','|---|---|---|---|---|---|---|---|---|---|---|---|']
for arm in ('all','no_hp','timing','tilt'):
    for lab,cvp in (('diagonal (committed)',None),('2x2 w0-v covariance',cp)):
        kw={'zero':('g0',)} if arm=='timing' else {}
        if cvp is not None and 'v' not in ARMS[arm]: continue
        R=Fitter(ARMS[arm],FLOORS,cov_pair=cvp,**kw).fit(); q=quants(R); f=five(R)
        L.append(f'| {arm} | {lab} | {R.chi2:.1f} | '+' | '.join(f'{q[k][0]:+.2f}+-{q[k][1]:.2f}' for k in ['h_kuu','h_kww','c_kuu','c_kww','gap_kuu','beta'])+f' | {f["mesh_x"][0]:+.2f}+-{f["mesh_x"][1]:.2f} | {f["cath_x"][0]:+.2f}+-{f["cath_x"][1]:.2f} | {f["cath_y"][0]:+.2f}+-{f["cath_y"][1]:.2f} |')
open(f'{OUT}/sec8.md','w').write('\n'.join(L)); print('\n'.join(L))
