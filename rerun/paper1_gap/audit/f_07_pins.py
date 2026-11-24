import sys; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
from scipy import stats
def wls(x,y,ex,ey,n=2000,seed=1):
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(ex)&np.isfinite(ey); x,y,ex,ey=x[ok],y[ok],ex[ok],ey[ok]
    r=stats.linregress(x,y)
    # errors in variables: orthogonal-ish via bootstrap of OLS on cells
    rng=np.random.default_rng(seed); b=[]
    for _ in range(n):
        k=rng.integers(0,len(x),len(x))
        if np.ptp(x[k])==0: continue
        b.append(np.polyfit(x[k],y[k],1)[0])
    # attenuation corrected slope: slope * var(x)/(var(x)-mean(ex^2))
    vx=np.var(x,ddof=1); att=vx/(vx-np.mean(ex**2)) if vx>np.mean(ex**2) else np.nan
    return len(x),r.slope,r.stderr,np.std(b),r.rvalue,att
print('7a: regress timing-free tilt d on w0/v across cells (expected slope +1)')
for r in OBS:
    for p,d in (('x','dX'),('y','dY')):
        o=OBS[r]
        x=np.array([c[f'w0_{p}']/c[f'v_{p}'] for c in o]); ex=np.array([c[f'w0_{p}_err']/c[f'v_{p}'] for c in o])
        y=np.array([c[d] for c in o]); ey=np.array([c[d+'_err'] for c in o])
        n,s,se,sb,rv,att=wls(x,y,ex,ey)
        print(f' {r} {p}: n={n} slope {s:+.2f} +- {se:.2f} (boot {sb:.2f}) r={rv:+.2f}  (x-noise attenuation factor {att:.2f}; slope*att {s*att:+.2f})')
print('7b: own-axis gradient of hp (central diff across cells, per mm/100mm) vs w0')
for r in OBS:
    for p in 'xy':
        o={ (c['i'],c['j']):c for c in OBS[r]}
        gx,w=[],[]
        for (i,j),c in o.items():
            if p=='x':
                nb=[o.get((i-1,j)),o.get((i+1,j))]; ax='U'
            else:
                nb=[o.get((i,j-1)),o.get((i,j+1))]; ax='W'
            nb=[q for q in nb if q is not None]
            lo=nb[0] if c[ax]>nb[0][ax] else None
            pts=nb+[c]; pts=sorted(pts,key=lambda q:q[ax])
            g=(pts[-1][f'hp_{p}']-pts[0][f'hp_{p}'])/(pts[-1][ax]-pts[0][ax])
            gx.append(g); w.append(c[f'w0_{p}'])
        s=stats.linregress(gx,w); sp=stats.spearmanr(gx,w)
        print(f' {r} {p}: slope w0 vs grad hp = {s.slope*1:+.4f} +- {s.stderr:.4f} um/ns per (mm/100mm), r={s.rvalue:+.2f} (spearman {sp[0]:+.2f}); expected +v/200*... = {37/200*1000/1000:+.3f}x(1+dc/dh)')
print('7c: ladder slope v across cells (um/ns), min/median/max')
for r in OBS:
    for p in 'xy':
        v=np.array([c[f'v_{p}'] for c in OBS[r]]); print(f' {r} {p}: {v.min():.1f} {np.median(v):.1f} {v.max():.1f}  all positive: {bool((v>0).all())}')
