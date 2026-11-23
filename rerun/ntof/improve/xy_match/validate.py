# study script (fixed paths), x/y matching study 2026-11-23; see README.md
import pandas as pd, numpy as np, pickle
out=pickle.load(open('/media/ucla/work/ntof_pass1/ana/xy_match/ana4.pkl','rb'))
ST=pd.concat([pd.read_parquet(f'/media/ucla/work/ntof_pass1/ana/scint_stack/tracks/stack_run_{r}.parquet') for r in (145,104,139)])
ev=ST.drop_duplicates(['run','subrun','event_id','arm'])[['run','subrun','event_id','arm']+[f'w{n}_amp_on' for n in range(1,9)]]
ev['run']=ev.run.astype(str).where(ev.run.astype(str).str.startswith('run_'),'run_'+ev.run.astype(str))
UOFF={'A':182.94,'C':181.99,'D':183.79}; KX={'A':1.0423,'C':0.989,'D':0.9825}; KY={'A':-1.019,'C':-0.9758,'D':-0.9901}
WE=np.array([-225.0,-125.0,-25.0,75.0,175.0]); LW=97.4*0.85; LS=234.6
bins=[1,5,8,12,16,20,30,1000]
def usrc(arm,p0,tan): return (UOFF[arm]-p0)-LS*KX[arm]*tan
def uwall(arm,p0,tan): return (UOFF[arm]-p0)+LW*KX[arm]*tan
def vwall(arm,p0,tan): return (p0-199.29)+LW*(-KY[arm])*tan   # tan_v ~ -tan_raw_y? sign checked below
for arm,a,m in out:
    look=a.set_index(['ek','x_rank','y_rank'])
    # --- x pointing: peak from late single chosen tracks
    late=a[a.gated1&(a.t_ms>=30)&(a.ncx==1)]
    u=usrc(arm,late.x_p0,late.x_tan_theta); pk=np.median(u); 
    def pfrac(uu): d=np.abs(uu-pk); return np.mean(d<15)-np.mean((d>60)&(d<120))*30/120   # excess over sideband-flat
    print(f'== {arm}  x-source peak {pk:.1f} mm, late pointing excess {pfrac(u):.3f}')
    # x_moved: old x (pass1) vs new x = chosen pair containing the pass-1 y
    xm=m[m.kind=='x_moved'].copy()
    ch=a[a.chosen][['ek','y_rank','x_p0','x_tan_theta']].rename(columns={'x_p0':'nx_p0','x_tan_theta':'nx_tan'})
    xm=xm.merge(ch,on=['ek','y_rank'],how='left')
    xm['tb']=pd.cut(xm.t_ms,bins)
    print(' x_moved  tbin  n  n_with_new | point-excess old  new')
    for tb,g in xm.groupby('tb',observed=True):
        h=g[g.nx_p0.notna()]
        print(f'   {str(tb):12s} {len(g):5d} {len(h):5d} | {pfrac(usrc(arm,g.x_p0,g.x_tan_theta)):.3f}  {pfrac(usrc(arm,h.nx_p0,h.nx_tan)) if len(h) else np.nan:.3f}  (old on same subset {pfrac(usrc(arm,h.x_p0,h.x_tan_theta)) if len(h) else np.nan:.3f})')
    # same-kind baseline
    sm=m[m.kind=='same'].assign(tb=lambda d:pd.cut(d.t_ms,bins))
    print('   unchanged pointing excess by tbin:', {str(k):round(pfrac(usrc(arm,g.x_p0,g.x_tan_theta)),3) for k,g in sm.groupby('tb',observed=True)})
    # --- wall along-bar for y_changed
    w=a.merge(ev,on=['run','subrun','event_id','arm'],how='inner')
    def lr_of(df):
        uw=uwall(arm,df.x_p0.values,df.x_tan_theta.values); g=np.digitize(uw,WE)-1
        a1=np.full(len(df),np.nan); a2=a1.copy()
        for gg in range(4):
            k=g==gg; a1[k]=df[f'w{2*gg+1}_amp_on'].values[k]; a2[k]=df[f'w{2*gg+2}_amp_on'].values[k]
        return g,np.log(a1/a2)
    cal=w[w.gated1&(w.t_ms>=30)&(w.ncx==1)&(w.ncy==1)]
    g,lr=lr_of(cal); best=None
    for sgn in (1,-1):
        vw=(cal.y_p0.values-199.29)+sgn*LW*cal.y_tan_theta.values
        rs=[]; par={}
        for gg in range(4):
            k=(g==gg)&np.isfinite(lr)
            if k.sum()<50: continue
            b,c=np.polyfit(vw[k],lr[k],1); par[gg]=(b,c); r=(lr[k]-c)/b-vw[k]; rs.append(np.median(np.abs(r)))
        if best is None or np.mean(rs)<best[0]: best=(np.mean(rs),sgn,par)
    _,sgn,par=best; print(f'   wall calib sign {sgn}, median |res| late singles {best[0]:.1f} mm, groups {list(par)}')
    def wres(df,p0col,tancol):
        g,lr=lr_of(df); vw=(df[p0col].values-199.29)+sgn*LW*df[tancol].values; r=np.full(len(df),np.nan)
        for gg,(b,c) in par.items():
            k=(g==gg)&np.isfinite(lr); r[k]=(lr[k]-c)/b-vw[k]
        return r
    yc=m[m.kind=='y_changed'].copy()
    new=a[a.chosen][['ek','x_rank','y_p0','y_tan_theta']].rename(columns={'y_p0':'ny_p0','y_tan_theta':'ny_tan'})
    yc=yc.merge(new,on=['ek','x_rank']).merge(ev,on=['run','subrun','event_id','arm'],how='inner')
    ro=wres(yc,'y_p0','y_tan_theta'); rn=wres(yc,'ny_p0','ny_tan'); ok=np.isfinite(ro)&np.isfinite(rn)
    yc=yc[ok].assign(ro=np.abs(ro[ok]),rn=np.abs(rn[ok]),tb=lambda d:pd.cut(d.t_ms,bins))
    print(' y_changed with wall: tbin n | frac |res|<40mm old new | new closer')
    for tb,h in yc.groupby('tb',observed=True):
        print(f'   {str(tb):12s} {len(h):5d} | {np.mean(h.ro<40):.3f} {np.mean(h.rn<40):.3f} | {np.mean(h.rn<h.ro):.3f}')
    sm=m[m.kind=='same'].merge(ev,on=['run','subrun','event_id','arm'])
    rs_=np.abs(wres(sm,'y_p0','y_tan_theta')); sm=sm.assign(r=rs_,tb=lambda d:pd.cut(d.t_ms,bins))
    print('   unchanged frac<40:', {str(k):round(np.nanmean(g.r[np.isfinite(g.r)]<40),3) for k,g in sm.groupby('tb',observed=True)})
