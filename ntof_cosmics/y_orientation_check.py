#!/usr/bin/env python3
"""
y_orientation_check.py -- is stage-3 global y mirrored?  (2026-10-13)

Two hardware-light tests, both with the position-independent wall time (the
mean of a wall group's two ends, which cancels propagation along the bar):

  cosmic   beam-off run_149, straight A-C cosmic lines (dev < 15 deg), wall A
           and wall C times on the n_TOF clock (clock_match).  Muons go DOWN, so
           the truly higher arm is hit first.  Result: the stored-HIGHER arm is
           hit LATER (88 % vs 7 %, Mann-Whitney p = 4e-11, step ~ +-2.3 ns =
           the 66 cm wall separation at c).  => stored y points down.
  pmt      beam runs, >= 12 ms, every arm: plastic time minus wall time against
           the track's stored v, per wall group (fixed effects).  All 8 bars
           are LATER at stored-high v (+0.15..+0.47 ns/100 mm, 10-40 sigma),
           flat in u (control).  With the plastic PMTs on top (Dylan), the
           stored top is the far end => stored y points down.

Run from the repo root:  python ntof_cosmics/y_orientation_check.py cosmic|pmt
Outputs: ntof_cosmics/results/y_orientation/{ac149.parquet,pmt_timing.csv}
"""
import sys
step = sys.argv[1] if len(sys.argv) > 1 else 'cosmic'
if step == 'cosmic':
    import sys, glob, numpy as np, pandas as pd, uproot
    sys.path.insert(0,'.')
    from ntof_cosmics import cosmic_wall_scale as CWS
    from ntof_cosmics import clock_match as CM
    K='ntof_cosmics/results/tracking/k_run_147'
    subs=sorted(set(p.split('pairs_run_149_')[1].rsplit('_',1)[0] for p in glob.glob(str(CM.OUT/'pairs_run_149_*.csv'))))
    trees={}
    def tree(ntof,name):
        k=(ntof,name)
        if k not in trees:
            want=['BunchNumber','detn','amp','tof']
            parts=[uproot.open(f)[name].arrays(want,library='np') for f in CM._parts(ntof)]
            d={c:np.concatenate([p[c] for p in parts]) for c in want}
            o=np.lexsort((d['tof'],d['BunchNumber'])); trees[k]={c:v[o] for c,v in d.items()}
        return trees[k]
    def walltime(W,key,k0):
        lo,hi=np.searchsorted(key,k0-150),np.searchsorted(key,k0+150)
        best=(-1,np.nan,np.nan)
        hits={}
        for i in range(lo,hi):
            dn=int(W['detn'][i]); a=float(W['amp'][i]); t=float(W['tof'][i])
            if dn not in hits or a>hits[dn][0]: hits[dn]=(a,t)
        for g in range(4):
            e1,e2=hits.get(2*g+1),hits.get(2*g+2)
            if e1 and e2 and e1[0]+e2[0]>best[0]: best=(e1[0]+e2[0],0.5*(e1[1]+e2[1]),e1[1]-e2[1])
        return best
    rows=[]
    for sub in subs:
        P=pd.read_parquet(f'{K}/pairs_run_149_{sub}.parquet'); P=P[P.pair=='AC'].drop(columns=['ntof_arm','res','matched','bunch'],errors='ignore')
        T=pd.read_parquet(f'{K}/tracks_run_149_{sub}.parquet',columns=['event_id','arm','track_id','p0_y','y_local','tan_raw_y','gated'])
        trig=CWS.matched_triggers(sub)
        if trig.empty or P.empty: continue
        P=P.merge(trig[['event_id','ntof','bunch','tof','ntof_arm']],on='event_id')
        for ntof,g in P.groupby('ntof'):
            WA,WC=tree(int(ntof),'WALA'),tree(int(ntof),'WALC')
            kA=WA['BunchNumber']*1e9+WA['tof']; kC=WC['BunchNumber']*1e9+WC['tof']
            for r in g.itertuples():
                k0=r.bunch*1e9+r.tof
                a=walltime(WA,kA,k0); c=walltime(WC,kC,k0)
                ya=T[(T.event_id==r.event_id)&(T.arm=='A')&(T.track_id==r.track1)]
                yc=T[(T.event_id==r.event_id)&(T.arm=='C')&(T.track_id==r.track2)]
                if len(ya)!=1 or len(yc)!=1: continue
                rows.append(dict(sub=sub,event_id=r.event_id,trig=r.ntof_arm,dev1=r.dev1_deg,dev2=r.dev2_deg,sep=r.sep_mm,
                     yA=float(ya.p0_y.iloc[0]),yC=float(yc.p0_y.iloc[0]),tyA=float(ya.tan_raw_y.iloc[0]),tyC=float(yc.tan_raw_y.iloc[0]),
                     tA=a[1],tC=c[1],eA=a[2],eC=c[2]))
    R=pd.DataFrame(rows); R.to_parquet('ntof_cosmics/results/y_orientation/ac149.parquet')
    print('AC pairs with clock match',len(R),' both walls',R[['tA','tC']].notna().all(axis=1).sum())
else:
    import glob, numpy as np, pandas as pd, pyarrow.parquet as pq
    cols=['run','arm','u_mm','v_mm','t_since_flash_ns','x_t0','y_t0']
    for e in range(1,9): cols+= [f'w{e}_dt_on',f'w{e}_amp_on']
    cols+=['p1_dt_on','p1_amp_on','p1_sat_on','p2_dt_on','p2_amp_on','p2_sat_on','l_dt_on','l_amp_on','l_sat_on']
    fs=sorted(glob.glob('/media/dylan/data/x17/scint_stack/tracks/*.parquet'))
    D=pd.concat([pq.read_table(f,columns=cols).to_pandas() for f in fs[::2]])
    D=D[(D.t_since_flash_ns>12e6)&(D.x_t0<=300)&(D.y_t0<=300)]
    bs=np.full(len(D),-1.0); bt=np.full(len(D),np.nan); bg=np.full(len(D),-1)
    for gi in range(4):
        a1,a2=D[f'w{2*gi+1}_amp_on'].fillna(0).values,D[f'w{2*gi+2}_amp_on'].fillna(0).values
        t1,t2=D[f'w{2*gi+1}_dt_on'].values,D[f'w{2*gi+2}_dt_on'].values
        s=np.where((a1>0)&(a2>0),a1+a2,-1); m=s>bs
        bs=np.where(m,s,bs); bt=np.where(m,0.5*(t1+t2),bt); bg=np.where(m,gi,bg)
    D['tw']=bt; D['grp']=bg
    rows=[]
    for arm,g in D.groupby('arm',observed=True):
        for el in ('p1','p2','l'):
            h=g[(g[f'{el}_amp_on']>0)&np.isfinite(g.tw)&(g.grp>=0)&~g[f'{el}_sat_on'].fillna(False).astype(bool)].copy()
            h['dt']=h[f'{el}_dt_on']-h.tw; h=h[h.dt.abs()<40]
            for coord in ('v_mm','u_mm'):
                # within-group regression (group fixed effects), robust-ish: clip residual tails
                x=h[coord]-h.groupby('grp')[coord].transform('mean'); y=h.dt-h.groupby('grp').dt.transform('median')
                k=y.abs()<10
                sl=np.polyfit(x[k],y[k],1)[0]*100 if k.sum()>200 else np.nan
                se=np.nan
                if k.sum()>200:
                    bs_=[np.polyfit(*np.c_[x[k],y[k]][np.random.default_rng(i).integers(0,k.sum(),k.sum())].T,1)[0]*100 for i in range(50)]
                    se=np.std(bs_)
                rows.append(dict(arm=arm,el=el,coord=coord,n=int(k.sum()),slope_ns_per_100mm=sl,err=se))
    R=pd.DataFrame(rows); pd.set_option('display.width',200); print(R.round(3).to_string(index=False))
    R.to_csv('ntof_cosmics/results/y_orientation/pmt_timing.csv',index=False)
