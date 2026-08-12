"""Is every DREAM bunch's true partner a fixed number of bunches away?

For every DREAM bunch b, test n_TOF bunch b+k over a window of k, counting
20 ns coincidences near lag 0. No FFT and no repo imports -- run this after
cross_bunch_matrix.py has said roughly where to look, on the .npz that
script dumps. Cheap enough for the laptop.

This is the measurement that solved the mystery class (2026-08-12): on
run_79/stat090_0002 x 224573 it found 129 of the 130 eligible DREAM bunches
(99 %) sharp at k = -280 +- 1, 45 counts in a 20 ns bin over a floor of 0,
at a residual of -250 ns +- 20 -- i.e. the n_TOF hits were present all along,
filed 280 bunches below the bunch the join had asked for. Cause and fix:
../join_mislock/README.md and ntof_dream_merge/bunch_join.py.

Prefer this over cross_bunch_matrix's own ranking: that script's robust-z
saturates (MAD = 0 on sparse rows) and its argmax then picks arbitrarily
among ties, so it UNDERCOUNTS the ridge. The 20 ns sharpness test is the
only thing either script should be believed on.
"""
import numpy as np, json, sys
z=np.load('crossbunch_input_run79_0002_224573.npz')
ev_b,ev_t,cb,ct,k=z['ev_b'],z['ev_t'],z['cb'],z['ct'],float(z['k_seed'])
KS=range(-300,-259)
WIN=60_000.0; BW=20.0
starts=np.searchsorted(cb,np.arange(0,1002)); 
def cand(c): return ct[starts[c]:starts[c+1]] if 0<=c<1000 else np.zeros(0)
cands={c:np.sort(cand(c)) for c in range(1,1001)}
rows=[]
for b in np.unique(ev_b):
    te=ev_t[ev_b==b]
    if te.size<10: continue
    pred=te*(1.0+k); best=None
    for kk in KS:
        c=b+kk
        if c<1 or c>1000: continue
        tc=cands[c]
        if tc.size<50: continue
        d=[]
        for t in pred:
            lo=np.searchsorted(tc,t-WIN); hi=np.searchsorted(tc,t+WIN)
            d.append(tc[lo:hi]-t)
        d=np.concatenate(d) if d else np.zeros(0)
        if d.size==0: continue
        edges=np.arange(-WIN,WIN+BW,BW); h,_=np.histogram(d,bins=edges)
        i=int(h.argmax()); ctr=0.5*(edges[:-1]+edges[1:])
        far=np.abs(ctr-ctr[i])>2000; floor=float(np.median(h[far]))
        sig=(h[i]-floor)/np.sqrt(max(floor,1.0))
        if best is None or sig>best[0]: best=(float(sig),int(kk),int(h[i]),floor,float(ctr[i]))
    if best: rows.append(dict(b=int(b),n=int(te.size),sigma=best[0],shift=best[1],
                              peak=best[2],floor=best[3],dt_ns=best[4]))
json.dump(rows,open('shift_ridge.json','w'),indent=1)
s=np.array([r['sigma'] for r in rows]); sh=np.array([r['shift'] for r in rows])
bb=np.array([r['b'] for r in rows]); dt=np.array([r['dt_ns'] for r in rows])
m=s>=8
print('%d bunches tested, %d SHARP (>=8 sigma at 20 ns)'%(len(rows),m.sum()))
el=bb>=281
print('  eligible (b>=281): %d, sharp %d (%.0f%%)'%(el.sum(),(m&el).sum(),100*np.mean(m[el])))
print('  ineligible (b<281): %d, sharp %d'%((~el).sum(),(m&~el).sum()))
if m.any():
    print('  shift: median %+.0f  p10 %+.0f p90 %+.0f  unique %s'%(np.median(sh[m]),
        np.percentile(sh[m],10),np.percentile(sh[m],90),sorted(set(sh[m].tolist()))))
    print('  dt_ns: median %+.0f  p10 %+.0f p90 %+.0f'%(np.median(dt[m]),
        np.percentile(dt[m],10),np.percentile(dt[m],90)))
    print('  peak counts: median %.0f, floor median %.1f'%(np.median([r['peak'] for r in rows if r['sigma']>=8]),
        np.median([r['floor'] for r in rows if r['sigma']>=8])))
