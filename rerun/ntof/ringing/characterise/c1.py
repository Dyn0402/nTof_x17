import sys, numpy as np, pandas as pd, uproot
sys.path.insert(0,'.')
from wft import io as wio
D='/media/ucla/work/ntof_pass1/runs/run_145/stat090_0000/decoded_root/'
tt=pd.read_parquet('/media/dylan/data/x17/sept26_prelim/trigtime/trigtime_run_145_stat090_0000.parquet').set_index('eventId')
for feu in (3,4,7,8):
    f=D+f'Mx17_stat090_0000_datrun_260805_14H06_000_{feu:02d}.root'
    r=wio.FeuReader(f)
    a=r.tree.arrays(['eventId','amplitude'],entry_stop=4000,library='np')
    ev=a['eventId']; ok=np.array([len(x)==r.n_sample*512 for x in a['amplitude']])
    A=np.stack([x.reshape(-1,512) for x,o in zip(a['amplitude'],ok) if o]).astype(np.float32)-r.ped   # (N, ns, 512)
    ev=ev[ok]; ms=tt.reindex(ev).t_pred_ns.to_numpy()/1e6
    N,ns,_=A.shape
    B=A.reshape(N,ns,8,64)
    cm=np.median(B,axis=3)                       # (N,ns,8)
    R=B-cm[...,None]                             # residual after block CM
    nz=r.noise
    bins=[-1,0.5,1,2,3,5,8,12,20,30,80]
    print(f'FEU {feu} ns {ns} N {N} noise med {np.median(nz):.1f}')
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(ms>=lo)&(ms<hi)
        if m.sum()<5: continue
        cmamp=np.abs(cm[m]).max(axis=(1,2))       # largest block-median excursion
        cmstd=cm[m].std(axis=1).mean()            # within-window variation of CM
        # residual excursion: count of channels below -5 sigma
        z=R[m].reshape(m.sum(),ns,512)/np.maximum(nz,1)
        nneg=(z.min(axis=1)<-5).sum(axis=1)
        npos=(z.max(axis=1)>5).sum(axis=1)
        print(f'  {lo:5}-{hi:<4} n {m.sum():5d}  |CM|max med {np.median(cmamp):7.1f}  CM std(t) {cmstd:6.1f}  n_neg5 med {np.median(nneg):5.0f}  n_pos5 med {np.median(npos):5.0f}')
    np.savez_compressed(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz',A=A[:4000].astype(np.float16),ms=ms,ev=ev,noise=nz)
