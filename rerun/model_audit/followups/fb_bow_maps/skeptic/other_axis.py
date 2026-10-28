import sys, numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/model_audit/followups/fb_bow_maps')
from fbcommon import load, view_mask, huber, DETS
EDGES = np.linspace(30, 370, 10)
rng = np.random.default_rng(5)
for name in DETS:
    T = load(name); v = T.attrs['v']
    for p in 'xy':
        m = view_mask(T, p); t = T[f'{p}_t'].to_numpy(); r = T[f'{p}_r'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
        other = (T.w if p == 'x' else T.u).to_numpy()          # OTHER axis: along the strips
        own = (T.u if p == 'x' else T.w).to_numpy()
        inner = np.abs(own - 200) < 110                          # keep away from the strip-plane edges on the own axis
        ma = m & np.isfinite(wv) & (np.abs(wv - v*t) < 0.15*v)
        hs, es, w0s = [], [], []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            k = m & inner & (other >= lo) & (other < hi); ka = ma & inner & (other >= lo) & (other < hi)
            if k.sum() < 50: hs.append(np.nan); es.append(np.nan); w0s.append(np.nan); continue
            X = np.c_[np.ones(k.sum()), t[k]]
            hs.append(huber(X, r[k], 0.5)[1])
            es.append(np.std([huber(X[q], r[k][q], 0.5)[1] for q in (rng.integers(0, k.sum(), k.sum()) for _ in range(20))]))
            w0s.append(huber(np.c_[np.ones(ka.sum()), t[ka]], wv[ka], 0.03*v)[0] if ka.sum() > 50 else np.nan)
        hs = np.array(hs) - np.nanmean(np.array(hs)[3:6]); es = np.array(es); w0s = np.array(w0s)
        pos = 0.5*(EDGES[:-1]+EDGES[1:])
        c2 = np.polyfit(pos[np.isfinite(hs)]-200, hs[np.isfinite(hs)], 2, w=1/np.maximum(es[np.isfinite(hs)], 0.05))
        print(f"{name} [{p}] h vs OTHER axis: " + ' '.join(f'{x:+5.2f}' for x in hs) + f"  err~{np.nanmean(es):.2f}  parabola depth@150: {c2[0]*150**2:+.2f} mm; inner-5-bin parabola@150: {np.polyfit(pos[2:7]-200, hs[2:7], 2)[0]*150**2:+.2f}")
        print(f"{name} [{p}] w0 vs OTHER axis: " + ' '.join(f'{x:+5.2f}' for x in w0s))
