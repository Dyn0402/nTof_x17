import sys, numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/model_audit/followups/fb_bow_maps')
from fbcommon import load, view_mask, huber, DETS
E = np.linspace(25, 375, 6)
for name in DETS:
    T = load(name); v = T.attrs['v']
    u = T.u.to_numpy(); w = T.w.to_numpy()
    for p in 'xy':
        m = view_mask(T, p); t = T[f'{p}_t'].to_numpy(); r = T[f'{p}_r'].to_numpy(); wv = T[f'{p}_wv'].to_numpy(); tau = T[f'{p}_q_u50'].to_numpy()
        H = []; M = []
        for i in range(5):
            for j in range(5):
                k = m & (u >= E[i]) & (u < E[i+1]) & (w >= E[j]) & (w < E[j+1])
                if k.sum() < 60 or (t[k] > 0.05).sum() < 10 or (t[k] < -0.05).sum() < 10: continue
                h = huber(np.c_[np.ones(k.sum()), t[k]], r[k], 0.5)[1]
                kk = k & np.isfinite(wv) & (np.abs(wv - v*t) < 0.15*v)
                b = huber(np.c_[np.ones(kk.sum()), t[kk]], wv[kk], 0.03*v)[1]
                tb = np.nanmedian(tau[kk])
                H.append(h); M.append((v - b) * tb * 1e-3)      # predicted fake height, mm
        H = np.array(H); M = np.array(M)
        A = np.c_[np.ones_like(M), M]; beta, *_ = np.linalg.lstsq(A, H, rcond=None)
        rr = H - A @ beta; r2 = 1 - rr.var() / H.var()
        print(f'{name} [{p}]  h_cell = {beta[0]:+.2f} + {beta[1]:+.2f} * (v-b)*tau   r2={r2:.2f}   spread h {H.std():.2f} mm, spread (v-b)tau {M.std():.2f} mm, n={len(H)}')
