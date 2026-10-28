import sys, numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/model_audit/followups/fb_bow_maps')
from fbcommon import load, view_mask, huber, DETS
E = np.linspace(25, 375, 6)
print("TILT-REMOVED (ln T, ln b regressed on 1, U, W, h):")
print("per cell: h (position route), T=u90-u50 (contained, |tan|<0.35), b=dw/dtan.  Fit ln T = c + sT*h and ln b = c + sb*h over cells.")
print("prediction mesh-away, flat cathode, v~E^1.2: sT = -(1+1.2)/30 = -0.073 /mm, sb = +0.040 /mm;  cathode-toward: mesh flat -> h has no geometric structure (slopes ~0);  reco bias: sT>0, sb<0")
for name in DETS:
    T = load(name); v = T.attrs['v']
    u = T.u.to_numpy(); w = T.w.to_numpy()
    lo, hi = 15.0, 385.0
    cont = ((T.u_mesh > lo) & (T.u_mesh < hi) & (T.w_mesh > lo) & (T.w_mesh < hi) & (T.u_cath > lo) & (T.u_cath < hi) & (T.w_cath > lo) & (T.w_cath < hi)).to_numpy()
    for p in 'xy':
        m = view_mask(T, p); t = T[f'{p}_t'].to_numpy(); r = T[f'{p}_r'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
        dur = (T[f'{p}_q_u90'] - T[f'{p}_q_u50']).to_numpy()
        H, D, B, Hi = [], [], [], []
        for i in range(5):
            for j in range(5):
                k = m & (u >= E[i]) & (u < E[i+1]) & (w >= E[j]) & (w < E[j+1])
                if k.sum() < 60 or (t[k] > 0.05).sum() < 10 or (t[k] < -0.05).sum() < 10: continue
                h = huber(np.c_[np.ones(k.sum()), t[k]], r[k], 0.5)[1]
                kd = k & cont & np.isfinite(dur) & (np.abs(t) < 0.35)
                if kd.sum() < 40: continue
                x = dur[kd]; Tm = np.mean(np.clip(x, *np.percentile(x, [5, 95])))
                kk = k & np.isfinite(wv) & (np.abs(wv - v*t) < 0.15*v)
                b = huber(np.c_[np.ones(kk.sum()), t[kk]], wv[kk], 0.03*v)[1]
                H.append(h); D.append(Tm); B.append(b); Hi.append((i, j))
        H, D, B = map(np.array, (H, D, B))
        UU = np.array([(i-2)/1.4 for i,j in Hi]); WW = np.array([(j-2)/1.4 for i,j in Hi]); A = np.c_[np.ones_like(H), UU, WW, H]
        sT = np.linalg.lstsq(A, np.log(D), rcond=None)[0][3]; sb = np.linalg.lstsq(A, np.log(B), rcond=None)[0][3]
        # bootstrap over cells
        rng = np.random.default_rng(1); bt = []
        for _ in range(200):
            q = rng.integers(0, len(H), len(H)); bt.append((np.linalg.lstsq(A[q], np.log(D[q]), rcond=None)[0][3], np.linalg.lstsq(A[q], np.log(B[q]), rcond=None)[0][3]))
        bt = np.array(bt)
        # rim-minus-centre contrast directly: cells with max(|i-2|,|j-2|)==2 vs the central 3x3
        rim = np.array([max(abs(i-2), abs(j-2)) == 2 for i, j in Hi]); cen = ~rim
        print(f"{name} [{p}] sT {sT:+.3f}±{bt[:,0].std():.3f}  sb {sb:+.3f}±{bt[:,1].std():.3f} /mm   rim-centre: h {H[rim].mean()-H[cen].mean():+.2f} mm, T {100*(D[rim].mean()/D[cen].mean()-1):+.1f} %, b {100*(B[rim].mean()/B[cen].mean()-1):+.1f} %   (n cells {len(H)})")
