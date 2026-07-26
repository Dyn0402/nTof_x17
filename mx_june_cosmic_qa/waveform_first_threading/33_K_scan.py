#!/usr/bin/env python3
"""Basis-length systematics: U50/u95 vs K on det3 1000V data, plus closure
toys (known U_true) fit at K=24/30 to verify no spurious tail growth."""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading')
import forward_model2 as fm2

B3 = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
      'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
hj = json.load(open(os.path.join(B3, 'hyper_v2.json')))
H3 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V = hj['v']
DT = 60.0
T0_TRUE = 430.0

ev3 = pickle.load(open(os.path.join(B3, 'wfcache.pkl'), 'rb'))['events']
split = json.load(open(os.path.join(B3, 'split_ref.json')))
sel = [e for e in split['test'] if e in ev3 and
       0.10 < np.hypot(ev3[e]['tan_x'], ev3[e]['tan_y']) < 0.45][:500]

def U50g(uk, prof, pb=(1, 6)):
    plat = np.median(prof[pb[0]:pb[1]])
    below = np.where(prof < 0.5 * plat)[0]
    below = below[below >= pb[1] - 1]
    if len(below) == 0 or plat <= 0:
        return np.nan
    j = below[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    return x0 + (0.5 * plat - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x1

_mode = {}

def fit_one(args):
    eid, K, toyU, seed = args
    fm2.K = K
    fm2.UK = (np.arange(K) + 0.5) * DT
    ev = ev3[eid]
    out = []
    for plane in ('x', 'y'):
        tn = ev[f'tan_{plane}']
        if abs(tn) < 0.08:
            continue
        P = ev[plane]
        pos = P['pos'].astype(np.float64)
        noise = np.maximum(P['noise'].astype(np.float64), 3.0)
        if toyU is not None:
            rng = np.random.default_rng(seed)
            KT = 18
            fm2.K = KT; fm2.UK = (np.arange(KT) + 0.5) * DT
            q = np.zeros(KT)
            for k in range(KT):
                lo = k * DT
                frac = np.clip((toyU - lo) / DT, 0, 1)
                if frac > 0:
                    q[k] = frac * rng.gamma(4.0, 0.25)
            q[0] += 3.0 * rng.gamma(4.0, 0.25)
            q *= 2500.0 / max(q.sum(), 1e-9)
            M = fm2.build_matrix(plane, pos, ev[f'ref_mesh_{plane}'],
                                 tn * V * 1e-3, T0_TRUE, H3)
            W = (M @ q).reshape(len(pos), fm2.NSAMP)
            W += rng.normal(0, 1, W.shape) * noise[:, None]
            Wp = dict(W=W.astype(np.float16), pos=P['pos'], noise=P['noise'],
                      ch=np.zeros(len(pos), np.int16))
            fm2.K = K; fm2.UK = (np.arange(K) + 0.5) * DT
        else:
            Wp = P
        Wd, nd, pd_, sd = fm2.prep_plane(Wp, plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 640.0, 30.0):
            c, qq = fm2.chi2_plane(plane, Wd, nd, pd_, sd,
                                   ev[f'ref_mesh_{plane}'], tn * V * 1e-3,
                                   float(t0), H3)
            if c < best[0]:
                best = (c, qq)
        qq = best[1]
        if qq is not None and 0 < qq.sum() and qq.max() < 2e4:
            out.append(qq / qq.sum())
    return out

def run(K, toyU=None):
    args = [(e, K, toyU, 5000 + i) for i, e in enumerate(sel)]
    profs = []
    with ProcessPoolExecutor(max_workers=14) as pool:
        for o in pool.map(fit_one, args, chunksize=6):
            profs.extend(o)
    L = max(len(p) for p in profs)
    P = np.zeros((len(profs), L))
    for i, p in enumerate(profs):
        P[i, :len(p)] = p
    uk = (np.arange(L) + 0.5) * DT
    mean = P.mean(axis=0)
    U = U50g(uk, mean)
    cum = np.cumsum(mean)
    u95 = uk[np.argmax(cum >= 0.95)]
    tail = float(mean[np.searchsorted(uk, 900):].sum())
    return U, u95, tail, uk, mean

if __name__ == '__main__':
    fm2.GAIN['x'] = np.ones(512) if False else fm2.GAIN['x']  # keep gains for data
    fig, ax = plt.subplots(figsize=(9, 5.5))
    print('== DATA K-scan (det3 1000V) ==')
    for K in (18, 21, 24, 27, 30):
        U, u95, tail, uk, mean = run(K)
        print(f'K={K:2d} (basis {K*60}ns): U50={U:.0f} u95={u95:.0f} '
              f'-> col {V*U*1e-3:.1f} / {V*u95*1e-3:.1f} mm  tail>900ns={tail*100:.1f}%',
              flush=True)
        ax.plot(uk, mean, 'o-', ms=2.5, label=f'K={K}: U50={U:.0f}')
    ax.axvline(25000 / V, ls=':', color='k'); ax.axvline(29000 / V, ls='--', color='k')
    ax.set_yscale('log'); ax.set_ylim(5e-5, 0.2)
    ax.set_xlabel('u [ns]'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('det3 1000V mean profile vs basis length')
    fig.tight_layout(); fig.savefig(os.path.join(B3, 'K_scan.png'), dpi=110)
    print('== CLOSURE toys (truth basis K=18) ==')
    for K, toyU in ((24, 793.0), (30, 793.0), (24, 700.0)):
        U, u95, tail, uk, mean = run(K, toyU)
        print(f'toy U_true={toyU:.0f} fit K={K}: U50={U:.0f} u95={u95:.0f} '
              f'tail>900={tail*100:.1f}%', flush=True)
    print('saved K_scan.png')
