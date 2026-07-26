#!/usr/bin/env python3
"""500V/300V endpoint with extended charge basis — direct 25-vs-29mm test.

At 500 V (v_fit=20.6): U(25mm)=1214ns vs U(29mm)=1408ns — both inside the
1920ns window (t0~400ns), separable with K=26 bins (1560ns).
At 300 V (v=12.0): U(25mm)=2083ns exceeds the visible window (~1500ns after
t0) — prediction under 25mm: profile runs flat to the window end, no edge.
"""
import os, sys, pickle, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading')
import forward_model2 as fm2
import forward_model3 as fm3

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
ffv = json.load(open(os.path.join(BASE, 'drift_scan_v.json')))
ffv.update(json.load(open(os.path.join(BASE, 'drift_scan_v_lowhv.json'))))
DT = 60.0

_store = {}

def set_K(K):
    fm2.K = K
    fm2.UK = (np.arange(K) + 0.5) * DT
    fm3._TT_CACHE.clear()

def fit_q(args):
    eid, v, K = args
    set_K(K)
    ev = _store[eid]
    out = []
    for plane in ('x', 'y'):
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        if abs(tn) < 0.08:
            continue
        p0l = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        W, noise, pos, sat = fm2.prep_plane(ev[plane], plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 640.0, 30.0):
            c, q = fm2.chi2_plane(plane, W, noise, pos, sat, p0l,
                                  tn * v * 1e-3, float(t0), H0)
            if c < best[0]:
                best = (c, q)
        q = best[1]
        if q is not None and 0 < q.sum() and q.max() < 2e4:
            out.append(q / q.sum())
    return out

def U50(uk, prof):
    plat = np.median(prof[1:6])
    below = np.where(prof < 0.5 * plat)[0]
    below = below[below >= 4]
    if len(below) == 0:
        return np.nan
    j = below[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    return x0 + (0.5 * plat - y0) / (y1 - y0) * (x1 - x0)

def boot(uk, P, n=200):
    rng = np.random.default_rng(1)
    vals = [U50(uk, np.median(P[rng.integers(0, len(P), len(P))], axis=0))
            for _ in range(n)]
    return np.nanmedian(vals), np.nanstd(vals)

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(9, 5.5))
    out = {}
    for hv, K in ((500, 26), (300, 30)):
        events = pickle.load(open(os.path.join(BASE, f'wfcache_{hv}V.pkl'), 'rb'))
        sel = [e for e, ev in events.items()
               if 0.10 < np.hypot(ev['tan_x'], ev['tan_y']) < 0.45][:300]
        _store.clear(); _store.update({e: events[e] for e in sel})
        v = ffv[str(hv)]['v']
        profs = []
        with ProcessPoolExecutor(max_workers=14) as pool:
            for o in pool.map(fit_q, [(e, v, K) for e in sel], chunksize=8):
                profs.extend(o)
        L = max(len(p) for p in profs)
        P = np.zeros((len(profs), L))
        for i, p in enumerate(profs):
            P[i, :len(p)] = p
        uk = (np.arange(L) + 0.5) * DT
        U, dU = boot(uk, P)
        med = np.median(P, axis=0)
        col = v * U * 1e-3
        out[hv] = dict(v=v, U=float(U), dU=float(dU), col=float(col))
        pred25 = 25000.0 / v
        pred29 = 29000.0 / v
        print(f'HV {hv} (K={K}): v={v:.1f}  U50={U:.0f}±{dU:.0f} ns  '
              f'column={col:.1f} mm   [U(25mm)={pred25:.0f}, U(29mm)={pred29:.0f}]',
              flush=True)
        ax.plot(uk, med, 'o-', ms=3, label=f'{hv} V: U50={U:.0f} ns')
        ax.axvline(pred25, ls=':', color=ax.lines[-1].get_color(), alpha=0.7)
        ax.axvline(pred29, ls='--', color=ax.lines[-1].get_color(), alpha=0.7)
    ax.set_xlabel('u [ns]  (dotted: U(25mm), dashed: U(29mm))')
    ax.set_ylabel('median normalized charge / 60 ns')
    ax.set_title('low-HV deconvolved profiles, extended charge basis')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, 'lowhv_endpoint.png'), dpi=110)
    json.dump(out, open(os.path.join(BASE, 'lowhv_endpoint.json'), 'w'), indent=1)
    print('saved lowhv_endpoint.png')
