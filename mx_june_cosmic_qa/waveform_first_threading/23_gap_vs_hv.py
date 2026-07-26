#!/usr/bin/env python3
"""Gap closure test v2 (robust): median profiles, outlier rejection,
interpolated U50 endpoint, extended charge basis for 700 V."""
import os, sys, pickle, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2

BASE = fm2.BASE
hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V1000 = hj['v']
ffv = json.load(open(os.path.join(BASE, 'drift_scan_v.json')))
DT = 60.0

def U50(uk, prof):
    """Endpoint: u where profile falls to 50% of its plateau (bins 1..plateau_end)."""
    plateau = np.median(prof[1:6])
    if plateau <= 0:
        return np.nan
    below = np.where(prof < 0.5 * plateau)[0]
    below = below[below >= 4]
    if len(below) == 0:
        return np.nan
    j = below[0]
    if j == 0:
        return uk[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    if y0 == y1:
        return x1
    return x0 + (0.5 * plateau - y0) / (y1 - y0) * (x1 - x0)

def U50_boot(uk, P, n=200, seed=1):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(P), len(P))
        vals.append(U50(uk, np.median(P[idx], axis=0)))
    vals = np.array(vals)
    return np.nanmedian(vals), np.nanstd(vals)

_store = {}
_K = [fm2.K]

def fit_q(args):
    eid, v, K = args
    fm2.K = K
    fm2.UK = (np.arange(K) + 0.5) * DT
    ev = _store[eid]
    out = []
    for plane in ('x', 'y'):
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        if abs(tn) < 0.08:
            continue
        p0l = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        P = ev[plane]
        W, noise, pos, sat = fm2.prep_plane(P, plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 700.0, 30.0):
            c, q = fm2.chi2_plane(plane, W, noise, pos, sat, p0l,
                                  tn * v * 1e-3, float(t0), H0)
            if c < best[0]:
                best = (c, q)
        q = best[1]
        if q is not None and 0 < q.sum() and q.max() < 2e4:
            out.append(q / q.sum())
    return out

def profile_for(hv, K):
    cache = os.path.join(BASE, f'wfcache_{hv}V.pkl')
    events = pickle.load(open(cache, 'rb'))
    sel = [e for e, ev in events.items()
           if 0.10 < np.hypot(ev['tan_x'], ev['tan_y']) < 0.45][:250]
    _store.clear(); _store.update({e: events[e] for e in sel})
    v = ffv[str(hv)]['v']
    profs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for out in pool.map(fit_q, [(e, v, K) for e in sel], chunksize=8):
            profs.extend(out)
    # pad to common length
    L = max(len(p) for p in profs)
    P = np.full((len(profs), L), 0.0)
    for i, p in enumerate(profs):
        P[i, :len(p)] = p
    return P, v

if __name__ == '__main__':
    results = {}
    for hv, K in ((700, 24), (900, 20), (1100, 18)):
        P, v = profile_for(hv, K)
        uk = (np.arange(P.shape[1]) + 0.5) * DT
        U, dU = U50_boot(uk, P)
        results[hv] = dict(v=v, U=U, dU=dU, gap_mm=v * U * 1e-3,
                           dgap=v * dU * 1e-3, n=len(P),
                           prof=np.median(P, axis=0).tolist(), uk=uk.tolist())
        print(f'HV {hv}: v={v:.1f}  U50={U:.0f}±{dU:.0f} ns  implied gap = '
              f'{v*U*1e-3:.1f}±{v*dU*1e-3:.1f} mm  (n={len(P)})', flush=True)

    # 1000 V from freefit2
    res2 = pickle.load(open(os.path.join(BASE, 'freefit2.pkl'), 'rb'))
    profs, t0s, tans = [], [], []
    for r in res2:
        for p in ('x', 'y'):
            if p not in r or 'error' in r[p] or 'q' not in r[p]:
                continue
            q = np.asarray(r[p]['q'], float)
            if q.sum() <= 0 or q.max() >= 2e4:
                continue
            profs.append(q / q.sum()); t0s.append(r[p]['t0'])
            tans.append(abs(r[p]['tan_ref']))
    P = np.array(profs); t0s = np.array(t0s); tans = np.array(tans)
    uk = (np.arange(P.shape[1]) + 0.5) * DT
    U, dU = U50_boot(uk, P)
    results[1000] = dict(v=V1000, U=U, dU=dU, gap_mm=V1000 * U * 1e-3,
                         dgap=V1000 * dU * 1e-3, n=len(P),
                         prof=np.median(P, axis=0).tolist(), uk=uk.tolist())
    print(f'HV 1000: v={V1000:.1f}  U50={U:.0f}±{dU:.0f}  implied gap = '
          f'{V1000*U*1e-3:.1f}±{V1000*dU*1e-3:.1f} mm  (n={len(P)})')
    for lo, hi in ((0.08, 0.15), (0.15, 0.25), (0.25, 0.45)):
        m = (tans >= lo) & (tans < hi)
        Ub, dUb = U50_boot(uk, P[m])
        print(f'  |tan| {lo}-{hi}: U50 = {Ub:.0f}±{dUb:.0f} (n={m.sum()})')
    m = t0s < 400
    Ue, dUe = U50_boot(uk, P[m])
    print(f'  t0<400 subset: U50 = {Ue:.0f}±{dUe:.0f} (n={m.sum()})')

    json.dump({str(k): {kk: vv for kk, vv in d.items() if kk not in ('prof', 'uk')}
               for k, d in results.items()},
              open(os.path.join(BASE, 'gap_vs_hv.json'), 'w'), indent=1)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for hv in sorted(results):
        d = results[hv]
        axs[0].plot(d['uk'], d['prof'], 'o-', ms=3,
                    label=f"{hv} V: U50={d['U']:.0f} ns")
    axs[0].set_xlabel('u [ns]'); axs[0].set_ylabel('median normalized charge')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    axs[0].set_title('deconvolved charge profiles by drift HV')
    hvv = sorted(results)
    axs[1].errorbar(hvv, [results[h]['gap_mm'] for h in hvv],
                    yerr=[results[h]['dgap'] for h in hvv], fmt='ko-', capsize=4)
    axs[1].axhline(29, color='r', ls='--', label='assumed 29 mm')
    axs[1].axhline(30, color='gray', ls=':', label='mechanical 30 mm')
    axs[1].set_xlabel('drift HV [V]'); axs[1].set_ylabel('v_fit x U50 [mm]')
    axs[1].set_title('implied charge-visible gap vs HV')
    axs[1].legend(); axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, 'gap_vs_hv.png'), dpi=110)
    print('saved gap_vs_hv.png')
