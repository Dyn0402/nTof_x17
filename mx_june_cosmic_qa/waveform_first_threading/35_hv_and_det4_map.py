#!/usr/bin/env python3
"""(1) Mean column vs drift HV (electrostatic sag ~E^2 vs mechanical tilt);
(2) det4 gap map (does another chamber show the same y-tilt?)."""
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
B4 = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det4_day_6-24-26/long_run/'
      'mx17_4/waveform_first')
hj = json.load(open(os.path.join(B3, 'hyper_v2.json')))
H3 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V3 = hj['v']
ffv = json.load(open(os.path.join(B3, 'drift_scan_v.json')))
DT = 60.0

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

def Uboot(uk, P, n=120):
    rng = np.random.default_rng(3)
    vals = [U50g(uk, P[rng.integers(0, len(P), len(P))].mean(axis=0))
            for _ in range(n)]
    return np.nanmedian(vals), np.nanstd(vals)

_ctx = {}

def fit_one(args):
    eid, v, K = args
    fm2.K = K
    fm2.UK = (np.arange(K) + 0.5) * DT
    ev = _ctx['events'][eid]
    H = _ctx['H']
    out = []
    for plane in ('x', 'y'):
        tn = ev[f'tan_{plane}']
        if abs(tn) < 0.08:
            continue
        P = ev[plane]
        if P['W'].size == 0:
            continue
        W, noise, pos, sat = fm2.prep_plane(P, plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 640.0, 30.0):
            c, q = fm2.chi2_plane(plane, W, noise, pos, sat,
                                  ev[f'ref_mesh_{plane}'], tn * v * 1e-3,
                                  float(t0), H)
            if c < best[0]:
                best = (c, q)
        q = best[1]
        if q is not None and 0 < q.sum() and q.max() < 2e4:
            out.append((ev['ref_mesh_x'], ev['ref_mesh_y'], q / q.sum()))
    return out

def collect(events, H, v, K, n_ev):
    sel = [e for e, ev in events.items()
           if max(abs(ev['tan_x']), abs(ev['tan_y'])) > 0.08][:n_ev]
    _ctx['events'] = {e: events[e] for e in sel}
    _ctx['H'] = H
    rows = []
    with ProcessPoolExecutor(max_workers=14) as pool:
        for o in pool.map(fit_one, [(e, v, K) for e in sel], chunksize=6):
            rows.extend(o)
    L = max(len(r[2]) for r in rows)
    P = np.zeros((len(rows), L))
    for i, r in enumerate(rows):
        P[i, :len(r[2])] = r[2]
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            P, (np.arange(L) + 0.5) * DT)

if __name__ == '__main__':
    print('== column vs drift HV (det3, mean estimator, extended K) ==')
    out_hv = {}
    for hv, K in ((900, 22), (1000, 24), (1100, 20)):
        if hv == 1000:
            events = pickle.load(open(os.path.join(B3, 'wfcache.pkl'), 'rb'))['events']
            v = V3
        else:
            events = pickle.load(open(os.path.join(B3, f'wfcache_{hv}V.pkl'), 'rb'))
            v = ffv[str(hv)]['v']
        X, Y, P, uk = collect(events, H3, v, K, 4000)
        U, dU = Uboot(uk, P)
        col = v * U * 1e-3
        out_hv[hv] = dict(v=v, U=U, dU=dU, col=col, dcol=v * dU * 1e-3, n=len(P))
        print(f'HV {hv}: v={v:.1f} U50={U:.0f}±{dU:.0f} -> col {col:.1f}±{v*dU*1e-3:.1f} mm '
              f'(n={len(P)})', flush=True)
        # y-tilt per HV (two halves)
        ymid = np.median(Y)
        for lab, m in (('y<med', Y < ymid), ('y>med', Y >= ymid)):
            Uh, dUh = Uboot(uk, P[m])
            print(f'   {lab}: {v*Uh*1e-3:.1f}±{v*dUh*1e-3:.1f} mm', flush=True)

    print('== det4 gap map ==')
    hj4 = json.load(open(os.path.join(B4, 'hyper_det4.json')))
    H4 = {k: hj4[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
    ev4 = pickle.load(open(os.path.join(B4, 'wfcache.pkl'), 'rb'))['events']
    tz4 = np.load(os.path.join(B4, 'templates_perplane.npz'))
    fm2.TGRID = tz4['grid']
    fm2.TMPL = {'x': tz4['tmpl_x'], 'y': tz4['tmpl_y']}
    fm2.GAIN = {'x': np.ones(512), 'y': np.ones(512)}
    fm2._smear_cache.clear()
    ns4 = next(iter(ev4.values()))['x']['W'].shape[1]
    fm2.set_nsamp(ns4)
    X, Y, P, uk = collect(ev4, H4, hj4['v'], 22, 6000)
    U, dU = Uboot(uk, P)
    print(f'det4 overall: col {hj4["v"]*U*1e-3:.1f}±{hj4["v"]*dU*1e-3:.1f} mm (n={len(P)})')
    xq = np.percentile(X, [0, 33, 66, 100]); yq = np.percentile(Y, [0, 33, 66, 100])
    M4 = np.full((3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            m = (X >= xq[i]) & (X < xq[i + 1]) & (Y >= yq[j]) & (Y < yq[j + 1])
            if m.sum() > 120:
                M4[i, j] = U50g(uk, P[m].mean(axis=0)) * hj4['v'] * 1e-3
    print('det4 column map [mm] (x terciles rows, y terciles cols):')
    print(np.round(M4, 1))
    json.dump(dict(hv=out_hv), open(os.path.join(B3, 'col_vs_hv.json'), 'w'),
              indent=1, default=float)
