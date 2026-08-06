#!/usr/bin/env python3
"""
gap_vpin_test.py — how much does the drift-gap column depend on the drift speed
ASSUMED by the model?

The column is a product of two things the fit produces: the arrival-time
endpoint T_end [ns] and the geometric drift speed v_geom = median(w / tan_ref)
[um/ns], with the reference supplying tan. Both are fitted with a model that
already contains an assumed v. This scans the assumed v and reports:

    v_in   ->   T_end,  v_geom(v_in),  column = T_end * v_geom

The physically meaningful point is the SELF-CONSISTENT one, v_geom(v_in) = v_in.
If the column is flat in v_in, the pin does not matter and a per-run pin is
harmless; if it slopes, every quoted column must be taken at its fixed point.

    ../../.venv/bin/python mx_june_wft/bench/gap_vpin_test.py <run_key> \
        --bundle <lp bundle> [--n 2500] [--jobs 4] [--v 34 35 36 37 38 39]
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

_CAL = None


def _init(bundle, v):
    global _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(bundle)
    _CAL.v_drift = float(v)
    _CAL.kw = {'x': 1.0, 'y': 1.0}
    wm.use_calibration(_CAL)


def _fit_one(payload):
    from wft import model as wm
    from wft import reco as wr
    eid, wins = payload
    out = {'eid': eid}
    for plane in ('x', 'y'):
        P = wins.get(plane)
        if P is None:
            continue
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        try:
            p0s, _w, t0s = wm.init_guess(P, plane)
            p0s, w0, t0s = wr._global_start(P, plane, p0s, t0s, wm.HYPER)
            r = wm.fit_plane_raw(P, plane, p0s, w0, t0s)
        except Exception:
            continue
        if r is None or not np.isfinite(r['chi2']):
            continue
        out[plane] = dict(q=np.asarray(r['q'], float), w=float(r['w']),
                          chi2dof=float(r['chi2'] / max(r['dof'], 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--plane', default='x')
    ap.add_argument('--v', type=float, nargs='+',
                    default=[34, 35, 36, 37, 38, 39, 40])
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from scipy.optimize import curve_fit
    from scipy.special import erfc

    cfg = get_config(args.run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    cal = CalibrationBundle.load(args.bundle)
    data = pickle.load(open(os.path.join(W, 'bench_cache.pkl'), 'rb'))
    events, box = data['events'], data['meta']['box']
    plane = args.plane

    payloads, truth = [], {}
    for eid, ev in sorted(events.items()):
        t = ev['truth']
        if not all(np.isfinite([t['ref_x'], t['ref_y'], t['tan_x'], t['tan_y']])):
            continue
        m = 3.0
        cont = (box['x'][0] + m + 15.5 * abs(t['tan_x']) <= t['ref_x']
                <= box['x'][1] - m - 15.5 * abs(t['tan_x'])
                and box['y'][0] + m + 15.5 * abs(t['tan_y']) <= t['ref_y']
                <= box['y'][1] - m - 15.5 * abs(t['tan_y']))
        if not cont:
            continue
        wins = {}
        for p in ('x', 'y'):
            cand = ev['wins'].get(p)
            s = ev['seeds'].get(p)
            if cand and s and s[0]['n_dropped'] == 0 and len(cand) == 1:
                wins[p] = cand[0]
        if plane not in wins:
            continue
        truth[eid] = t
        payloads.append((eid, wins))
        if len(payloads) >= args.n:
            break
    print(f'{len(payloads):,} contained events, plane {plane}, '
          f'bundle v = {cal.v_drift:.2f}')

    u = None
    rows = []
    for v_in in args.v:
        outs = []
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                 initargs=(args.bundle, v_in)) as pool:
            for o in pool.map(_fit_one, payloads, chunksize=8):
                outs.append(o)
        if u is None:
            from wft import model as wm
            wm.use_calibration(cal)
            u = wm.UK.copy()
        prof, ws, tans = [], [], []
        for o in outs:
            d = o.get(plane)
            if d is None or d['chi2dof'] > 250:
                continue
            q = d['q']
            if q.sum() <= 0:
                continue
            prof.append(q / q.sum())
            ws.append(d['w'] * 1e3)
            tans.append(truth[o['eid']][f'tan_{plane}'])
        P = np.array(prof)
        ws, tans = np.array(ws), np.array(tans)
        s = (np.abs(tans) > 0.10) & (np.abs(tans) < 0.40)
        w0 = cal.w0.get(plane, 0.0)
        v_geom = float(np.median((ws[s] - w0) / tans[s]))
        m = P.mean(axis=0)
        e = np.maximum(P.std(axis=0) / np.sqrt(len(P)), 1e-5)

        def sharp(uu, A, T, sig):
            return A * 0.5 * erfc((uu - T) / (np.sqrt(2) * sig))

        sel = u < 1050
        p, c = curve_fit(sharp, u[sel], m[sel], p0=[m[:5].mean(), 700, 60],
                         sigma=e[sel], absolute_sigma=True, maxfev=40000,
                         bounds=([0, 200, 10], [np.inf, 1100, 400]))
        T, Te = float(p[1]), float(np.sqrt(c[1, 1]))
        col = T * v_geom / 1e3
        rows.append(dict(v_in=v_in, v_geom=v_geom, T_end=T, T_err=Te,
                         column=col, n=len(P)))
        print(f'  v_in {v_in:5.1f} -> v_geom {v_geom:6.2f}  '
              f'T_end {T:6.0f}+-{Te:3.0f} ns  column {col:6.2f} mm  '
              f'(n={len(P)})', flush=True)

    # self-consistent point: interpolate where v_geom(v_in) = v_in
    vi = np.array([r['v_in'] for r in rows])
    vg = np.array([r['v_geom'] for r in rows])
    col = np.array([r['column'] for r in rows])
    d = vg - vi
    fix = np.nan
    if np.any(d > 0) and np.any(d < 0):
        k = np.argsort(vi)
        fix = float(np.interp(0.0, d[k][::-1] if d[k][0] > d[k][-1] else d[k],
                              vi[k][::-1] if d[k][0] > d[k][-1] else vi[k]))
        col_fix = float(np.interp(fix, vi[k], col[k]))
        print(f'\n  self-consistent v = {fix:.2f} um/ns -> column {col_fix:.2f} mm')
    else:
        print('\n  no self-consistent point inside the scanned range')
    out = os.path.join(W, 'gap_study', 'vpin_scan.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(run_key=args.run_key, bundle=args.bundle, plane=plane,
                   rows=rows, v_selfconsistent=fix), open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
