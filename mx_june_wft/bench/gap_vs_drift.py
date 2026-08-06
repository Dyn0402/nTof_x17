#!/usr/bin/env python3
"""
gap_vs_drift.py — is the charge-visible column invariant under the drift field?

The strongest single test that the column is GEOMETRY and not a timing
systematic: change the drift speed and see whether the column in mm stays put
while the column in ns moves. Uses the R&D drift-scan waveform caches
(``waveform_first/wfcache_<HV>V.pkl``, ref-pinned windows for the 6-27 det3
drift scan) with the production RC-ladder bundle.

Per HV point, two passes:
  1. fit every event with the bundle as-is, measure the GEOMETRIC drift speed
     v_geom = median((w*1e3 - w0) / tan_ref) against the M3 reference
  2. refit with v_drift := v_geom (self-consistent), stack the normalised
     charge-arrival profiles of contained tracks, fit the erfc endpoint

Only points whose full column fits inside the 1080 ns readout window are
meaningful: at v < ~28 um/ns a 30 mm column is truncated by the window, so the
300/500/700 V points are reported but flagged.

    ../../.venv/bin/python mx_june_wft/bench/gap_vs_drift.py \
        --bundle <lp bundle> [--jobs 8]
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

WF = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
      'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
HV_POINTS = [700, 900, 1100]        # 1000 V = the long run itself, added below
WINDOW_NS = 1080.0
BOX = dict(x=(20.0, 380.0), y=(30.0, 370.0))   # active box, raw strip frame

_CAL = None


def _init(bundle_path, v_override):
    global _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(bundle_path)
    if v_override:
        _CAL.v_drift = float(v_override)
        _CAL.kw = {'x': 1.0, 'y': 1.0}
    wm.use_calibration(_CAL)


def _fit_one(payload):
    from wft import model as wm
    from wft import reco as wr
    eid, ev = payload
    out = {'eid': eid, 'tan_x': ev['tan_x'], 'tan_y': ev['tan_y'],
           'ref_x': ev['ref_mesh_x'], 'ref_y': ev['ref_mesh_y']}
    for plane in ('x', 'y'):
        P = ev.get(plane)
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


def run_pass(cache, bundle, jobs, v_override=None):
    payloads = [(eid, ev) for eid, ev in sorted(cache.items())]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                             initargs=(bundle, v_override)) as pool:
        for out in pool.map(_fit_one, payloads, chunksize=8):
            rows.append(out)
    return rows


def v_geom_from(rows, plane, w0):
    tan = np.array([r['tan_x'] if plane == 'x' else r['tan_y'] for r in rows
                    if plane in r])
    w = np.array([r[plane]['w'] * 1e3 for r in rows if plane in r])
    s = (np.abs(tan) > 0.10) & (np.abs(tan) < 0.40)
    if s.sum() < 20:
        return np.nan, int(s.sum())
    return float(np.median((w[s] - w0) / tan[s])), int(s.sum())


def endpoint(rows, plane, v_geom):
    from scipy.optimize import curve_fit
    from scipy.special import erfc
    from wft import model as wm
    u = wm.UK.copy() if hasattr(wm, 'UK') else (np.arange(18) + 0.5) * 60.0
    prof = []
    for r in rows:
        d = r.get(plane)
        if d is None or d['chi2dof'] > 250:
            continue
        tan = r['tan_x'] if plane == 'x' else r['tan_y']
        if not np.isfinite(tan):
            continue
        m = 3.0 + 15.5 * abs(tan)
        if not (BOX['x'][0] + m <= r['ref_x'] <= BOX['x'][1] - m
                and BOX['y'][0] + m <= r['ref_y'] <= BOX['y'][1] - m):
            continue
        q = d['q']
        if q.sum() <= 0:
            continue
        prof.append(q / q.sum())
    if len(prof) < 50:
        return None
    P = np.array(prof)
    mprof = P.mean(axis=0)
    err = np.maximum(P.std(axis=0) / np.sqrt(len(P)), 1e-5)

    def sharp(uu, A, T, sig):
        return A * 0.5 * erfc((uu - T) / (np.sqrt(2) * sig))

    sel = u < 1050
    try:
        p, c = curve_fit(sharp, u[sel], mprof[sel],
                         p0=[mprof[:5].mean(), 700, 60],
                         sigma=err[sel], absolute_sigma=True, maxfev=40000,
                         bounds=([0, 200, 10], [np.inf, 1100, 400]))
    except Exception as exc:
        # at low drift field the column runs past the 1080 ns basis and the
        # edge model has nothing to fit: report it rather than crash
        return dict(n=len(P), error=str(exc), truncated=True)
    T, Te = float(p[1]), float(np.sqrt(c[1, 1]))
    return dict(n=len(P), T_end=T, T_err=Te, sig_e=float(p[2]),
                gap_mm=T * v_geom / 1e3, gap_err=Te * v_geom / 1e3,
                truncated=bool(T + 2 * float(p[2]) > WINDOW_NS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(REPO, 'mx_june_wft',
                                                  'gap_vs_drift.json'))
    args = ap.parse_args()

    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(args.bundle)
    caches = [(hv, os.path.join(WF, f'wfcache_{hv}V.pkl')) for hv in HV_POINTS]
    caches.append((1000, os.path.join(WF, 'wfcache.pkl')))
    results = {}
    for hv, path in sorted(caches):
        if not os.path.exists(path):
            print(f'-- {hv} V: no cache at {path}')
            continue
        cache = pickle.load(open(path, 'rb'))
        if 'events' in cache and 'meta' in cache:     # the 1000 V long-run cache
            cache = cache['events']
        print(f'\n== drift {hv} V: {len(cache):,} cached events', flush=True)
        rows = run_pass(cache, args.bundle, args.jobs)
        res = {'n_events': len(cache)}
        for plane in ('x', 'y'):
            vg, nv = v_geom_from(rows, plane, cal.w0.get(plane, 0.0))
            res[plane] = dict(v_geom=vg, n_v=nv)
        # pass 2: self-consistent v on the X plane (the plane we quote)
        vx = res['x']['v_geom']
        if np.isfinite(vx):
            rows2 = run_pass(cache, args.bundle, args.jobs, v_override=vx)
            for plane in ('x', 'y'):
                vg2, _ = v_geom_from(rows2, plane, cal.w0.get(plane, 0.0))
                e = endpoint(rows2, plane, vg2 if np.isfinite(vg2) else vx)
                res[plane].update(v_geom_pass2=vg2, endpoint=e)
                if e and 'error' in e:
                    print(f'  {plane}: v_geom {vg2:.2f} um/ns  '
                          f'endpoint fit failed (column past the basis end): '
                          f'{e["error"][:60]}', flush=True)
                elif e:
                    print(f'  {plane}: v_geom {vg2:.2f} um/ns  '
                          f'T_end {e["T_end"]:.0f}+-{e["T_err"]:.0f} ns  '
                          f'-> column {e["gap_mm"]:.2f}+-{e["gap_err"]:.2f} mm'
                          + ('   [WINDOW-TRUNCATED]' if e['truncated'] else ''),
                          flush=True)
        results[hv] = res

    with open(args.out, 'w') as f:
        json.dump(dict(bundle=args.bundle, results=results), f, indent=1,
                  default=float)
    print('\nwrote', args.out)
    print('\n== summary (X plane) ==')
    print(f"{'drift V':>8} {'v_geom':>8} {'T_end ns':>10} {'column mm':>12}")
    for hv, r in sorted(results.items()):
        e = r.get('x', {}).get('endpoint')
        if not e or 'T_end' not in e:
            print(f"{hv:>8} {r['x'].get('v_geom_pass2', float('nan')):>8.2f} "
                  f"{'--':>10} {'endpoint past the basis end':>12}")
            continue
        print(f"{hv:>8} {r['x']['v_geom_pass2']:>8.2f} "
              f"{e['T_end']:>10.0f} {e['gap_mm']:>8.2f}+-{e['gap_err']:.2f}"
              + ('  [truncated]' if e['truncated'] else ''))


if __name__ == '__main__':
    main()
