#!/usr/bin/env python3
"""timing_row.py <colsum.npz> --bundle <calib_bundle dir> --out <stem> [--meta k=v ...] [--nboot 50]
-- Paper I §7: the chamber's time resolution on one bench row, from the waveforms (no hit times, no track model, no M3).

Input: colsum.py's per-trigger strip sum S(t) of the track's cluster (X and Y), with the DREAM fine timestamp ftst.
Primary estimator 'edge' (estimators.py): the column START, i.e. the arrival of the charge made next to the mesh, from a
step of the row's own pass template fitted to the rising edge of S (samples up to the first above EDGE_FRAC = 0.25 of the
maximum), + 10 ns x ftst.  It is the particle's crossing time plus a fixed delay, for any track angle.

Clean triggers: both views have a cluster with >= 5 pulse strips, one cluster, no saturated strip.

Per row (all widths are s68 = half the 16-84 % range, errors by bootstrap over triggers):
  sigma_xy     s68(t_X - t_Y) / sqrt 2: the part of each view's time that the other view does not share (noise,
               sampling, view-specific response).  X and Y are read by different FEUs: their own ftst corrects each.
  sigma_x/y    s68 of t_X (t_Y) against the trigger: one view's absolute time resolution (trigger included).
  sigma_evt    s68 of (t_X + t_Y)/2 against the trigger: the event time from both views.
  sigma_common sqrt(sigma_evt^2 - sigma_xy^2 / 2): the part common to X and Y (the same electrons arrive in both).
  trig_cov     cov(start, end) of the column, mean of the views, central 90 %: a trigger-time error shifts both ends of
               the column while the ionization at the mesh and at the cathode is independent, so this is sigma_trigger^2
               (+ any other term common to both ends).  'end' = the last 30 %-of-maximum crossing of S.
Also: the same widths for the comparison estimators (colsum's box t0, the box with T fixed, cf 10 %, edge 15 / 40 %),
the walk (median t_evt in quartiles of the early amplitude), the per-ftst-class medians, and sigma in mm = sigma_t x
(gap / median T) (timing v of the row: the gap from rerun/calib/physics_inputs.json).
Writes <stem>.json and <stem>.npz (per clean trigger: acq, eid, ts, edge start and end per view, early amplitude).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402

PRIMARY = 0.25
GAP_MM = {'mx17_2': 31.12, 'mx17_3': 28.08, 'mx17_4': 30.48, 'mx17_6': 28.0, 'mx17_7': 29.44}   # physics_inputs.json


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan
    q = np.percentile(a, [16, 84])
    return float((q[1] - q[0]) / 2)


def trail(y, frac=0.3):
    m = y.max()
    if not m > 0:
        return np.nan
    k = len(y) - 1 - int(np.argmax(y[::-1] >= frac * m))
    if k >= len(y) - 1:
        return np.nan
    y0, y1 = y[k], y[k + 1]
    return float((k + (y0 - frac * m) / max(y0 - y1, 1e-9)) * E.DTS)


def tcov(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 50:
        return np.nan
    qa, qb = np.percentile(a, [5, 95]), np.percentile(b, [5, 95])
    s = (a > qa[0]) & (a < qa[1]) & (b > qb[0]) & (b < qb[1])
    return float(np.cov(a[s], b[s])[0, 1])


def widths(tx, ty):
    D = tx - ty; M = (tx + ty) / 2
    sxy = s68(D) / np.sqrt(2); sev = s68(M)
    return dict(sigma_xy=sxy, sigma_x=s68(tx), sigma_y=s68(ty), sigma_evt=sev,
                sigma_common=float(np.sqrt(max(sev ** 2 - sxy ** 2 / 2, 0.0))), bias_xy=float(np.nanmedian(D)))


def boot(fn, n, nboot, rng):
    vals = [fn(rng.integers(0, n, n)) for _ in range(nboot)]
    keys = vals[0].keys()
    return {k: float(np.nanstd([v[k] for v in vals])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('colsum'); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--meta', nargs='*', default=[]); ap.add_argument('--nboot', type=int, default=50)
    ap.add_argument('--det', default=None)
    a = ap.parse_args()
    z = np.load(a.colsum); d = {k: z[k] for k in z.files}
    b = np.load(os.path.join(a.bundle, 'arrays.npz'))
    kern = E.kernel(b['grid'], b['tmpl_x'])
    ns = d['x_S'].shape[1]
    clean = (d['x_has'] & d['y_has'] & (d['x_nsig'] >= 5) & (d['y_nsig'] >= 5) & (d['x_ncl'] == 1)
             & (d['y_ncl'] == 1) & ~d['x_sat'] & ~d['y_sat'])
    idx = np.flatnonzero(clean)
    out = dict(colsum=a.colsum, bundle=a.bundle, n_triggers=int(len(d['eid'])), n_clean=int(len(idx)),
               primary=f'edge{int(PRIMARY * 100)}', **dict(kv.split('=', 1) for kv in a.meta))
    if len(idx) < 200:
        out['skip'] = 'fewer than 200 clean triggers'
        json.dump(out, open(a.out + '.json', 'w'), indent=1); return
    t, early = {}, {}
    for p in 'xy':
        S = d[f'{p}_S'][idx].astype(float)
        off = E.FTST_NS * d[f'{p}_ftst'][idx].astype(float)
        T = float(np.nanmedian(d[f'{p}_T'][idx]))
        out[f'T_median_{p}'] = T
        r = {'box': d[f'{p}_t0'][idx].astype(float) + off}
        fx = E.FixT(kern, ns, T)
        r['fixT'] = np.array([fx.fit(s) for s in S]) + off
        r['cf10'] = np.array([E.cf(s, 0.10) for s in S]) + off
        for f in (0.15, PRIMARY, 0.40):
            E.EDGE_FRAC = f
            ed = E.Edge(kern, ns)
            r[f'edge{int(round(f * 100))}'] = np.array([ed.fit(s) for s in S]) + off
        r['end'] = np.array([trail(s) for s in S]) + off
        early[p] = np.array([s[: int(np.argmax(s > PRIMARY * s.max())) + 1].max() for s in S])
        t[p] = r
    rng = np.random.default_rng(1)
    pk = out['primary']
    est = {}
    for k in t['x']:
        if k == 'end':
            continue
        w = widths(t['x'][k], t['y'][k])
        w['err'] = boot(lambda ii: widths(t['x'][k][ii], t['y'][k][ii]), len(idx), a.nboot if k == pk else 10, rng)
        est[k] = w
    out['estimators'] = est
    sm = (t['x'][pk] + t['y'][pk]) / 2; em = (t['x']['end'] + t['y']['end']) / 2
    cv = tcov(sm, em)
    cvb = [tcov(sm[ii], em[ii]) for ii in (rng.integers(0, len(idx), len(idx)) for _ in range(a.nboot))]
    out['trig_cov_ns2'] = cv; out['trig_cov_err_ns2'] = float(np.nanstd(cvb))
    out['startstart_cov_ns2'] = tcov(t['x'][pk], t['y'][pk])
    amp = early['x'] + early['y']
    qs = np.percentile(amp, [0, 25, 50, 75, 100])
    out['walk_median_evt_by_amp_quartile'] = [float(np.nanmedian(sm[(amp >= qs[i]) & (amp <= qs[i + 1])])) for i in range(4)]
    out['walk_sigma_evt_by_amp_quartile'] = [s68(sm[(amp >= qs[i]) & (amp <= qs[i + 1])]) for i in range(4)]
    out['amp_quartile_edges'] = qs.tolist()
    for p in 'xy':
        f = d[f'{p}_ftst'][idx]
        out[f'ftst_median_{p}'] = {int(c): float(np.nanmedian(t[p][pk][f == c])) for c in np.unique(f)}
    det = a.det or next((m for m in GAP_MM if m in a.colsum), None)
    if det in GAP_MM:
        v = GAP_MM[det] / np.nanmean([out['T_median_x'], out['T_median_y']]) * 1e3      # um/ns
        out['v_timing_um_ns'] = float(v)
        out['mm'] = {k: float(est[pk][k] * v / 1e3) for k in ('sigma_xy', 'sigma_x', 'sigma_evt', 'sigma_common')}
    json.dump(out, open(a.out + '.json', 'w'), indent=1)
    np.savez_compressed(a.out + '.npz', acq=d['acq'][idx], eid=d['eid'][idx], ts=d['x_ts'][idx],
                        tx=t['x'][pk].astype(np.float32), ty=t['y'][pk].astype(np.float32),
                        ex=t['x']['end'].astype(np.float32), ey=t['y']['end'].astype(np.float32),
                        amp=amp.astype(np.float32), acq_tags=d['acq_tags'])
    e = est[pk]
    print(f'{os.path.basename(a.out)}: n {len(idx)}  sigma_xy {e["sigma_xy"]:.1f}  x {e["sigma_x"]:.1f}  evt {e["sigma_evt"]:.1f}'
          f'  common {e["sigma_common"]:.1f}  trig cov {cv:.0f} +- {out["trig_cov_err_ns2"]:.0f} ns^2', flush=True)


if __name__ == '__main__':
    main()
