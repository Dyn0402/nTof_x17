#!/usr/bin/env python3
"""tilt.py -- FA item 3: the chamber tilt against M3 for every reconstructed bench row, rigid tilt split from the bow.

The mesh height under the M3 reference is modelled as a quadric about the chamber centre (strip frame, P = p - 200 mm,
Q = q - 200 mm):  z(P, Q) = z0 + s_p P + s_q Q + k_pp P^2 + k_pq P Q + k_qq Q^2.
  * rigid tilt  = the slopes at the centre (s_x, s_y): a remount changes it;
  * bow (dish)  = the curvatures (k): the board's own shape, static per chamber whatever its mounting.
Two independent routes measure both, on the same events (geometry from the wft fit only):
  angle route    the fitted lateral speed w of view p, robust line  w = v tan_p + a + g P + h Q  (|tan_p| < 0.35):
                 the local angle offset is the local mesh slope, so  a = v s_p,  g = 2 v k_pp,  h = v k_pq;
  position route the mesh-position residual r_p = p0 - ref_mesh_p is tan_p x z(P, Q) (+ alignment terms e P + f Q):
                 r_p = c + tan_p (dz + s_p P + s_q Q + k_pp P^2 + k_pq P Q + k_qq Q^2) + e P + f Q,
                 so each view measures both slopes; the slope along p is combined from view p (tan_p P) and view q (tan_q P).
Sign: as ws_v_w0 (w0_study.py, r = 0.81 between the routes): positive = the angle offset w0/v; its absolute direction
is pinned by M3's z axis pointing up (ws_v_w0 F9), not by this comparison.
Rows: the five golden tables (ws_v_w0), the det3 dry runs (d700, d500, 6-25), and every FA row pulled from EOS
(<FA>/<run>/<subrun>/<det>/fa/table.parquet).  Output results/tilt_rows.json, results/tilt_table.md.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_common import ROOT, WS, FA, RES, CENTRE, huber, load_table, GOLD   # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
MANIFEST = os.path.join(REPO, 'mx_june_wft', 'condor', 'campaign_manifest.csv')
DETN = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
TAN_MAX = 0.35
MATCH_MM = 5.0         # |p0 - ref_mesh| for a reference ray that points at this track
RNG = np.random.default_rng(3)


def manifest():
    out = {}
    for r in csv.DictReader(open(MANIFEST)):
        out[(r['run'], r['subrun'], r['det'])] = r
    return out


def angle_route(t, p, q, nboot=60):
    ok = t[f'{p}_ok'].astype(bool).to_numpy() & np.isfinite(t[f'{p}_w']).to_numpy()
    tan = t[f'tan_ref_{p}'].to_numpy(); w = t[f'{p}_w'].to_numpy() * 1e3
    P = t[f'ref_mesh_{p}'].to_numpy() - CENTRE; Q = t[f'ref_mesh_{q}'].to_numpy() - CENTRE
    # the same reference match as the position route: on rows with a degraded M3 (6-17, 6-22/6-23 bottom slot) most
    # M3 rays do not point at the chamber's track and the line would fit noise
    match = (np.abs(t[f'{p}_p0'] - t[f'ref_mesh_{p}']) < MATCH_MM).to_numpy()
    m = ok & match & np.isfinite(tan) & (np.abs(tan) < TAN_MAX) & np.isfinite(P) & np.isfinite(Q)
    X = np.column_stack([tan, np.ones_like(tan), P, Q])[m]; y = w[m]
    b, cov, n = huber(X, y)
    bs = []
    idx = np.arange(len(y))
    for _ in range(nboot):
        i = RNG.choice(idx, len(idx))
        bs.append(huber(X[i], y[i], it=15)[0])
    bs = np.array(bs)
    v = b[0]; s = np.degrees(np.arctan(b[1] / v))
    sd = np.degrees(np.arctan(bs[:, 1] / bs[:, 0]))
    return dict(n=n, v=float(v), v_err=float(bs[:, 0].std()), a=float(b[1]), a_err=float(bs[:, 1].std()),
                tilt_deg=float(s), tilt_err=float(sd.std()),
                k_pp=float(b[2] / (2 * v)), k_pp_err=float(np.std(bs[:, 2] / (2 * bs[:, 0]))),
                k_pq=float(b[3] / v), k_pq_err=float(np.std(bs[:, 3] / bs[:, 0])))


def position_route(t, p, q):
    ok = t[f'{p}_ok'].astype(bool) & (np.abs(t[f'{p}_p0'] - t[f'ref_mesh_{p}']) < 5)
    s = t[ok]
    r = (s[f'{p}_p0'] - s[f'ref_mesh_{p}']).to_numpy()
    tp = s[f'tan_ref_{p}'].to_numpy()
    P = s[f'ref_mesh_{p}'].to_numpy() - CENTRE; Q = s[f'ref_mesh_{q}'].to_numpy() - CENTRE
    m = np.abs(tp) < 0.6
    X = np.column_stack([np.ones_like(r), tp, tp * P, tp * Q, tp * P ** 2, tp * P * Q, tp * Q ** 2, P, Q])[m]
    b, cov, n = huber(X, r[m])
    e = np.sqrt(np.diag(cov))
    names = ['c', 'dz', 's_p', 's_q', 'k_pp', 'k_pq', 'k_qq', 'e', 'f']
    return dict(n=n, **{k: float(v) for k, v in zip(names, b)}, **{k + '_err': float(v) for k, v in zip(names, e)})


def combine(a, ea, b, eb):
    w1, w2 = 1 / ea ** 2, 1 / eb ** 2
    return float((a * w1 + b * w2) / (w1 + w2)), float(1 / np.sqrt(w1 + w2))


def analyse(t):
    R = {}
    for p, q in (('x', 'y'), ('y', 'x')):
        R[f'ang_{p}'] = angle_route(t, p, q)
        R[f'pos_{p}'] = position_route(t, p, q)
    for p, q in (('x', 'y'), ('y', 'x')):
        sp, ep = combine(R[f'pos_{p}']['s_p'], R[f'pos_{p}']['s_p_err'], R[f'pos_{q}']['s_q'], R[f'pos_{q}']['s_q_err'])
        kp, ekp = combine(R[f'pos_{p}']['k_pp'], R[f'pos_{p}']['k_pp_err'], R[f'pos_{q}']['k_qq'], R[f'pos_{q}']['k_qq_err'])
        R[f'tilt_{p}'] = dict(angle_deg=R[f'ang_{p}']['tilt_deg'], angle_err=R[f'ang_{p}']['tilt_err'],
                              pos_deg=float(np.degrees(sp)), pos_err=float(np.degrees(ep)),
                              curv_angle_per_km=1e6 * R[f'ang_{p}']['k_pp'], curv_angle_err=1e6 * R[f'ang_{p}']['k_pp_err'],
                              curv_pos_per_km=1e6 * kp, curv_pos_err=1e6 * ekp)
    return R


def rows():
    """(label, det, run, subrun, table path)"""
    out = []
    for det, key in GOLD.items():
        out.append((key, det, None, None, os.path.join(WS, f'{det}.parquet')))
    out += [('det3 sat 700 V (dry run)', 'det3', 'mx17_det3_saturday_scan_6-27-26', 'drift_scan_resist_490V_drift_700V',
             os.path.join(WS, 'det3_d700.parquet')),
            ('det3 sat 500 V (dry run)', 'det3', 'mx17_det3_saturday_scan_6-27-26', 'drift_scan_resist_490V_drift_500V',
             os.path.join(WS, 'det3_d500.parquet')),
            ('det3 6-25 (dry run)', 'det3', 'mx17_det3_day_6-25-26', 'long_run', os.path.join(WS, 'det3_day625.parquet'))]
    for f in sorted(glob.glob(os.path.join(FA, '*', '*', 'mx17_*', 'fa', 'table.parquet'))):
        parts = f.split(os.sep)
        run, sub, dn = parts[-5], parts[-4], parts[-3]
        out.append((f'{DETN[dn]} {run} {sub}', DETN[dn], run, sub, f))
    return out


def main():
    man = manifest()
    gold_rs = {'o22_long_det2': ('mx17_det2_det3_overnight_6-22-26', 'longer_run'),
               'sat_det3': ('mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V'),
               'g_det4': ('mx17_det4_day_6-24-26', 'long_run'), 'g_det6_long': ('mx17_det6_det7_overnight_6-26-26', 'long_run'),
               'g_det7_long': ('mx17_det6_det7_overnight_6-26-26', 'long_run')}
    res = []
    for label, det, run, sub, path in rows():
        if run is None:
            run, sub = gold_rs[label]
        dn = {v: k for k, v in DETN.items()}[det]
        mrow = man.get((run, sub, dn), {})
        t = load_table(path)
        if len(t) < 150:
            print('skip (too few events)', label, len(t)); continue
        try:
            R = analyse(t)
        except Exception as e:                         # a row whose fit cannot be done is listed, not hidden
            print('FAILED', label, e); res.append(dict(label=label, det=det, run=run, subrun=sub, failed=str(e))); continue
        date = run.split('_')[-1]
        R.update(label=label, det=det, run=run, subrun=sub, date=date, n_events=len(t),
                 slot_z=float(mrow.get('det_z') or np.nan), feu_x=mrow.get('feu_x'), feu_y=mrow.get('feu_y'),
                 drift_V=float(mrow.get('drift_V') or np.nan), resist_V=float(mrow.get('resist_V') or np.nan))
        # the angle route needs a working ladder: v of each view above half the expected v at this drift HV (det3's
        # column-duration v(E): 36 um/ns x (V/1000)^0.87) and X, Y within 10 %.  6-17 det2 and the 6-22/6-23 bottom-slot
        # det3 rows fail it (v 0-15): the fitted w does not follow the M3 angle there; their position route stands.
        vexp = 36.0 * (R['drift_V'] / 1000.0) ** 0.87 if np.isfinite(R['drift_V']) else 30.0
        vx, vy = R['ang_x']['v'], R['ang_y']['v']
        R['angle_valid'] = bool(vx > 0.5 * vexp and vy > 0.5 * vexp and abs(vx / vy - 1) < 0.1)
        res.append(R)
        print(f"{label[:70]:70s} n {len(t):5d} | X ang {R['tilt_x']['angle_deg']:+.3f}±{R['tilt_x']['angle_err']:.3f} pos "
              f"{R['tilt_x']['pos_deg']:+.3f}±{R['tilt_x']['pos_err']:.3f} | Y ang {R['tilt_y']['angle_deg']:+.3f}±"
              f"{R['tilt_y']['angle_err']:.3f} pos {R['tilt_y']['pos_deg']:+.3f}±{R['tilt_y']['pos_err']:.3f} | curv/km X "
              f"{R['tilt_x']['curv_angle_per_km']:+.0f}/{R['tilt_x']['curv_pos_per_km']:+.0f} Y {R['tilt_y']['curv_angle_per_km']:+.0f}/"
              f"{R['tilt_y']['curv_pos_per_km']:+.0f}")
    json.dump(res, open(os.path.join(RES, 'tilt_rows.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
