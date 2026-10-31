#!/usr/bin/env python3
"""charge_rows.py --registry R --root ROOT --out-dir D [--only k1,k2] -- Paper I charge on the pass-1 bench tables.

Per bench row of the registry (cls != excluded): the fitted view charge q_sum of single-track, M3-matched, spark-free
events in the interior of the chamber, normalised to normal incidence with the M3 angles, on UNSATURATED events
(no strip at the 3550 rule in either view; PASS_PLAN §4, calib/saturation/DECISION_2026-10-29.md K3).  Also the
tan-sign facts per row (sign of kw, corr(w, M3 tan)) for the bench tan_y sign check.  Read-only on Analysis_pass1.

Writes <out-dir>/rows/<key>.json (statistics) and <out-dir>/events/<key>.npz (the selected per-event arrays).

The selection, in order (each count is kept in the json):
  base      x_ok and y_ok, not spark, the event_id once in the table (one track), M3-matched in both views
  interior  the mid-gap position (p0 + 15 mm * tan) at least EDGE_MM inside the chamber's measured active box
            (common/mx17_active_area.TRUE_ACTIVE_BY_DET), both coordinates; det7 additionally X >= DET7_XMIN at both
            ends of the column (X readout card 1 excluded: det7 X ch 256-511 only)
  unsat     x_n_sat_strips == 0 and y_n_sat_strips == 0
The charge per view is q_sum * cos(theta) with cos(theta) = 1 / sqrt(1 + tx^2 + ty^2) from the M3 tangents.
No chi2 gate: a quality gate fails preferentially on the bright events (det3-hv-charge memory, the gate trap).
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

ROOT = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
EDGE_MM = 30.0                       # interior fiducial: >= 30 mm inside the active box (PASS_PLAN §4, S4 re-check)
DET7_XMIN = 199.68 + 5.0             # det7 X: ch 256 = 199.68 mm (mx17_m1_map.csv), plus a 5 mm footprint margin
MID_MM = 15.0                        # mid-gap depth used for the fiducial position
N_BOOT = 200
# common/mx17_active_area.TRUE_ACTIVE_BY_DET (copied so the script runs from a bare checkout on lxplus)
ACTIVE = {'mx17_2': {'x': (0.0, 398.6), 'y': (19.7, 378.8)},
          'mx17_3': {'x': (0.0, 398.6), 'y': (18.0, 379.88)},
          'mx17_4': {'x': (0.0, 398.6), 'y': (18.4, 378.6)},
          'mx17_6': {'x': (0.0, 398.6), 'y': (17.9, 380.7)},
          'mx17_7': {'x': (0.0, 398.6), 'y': (20.4, 379.3)}}


def boot_median(a, rng):
    if len(a) < 10:
        return float('nan')
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    return float(np.std(np.median(a[idx], axis=1)))


def trunc_mean(a, keep=0.7):
    if len(a) == 0:
        return float('nan')
    s = np.sort(a)
    return float(np.mean(s[:max(1, int(keep * len(s)))]))


def stats(a, rng):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return dict(n=int(len(a)), median=float(np.median(a)) if len(a) else float('nan'),
                median_err=boot_median(a, rng), tmean70=trunc_mean(a),
                p20=float(np.percentile(a, 20)) if len(a) else float('nan'),
                p80=float(np.percentile(a, 80)) if len(a) else float('nan'))


def one_row(r, root, out_dir, rng):
    wdir = os.path.join(root, r['run'], r['subrun'], r['det'], 'wft')
    m = json.load(open(os.path.join(wdir, 'events.meta.json')))
    ac = m['angle_constants']
    df = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    n_all = int(df.event_id.nunique())
    once = df.event_id.value_counts()
    df = df[df.event_id.isin(once.index[once == 1])].set_index('event_id')
    z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
    tr = {}
    for p in ('x', 'y'):
        tr[p] = pd.Series(z[f'{p}_tan_ref'], index=z[f'{p}_eid'].astype(int))
        tr[p] = tr[p][~tr[p].index.duplicated()]
    # tan-sign facts on every M3-matched plane (no other cut): sign of the slope of w against the M3 tangent
    sign = {}
    for p in ('x', 'y'):
        w = z[f'{p}_w'] * 1e3
        t = z[f'{p}_tan_ref']
        k = np.isfinite(w) & np.isfinite(t)
        sign[p] = dict(n=int(k.sum()), corr_w_tanref=float(np.corrcoef(w[k], t[k])[0, 1]) if k.sum() > 10 else None,
                       slope_w_vs_tanref=float(np.polyfit(t[k], w[k], 1)[0]) if k.sum() > 10 else None,
                       kw=float(ac['kw'][p]), w0=float(ac['w0'][p]),
                       frac_same_sign=float(np.mean(np.sign(z[f'{p}_tan_fit'][k]) == np.sign(t[k])))
                       if k.sum() else None)
    base = df.x_ok.astype(bool) & df.y_ok.astype(bool) & ~df.spark.astype(bool)
    base &= df.index.isin(tr['x'].index) & df.index.isin(tr['y'].index)
    d = df[base].copy()
    d['tx'] = tr['x'].reindex(d.index).to_numpy()
    d['ty'] = tr['y'].reindex(d.index).to_numpy()
    cos = 1.0 / np.sqrt(1.0 + d.tx ** 2 + d.ty ** 2)
    xm = d.x_p0 + MID_MM * d.x_tan_theta
    ym = d.y_p0 + MID_MM * d.y_tan_theta
    box = ACTIVE[r['det']]
    inside = ((xm >= box['x'][0] + EDGE_MM) & (xm <= box['x'][1] - EDGE_MM) &
              (ym >= box['y'][0] + EDGE_MM) & (ym <= box['y'][1] - EDGE_MM))
    if r['det'] == 'mx17_7':
        inside &= (d.x_p0 >= DET7_XMIN) & (d.x_p0 + 2 * MID_MM * d.x_tan_theta >= DET7_XMIN)
    d = d[inside.to_numpy()]
    cos = cos[inside.to_numpy()]
    sat_x = d.x_n_sat_strips.to_numpy() > 0
    sat_y = d.y_n_sat_strips.to_numpy() > 0
    unsat = ~sat_x & ~sat_y
    qx = d.x_q_sum.to_numpy() * cos.to_numpy()
    qy = d.y_q_sum.to_numpy() * cos.to_numpy()
    out = dict(key=r['key'], det=r['det'], run=r['run'], subrun=r['subrun'], cls=r['cls'],
               resist_V=float(r['resist_V']), drift_V=float(r['drift_V']), code=m.get('provenance', {}),
               counts=dict(all=n_all, single_track=int(len(df)), base=int(base.sum()), interior=int(len(d)),
                           unsat=int(unsat.sum())),
               f_sat=dict(x=float(sat_x.mean()) if len(d) else None, y=float(sat_y.mean()) if len(d) else None,
                          any=float((sat_x | sat_y).mean()) if len(d) else None),
               unsat=dict(x=stats(qx[unsat], rng), y=stats(qy[unsat], rng), sum=stats((qx + qy)[unsat], rng)),
               all=dict(x=stats(qx, rng), y=stats(qy, rng), sum=stats(qx + qy, rng)),
               per_view_unsat=dict(x=stats(qx[~sat_x], rng), y=stats(qy[~sat_y], rng)),
               sign=sign, t_h=None)
    if unsat.sum() > 10:
        lr = np.log(qx[unsat] / qy[unsat])
        lr = lr[np.isfinite(lr)]
        q16, q50, q84 = np.percentile(lr, [16, 50, 84])
        out['xy'] = dict(n=int(len(lr)), ln_ratio_median=float(q50), ln_ratio_s68=float(0.5 * (q84 - q16)),
                         asym_median=float(np.median((qx[unsat] - qy[unsat]) / (qx[unsat] + qy[unsat]))))
    if 'timestamp' in d:
        out['t_span_ticks'] = float(d.timestamp.max() - d.timestamp.min()) if len(d) else None
    os.makedirs(os.path.join(out_dir, 'rows'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'events'), exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, 'rows', r['key'] + '.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out_dir, 'events', r['key'] + '.npz'), event_id=d.index.to_numpy(),
                        qx=qx, qy=qy, cos=cos.to_numpy(), sat_x=sat_x, sat_y=sat_y,
                        nsx=d.x_n_sat_strips.to_numpy(), nsy=d.y_n_sat_strips.to_numpy(),
                        tx=d.tx.to_numpy(), ty=d.ty.to_numpy(), xm=xm[inside.to_numpy()].to_numpy(),
                        ym=ym[inside.to_numpy()].to_numpy(), timestamp=d.timestamp.to_numpy(),
                        acq=d.acq_tag.astype(str).to_numpy().astype('U'), quality_ok_x=d.x_quality_ok.to_numpy(),
                        quality_ok_y=d.y_quality_ok.to_numpy(), u90x=d.x_q_u90.to_numpy(), u90y=d.y_q_u90.to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', required=True)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    rng = np.random.default_rng(12345)
    rows = [r for r in csv.DictReader(open(a.registry)) if r['cls'] != 'excluded']
    if a.only:
        keep = set(a.only.split(','))
        rows = [r for r in rows if r['key'] in keep]
    bad = []
    for r in rows:
        try:
            o = one_row(r, a.root, a.out_dir, rng)
            print(r['key'], o['counts'], 'f_sat', round(o['f_sat']['any'] or -1, 3),
                  'med_sum', round(o['unsat']['sum']['median'], 1), flush=True)
        except Exception as e:                                          # noqa: BLE001
            bad.append((r['key'], repr(e)))
            print('FAIL', r['key'], repr(e), flush=True)
    json.dump(bad, open(os.path.join(a.out_dir, 'failures.json'), 'w'), indent=1)
    print('done', len(rows), 'failed', len(bad))


if __name__ == '__main__':
    main()
