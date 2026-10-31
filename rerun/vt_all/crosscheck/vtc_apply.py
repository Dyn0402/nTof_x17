#!/usr/bin/env python3
"""vtc_apply.py <key> [...] --vt-dir D --out-root R -- the corrected table of a bench pass-1 row whose v(t) gate says
'correct' and that has a line model (vt_bench's <key>.vt_map.json): the same columns and layout as vt_within_rows'
vt_apply.py, so every corrected row reads the same way (VT_PLAN §4, CROSSCHECK_2026-10-31 §3).

    <R>/<run>/<subrun>/<det>/events_vt.parquet   event_id, acq_tag, t_h, v_rel, {x,y}_tan_vt, {x,y}_theta_deg_vt,
                                                 {x,y}_mid_vt, z_end_vt
    <R>/<run>/<subrun>/<det>/vt_map.json         the v_map['t'] block (model, clock, t_h, v_rel, v_drift, source,
                                                 pass1_table)

tan = (w - w0) / (kw v r(t) S(U, W)) - a(U, W) with the row's own w0, kw, v and the chamber's bow block (exactly the
pass's mapping with v -> v r); mid = p0 + 15 tan; z_end = v kw_mean r u_end (vt_apply's column depth).  r(t) is the
line, clamped to the row (np.interp on its two end points).  Read-only on Analysis_pass1; refuses an out-root inside it.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'rerun', 'vt_all'), REPO]
import vt_bench as vb  # noqa: E402
from vt_common import bow_aS, tan_map  # noqa: E402
import wft.reco as wr  # noqa: E402


def apply(key, reg, vt_dir, out_root, root):
    r = reg[key]
    wdir = os.path.join(root, r['run'], r['subrun'], r['det'], 'wft')
    m, cal, df, V = vb.load_row(wdir, root)
    vm = json.load(open(os.path.join(vt_dir, f'{key}.vt_map.json')))
    assert vm['pass1_key'] == key and vm['model'] == 'line'
    rr = np.interp(df.t_h.to_numpy(), vm['t_h'], vm['v_rel'])
    tcx = (df.x_w * 1e3 - cal.w0['x']) / (cal.kw['x'] * cal.v_drift)
    tcy = (df.y_w * 1e3 - cal.w0['y']) / (cal.kw['y'] * cal.v_drift)
    U = ((df.x_p0 + wr.BOW_MID_MM * tcx) - 200.0) / 100.0
    W = ((df.y_p0 + wr.BOW_MID_MM * tcy) - 200.0) / 100.0
    U = U.where(np.isfinite(U) & (U.abs() <= wr.BOW_UW_MAX) & df.x_ok, 0.0).fillna(0.0).to_numpy()
    W = W.where(np.isfinite(W) & (W.abs() <= wr.BOW_UW_MAX) & df.y_ok, 0.0).fillna(0.0).to_numpy()
    out = pd.DataFrame(dict(event_id=df.index.to_numpy(), acq_tag=df.acq_tag.to_numpy(), t_h=df.t_h.to_numpy(), v_rel=rr))
    summ = dict(key=key, v_rel_min=float(rr.min()), v_rel_max=float(rr.max()))
    for p in ('x', 'y'):
        a, S = bow_aS(cal, p, U, W)
        ok = df[f'{p}_ok'].to_numpy()
        w = df[f'{p}_w'].to_numpy() * 1e3
        t0 = np.where(ok, tan_map(cal, p, w, a, S), np.nan)
        t1 = np.where(ok, tan_map(cal, p, w, a, S, rr), np.nan)
        # closure: r = 1 must give back the pass table's own tangent
        tt = df[f'{p}_tan_theta'].to_numpy()
        k = ok & np.isfinite(t0) & np.isfinite(tt)
        summ[f'{p}_closure_max'] = float(np.max(np.abs(t0[k] - tt[k]))) if k.any() else np.nan
        out[f'{p}_tan_vt'] = t1
        out[f'{p}_theta_deg_vt'] = np.degrees(np.arctan(t1))
        out[f'{p}_mid_vt'] = df[f'{p}_p0'].to_numpy() + 15.0 * t1
        sl = k & (np.abs(t0) < 1.0)
        dth = np.degrees(np.arctan(t1[sl]) - np.arctan(t0[sl]))
        summ[f'{p}_dtheta_p95_deg'] = float(np.percentile(np.abs(dth), 95))
    km = 0.5 * (cal.kw['x'] + cal.kw['y'])
    uend = df[['x_q_uend', 'y_q_uend']].where(df[['x_ok', 'y_ok']].to_numpy()).mean(axis=1).to_numpy()
    out['z_end_vt'] = cal.v_drift * km * uend * 1e-3 * rr
    d = os.path.join(out_root, r['run'], r['subrun'], r['det'])
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, 'events_vt.parquet'), index=False)
    json.dump(dict(vm, source='rerun/vt_all/vt_bench.py (model) + rerun/vt_all/crosscheck/vtc_apply.py (table)',
                   pass1_table=wdir), open(os.path.join(d, 'vt_map.json'), 'w'), indent=1)
    print(json.dumps(summ), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+'); ap.add_argument('--vt-dir', required=True); ap.add_argument('--out-root', required=True)
    ap.add_argument('--root', default=vb.ROOT); ap.add_argument('--summary', default=None)
    a = ap.parse_args()
    if 'Analysis_pass1' in os.path.abspath(a.out_root):
        ap.error('--out-root must not be inside Analysis_pass1 (read-only)')
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}
    S = [apply(k, reg, a.vt_dir, a.out_root, a.root) for k in a.keys]
    if a.summary:
        json.dump(S, open(a.summary, 'w'), indent=1)


if __name__ == '__main__':
    main()
