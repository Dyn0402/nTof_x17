#!/usr/bin/env python3
"""fd_pass1.py --out F.json -- the data side of the FD angle budget re-measured on pass 1 (2026-10-31), both arms.

Per golden / headline row and view, on the pass table (Analysis_pass1, read-only), arms as rerun/paper1_res/res_rows.py:
  pass   the pass table's tan_theta (the bow block is applied in the pass)
  vt     v -> v r(t) + w0/kw re-folded (res_rows.vt_arm), identical to pass on a row with no v(t) map
samples
  full   every M3-matched event of angles/per_event.npz (FD's 'measured' sample)
  ig     03_angles' interior_good sample (paper1_eff/angles/<rel>/per_event.npz: the Paper I headline sample)
statistics (FD's convention, tilt_map_xval.py / measured_chi2.py): s68 = 68th percentile of |d - median| over |d| <= 5 deg;
  slope  the same after removing the global d(tan_ref) line (the angle response is not a tilt)
  1d40 / 2d40  after the cross-validated (even/odd event id) cell-median correction, 40 mm cells along the view's own
         coordinate / in (x, y) -- what is left of the position-dependent tilt once the pass's bow block is applied
  v_rms  rms of r(t) over the row's events (the v(t) map's v_rel; 0 where the row has no map), for the L8 rescale
Run on lxplus with RES_CODE=<code archive with rerun/paper1_res> (lock venv).  budget_pass1.py turns it into the budget.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

CODE = os.environ['RES_CODE']
sys.path[:0] = [os.path.join(CODE, 'rerun', 'paper1_res')]
import res_rows as rr  # noqa: E402

ROWS = ('c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'o22_long_det2',
        'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3', 'sat_det3',
        'g_det4', 'g_det6_long', 'g_det7_long')
CELL_MM = 40.0


def s68(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    if len(a) < 50:
        return np.nan
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def xval(d, keys, half):
    df = pd.DataFrame({'d': d, 'h': half, **{f'k{i}': k for i, k in enumerate(keys)}})
    kc = [f'k{i}' for i in range(len(keys))]
    corr = np.zeros(len(d))
    for h in (0, 1):
        tr = df[(df.h != h) & (df.d.abs() <= 5)]
        med = tr.groupby(kc).d.agg(['median', 'size'])
        med = med[med['size'] >= 8]['median']
        te = df[df.h == h]
        idx = pd.MultiIndex.from_frame(te[kc]) if len(kc) > 1 else pd.Index(te[kc[0]])
        corr[te.index] = med.reindex(idx).fillna(0.0).to_numpy()
    return d - corr


def measure(eid, tref, tfit, pv, po):
    d = np.degrees(np.arctan(tfit) - np.arctan(tref))
    ok = np.isfinite(d) & np.isfinite(pv) & np.isfinite(po) & np.isfinite(tref)
    d, tref, pv, po, half = d[ok], tref[ok], pv[ok], po[ok], (eid[ok] % 2).astype(int)
    out = dict(n=int(ok.sum()), s68=s68(d))
    k = np.abs(d) < 5
    ds = d - np.polyval(np.polyfit(tref[k], d[k], 1), tref)
    out['slope'] = s68(ds)
    out['1d40'] = s68(xval(ds, (np.floor(pv / CELL_MM).astype(int),), half))
    out['2d40'] = s68(xval(ds, (np.floor(pv / CELL_MM).astype(int), np.floor(po / CELL_MM).astype(int)), half))
    return out


def do_row(r, by, vtrows):
    wdir = os.path.join(rr.P1, rr.rel(r), 'wft')
    m, cal = rr.load_calib(wdir)
    df = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    df = df[~df.event_id.duplicated(keep=False)].set_index('event_id')
    vrel, vrms = None, 0.0
    if r['key'] in vtrows:
        vrel = pd.read_parquet(vtrows[r['key']]['table']).drop_duplicates('event_id').set_index('event_id')['v_rel']
        vrms = float(np.nanstd(vrel.reindex(df.index).to_numpy()))
    V = rr.vt_arm(wdir, df, cal, vrel)
    Z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
    ANG = np.load(os.path.join(rr.EFF, 'angles', rr.rel(r), 'per_event.npz'))
    out = dict(key=r['key'], det=rr.SHORT[r['det']], cls=r['cls'], v_rms=vrms, vt_map=vrel is not None, views={})
    for p, o in (('x', 'y'), ('y', 'x')):
        res = {}
        for smp, src in (('full', Z), ('ig', ANG)):
            e = src[f'{p}_eid'].astype(int)
            k = ~pd.Index(e).duplicated(keep=False) & np.isin(e, df.index.to_numpy())
            e = e[k]
            tref = src[f'{p}_tan_ref'][k]
            pv = df[f'{p}_p0'].reindex(e).to_numpy()
            po = df[f'{o}_p0'].reindex(e).to_numpy()
            t_pass = df[f'{p}_tan_theta'].reindex(e).to_numpy()
            t_vt = V[p]['tan'].reindex(e).to_numpy()
            t_vt = np.where(np.isfinite(t_vt), t_vt, t_pass)
            res[smp] = dict(pass_=measure(e, tref, t_pass, pv, po), vt=measure(e, tref, t_vt, pv, po))
        out['views'][p] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = list(csv.DictReader(open(os.path.join(CODE, 'rerun', 'pass_registry.csv'))))
    by = {r['key']: r for r in reg}
    vtrows = {r['key']: r for r in csv.DictReader(open(os.path.join(CODE, 'rerun', 'vt_all', 'results',
                                                                    'vt_corrected_rows.csv')))}
    out = {}
    for k in ROWS:
        out[k] = do_row(by[k], by, vtrows)
        v = out[k]['views']
        print(k, out[k]['v_rms'], {p: {s: (round(v[p][s]['pass_']['s68'], 3), round(v[p][s]['vt']['s68'], 3),
                                          round(v[p][s]['vt']['2d40'], 3)) for s in v[p]} for p in v}, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
