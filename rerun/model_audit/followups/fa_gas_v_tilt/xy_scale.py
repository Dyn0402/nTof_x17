#!/usr/bin/env python3
"""xy_scale.py -- FA item 2: is det3's X-Y ladder difference (+2.1 %) a drift-velocity difference or a per-view tan scale?

Physics first: both views read the SAME electrons drifting through the SAME gas over the SAME gap, so the drift
velocity cannot differ between views.  What the ladder measures is dw/dtan_ref, i.e. v times the view's angle scale.
Tests (all event-paired: the same tracks in X and Y):
  A  ladder ratio R = v_X / v_Y from robust lines w = v tan + a (intercept free; no head-on median), bootstrap;
     per chamber, per quadrant, per time fifth, and in a 3 x 3 grid of cells (does it follow position?);
  B  the position map: the same ratio after subtracting the ws_widthtilt w0(p) bow map from w (the local tilt);
  C  the column duration per view from the strip sum (colsum.npz; no angle, no M3): T_X / T_Y -- a v difference would
     show here, an angle-scale difference cannot;
  D  arms through the same fitter: free t0 (t0i) vs t0 pinned at the trigger latency (lat), sigma0_x 0.20 (s0x20),
     no per-event sigma0 choice (noprof) -- held-out 1320 events; and the full-row condor arms base / s0c (one common
     sigma0, no choice) / pin / pins0c when pulled (<FA>/<row>/wft/arm_<name>/events.parquet);
  E  per bench row (run, mounting) and per det3 HV row (FA item-3 tables).
Output results/xy_scale.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_common import ROOT, WS, FA, RES, DETS, GOLD_ROW, huber, golden_table, load_table, load_colsum, tmean  # noqa: E402

RNG = np.random.default_rng(11)
TAN_LO, TAN_HI = 0.0, 0.45


def slope(w, tr):
    m = np.isfinite(w) & np.isfinite(tr) & (np.abs(tr) < TAN_HI)
    b, _, _ = huber(np.column_stack([tr[m], np.ones(m.sum())]), w[m], it=20)
    return b[0]


def ratio(wx, tx, wy, ty, nb=150):
    """paired v_X / v_Y - 1 [%] with an event bootstrap."""
    r0 = 100 * (slope(wx, tx) / slope(wy, ty) - 1)
    n = len(wx); bs = []
    for _ in range(nb):
        i = RNG.integers(0, n, n)
        bs.append(100 * (slope(wx[i], tx[i]) / slope(wy[i], ty[i]) - 1))
    return float(r0), float(np.std(bs)), int(n)


def paired(t, wcol='w', w0map=None):
    ok = (t.x_ok.astype(bool) & t.y_ok.astype(bool) & np.isfinite(t[f'x_{wcol}']) & np.isfinite(t[f'y_{wcol}'])
          & (np.abs(t.x_p0 - t.ref_mesh_x) < 5) & (np.abs(t.y_p0 - t.ref_mesh_y) < 5)).to_numpy()   # M3 ray points at this track
    t = t[ok]
    wx, wy = t[f'x_{wcol}'].to_numpy() * 1e3, t[f'y_{wcol}'].to_numpy() * 1e3
    if w0map is not None:
        wx = wx - np.interp(t.ref_mesh_x.to_numpy(), w0map['x']['p'], w0map['x']['w0'])
        wy = wy - np.interp(t.ref_mesh_y.to_numpy(), w0map['y']['p'], w0map['y']['w0'])
    return t, wx, t.tan_ref_x.to_numpy(), wy, t.tan_ref_y.to_numpy()


def splits(t, wx, tx, wy, ty):
    out = {'all': ratio(wx, tx, wy, ty)}
    X, Y = t.ref_mesh_x.to_numpy(), t.ref_mesh_y.to_numpy()
    q = (X > 200).astype(int) * 2 + (Y > 200).astype(int)
    out['quadrant'] = [ratio(wx[q == k], tx[q == k], wy[q == k], ty[q == k], 80) for k in range(4)]
    tt = t['evttime'].rank(method='first').to_numpy() if 'evttime' in t and t['evttime'].notna().all() else np.arange(len(t))
    f = np.minimum((tt / len(t) * 5).astype(int), 4)
    out['time'] = [ratio(wx[f == k], tx[f == k], wy[f == k], ty[f == k], 80) for k in range(5)]
    edges = [0, 133, 266, 400]
    cells = []
    for i in range(3):
        for j in range(3):
            m = (X >= edges[i]) & (X < edges[i + 1]) & (Y >= edges[j]) & (Y < edges[j + 1])
            if m.sum() > 250:
                cells.append(dict(ix=i, iy=j, r=ratio(wx[m], tx[m], wy[m], ty[m], 60)))
    out['cells'] = cells
    c = np.array([c['r'][0] for c in cells]); e = np.array([c['r'][1] for c in cells])
    if len(c) > 1:
        wm = np.sum(c / e ** 2) / np.sum(1 / e ** 2)
        out['cells_chi2'] = [float(np.sum((c - wm) ** 2 / e ** 2)), len(c) - 1]
    return out


def colsum_ratio(det):
    f = os.path.join(FA, GOLD_ROW[det], 'fa', 'colsum.npz')
    if not os.path.exists(f):
        return None
    df, _, _ = load_colsum(f)
    ok = (df.x_has & df.y_has & (df.x_nsig >= 5) & (df.y_nsig >= 5) & (df.x_ncl == 1) & (df.y_ncl == 1)
          & np.isfinite(df.x_T) & np.isfinite(df.y_T)).to_numpy()
    ns = ok & ~df.x_sat.to_numpy() & ~df.y_sat.to_numpy()
    out = {}
    for name, m in (('all', ok), ('unsaturated', ns)):
        d = (df.x_T.to_numpy()[m] / df.y_T.to_numpy()[m])
        lr = np.log(d)
        bs = [tmean(lr[RNG.integers(0, len(lr), len(lr))]) for _ in range(200)]
        out[name] = dict(n=int(m.sum()), TX_over_TY_pct=float(100 * (np.exp(tmean(lr)) - 1)), err=float(100 * np.std(bs)),
                         TX=tmean(df.x_T.to_numpy()[m]), TY=tmean(df.y_T.to_numpy()[m]),
                         TY_tmpl_y=tmean(df.y_T_ty.to_numpy()[m]))
    return out


def heldout_arms(det):
    out = {}
    src = {'t0i': f'{ROOT}/heldout/{det}_t0i.npz', 'lat': f'{ROOT}/ws_t0/heldout/{det}_lat.npz',
           'base': f'{ROOT}/heldout2/{det}_base.npz', 's0x20': f'{ROOT}/heldout2/{det}_s0x20.npz',
           'noprof': f'{ROOT}/heldout2/{det}_noprof.npz', 't0fix': f'{ROOT}/heldout/{det}_t0fix.npz'}
    P = {k: np.load(v)['per'] for k, v in src.items() if os.path.exists(v)}
    common = None
    for k, A in P.items():
        ids = set(A[A[:, 1] == 0, 0].astype(int)) & set(A[A[:, 1] == 1, 0].astype(int))
        common = ids if common is None else common & ids
    common = np.array(sorted(common or []))
    for k, A in P.items():
        X = {int(r[0]): r for r in A[A[:, 1] == 0]}; Y = {int(r[0]): r for r in A[A[:, 1] == 1]}
        RX = np.array([X[i] for i in common]); RY = np.array([Y[i] for i in common])
        out[k] = ratio(RX[:, 3], RX[:, 2], RY[:, 3], RY[:, 2])
    return out


def condor_arms(det):
    t = golden_table(det).set_index('event_id')
    out = {}
    for f in sorted(glob.glob(os.path.join(FA, GOLD_ROW[det], 'wft', 'arm_*', 'events.parquet'))):
        arm = f.split(os.sep)[-2][4:]
        a = pd.read_parquet(f).drop_duplicates('event_id', keep=False).set_index('event_id')
        j = t[['tan_ref_x', 'tan_ref_y', 'ref_mesh_x', 'ref_mesh_y', 'evttime']].join(
            a[['x_ok', 'y_ok', 'x_w', 'y_w', 'x_p0', 'y_p0']], how='inner').reset_index()
        tt, wx, tx, wy, ty = paired(j)
        out[arm] = dict(n=len(tt), **splits(tt, wx, tx, wy, ty))
    return out


def rows():
    """per FA row table (item 3): det3 HV rows, every run of every chamber."""
    out = []
    for f in sorted(glob.glob(os.path.join(FA, '*', '*', 'mx17_*', 'fa', 'table.parquet'))):
        parts = f.split(os.sep)
        t = load_table(f)
        tt, wx, tx, wy, ty = paired(t)
        if len(tt) < 200:
            continue
        out.append(dict(run=parts[-5], subrun=parts[-4], det=parts[-3], n=len(tt), r=ratio(wx, tx, wy, ty, 100)))
    for name in ('det3_d700', 'det3_d500', 'det3_day625'):
        tt, wx, tx, wy, ty = paired(load_table(os.path.join(WS, f'{name}.parquet')))
        out.append(dict(run=name, subrun='', det='mx17_3', n=len(tt), r=ratio(wx, tx, wy, ty, 100)))
    return out


def main():
    amap = json.load(open(os.path.join(ROOT, 'ws_widthtilt', 'angle_maps.json')))
    res = {}
    for det in DETS:
        t = golden_table(det)
        R = dict(golden=splits(*paired(t)), golden_w0map=splits(*paired(t, w0map=amap[det])),
                 colsum=colsum_ratio(det), heldout=heldout_arms(det), condor_arms=condor_arms(det))
        res[det] = R
        g, gm = R['golden'], R['golden_w0map']
        print(f"{det}: X/Y-1 {g['all'][0]:+.2f}±{g['all'][1]:.2f} %  w0-map {gm['all'][0]:+.2f}±{gm['all'][1]:.2f}  "
              f"quadrants {' '.join(f'{a:+.1f}' for a, b, n in g['quadrant'])}  cells chi2 {g.get('cells_chi2')}  "
              f"w0map cells chi2 {gm.get('cells_chi2')}")
        print('   heldout', {k: f'{v[0]:+.2f}±{v[1]:.2f}' for k, v in R['heldout'].items()})
        if R['colsum']:
            print('   colsum', R['colsum'])
        for k, v in R['condor_arms'].items():
            print(f"   arm {k}: {v['all'][0]:+.2f}±{v['all'][1]:.2f} (n {v['n']})")
    res['rows'] = rows()
    for r in res['rows']:
        print(f"   row {r['det']} {r['run']} {r['subrun']}: {r['r'][0]:+.2f}±{r['r'][1]:.2f} (n {r['n']})")
    json.dump(res, open(os.path.join(RES, 'xy_scale.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
