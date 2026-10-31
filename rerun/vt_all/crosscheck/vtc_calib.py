#!/usr/bin/env python3
"""vtc_calib.py --out OUT.json [--vtc DIR] -- what the in-row v(t) of the three golden calibration rows (g_det4,
g_det6_long, g_det7_long) does to what was calibrated on them with a constant v (CROSSCHECK_2026-10-31 §2).
Table level only: nothing is refitted, no calibration is re-run.

Per golden row, for its G1 table (Analysis_pass1/<row>/wft/calib/r3, the CAL4 reconstruction G1 scores) and its pass
table (<row>/wft):
  G1 s68     per view, 03_angles' definition (|dtheta - median| 68th percentile, fitted vs M3, every finite pair):
               const      as calibrated (closure against angular_resolution.json);
               vt         tan = (w - w0) / (kw v r(t) S) - a, w0 and kw as calibrated;
               vt+fold    the same, then w0 and kw re-measured on the corrected mapping (set_w0kw's ladder): what the
                          fold of the calibration would have given with v(t) inside;
             for each r(t) model: 'line' (vt_bench's vt_map.json) and, for det6/det7, 'night' (vt_within_rows' applied
             one-curve-per-chamber model, read per event from vt_correction/.../events_vt.parquet).
  kw / w0    the re-measured kw and w0 against the calibrated ones.
  bow        the bow's scale map S = 1 + su U + sw W is a fit of the angle scale over the plate; a v(t) correlated with
             where the tracks sit would leak into it.  The leak is the least-squares slope of r(t_i) on (U_i, W_i) over
             the kw-window events (r - 1 = du U + dw W + c): compare du, dw with the block's su, sw.
  S4         the head-on stack T is a time average.  For det4, the two S4 samples (the hits-chain selection the pre-pass
             input read, and the ref-only sample S4 used) are placed on the clock (colsum eid -> t); the T ratio a pure
             time-average effect predicts is mean(1/r) over one sample over the other, with r from the ladder line and
             from the column duration (vtc_measure's blocks, piecewise linear).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'rerun', 'vt_all'), REPO]
import vt_bench as vb  # noqa: E402
from vt_common import ladder, kw_window, bow_aS, tan_map  # noqa: E402

P1 = vb.ROOT
VTA = '/eos/user/d/dneff/x17rerun/vt_all/bench'
VTC = '/eos/user/d/dneff/x17rerun/vt_correction'
S4I = '/eos/user/d/dneff/x17rerun/s4_inputs/Analysis'
S4F = '/eos/user/d/dneff/x17rerun/s4_followup/Analysis_prod'
GOLD = {'g_det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'g_det6_long': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'g_det7_long': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}


def s68(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return float(np.percentile(np.abs(d - np.median(d)), 68)) if len(d) else np.nan


def line_r(vm):
    t, r = np.asarray(vm['t_h']), np.asarray(vm['v_rel'])
    return lambda x: np.interp(np.asarray(x, float), t, r)


def row_views(wdir):
    """the row as vt_bench reads it, plus per view the event ids, U, W and the bow a, S for the matched events."""
    m, cal, df, V = vb.load_row(wdir, P1)
    tcx = (df.x_w * 1e3 - cal.w0['x']) / (cal.kw['x'] * cal.v_drift)
    tcy = (df.y_w * 1e3 - cal.w0['y']) / (cal.kw['y'] * cal.v_drift)
    import wft.reco as wr
    U = ((df.x_p0 + wr.BOW_MID_MM * tcx) - 200.0) / 100.0
    W = ((df.y_p0 + wr.BOW_MID_MM * tcy) - 200.0) / 100.0
    U = U.where(np.isfinite(U) & (U.abs() <= wr.BOW_UW_MAX) & df.x_ok, 0.0).fillna(0.0).to_numpy()
    W = W.where(np.isfinite(W) & (W.abs() <= wr.BOW_UW_MAX) & df.y_ok, 0.0).fillna(0.0).to_numpy()
    z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
    for p in ('x', 'y'):
        e = z[f'{p}_eid'].astype(int)
        k = np.isin(e, df.index.to_numpy())
        pos = df.index.get_indexer(e[k])
        a, S = bow_aS(cal, p, U[pos], W[pos])
        V[p].update(eid=e[k], a=a, S=S, U=U[pos], Wc=W[pos], tan_ref=z[f'{p}_tan_ref'][k], tan_fit=z[f'{p}_tan_fit'][k])
    return m, cal, df, V


def score(cal, V, rfun, fold):
    out = {}
    for p in ('x', 'y'):
        v = V[p]
        rr = rfun(v['t_h'], v['eid'])
        c = dict(w0=cal.w0[p], kw=cal.kw[p])
        if fold:
            # set_w0kw on the corrected mapping: w' = w0 + (w - w0) / r is the corrected w on the constant mapping
            wc = cal.w0[p] + (v['w'] - cal.w0[p]) / rr
            w0n, kwn, _, kwe = ladder(cal, wc, v['tp'])        # absolute: w0 = median(w - v t'), kw = (w - w0) / (v t')
            c = dict(w0=w0n, kw=kwn, kw_err=kwe)
        tan = (v['w'] - c['w0']) / (c['kw'] * cal.v_drift * rr * v['S']) - v['a']
        d = np.degrees(np.arctan(tan) - np.arctan(v['tan_ref']))
        out[p] = dict(s68=s68(d), n=int(np.isfinite(d).sum()), **c)
    return out


def bow_leak(cal, V, rfun):
    out = {}
    for p in ('x', 'y'):
        v = V[p]
        k = kw_window(v['tp'])
        r = rfun(v['t_h'][k], v['eid'][k]) - 1.0
        A = np.column_stack([v['U'][k], v['Wc'][k], np.ones(k.sum())])
        c, *_ = np.linalg.lstsq(A, r, rcond=None)
        s = (cal.bow or {}).get(f's_{p}') or {}
        out[p] = dict(du=float(c[0]), dw=float(c[1]), su=s.get('su'), sw=s.get('sw'),
                      U_t_corr=float(np.corrcoef(v['t_h'][k], v['U'][k])[0, 1]),
                      W_t_corr=float(np.corrcoef(v['t_h'][k], v['Wc'][k])[0, 1]))
    return out


def colsum_clock(path):
    z = np.load(path, allow_pickle=True)
    t0 = np.array([(vb.stem_time(t) - vb.T_REF).total_seconds() / 3600 for t in z['acq_tags']])
    ts = np.where(z['x_ts'] >= 0, z['x_ts'], z['y_ts']).astype(float)
    t = t0[z['acq'].astype(int)] + ts * vb.TICK_S / 3600
    k = (ts >= 0)
    o = np.argsort(z['eid'][k])
    E, T = z['eid'][k][o].astype(float), t[k][o]
    # colsum keeps only triggers with a cluster: interpolate in the (global, monotonic) event id (vtc_measure.eid_clock)
    return lambda e: np.where((np.asarray(e) >= E[0]) & (np.asarray(e) <= E[-1]), np.interp(np.asarray(e, float), E, T), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--vtc', default=None, help='vtc_measure outputs (<key>.json)')
    a = ap.parse_args()
    R = {}
    for key, row in GOLD.items():
        det = row.split('/')[-1]
        vm = json.load(open(os.path.join(VTA, f'{key}.vt_map.json')))
        models = {'line': line_r(vm)}
        nf = os.path.join(VTC, row, 'events_vt.parquet')
        if os.path.exists(nf):
            n = pd.read_parquet(nf).set_index('event_id')['v_rel']
            models['night'] = n
        R[key] = dict(row=row, vt_map=vm)
        for tab, wdir in (('g1', os.path.join(P1, row, 'wft', 'calib', 'r3')), ('pass', os.path.join(P1, row, 'wft'))):
            m, cal, df, V = row_views(wdir)
            ref = json.load(open(os.path.join(wdir, 'angles', 'angular_resolution.json')))['planes']
            T = dict(n_events=int(len(df)), closure={p: ref[p]['s68_deg'] for p in 'xy'},
                     const=score(cal, V, lambda t, e: np.ones_like(t), False))
            for mn, mod in models.items():
                if mn == 'night':
                    f = (lambda S: (lambda t, e: S.reindex(e).fillna(1.0).to_numpy()))(mod)
                    T['night_coverage'] = {p: float(np.isin(V[p]['eid'], mod.index).mean()) for p in 'xy'}
                else:
                    f = (lambda g: (lambda t, e: g(t)))(mod)
                T[f'vt_{mn}'] = score(cal, V, f, False)
                T[f'vt_{mn}_fold'] = score(cal, V, f, True)
                T[f'bow_leak_{mn}'] = bow_leak(cal, V, f)
            R[key][tab] = T
            print(key, tab, {k: {p: round(v[p]['s68'], 3) for p in 'xy'} for k, v in T.items()
                             if isinstance(v, dict) and isinstance(v.get('x'), dict) and 's68' in v['x']}, flush=True)
    # S4: the two det4 samples on the clock
    row = GOLD['g_det4']
    clk = colsum_clock(os.path.join(S4I, row, 'fa', 'colsum.npz'))
    H = pickle.load(open(os.path.join(S4I, row, 's4', 'headon.pkl'), 'rb'))
    sel = json.load(open(os.path.join(S4F, row, 's4f', 'selections.json')))
    head = lambda ks: [k for k in ks if k in H and abs(H[k]['tan_x']) < 0.03 and H[k]['x']['W'].max() < 3500]
    samples = {'ref_only': head(list(H)), 'hits_rr1': head([int(k) for k in sel['hits_rr1']])}
    vm = json.load(open(os.path.join(VTA, 'g_det4.vt_map.json')))
    rl = line_r(vm)
    rc = None
    if a.vtc and os.path.exists(os.path.join(a.vtc, 'g_det4.json')):
        B = json.load(open(os.path.join(a.vtc, 'g_det4.json')))['survey']['blocks']
        tb = np.array([b['t'] for b in B]); rb = np.array([b['col']['r'] for b in B])
        rc = lambda t: np.interp(t, tb, rb)
    s4 = {}
    for nm, ks in samples.items():
        t = clk(np.array(ks, float))
        s4[nm] = dict(n=len(ks), n_clock=int(np.isfinite(t).sum()), t_mean=float(np.nanmean(t)),
                      t_q=[float(x) for x in np.nanquantile(t, [0.1, 0.5, 0.9])],
                      inv_r_line=float(np.nanmean(1 / rl(t))),
                      inv_r_col=float(np.nanmean(1 / rc(t))) if rc else None)
    s4['pred_T_ratio_line'] = s4['ref_only']['inv_r_line'] / s4['hits_rr1']['inv_r_line']
    if rc:
        s4['pred_T_ratio_col'] = s4['ref_only']['inv_r_col'] / s4['hits_rr1']['inv_r_col']
    R['s4_det4'] = s4
    print('S4 det4', json.dumps(s4, default=float), flush=True)
    json.dump(R, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
