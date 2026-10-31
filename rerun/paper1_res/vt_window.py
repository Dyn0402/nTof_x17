#!/usr/bin/env python3
"""vt_window.py --out OUT.json -- does an in-row v(t) truncate the reconstruction's v-scaled windows?
(CROSSCHECK_2026-10-31 open risk; PASS1_RESOLUTION_2026-10-31.md §3).  Table level, READ-ONLY on Analysis_pass1.

The only v-scaled window of the bench reconstruction is the plausible column end, wft.reco.u_window:
[U_MIN_FRAC, U_MAX_FRAC] x T with T = 30 mm / v_bundle (it ranks the candidates and gates 'plausible').  A row whose
true v is r(t) v_bundle has its columns ending at u ~ T / r.  Two table-level tests per row of vt_corrected_rows.csv,
per view, in four equal-count blocks of r(t):
  edge     the fraction of fitted column ends (q_uend, ok fits) within 10 % of either window edge, and beyond it, per
           block: if the window truncated, the upper-edge fraction would rise in the low-r (slow) blocks and the
           column-end distribution would pile against U_MAX_FRAC T;
  detect   the fraction of interior_good M3 rays (effbudget rays) with the view's fit ok, per block (the ray's time is
           interpolated in the event id from the table, monotonic in time): truncation would lose fits where r is low.
A line of each against r over the blocks (slope per unit r, with its error) is the verdict.  The window is not a hard
gate on a single track (it ranks candidates, guards the two-track split and sets the isochronous flag), so fits with a
column end outside it stay in the table: the direct truncation signature is the outside fraction rising where r is
low.  The detection fraction also follows gain and occupancy over the row, so its slope alone does not separate them.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('RES_CODE') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO]
from wft.reco import U_MIN_FRAC, U_MAX_FRAC  # noqa: E402
from common.mx17_fiducial import fiducial_mask  # noqa: E402

P1 = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
EFF = '/eos/user/d/dneff/x17rerun/paper1_eff'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
NB = 4
EDGE = 0.10


def line(x, y, e):
    k = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    if k.sum() < 3:
        return dict(slope=np.nan, err=np.nan)
    w = 1 / e[k] ** 2
    A = np.column_stack([np.ones(k.sum()), x[k] - np.average(x[k], weights=w)])
    Aw = A * np.sqrt(w)[:, None]
    c, *_ = np.linalg.lstsq(Aw, y[k] * np.sqrt(w), rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    return dict(slope=float(c[1]), err=float(np.sqrt(cov[1, 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--vt-rows', default=os.path.join(REPO, 'rerun', 'vt_all', 'results', 'vt_corrected_rows.csv'))
    ap.add_argument('--registry', default=os.path.join(REPO, 'rerun', 'pass_registry.csv'))
    a = ap.parse_args()
    reg = {r['key']: r for r in csv.DictReader(open(a.registry))}
    out = {}
    for vr in csv.DictReader(open(a.vt_rows)):
        r = reg[vr['key']]
        rel = f"{r['run']}/{r['subrun']}/{r['det']}"
        wdir = os.path.join(P1, rel, 'wft')
        m = json.load(open(os.path.join(wdir, 'events.meta.json')))
        v = float(m['bundle']['v_drift'])
        T = 30000.0 / v
        lo, hi = U_MIN_FRAC * T, U_MAX_FRAC * T
        df = pd.read_parquet(os.path.join(wdir, 'events.parquet')).drop_duplicates('event_id', keep=False)
        vt = pd.read_parquet(vr['table']).drop_duplicates('event_id').set_index('event_id')['v_rel']
        df = df.set_index('event_id')
        df['r'] = vt.reindex(df.index).to_numpy()
        df = df[np.isfinite(df.r)].sort_index()
        if len(df) < 400:
            continue
        q = np.quantile(df.r, np.linspace(0, 1, NB + 1)); q[0] -= 1; q[-1] += 1
        df['blk'] = np.digitize(df.r, q[1:-1])
        rays = pd.read_parquet(os.path.join(EFF, 'budget', rel, f"rays_{SHORT[r['det']]}.parquet"))
        ig = fiducial_mask('interior_good', rays.mesh_x.to_numpy(), rays.mesh_y.to_numpy(), rays.tan_x.to_numpy(),
                           rays.tan_y.to_numpy(), r['det'])
        R = rays[ig & (rays.event_id >= df.index.min()) & (rays.event_id <= df.index.max())].copy()
        # the ray's r: interpolated in the (monotonic) event id
        R['r'] = np.interp(R.event_id.to_numpy(float), df.index.to_numpy(float), df.r.to_numpy())
        R['blk'] = np.digitize(R.r, q[1:-1])
        row = dict(key=vr['key'], det=SHORT[r['det']], v_bundle=v, T_ns=T, window_ns=[lo, hi], r_min=float(df.r.min()),
                   r_max=float(df.r.max()), views={})
        for p in 'xy':
            ok = df[f'{p}_ok'].astype(bool) & np.isfinite(df[f'{p}_q_uend'])
            B = []
            for b in range(NB):
                s = ok & (df.blk == b)
                u = df.loc[s, f'{p}_q_uend'].to_numpy()
                rb = df.loc[df.blk == b, 'r']
                Rb = R[R.blk == b]
                n = len(u)
                hi_f = float(np.mean(u > hi * (1 - EDGE))) if n else np.nan
                lo_f = float(np.mean(u < lo * (1 + EDGE))) if n else np.nan
                out_f = float(np.mean((u > hi) | (u < lo))) if n else np.nan
                det = float(Rb[f'{p}_ok'].astype(bool).mean()) if len(Rb) else np.nan
                B.append(dict(r=float(rb.mean()), n=n, u_med_over_T=float(np.median(u) / T) if n else np.nan,
                              u_p99_over_T=float(np.percentile(u, 99) / T) if n else np.nan,
                              hi_edge=hi_f, lo_edge=lo_f, outside=out_f, n_rays=int(len(Rb)), detect=det,
                              hi_edge_err=float(np.sqrt(max(hi_f * (1 - hi_f), 1 / n) / n)) if n else np.nan,
                              lo_edge_err=float(np.sqrt(max(lo_f * (1 - lo_f), 1 / n) / n)) if n else np.nan,
                              outside_err=float(np.sqrt(max(out_f * (1 - out_f), 1 / n) / n)) if n else np.nan,
                              detect_err=float(np.sqrt(max(det * (1 - det), 1 / len(Rb)) / len(Rb))) if len(Rb) else np.nan))
            rr = np.array([b['r'] for b in B])
            row['views'][p] = dict(blocks=B,
                                   hi_edge_vs_r=line(rr, np.array([b['hi_edge'] for b in B]),
                                                     np.array([b['hi_edge_err'] for b in B])),
                                   lo_edge_vs_r=line(rr, np.array([b['lo_edge'] for b in B]),
                                                     np.array([b['lo_edge_err'] for b in B])),
                                   detect_vs_r=line(rr, np.array([b['detect'] for b in B]),
                                                    np.array([b['detect_err'] for b in B])),
                                   u_med_vs_r=line(rr, np.array([b['u_med_over_T'] for b in B]),
                                                   np.full(NB, 0.005)),
                                   outside_vs_r=line(rr, np.array([b['outside'] for b in B]),
                                                     np.array([b['outside_err'] for b in B])))
        out[vr['key']] = row
        print(vr['key'], {p: (round(row['views'][p]['blocks'][0]['u_p99_over_T'], 3),
                              round(row['views'][p]['detect_vs_r']['slope'], 3),
                              round(row['views'][p]['detect_vs_r']['err'], 3)) for p in 'xy'}, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
