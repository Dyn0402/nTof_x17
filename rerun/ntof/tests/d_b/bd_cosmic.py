#!/usr/bin/env python3
"""bd_cosmic.py -- run_149 cosmic checks for D and B on the pass-1 v1 fits (FINDINGS.md items 1-3).

    bd_cosmic.py dv        # D's timing v on line muons: box fit to D's stacked strip sum on the centroid-line events
    bd_cosmic.py eff       # B's cosmic efficiency on D-predicted crossings (C on A-predicted crossings as reference)
    bd_cosmic.py bpos      # B positions vs v: the same B tracks fitted at v 42.6 and 30.0; B fit p0 vs B centroid

dv: the events of line_kw.py's strict-partner lines (B, A or C centroid + D fitted p0, miss < 60 mm), D's strip sums
from centroids.py, stacked on the trigger, colsum.GridFit with the pass-1 X template (as v in situ for A and C, where the
A-C line muons were used).  v = gap 29.44 mm / T; bootstrap error.
eff: a D (A) single gated track (D: on the wall-listed triggers only, which are chosen without looking at B) whose line crosses B's (C's) mesh plane inside |x|, |y| < 150 mm; B (C) is "lit" when
its main cluster centroid lies within 40 mm of the crossing in both views.  Accidentals: the same with the partner's
centroids of the event 1, 2, ..., 5 places later; the efficiency is lit minus accidental.
bpos: B fits at the two v on the same events: the p0 shift between them (positions must not depend on v), how often
the winning candidate changes (the plausibility gate is in tan, so in v), the paired-track yield, and the one place v
reaches a position downstream: the bow mesh-height term p0 - tan h (|delta tan| h).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, HERE, os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt')]
import line_kw as LK  # noqa: E402

W = '/media/ucla/work/ntof_pass1'
OUT = f'{W}/d_b/bd_cosmic'


def dv():
    import colsum
    colsum.T0_GRID = np.arange(-300.0, 900.0 + 1e-9, 5.0)
    colsum.T_GRID = np.arange(300.0, 1500.0 + 1e-9, 5.0)
    M = pd.read_parquet(f'{W}/d_b/line_kw/lines_D_v1.parquet')
    M = M[M.miss_mm < 60]
    rng = np.random.default_rng(3)
    t = np.load(f'{W}/colsum/tmpl/tmpl_D.npz')
    g = colsum.GridFit(colsum.Kern(t['grid'], t['tmpl_x'] / t['tmpl_x'].max()), 20)
    rows, stacks = [], {}
    for part in ('all', 'B', 'A', 'C'):
        m = M if part == 'all' else M[M.partner == part]
        ev = m[['subrun', 'event_id']].drop_duplicates()
        for p in 'xy':
            S = []
            for sub, gg in ev.groupby('subrun'):
                z = np.load(f'{W}/d_b/centroids/r149_{sub}.npz')
                idx = {int(e): i for i, e in enumerate(z['D_eid'])}
                ii = [idx[e] for e in gg.event_id if e in idx and z[f'D_{p}_has'][idx[e]] and not z[f'D_{p}_sat'][idx[e]]]
                S.append(z[f'D_{p}_S'][ii].astype(float))
            if not S or not sum(len(x) for x in S):
                continue
            S = np.concatenate(S)
            S /= np.maximum(S.max(1, keepdims=True), 1)
            if len(S) < 20:
                continue
            med = np.median(S, 0)
            r = g.fit(med)
            bT = [g.fit(np.median(S[rng.integers(0, len(S), len(S))], 0))[1] for _ in range(60)]
            eT = float(np.std(bT))
            rows.append(dict(partner=part, view=p, n=len(S), t0=r[0], T=r[1], T_err=eT, v=LK.GAP['D'] * 1e3 / r[1],
                             v_err=LK.GAP['D'] * 1e3 / r[1] ** 2 * eT, end_ns=r[0] + r[1]))
            stacks[f'{part}_{p}'] = med.tolist()
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/dv_lines.csv', index=False)
    json.dump(stacks, open(f'{OUT}/dv_stacks.json', 'w'))
    print(R.round(2).to_string(index=False))


def _crossing(T, tr_to):
    """global lines of T -> local (x, y) at the mesh plane (local depth 0) of the chamber tr_to."""
    P = T[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    d = T[['d_x', 'd_y', 'd_z']].to_numpy(float)
    n = tr_to.R[:, 2]
    c = tr_to.center
    with np.errstate(divide='ignore', invalid='ignore'):
        s = ((c - P) @ n) / (d @ n)
    X = P + s[:, None] * d
    loc = (X - c) @ tr_to.R
    return loc[:, 0], loc[:, 1]


def _eff_one(T, cen_by_sub, parm, tr, win=40.0, half=150.0, shifts=(1, 2, 3, 4, 5)):
    from sept26_prelim_analysis import build_tracks as BT
    xl, yl = _crossing(T, tr)
    T = T.assign(xc=xl, yc=yl)
    T = T[(np.abs(T.xc) < half) & (np.abs(T.yc) < half)]
    lit, acc, n = 0, [], 0
    for sub, g in T.groupby('subrun'):
        C = cen_by_sub[sub]
        cx = BT.IN_PLANE_SIGN * (C.cx.to_numpy(float) - BT.STRIP_MAP_HALF)
        cy = BT.IN_PLANE_SIGN_Y * (C.cy.to_numpy(float) - BT.STRIP_MAP_HALF)
        pos = dict(zip(C.event_id.to_numpy(), zip(cx, cy)))
        eids = np.array(sorted(pos))
        for e, x, y in zip(g.event_id, g.xc, g.yc):
            n += 1
            if e in pos and abs(pos[e][0] - x) < win and abs(pos[e][1] - y) < win:
                lit += 1
        # accidentals: the partner's clusters of later triggers
        for s in shifts:
            k = 0
            for e, x, y in zip(g.event_id, g.xc, g.yc):
                j = np.searchsorted(eids, e) + s
                if j < len(eids):
                    p = pos[eids[j]]
                    k += abs(p[0] - x) < win and abs(p[1] - y) < win
            acc.append(k)
    a = float(np.mean(acc)) if acc else 0.0
    eff = (lit - a) / max(n, 1)
    return dict(partner=parm, n_pred=n, n_lit=lit, n_acc=a, eff=eff, eff_err=float(np.sqrt(max(lit, 1)) / max(n, 1)))


def eff():
    trs = LK.transforms()
    rows = []
    files = sorted(glob.glob(f'{W}/d_b/reco/r149_cosbounce_cos_*_D_v1_[a-z].parquet'))
    T = []
    for f in files:
        sub = 'cosbounce_' + os.path.basename(f).split('_cosbounce_')[1].split('_D_')[0]
        T.append(LK.fitted(f, 'D', trs['mx17_D']).assign(subrun=sub))
    T = pd.concat(T, ignore_index=True)
    # only D tracks whose event was listed for the WALL (D cluster + one D wall group on a clock-matched trigger):
    # the line candidates were listed because A or B had a cluster, which would bias B's efficiency up
    import events as EV
    keep = []
    for s in T.subrun.unique():
        _, wall = EV.candidates(s)
        keep.append(T[(T.subrun == s) & T.event_id.isin(wall)])
    T = pd.concat(keep, ignore_index=True)
    for parm in 'BC':
        cen = {s: LK.centroids(s, parm, strict=False) for s in T.subrun.unique()}
        rows.append(_eff_one(T, cen, parm, trs[f'mx17_{parm}']) | dict(tracks='D (v1_D) on wall-listed triggers, run_149'))
    # reference: C on A-predicted crossings, A from the held-out v1k fits of cos_0001
    TA = LK.fitted(f'{W}/heldout/r149s1_A_v1k.parquet', 'A', trs['mx17_A']).assign(subrun='cosbounce_cos_0001')
    for parm in 'CB':
        cen = {'cosbounce_cos_0001': LK.centroids('cosbounce_cos_0001', parm, strict=False)}
        rows.append(_eff_one(TA, cen, parm, trs[f'mx17_{parm}']) | dict(tracks='A (v1k_A), run_149 cos_0001'))
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/eff.csv', index=False)
    print(R.round(3).to_string(index=False))


def bpos():
    from sept26_prelim_analysis import build_tracks as BT
    rows = []
    for samp, tag in (('run_149 cos_0000', 'r149_c0'), ('run_145 stat090_0000 _000', 'r145_s0')):
        a = pd.read_parquet(f'{W}/d_b/reco/{tag}_v1_B.parquet')
        b = pd.read_parquet(f'{W}/d_b/reco/{tag}_v30_B.parquet')
        j = a.merge(b, on='event_id', suffixes=('_42', '_30'))
        for p in 'xy':
            ok = j[f'{p}_ok_42'].astype(bool) & j[f'{p}_ok_30'].astype(bool)
            same = ok & (np.abs(j[f'{p}_p0_42'] - j[f'{p}_p0_30']) < 20)
            dp = (j[f'{p}_p0_42'] - j[f'{p}_p0_30'])[same]
            dt = (j[f'{p}_tan_theta_42'] / j[f'{p}_tan_theta_30'])[same & (np.abs(j[f'{p}_tan_theta_30']) > 0.1)]
            bow = ((j[f'{p}_tan_theta_42'] - j[f'{p}_tan_theta_30']) * j[f'{p}_bow_h_42'])[same].abs()
            rows.append(dict(sample=samp, view=p, n_events=len(j), n_ok_42=int(j[f'{p}_ok_42'].sum()),
                             n_paired_42=int((a.n_tracks > 0).sum()), n_paired_30=int((b.n_tracks > 0).sum()),
                             bow_term_p50=float(bow.median()), bow_term_p95=float(np.percentile(bow, 95)),
                             n_ok_30=int(j[f'{p}_ok_30'].sum()), n_same_track=int(same.sum()),
                             frac_other_winner=float(1 - same.sum() / max(ok.sum(), 1)),
                             dp0_med=float(dp.median()), dp0_68=float(np.percentile(np.abs(dp), 68)),
                             dp0_95=float(np.percentile(np.abs(dp), 95)), tan_ratio_42_over_30=float(dt.median()),
                             q_ok_42=int(j[f'{p}_quality_ok_42'].sum()), q_ok_30=int(j[f'{p}_quality_ok_30'].sum())))
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/bpos.csv', index=False)
    print(R.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['dv', 'eff', 'bpos'])
    a = ap.parse_args()
    dict(dv=dv, eff=eff, bpos=bpos)[a.step]()


if __name__ == '__main__':
    main()
