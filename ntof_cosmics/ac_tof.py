#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ac_tof.py -- the arm-to-arm time resolution sigma_t of the n_TOF scintillators, measured on cosmic muons that cross
arms A and C, with the muon's own time of flight predicted from the Micromegas line.

Why: the LNL two-arm proposal assumes sigma_t = 0.3 ns for the cosmic TOF veto (x17_facility_search
lnl/FEASIBILITY_SIM.md §4, "(ask)").  It has never been measured.  A first look on beam-off run_149 (lnl_x17
sim2arm/ntof_timing_firstlook/) gave an upper bound of ~1.9 ns per arm, with the muons' own TOF still in it.

Sample: beam runs, 20-80 ms after the flash (as inbeam_through_goers.py), exactly one gated track in A and in C,
the two lines within SEP_MAX mm of each other, the joined line > JDCA_MIN mm from the beam axis (no capsule pairs).
Timing: the scintillator stack tables (ntof_scint_stack/extract.py: every channel's largest hit in (-100, +60) ns of
the DREAM trigger, n_TOF `tof` rounded to 1 ns per end).  Per arm, the SiPM wall group with both ends firing and the
largest geometric-mean amplitude; time = mean of the two ends (cancels the light path along the vertical bars).

Geometry (global frame, y up = the beam, COORDINATES.md; A strip plane z = +234.6, C z = -234.6; walls 97.4 mm
behind the strips, |z| = 332.0, ntof_scint_stack geometry.csv): the A-C line from the two strip-plane points gives the
wall crossings W_A, W_C, the path L between them, and the predicted

    dt_pred = t_A - t_C = sign(y(W_C) - y(W_A)) * L / c      (cosmic muons go DOWN: the higher wall fires first)

The pinned external fact is "muons go down"; the slope of measured dt against dt_pred is printed as the check (1 if
the sign, the frame and the speed are right).

Corrections, in order: (1) one offset per wall group per arm (7 free: A's group 0 fixed), (2) a linear time walk in
log(amplitude) per arm.  Fitted on even events, scored on odd ones (and the other way round), so the quoted width is
not tuned to its own sample.  Width estimators: s68 (half the 16-84 % range) and a Gaussian core fit (iterated 2-sigma).
Per arm: width / sqrt(2) (arms assumed equal).  The 1 ns rounding of each end contributes 1/sqrt(12)/sqrt(2) = 0.20 ns
per arm (mean of two ends); "detector" subtracts it in quadrature.

    .venv/bin/python ntof_cosmics/ac_tof.py build      # pairs + wall/plastic times -> OUT/ac_tof_events.parquet
    .venv/bin/python ntof_cosmics/ac_tof.py ana        # widths, corrections, figures -> OUT/ac_tof_summary.csv
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ntof_cosmics.cosmic_tracks import _axis_dca  # noqa: E402
from sept26_prelim_analysis.source_imaging import _dca_two_lines  # noqa: E402

CAMPAIGN = Path('/media/dylan/data/x17/sept26_prelim/stage3_fullpass/tracks_campaign.parquet')
STACK = Path('/media/dylan/data/x17/scint_stack/tracks')
OUT = Path(__import__('os').environ.get('WORK_DATA', '/media/dylan/data/x17')) / 'nTof_x17' / 'ac_tof'
COLS = ['run', 'subrun', 'event_id', 'arm', 'gated', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 't_since_flash_ns']
C_MM_NS = 299.792458
Z_STRIP = 234.6
L_WALL = 97.4                       # strips -> wall (geometry.csv, all runs)
L_PLAS = 189.6                      # strips -> plastic, mean of A (190.6) and C (188.6) in geometry.csv
SEP_MAX = 20.0
JDCA_MIN = 60.0
MS = (20.0, 80.0)
SIG_RND = 1 / np.sqrt(12) / np.sqrt(2)      # 1 ns rounding, mean of two ends, per arm [ns]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def pairs() -> pd.DataFrame:
    t = pq.read_table(CAMPAIGN, columns=COLS, filters=[('arm', 'in', ['A', 'C']), ('gated', '==', True)]).to_pandas()
    t['ms'] = t.t_since_flash_ns / 1e6
    t = t[(t.ms >= 10.0)]
    k = ['run', 'subrun', 'event_id']
    n = t.groupby(k + ['arm'], observed=True).size().unstack(fill_value=0)
    ok = n[(n.get('A', 0) == 1) & (n.get('C', 0) == 1)].index
    t = t.set_index(k)
    t = t[t.index.isin(ok)].reset_index()
    a = t[t.arm == 'A'].sort_values(k).reset_index(drop=True)
    c = t[t.arm == 'C'].sort_values(k).reset_index(drop=True)
    P1, D1 = a[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), a[['d_x', 'd_y', 'd_z']].to_numpy(float)
    P2, D2 = c[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), c[['d_x', 'd_y', 'd_z']].to_numpy(float)
    _, sep = _dca_two_lines(P1, D1, P2, D2)
    J = P2 - P1
    Ju = J / np.linalg.norm(J, axis=1)[:, None]
    out = a[k + ['ms']].copy()
    out['run'] = out.run.astype(str)
    out['subrun'] = out.subrun.astype(str)
    out['sep'] = sep
    out['jdca'] = _axis_dca(P1, Ju)[0]
    out['vert_deg'] = np.degrees(np.arccos(np.abs(Ju[:, 1])))

    def at_z(z):        # the A-C line at plane z
        s = (z - P1[:, 2]) / J[:, 2]
        return P1 + s[:, None] * J
    for nm, zz in (('wA', Z_STRIP + L_WALL), ('wC', -(Z_STRIP + L_WALL)),
                   ('pA', Z_STRIP + L_PLAS), ('pC', -(Z_STRIP + L_PLAS))):
        X = at_z(zz)
        out[f'{nm}_x'], out[f'{nm}_y'] = X[:, 0], X[:, 1]
    for lay in ('w', 'p'):
        dX = np.c_[out[f'{lay}C_x'] - out[f'{lay}A_x'], out[f'{lay}C_y'] - out[f'{lay}A_y']]
        dz = 2 * (Z_STRIP + (L_WALL if lay == 'w' else L_PLAS))
        L = np.sqrt(dX[:, 0] ** 2 + dX[:, 1] ** 2 + dz ** 2)
        out[f'L_{lay}'] = L
        out[f'pred_{lay}'] = np.sign(dX[:, 1]) * L / C_MM_NS
    return out


SCOLS = ['subrun', 'event_id', 'arm', 'run'] + [f'w{i}_{q}_on' for i in range(1, 9) for q in ('amp', 'dt', 'sat')] \
    + [f'p{i}_{q}_on' for i in (1, 2) for q in ('amp', 'dt', 'sat')]


def stack() -> pd.DataFrame:
    fr = []
    for f in sorted(STACK.glob('stack_run_*.parquet')):
        s = pq.read_table(f, columns=SCOLS, filters=[('arm', 'in', ['A', 'C'])]).to_pandas()
        s['subrun'] = s.subrun.astype(str)
        s['run'] = s.run.astype(str)
        fr.append(s.drop_duplicates(['subrun', 'event_id', 'arm']))
    return pd.concat(fr, ignore_index=True)


def wall_time(s: pd.DataFrame) -> pd.DataFrame:
    """per (event, arm): the wall group with both ends in the window and the largest sqrt(a1 a2)."""
    best = np.full(len(s), -1)
    bamp = np.zeros(len(s))
    ngrp = np.zeros(len(s), int)
    for g in range(4):
        a1, a2 = s[f'w{2 * g + 1}_amp_on'].to_numpy(float), s[f'w{2 * g + 2}_amp_on'].to_numpy(float)
        t1, t2 = s[f'w{2 * g + 1}_dt_on'].to_numpy(float), s[f'w{2 * g + 2}_dt_on'].to_numpy(float)
        ok = np.isfinite(t1) & np.isfinite(t2) & (a1 > 0) & (a2 > 0) & (np.abs(t1 - t2) < 15)
        gm = np.where(ok, np.sqrt(np.clip(a1, 0, None) * np.clip(a2, 0, None)), 0.0)
        ngrp += ok
        up = gm > bamp
        best, bamp = np.where(up, g, best), np.where(up, gm, bamp)
    o = s[['run', 'subrun', 'event_id', 'arm']].copy()
    o['wg'], o['wamp'], o['wngrp'] = best, bamp, ngrp
    tm, td, sat = (np.full(len(s), np.nan) for _ in range(3))
    for g in range(4):
        m = best == g
        t1, t2 = s[f'w{2 * g + 1}_dt_on'].to_numpy(float), s[f'w{2 * g + 2}_dt_on'].to_numpy(float)
        tm[m], td[m] = 0.5 * (t1 + t2)[m], (t1 - t2)[m]
        sat[m] = (s[f'w{2 * g + 1}_sat_on'].fillna(0).to_numpy(float) + s[f'w{2 * g + 2}_sat_on'].fillna(0).to_numpy(float))[m]
    o['wt'], o['wtd'], o['wsat'] = tm, td, sat
    pa = s[['p1_amp_on', 'p2_amp_on']].fillna(0).to_numpy(float)
    pt = s[['p1_dt_on', 'p2_dt_on']].to_numpy(float)
    pb = np.where(pa[:, 1] > pa[:, 0], 1, 0)
    o['pb'] = np.where(pa.max(1) > 0, pb, -1)
    o['pamp'] = pa[np.arange(len(s)), pb]
    o['pt'] = pt[np.arange(len(s)), pb]
    o['psat'] = s[['p1_sat_on', 'p2_sat_on']].fillna(0).to_numpy(float)[np.arange(len(s)), pb]
    return o


def build() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pairs()
    print(f'A-C single-track events at >= 10 ms: {len(P)}', flush=True)
    W = wall_time(stack())
    k = ['run', 'subrun', 'event_id']
    A = W[W.arm == 'A'].drop(columns='arm').add_suffix('_A').rename(columns={f'{c}_A': c for c in k})
    C = W[W.arm == 'C'].drop(columns='arm').add_suffix('_C').rename(columns={f'{c}_C': c for c in k})
    E = P.merge(A, on=k, how='inner').merge(C, on=k, how='inner')
    print(f'joined with the stack tables: {len(E)}', flush=True)
    E.to_parquet(OUT / 'ac_tof_events.parquet', index=False)
    return 0


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def s68(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    q = np.percentile(x, [15.865, 84.135])
    return 0.5 * (q[1] - q[0])


def core(x, k=2.0, it=10):
    x = x[np.isfinite(x)]
    mu, sd = np.median(x), s68(x)
    for _ in range(it):
        m = np.abs(x - mu) < k * sd
        mu, sd0 = x[m].mean(), x[m].std()
        # undo the truncation at ±k sigma of a Gaussian
        from scipy.stats import norm
        f = 1 - 2 * k * norm.pdf(k) / (2 * norm.cdf(k) - 1)
        sd = sd0 / np.sqrt(f)
    return float(sd)


def fit_corr(d: pd.DataFrame, r: np.ndarray, walk=True):
    """least squares: r = o_A[gA] - o_C[gC] + wA log aA + wC log aC (+ const); returns a predictor."""
    gA, gC = d.wg_A.to_numpy(int), d.wg_C.to_numpy(int)
    cols = [np.ones(len(d))]
    for g in range(1, 4):
        cols.append((gA == g).astype(float))
    for g in range(1, 4):
        cols.append(-(gC == g).astype(float))
    if walk:
        cols += [np.log(d.wamp_A.to_numpy(float)), np.log(d.wamp_C.to_numpy(float))]
    X = np.c_[tuple(cols)]
    ok = np.isfinite(r)
    for _ in range(3):         # iterate with a 3 s68 clip: the tails are not Gaussian
        beta, *_ = np.linalg.lstsq(X[ok], r[ok], rcond=None)
        res = r - X @ beta
        ok = np.isfinite(r) & (np.abs(res - np.nanmedian(res)) < 3 * s68(res))

    def pred(dd):
        gA, gC = dd.wg_A.to_numpy(int), dd.wg_C.to_numpy(int)
        cc = [np.ones(len(dd))] + [(gA == g).astype(float) for g in range(1, 4)] + [-(gC == g).astype(float) for g in range(1, 4)]
        if walk:
            cc += [np.log(dd.wamp_A.to_numpy(float)), np.log(dd.wamp_C.to_numpy(float))]
        return np.c_[tuple(cc)] @ beta
    return pred, beta


def crossval(d, r, walk=True):
    ev = (d.event_id.to_numpy() % 2) == 0
    out = np.full(len(d), np.nan)
    for tr in (ev, ~ev):
        pr, _ = fit_corr(d[tr], r[tr], walk)
        out[~tr] = r[~tr] - pr(d[~tr])
    return out


def ana() -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    E = pd.read_parquet(OUT / 'ac_tof_events.parquet')
    base = (E.sep < SEP_MAX) & (E.jdca > JDCA_MIN) & (E.ms >= MS[0]) & (E.ms < MS[1])
    wall = base & (E.wg_A >= 0) & (E.wg_C >= 0) & (E.wsat_A == 0) & (E.wsat_C == 0)
    d = E[wall].reset_index(drop=True)
    dt = (d.wt_A - d.wt_C).to_numpy()
    pred = d.pred_w.to_numpy()
    rows = []

    def row(name, x, n=None, note=''):
        s, c = s68(x), core(x)
        rows.append(dict(sample=name, n=int(np.isfinite(x).sum()) if n is None else n, s68_ns=s, core_ns=c,
                         per_arm_s68=s / np.sqrt(2), per_arm_core=c / np.sqrt(2),
                         per_arm_core_detector=np.sqrt(max((c / np.sqrt(2)) ** 2 - SIG_RND ** 2, 0)), note=note))
        print(f'{name:58s} n {rows[-1]["n"]:5d}  s68 {s:5.2f}  core {c:5.2f} ns  per arm core {c / np.sqrt(2):4.2f}', flush=True)

    print(f'events: base {int(base.sum())}, with a wall group both ends in A and C (no saturation) {len(d)}')
    # check: the slope of dt against the predicted muon TOF
    m = np.isfinite(dt)
    sl = np.polyfit(pred[m], dt[m], 1)
    rr = np.corrcoef(pred[m], dt[m])[0, 1]
    print(f'check: dt = {sl[0]:.3f} * pred + {sl[1]:.3f} ns (corr {rr:.3f}); |pred| median {np.median(np.abs(pred)):.2f} ns')
    row('raw dt (A - C)', dt)
    row('dt - predicted muon TOF', dt - pred)
    r_off = crossval(d, dt - pred, walk=False)
    row('- TOF, group offsets (cross-validated)', r_off)
    r_all = crossval(d, dt - pred, walk=True)
    row('- TOF, group offsets + time walk (cross-validated)', r_all)
    pr_full, beta = fit_corr(d, dt - pred, walk=True)
    r_in = dt - pred - pr_full(d)
    row('same, fitted on the whole sample (not cross-validated)', r_in)
    # subsets of the cross-validated residual
    big = (d.wamp_A > np.median(d.wamp_A)) & (d.wamp_C > np.median(d.wamp_C))
    row('  both amplitudes above median', r_all[big.to_numpy()])
    one = (d.wngrp_A == 1) & (d.wngrp_C == 1)
    row('  exactly one wall group fired per arm', r_all[one.to_numpy()])
    tight = (d.sep < 10).to_numpy()
    row('  sep < 10 mm', r_all[tight])
    # same without the TOF prediction (what the first look measured), offsets + walk
    row('no TOF prediction, offsets + walk (cross-validated)', crossval(d, dt, walk=True))
    # a looser sample: >= 10 ms
    b2 = (E.sep < SEP_MAX) & (E.jdca > JDCA_MIN) & (E.ms >= 10) & (E.wg_A >= 0) & (E.wg_C >= 0) & (E.wsat_A == 0) & (E.wsat_C == 0)
    d2 = E[b2].reset_index(drop=True)
    row('>= 10 ms, - TOF, offsets + walk (cross-validated)', crossval(d2, (d2.wt_A - d2.wt_C - d2.pred_w).to_numpy()))
    # plastics: one PMT per bar, so the light path along the bar is NOT cancelled
    pl = base & (E.pb_A >= 0) & (E.pb_C >= 0) & (E.pamp_A > 300) & (E.pamp_C > 300) & (E.psat_A == 0) & (E.psat_C == 0)
    q = E[pl].reset_index(drop=True)
    pdt = (q.pt_A - q.pt_C - q.pred_p).to_numpy()
    row('plastic: dt - TOF (amp > 300)', pdt)
    # plastics: offsets per bar pair + linear in the crossing position (x, y) on each bar, cross-validated
    X = np.c_[np.ones(len(q)), q.pb_A, q.pb_C, q.pA_x, q.pA_y, q.pC_x, q.pC_y, np.log(q.pamp_A), np.log(q.pamp_C)]
    ev = (q.event_id.to_numpy() % 2) == 0
    rp = np.full(len(q), np.nan)
    for tr in (ev, ~ev):
        ok = tr & np.isfinite(pdt)
        for _ in range(3):
            b, *_ = np.linalg.lstsq(X[ok], pdt[ok], rcond=None)
            res = pdt - X @ b
            ok = tr & np.isfinite(pdt) & (np.abs(res - np.nanmedian(res[tr])) < 3 * s68(res[tr]))
        rp[~tr] = (pdt - X @ b)[~tr]
    row('plastic: - TOF, bar offsets + position + walk (cross-validated)', rp)
    # wall + plastic combined per arm (mean), on events with both
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'ac_tof_summary.csv', index=False)
    meta = dict(slope=sl[0], intercept=sl[1], corr=rr, n_wall=len(d), n_plastic=len(q), beta=list(map(float, beta)),
                sep_max=SEP_MAX, jdca_min=JDCA_MIN, ms=MS, sig_rounding_per_arm=SIG_RND)
    (OUT / 'ac_tof_meta.json').write_text(json.dumps(meta, indent=1))
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist2d(pred[m], dt[m], bins=[40, 60], range=[[-3.5, 3.5], [-12, 12]], cmap='Blues')
    xx = np.linspace(-3.5, 3.5, 2)
    ax[0].plot(xx, xx + np.median(dt - pred), 'r--', lw=1, label='slope 1')
    ax[0].plot(xx, np.polyval(sl, xx), 'k-', lw=1, label=f'fit slope {sl[0]:.2f}')
    ax[0].set(xlabel='predicted muon TOF  t_A − t_C  [ns]', ylabel='measured t_A − t_C (walls) [ns]', title='walls: the muon TOF is visible')
    ax[0].legend(fontsize=8)
    bins = np.arange(-10, 10.01, 0.5)
    for x, lab in ((dt - np.median(dt), 'raw'), (r_all - np.nanmedian(r_all), '− TOF, offsets, walk')):
        ax[1].hist(x, bins, histtype='step', label=f'{lab}: core {core(x):.2f} ns')
    ax[1].set(xlabel='t_A − t_C residual [ns]', ylabel='events / 0.5 ns', title=f'SiPM walls, {len(d)} A–C muons')
    ax[1].legend(fontsize=8)
    ax[2].hist(rp - np.nanmedian(rp), bins, histtype='step', color='C2', label=f'plastics: core {core(rp):.2f} ns')
    ax[2].set(xlabel='t_A − t_C residual [ns]', title=f'plastics, {len(q)} A–C muons')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    (OUT / 'figures').mkdir(exist_ok=True)
    fig.savefig(OUT / 'figures' / 'ac_tof.png', dpi=120)
    print('->', OUT / 'ac_tof_summary.csv', OUT / 'figures' / 'ac_tof.png')
    return 0


if __name__ == '__main__':
    raise SystemExit({'build': build, 'ana': ana}[sys.argv[1] if len(sys.argv) > 1 else 'ana']())
