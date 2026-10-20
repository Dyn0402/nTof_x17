#!/usr/bin/env python3
"""
trig_analysis.py -- the hardware wall trigger against time since the flash, from
the emulated candidates of trigger_cand.py (condor -> flash_tail_trigger).

    python -m ntof_scint_recovery.trig_analysis [--dir DIR]

* matched: a DREAM trigger with t_pred - t in MATCH_NS (the match peak sits at
  -75..+25 ns);
* live: DREAM's gate is open (t >= GATE_US, the earliest trigger of any bunch is
  at ~993 us) and no trigger in the previous DEAD_US (DREAM's dead time varies up
  to ~600 us with the event: efficiency is ~0 within 205 us, ~0.3 at 250-600 us,
  ~1 beyond);
* the plastic well above its threshold (> 1.2x), so the wall decides.

Turn-on: efficiency against wall_sum / threshold in time bins; its 50 % point
from a logistic fit.  If the analog trigger sum shares the digitized wall
undershoot, the turn-on moves right early by (2 x the per-end baseline shift at
that time) / threshold.  The expected shift is computed from quick.py's per-wall
median baseline curves and printed next to the measured one.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.quick import PRODUCTION_FROM

WALL_THR = {0: 26.0, 1: 35.0, 2: 38.0, 3: 37.0}
PLAS_THR = {0: 112.0, 1: 136.0, 2: 151.0, 3: 130.0}
MATCH_NS = (-100.0, 40.0)
GATE_US = 995.0
DEAD_US = 600.0
T_BINS = ((995, 1100), (1100, 1300), (1300, 2000), (2000, 4000), (4000, 10000))
X_EDGES = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 3.0, 10.0])


def logistic(x, x50, w, top):
    return top / (1 + np.exp(-(x - x50) / w))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dir', type=Path, default=out_dir('trigger'))
    a = ap.parse_args(argv)
    keys = ('run', 'arm', 't_us', 'wall_sum', 'plas', 'ppp', 'dt_ns', 'prev_us', 'has_slim', 'bunch', 'first_us')
    acc = {k: [] for k in keys}
    for f in sorted(a.dir.glob('trigcand_*.npz')):
        z = np.load(f)
        if int(z['run']) < PRODUCTION_FROM or not len(z['t_us']):
            continue
        n = len(z['t_us'])
        for k in keys:
            acc[k].append(np.full(n, int(z['run'])) if k == 'run' else z[k])
    D = {k: np.concatenate(v) for k, v in acc.items()}
    wt = np.vectorize(WALL_THR.get)(D['arm'])
    pt = np.vectorize(PLAS_THR.get)(D['arm'])
    x, y = D['wall_sum'] / wt, D['plas'] / pt
    match = (D['dt_ns'] > MATCH_NS[0]) & (D['dt_ns'] < MATCH_NS[1])
    live = D['has_slim'] & (D['t_us'] >= GATE_US) & (D['prev_us'] > DEAD_US)
    cls = np.where(D['ppp'] >= 6.5e12, 'ded', np.where(D['ppp'] >= 1e11, 'par', 'none'))
    # expected shift of the wall SUM from the per-end median baseline (quick curves), per class and time bin
    C = np.load(out_dir('quick') / 'curves.npz')
    tcq = C['tc_us']
    out = {'x_edges': X_EDGES, 't_bins': np.array(T_BINS)}
    rows = []
    for c in ('par', 'ded'):
        for lo, hi in T_BINS:
            sel = live & (cls == c) & (D['t_us'] >= lo) & (D['t_us'] < hi) & (y > 1.2)
            n, _ = np.histogram(x[sel], X_EDGES)
            k, _ = np.histogram(x[sel & match], X_EDGES)
            eff = k / np.maximum(n, 1)
            xc = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])
            ok = n >= 5
            try:
                p, cov = curve_fit(logistic, xc[ok], eff[ok], p0=[1.0, 0.1, 0.95],
                                   sigma=np.sqrt(np.maximum(eff[ok] * (1 - eff[ok]), 0.02) / n[ok]),
                                   bounds=([0.3, 0.005, 0.3], [3, 1, 1.05]), maxfev=20000)
                x50, ex50, top = p[0], float(np.sqrt(cov[0, 0])), p[2]
            except Exception:                    # noqa: BLE001
                x50 = ex50 = top = np.nan
            # expected: 2 x the median wall-end shift in this bin, averaged over the four walls, in units of the threshold
            m = (tcq >= lo) & (tcq < hi)
            sh = [float(np.nanmean(C[f'{w}_{c}_shift'][1][m])) for w in ('WALA', 'WALB', 'WALC', 'WALD')
                  if f'{w}_{c}_shift' in C.files]
            exp_mV = -2 * float(np.mean(sh)) if sh else np.nan
            out[f'{c}_{lo}_n'] = n
            out[f'{c}_{lo}_eff'] = eff
            rows.append(dict(cls=c, t_lo_us=lo, t_hi_us=hi, n_live=int(sel.sum()), x50=x50, x50_err=ex50,
                             plateau=top, expected_sum_undershoot_mV=exp_mV,
                             mean_threshold_mV=float(np.mean(list(WALL_THR.values())))))
    for c in ('par', 'ded'):
        ref = [r for r in rows if r['cls'] == c][-1]['x50']
        for r in rows:
            if r['cls'] == c:
                r['dx50'] = r['x50'] - ref
                r['dthr_mV_measured'] = r['dx50'] * r['mean_threshold_mV']
    # the robust part: when DREAM's gate opens, per bunch (its first recorded trigger)
    key = D['run'].astype(np.int64) * 100000 + D['bunch'].astype(np.int64)
    _, first_idx = np.unique(key, return_index=True)
    ft = D['first_us'][first_idx]
    out['first_trigger_us'] = ft[np.isfinite(ft)]
    # efficiency of strong candidates against time since the previous trigger: DREAM's busy behaviour
    arm = D['arm'].astype(int)
    strong = D['has_slim'].astype(bool) & (x > 1.5) & (y > 1.5) & (D['t_us'] > GATE_US)
    pe = np.array([0, 206, 300, 600, 1000, 2000, 5000, 1e5])
    out['prev_edges_us'] = pe
    out['eff_vs_prev'] = np.array([match[strong & (D['prev_us'] >= a) & (D['prev_us'] < b)].mean()
                                   if (strong & (D['prev_us'] >= a) & (D['prev_us'] < b)).any() else np.nan
                                   for a, b in zip(pe[:-1], pe[1:])])
    out['n_vs_prev'] = np.array([(strong & (D['prev_us'] >= a) & (D['prev_us'] < b)).sum() for a, b in zip(pe[:-1], pe[1:])])
    fw = strong & np.isinf(D['prev_us']) & (D['t_us'] < 1050)
    out['eff_first_window'] = match[fw].mean() if fw.any() else np.nan
    out['n_first_window'] = int(fw.sum())
    # expected undershoot of the wall SUM (2 x per-end median shift) against time, per class, over the mean threshold
    for c in ('par', 'ded'):
        sh = np.nanmean([C[f'{w}_{c}_shift'][1] for w in ('WALA', 'WALB', 'WALC', 'WALD') if f'{w}_{c}_shift' in C.files], axis=0)
        out[f'sum_undershoot_{c}_mV'] = -2 * sh
    out['tcq_us'] = tcq
    o = out_dir('trigger_ana')
    np.savez_compressed(o / 'trig_curves.npz', **out)
    print(f"first DREAM trigger per bunch: percentiles 1/5/50/95 = {np.percentile(out['first_trigger_us'], [1, 5, 50, 95]).round(1)} us; "
          f"first-window efficiency {out['eff_first_window']:.2f} (n={out['n_first_window']})")
    with open(o / 'trig_turnon.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(D["t_us"]):,} candidates, live {live.sum():,}, live & matched {(live & match).sum():,} -> {o}')
    for r in rows:
        print(f"  {r['cls']} {r['t_lo_us']:5.0f}-{r['t_hi_us']:5.0f} us: n={r['n_live']:6d}  x50 {r['x50']:.3f}±{r['x50_err']:.3f} "
              f"(shift {r['dthr_mV_measured']:+5.1f} mV, expected if shared {r['expected_sum_undershoot_mV']:+5.1f} mV)  plateau {r['plateau']:.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
