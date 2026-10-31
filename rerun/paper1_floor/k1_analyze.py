#!/usr/bin/env python3
"""k1_analyze.py <raw dir> [--out results/k1_test.json] -- does the locked model's K1 (plane-to-plane shape scatter of
bright pulses) explain the unexplained angle remainder?  (K1_TEST_2026-11-01.md)

Reads the K1 campaign's per-event toy rows (condor/make_k1_jobs.py; ladder.py columns: angle, dtheta, chi2/dof, dof,
sigma0 used, dt0, sat_event, n_strips, n_e, q_sum, centre residual, centre peak) and, per chamber, view, rung and
size, gives the signatures the data were characterised by (README.md, results/floor_budget.json, chi2_scaling.json):
  s68_w       angle s68 with the data's |theta| mix (pass1_basis weights of the 3 / 9 / 16 deg points)
  a, b        s68(theta)^2 = a^2 + (b tan theta)^2 over the three angle points (a: the head-on part)
  rc_sd       scatter of (data - model) / model peak on the brightest unsaturated strip at its peak (T16's statistic;
              data det3: X 0.063, Y 0.058)
  cd_q50      median chi2/dof of unsaturated events, read at the data's median charge by a log-log fit over q deciles
  cd_slope    d log(chi2/dof - 1) / d log q_sum over unsaturated events (data: 1.4-2.0)
  sel_share   variance removed by keeping the best chi2/dof quarter in (angle x q tercile) cells (data: 10-31 %)
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, '..', 'model_audit', 'followups', 'fd_floor_realism', 'results')
INPUTS = json.load(open(os.path.join(FD, 'toy_inputs_pass1.json')))
ANG = (3.0, 9.0, 16.0)


def s68(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    return float(np.percentile(np.abs(a - np.median(a)), 68)) if len(a) > 5 else np.nan


def s68_w(rows, w):
    """weighted pooled s68 (each angle point weighted by the data's mix / its count), as pass1_basis.s68_w"""
    d, ang = rows[:, 1], rows[:, 0]
    k = np.isfinite(d) & (np.abs(d) <= 5)
    d, ang = d[k], ang[k]
    ww = np.zeros(len(d))
    for a, wa in zip(ANG, w):
        m = ang == a
        if m.any():
            ww[m] = wa / m.sum()
    if ww.sum() <= 0:
        return np.nan

    def wq(x, q):
        o = np.argsort(x)
        c = np.cumsum(ww[o]) - 0.5 * ww[o]
        return float(np.interp(q * ww.sum(), c, x[o]))
    med = wq(d, 0.5)
    return wq(np.abs(d - med), 0.68)


def angle_form(rows):
    t2, s2 = [], []
    for a in ANG:
        r = rows[rows[:, 0] == a]
        if len(r) > 20:
            t2.append(np.tan(np.radians(a)) ** 2)
            s2.append(s68(r[:, 1]) ** 2)
    if len(t2) < 2:
        return np.nan, np.nan
    b2, a2 = np.polyfit(t2, s2, 1)
    return float(np.sqrt(max(a2, 0))), float(np.sqrt(max(b2, 0)))


def chi2_q(rows, q50):
    r = rows[(rows[:, 6] == 0) & np.isfinite(rows[:, 9]) & (rows[:, 9] > 0) & np.isfinite(rows[:, 2])]
    if len(r) < 50:
        return np.nan, np.nan
    edges = np.quantile(r[:, 9], np.linspace(0, 1, 11))
    qs, cs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r[:, 9] >= lo) & (r[:, 9] <= hi)
        if m.sum() > 5:
            qs.append(np.median(r[m, 9]))
            cs.append(np.median(r[m, 2]))
    qs, cs = np.array(qs), np.array(cs)
    ok = cs > 1.5
    sl = float(np.polyfit(np.log(qs[ok]), np.log(cs[ok] - 1), 1)[0]) if ok.sum() >= 4 else np.nan
    c = np.polyfit(np.log(qs), np.log(cs), 1)
    return float(np.exp(np.polyval(c, np.log(q50)))), sl


def sel_share(rows):
    num = den = 0.0
    for a in ANG:
        r = rows[(rows[:, 0] == a) & np.isfinite(rows[:, 1]) & np.isfinite(rows[:, 2])]
        if len(r) < 60:
            continue
        keep = np.zeros(len(r), bool)
        qe = np.quantile(r[:, 9], [0, 1 / 3, 2 / 3, 1]) if np.isfinite(r[:, 9]).all() else None
        cells = [np.ones(len(r), bool)] if qe is None else \
            [(r[:, 9] >= lo) & (r[:, 9] <= hi) for lo, hi in zip(qe[:-1], qe[1:])]
        for m in cells:
            idx = np.flatnonzero(m)
            if len(idx) < 8:
                continue
            thr = np.quantile(r[idx, 2], 0.25)
            keep[idx[r[idx, 2] <= thr]] = True
        sa, sb = s68(r[:, 1]), s68(r[keep, 1])
        num += sa * sa - sb * sb
        den += sa * sa
    return float(num / den) if den > 0 else np.nan


def summarise(rows, det):
    w = INPUTS[det]['ang_w']['ig']
    a, b = angle_form(rows)
    cd, sl = chi2_q(rows, INPUTS[det]['q50']['ig'])
    rc = rows[:, 10]
    rc = rc[np.isfinite(rc)]
    return dict(n=int(len(rows)), s68_w=s68_w(rows, w), a=a, b=b,
                s68_per_angle={str(int(x)): s68(rows[rows[:, 0] == x, 1]) for x in ANG},
                rc_mean=float(np.mean(rc)) if len(rc) else np.nan, rc_sd=float(np.std(rc)) if len(rc) else np.nan,
                cd_q50=cd, cd_slope=sl, chi2dof_med=float(np.nanmedian(rows[:, 2])), sel_share=sel_share(rows),
                sat_event=float(np.mean(rows[:, 6])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'k1_test.json'))
    a = ap.parse_args()
    groups = {}
    pat = re.compile(r'^(det\d)_(L8w?|K1[GTWSL]w?)(?:_k([0-9.e-]+))?(?:_wt([0-9.]+))?_a(\d+)(_y)?\.npy$')
    for f in sorted(glob.glob(os.path.join(a.raw, '*.npy'))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        det, rung, amp, wt, ang, y = m.groups()
        key = (det, 'y' if y else 'x', rung, amp or '', wt or '')
        groups.setdefault(key, []).append(np.load(f))
    out = {}
    for (det, view, rung, amp, wt), parts in sorted(groups.items()):
        rows = np.concatenate(parts)
        s = summarise(rows, det)
        s.update(det=det, view=view, rung=rung, amp=float(amp) if amp else 0.0, wt=float(wt) if wt else 0.0,
                 n_angles=len(parts))
        out[f'{det}|{view}|{rung}|{amp}|{wt}'] = s
        print(f"{det} {view} {rung:5s} amp {amp or '-':>5s} wt {wt or '-':>4s}: s68_w {s['s68_w']:.3f} a {s['a']:.2f} "
              f"b {s['b']:.2f} rc_sd {s['rc_sd']:.3f} cd@q50 {s['cd_q50']:.1f} slope {s['cd_slope']:.2f} "
              f"sel {s['sel_share']:.2f} ({s['n']})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
