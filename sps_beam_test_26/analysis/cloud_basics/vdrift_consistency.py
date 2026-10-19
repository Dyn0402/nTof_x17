#!/usr/bin/env python3
"""vdrift_consistency.py -- do the timing-free gaps, the measured drift times and Magboltz mesh?  (FINDINGS §40)

For every bench dataset with an unsaturated head-on stack (unsat_width.json) the stack fit gives the drift
duration T (the charge column's arrival length, free in the fit).  With the gap G of gaps.py:

    v = G / T,    E = V_drift / G           (every earlier comparison used E = V / 2.88 cm for all chambers)

and Magboltz (Ar/iso 95/5 + water, gasmodel 'bench', air_hs grid) has one free parameter per gas fill, the
water fraction.  Tests:

  1. det3's 6-27 drift scan (700 / 900 / 1100 V + the 1000 V long run, + the two 1000 V amplification-scan
     points): ONE water must fit every field.  Profiled over G it gives a gap from timing + Magboltz alone,
     to compare with the timing-free 28.08 mm.
  2. Every single-field chamber: the water implied by v, then Magboltz's D_T at that water against the
     stack's measured D_T (gap-corrected).  Scanning G turns each (v, D_T) pair into a third gap estimate.
  3. det6 and det7 (6-26, the same run and field): the same gas would give the same v.

    vdrift_consistency.py  -> results/vdrift_consistency.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gaps                                                  # noqa: E402
from gasmodel import GasGrid                                 # noqa: E402

R = os.path.join(HERE, 'results')
UW = json.load(open(os.path.join(R, 'unsat_width.json')))
G = GasGrid('bench', 'air_hs')
WMAX = float(G.W.max())
WS = np.arange(0.0, WMAX + 1e-9, 0.005)
V_SYS = 0.015            # fractional systematic on T per stack (template / t0 / end definition); bench_ladder, §21

SCAN = [  # key, drift V, label
    ('sat_det3__drift_scan_resist_490V_drift_700V', 700, '700 V scan'),
    ('sat_det3__drift_scan_resist_490V_drift_900V', 900, '900 V scan'),
    ('sat_det3__long_run_resist_490V_drift_1000V', 1000, '1000 V long run'),
    ('sat_det3__hv_scan_resist_445V_drift_1000V', 1000, '1000 V, mesh 445'),
    ('sat_det3__hv_scan_resist_465V_drift_1000V', 1000, '1000 V, mesh 465'),
    ('sat_det3__drift_scan_resist_490V_drift_1100V', 1100, '1100 V scan'),
]
SINGLE = [  # key, det, drift V, gas group
    ('sat_det3__long_run_resist_490V_drift_1000V', 'det3', 1000, '6-27'),
    ('o22_long_det2', 'det2', 1000, '6-22'),
    ('g_det4', 'det4', 600, '6-24'),
    ('g_det6_long', 'det6', 700, '6-26'),
    ('g_det6_longer', 'det6', 700, '6-26'),
    ('g_det7_long', 'det7', 700, '6-26'),
    ('g_det7_long__longer_run', 'det7', 700, '6-26'),
    ('g_det7_long__hvpool_det7_480-505V_drift_700V', 'det7', 700, '6-26'),
]


def tv(key):
    """(T [ns], fractional error on T) of a dataset's unsaturated stack."""
    u = UW[key]['unsat']
    return u['T'], np.hypot(u['v_err'] / u['v'], V_SYS)


def vgrid(E):
    return np.array([G(w, 0.0, E)['v'] for w in WS])


def water_branches(v, E):
    """Every water fraction on the grid where Magboltz v(E) crosses v."""
    vv = vgrid(E) - v
    out = []
    for i in range(len(WS) - 1):
        if vv[i] == 0 or vv[i] * vv[i + 1] < 0:
            out.append(float(WS[i] + (WS[i + 1] - WS[i]) * vv[i] / (vv[i] - vv[i + 1])))
    return out


def scan_fit(gap):
    """det3 scan: chi2(water) at this gap, minimised over water."""
    pts = []
    for key, V, lab in SCAN:
        if key not in UW or not UW[key].get('unsat', {}).get('T'):
            continue
        T, fe = tv(key)
        pts.append((V / (gap / 10.0), gap / T * 1e3, fe, lab, key))
    chi = []
    for w in WS:
        c = sum(((v - G(w, 0.0, E)['v']) / (fe * v)) ** 2 for E, v, fe, _, _ in pts)
        chi.append(c)
    chi = np.array(chi)
    i = int(np.argmin(chi))
    # water error from delta chi2 = 1
    ok = WS[chi <= chi[i] + 1.0]
    return dict(gap=gap, water=float(WS[i]), water_lo=float(ok.min()), water_hi=float(ok.max()), chi2=float(chi[i]),
                ndf=len(pts) - 1, points=[dict(E=E, v=v, v_err=fe * v, label=lab, key=k,
                                               v_mb=float(G(WS[i], 0.0, E)['v'])) for E, v, fe, lab, k in pts])


def single(key, det, V, gap):
    T, fe = tv(key)
    v, E = gap / T * 1e3, V / (gap / 10.0)
    u = UW[key]['unsat']
    DT = u['DT'] * np.sqrt(30.0 / gap)          # fit frame GAP = 30: v x gap/30, D_T x sqrt(30/gap)
    DTe = u['DT_err'] * np.sqrt(30.0 / gap)
    br = water_branches(v, E)
    return dict(key=key, det=det, gap=gap, E=E, v=v, v_err=fe * v, DT=DT, DT_err=DTe,
                branches=[dict(water=w, DT_mb=float(G(w, 0.0, E)['DT']), DL_mb=float(G(w, 0.0, E)['DL'])) for w in br])


def main():
    out = {'v_sys': V_SYS, 'water_max': WMAX}
    # 1. det3 scan, gap fixed and profiled
    g3 = gaps.gap('det3')
    out['scan_fixed'] = scan_fit(g3)
    prof = [scan_fit(g) for g in np.arange(25.0, 33.01, 0.25)]
    out['scan_profile'] = [dict(gap=p['gap'], chi2=p['chi2'], water=p['water']) for p in prof]
    c = np.array([p['chi2'] for p in prof]); gg = np.array([p['gap'] for p in prof])
    i = int(np.argmin(c))
    okg = gg[c <= c[i] + 1.0]
    out['scan_gap'] = dict(best=float(gg[i]), lo=float(okg.min()), hi=float(okg.max()), chi2=float(c[i]))
    f = out['scan_fixed']
    print(f'det3 scan at the timing-free gap {g3:.2f}: water {f["water"]:.3f} % [{f["water_lo"]:.3f}, {f["water_hi"]:.3f}]  '
          f'chi2 {f["chi2"]:.1f} / {f["ndf"]}')
    for p in f['points']:
        print(f'   {p["label"]:18s} E {p["E"]:6.1f}  v {p["v"]:6.2f} ± {p["v_err"]:.2f}   Magboltz {p["v_mb"]:6.2f}   pull {(p["v"] - p["v_mb"]) / p["v_err"]:+.1f}')
    w3 = f['water']
    for p in f['points']:
        u = UW[p['key']]['unsat']
        p['DT'] = u['DT'] * np.sqrt(30.0 / g3); p['DT_err'] = u['DT_err'] * np.sqrt(30.0 / g3)
        p['DT_mb'] = float(G(w3, 0.0, p['E'])['DT']); p['DL_mb'] = float(G(w3, 0.0, p['E'])['DL'])
        print(f'   {p["label"]:18s} D_T {p["DT"]:4.0f} ± {p["DT_err"]:3.0f}  Magboltz {p["DT_mb"]:4.0f}  '
              f'(excess in quadrature {np.sign(p["DT"] - p["DT_mb"]) * np.sqrt(abs(p["DT"] ** 2 - p["DT_mb"] ** 2)):+.0f})')
    print(f'det3 scan gap from timing + Magboltz alone: {gg[i]:.2f} [{okg.min():.2f}, {okg.max():.2f}] mm (chi2 {c[i]:.1f})')
    # 2. single-field chambers, at the adopted gap and scanned
    out['single'] = []
    for key, det, V, grp in SINGLE:
        if key not in UW or not UW[key].get('unsat', {}).get('T'):
            continue
        g = gaps.gap(det)
        r = single(key, det, V, g)
        r['group'] = grp
        r['scan'] = []
        for gs in np.arange(26.0, 34.01, 0.25):
            q = single(key, det, V, gs)
            r['scan'].append(dict(gap=float(gs), v=q['v'], E=q['E'], DT=q['DT'],
                                  branches=[(b['water'], b['DT_mb']) for b in q['branches']]))
        # the gap at which the measured D_T equals Magboltz's at the v-implied water (each branch)
        cross = []
        for b in range(3):
            xs = [(q['gap'], q['DT'] - q['branches'][b][1]) for q in r['scan'] if len(q['branches']) > b]
            for (g0, d0), (g1, d1) in zip(xs[:-1], xs[1:]):
                if d0 * d1 < 0:
                    cross.append(float(g0 + (g1 - g0) * d0 / (d0 - d1)))
        r['dt_gap'] = cross
        out['single'].append(r)
        br = ', '.join(f'{b["water"]:.2f} % (D_T {b["DT_mb"]:.0f})' for b in r['branches']) or f'none (v outside Magboltz 0–{WMAX:g} %)'
        print(f'{key:46s} gap {g:5.2f}  E {r["E"]:5.1f}  v {r["v"]:5.2f} ± {r["v_err"]:.2f}  D_T {r["DT"]:4.0f} ± {r["DT_err"]:3.0f}  '
              f'water: {br}   D_T-match gap: ' + ', '.join(f'{x:.1f}' for x in cross))
    json.dump(out, open(os.path.join(R, 'vdrift_consistency.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
