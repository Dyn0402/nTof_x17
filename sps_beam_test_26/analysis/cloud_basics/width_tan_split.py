#!/usr/bin/env python3
"""width_tan_split.py -- is the bench's dry-like drift diffusion an inclination leak?

FINDINGS §10 fits bench drift diffusion at dry-Magboltz size (D_T ~ 460-510 um/sqrt(cm)), while
the bench v(E) needs ~1 % water (§25), where Magboltz gives D_T ~ 270.  "Head-on" bench tracks
are |tan_ref| < 0.03, so each event's charge also walks tan*z across the strips during the drift,
which adds tan^2 (z - zbar)^2 to the width about the event centroid and grows with time like
diffusion does.

Model-free (width_vs_time.py moments, X view only: X has no resistive spread): the width growth
d(var)/dt over the drift plateau, in |tan_ref| bins.  Diffusion is independent of tan; the
inclination term goes as tan^2.  A linear fit of the growth against <tan^2> gives the tan = 0
intercept, converted to D_T with the chamber's v.

    width_tan_split.py        -> results/width_tan_split.json
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import width_vs_time as wvt                                      # noqa: E402

BINS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.08)]
T_LO, T_HI = 100.0, 600.0          # ns after the 50 % rise: inside every bench drift (~800 ns)


def growth(ev, lo, hi):
    sub = {k: e for k, e in ev.items() if 'x' in e and lo <= abs(e['tan_x']) < hi}
    n, num, den = wvt.stack(sub, 'x', hi)
    ok = den / max(den.max(), 1e-9) > 0.05
    var = np.where(ok, num / np.where(ok, den, 1) - wvt.PITCH ** 2 / 12, np.nan)
    sel = ok & (wvt.GRID >= T_LO) & (wvt.GRID <= T_HI)
    p = np.polyfit(wvt.GRID[sel], var[sel], 1)
    t2 = np.mean([e['tan_x'] ** 2 for e in sub.values()])
    return n, float(p[0]), float(t2), float(np.interp(0.0, wvt.GRID[ok], var[ok]))


def main():
    pin = json.load(open(os.path.join(HERE, 'results', 'bench_pinned.json')))
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        ev = pickle.load(open(wvt.SOURCES[det], 'rb'))
        v = pin[det]['v']                                       # um/ns
        rows = []
        for lo, hi in BINS:
            n, slope, t2, var0 = growth(ev, lo, hi)
            rows.append(dict(lo=lo, hi=hi, n=n, slope_mm2_per_ns=slope, tan2=t2, var_t0=var0))
        # weighted (by n) linear fit of slope vs <tan^2>; bootstrap over events for the error
        x = np.array([r['tan2'] for r in rows]); y = np.array([r['slope_mm2_per_ns'] for r in rows])
        w = np.array([r['n'] for r in rows], float)
        A = np.c_[np.ones_like(x), x] * np.sqrt(w)[:, None]
        c0, c1 = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        # var growth per ns = D_T^2 [mm^2/cm] * v [mm/ns] / 10
        to_DT = lambda s: 1e3 * np.sqrt(max(s, 0) * 10.0 / (v * 1e-3))   # um/sqrt(cm)
        out[det] = dict(v=v, rows=rows, intercept=float(c0), tan2_coef=float(c1),
                        DT_tan0=to_DT(c0), DT_lowest_bin=to_DT(rows[0]['slope_mm2_per_ns']),
                        DT_cut003=to_DT(np.average(y[:3], weights=w[:3])))
        print(f'{det} v={v:.1f}: ' + ' | '.join(f'[{r["lo"]:.2f},{r["hi"]:.2f}) n={r["n"]} '
                                                 f'dvar/dt={r["slope_mm2_per_ns"]*1e3:.3f}e-3' for r in rows))
        print(f'      D_T: |tan|<0.03 pooled {out[det]["DT_cut003"]:.0f}, lowest bin {out[det]["DT_lowest_bin"]:.0f}, '
              f'tan->0 intercept {out[det]["DT_tan0"]:.0f} um/sqrt(cm); tan^2 coef {c1:.3g} mm^2/ns', flush=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'width_tan_split.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
