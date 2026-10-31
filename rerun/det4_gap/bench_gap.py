#!/usr/bin/env python3
"""bench_gap.py <incl.pkl> [--key g_det4] [--out results/bench_gap_<key>.json] -- the bench drift gap with the SPS estimator.

Same per-event estimator as gap_geom.py (strips on the time-position line, attenuated box blurred by the footprint,
half-height edges, L = |b - a|), on the bench events of S4's inclined-track cache (rerun/calib/s4/s4_stacks.py:
pass-1 waveforms, M3 rays through the pass alignment).  Here the track angle is M3's tan_x per event, and the gap is
the slope of L against |tan_x| over |tan_x| in [TAN_LO, TAN_HI] (the footprint and any constant edge offset go into
the intercept), with median(L / |tan_x|) as a check.  Unlike gap_extent.py, no mesh position (ref_mesh) enters, so an
alignment offset or tilt of the reference along x cannot split the +-tan halves; only M3's tan scale can bias it.
--view x|y; interior events only (>= 30 mm from the X edges, >= 26 mm from the Y edges).  On det4 the X view crosses
its dead amplification stripes (x intervals) and reads short; --view y --cross-max 0.05 keeps tracks at one x, whose
whole Y column is live or dead together.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_geom as G                                         # noqa: E402

TAN_LO, TAN_HI = 0.18, 0.6
X_LO, X_HI = 30.0, 368.6          # interior in x (active X 0-398.6 mm)


VIEW = 'x'
CROSS_MAX = None      # Y view: |tan_x| below this keeps the column at one x (det4's dead stripes are x intervals)
Y_LO, Y_HI = 26.0, 372.0


def to_npz_like(ev, v):
    W = np.asarray(ev[v]['W'], float); pos = np.asarray(ev[v]['pos'], float)
    return pos, np.clip(W, 0, None).sum(1), W.max(1), W.argmax(1).astype(float)


def extent(ev, s, nsmp):
    y, q, pk, t = to_npz_like(ev, VIEW)
    use = pk >= G.PK_LINE
    if use.sum() < G.MIN_STRIPS:
        return None
    a0, b0 = G.theil_sen(y[use], t[use])
    keep = (pk >= G.PK_MIN) & (np.abs(t - (a0 + b0 * y)) <= G.T_TOL)
    if keep.sum() < G.MIN_STRIPS:
        return None
    tk = t[keep]
    if tk.min() < G.T_FIRST or tk.max() > nsmp - 1 - G.T_LAST:
        return None
    qq = np.where(keep, q, 0.0); yk = y[keep]
    ym, yc = yk[np.argmin(tk)], yk[np.argmax(tk)]
    lo = [0, -2.0, min(ym, yc) - 8, min(ym, yc) - 8]; hi = [np.inf, 2.0, max(ym, yc) + 8, max(ym, yc) + 8]
    try:
        r = least_squares(lambda p: G.box_model(p, y, s) - qq, [np.median(qq[keep]), 0.0, ym, yc], bounds=(lo, hi),
                          loss='soft_l1', f_scale=max(np.median(qq[keep]) * 0.3, 1.0))
    except Exception:
        return None
    A, k, a, b = r.x
    if abs(b - a) > abs(yc - ym) + 10:
        return None
    if min(a, b) - y.min() < G.CONTAIN_MM or y.max() - max(a, b) < G.CONTAIN_MM:
        return None                   # the column runs off the window: its extent is cut
    return abs(b - a), abs(yc - ym)


def footprint(evs):
    out = []
    for e in evs:
        y, q, pk, t = to_npz_like(e, VIEW)
        if len(y) < 5 or pk.max() < 300:
            continue
        c = y[np.argmax(pk)]; m = np.abs(y - c) <= 6 * 0.78
        yy, qq = y[m], np.where(pk[m] >= G.PK_MIN, q[m], 0.0)
        try:
            r = least_squares(lambda p: p[0] * np.exp(-0.5 * ((yy - p[1]) / p[2]) ** 2) - qq, [qq.max(), c, 0.8],
                              bounds=([0, c - 3, 0.2], [np.inf, c + 3, 4.0]))
            out.append(r.x[2])
        except Exception:
            pass
    return float(np.median(out)), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('incl'); ap.add_argument('--key', default='g_det4'); ap.add_argument('--out')
    ap.add_argument('--view', default='x'); ap.add_argument('--cross-max', type=float)
    a = ap.parse_args()
    global VIEW, CROSS_MAX
    VIEW, CROSS_MAX = a.view, a.cross_max
    o = 'y' if VIEW == 'x' else 'x'
    D = pickle.load(open(a.incl, 'rb'))
    evs = [e for e in D.values() if X_LO <= e['ref_mesh_x'] <= X_HI and Y_LO <= e['ref_mesh_y'] <= Y_HI]
    if CROSS_MAX is not None:
        evs = [e for e in evs if abs(e[f'tan_{o}']) < CROSS_MAX]
    nsmp = np.asarray(evs[0][VIEW]['W']).shape[1]
    s, ns = footprint([e for e in evs if abs(e[f'tan_{VIEW}']) < 0.03])
    print(f'{a.key}: {len(evs)} interior events, {nsmp} samples, head-on footprint sigma {s:.3f} mm ({ns})', flush=True)
    rows = []
    for e in evs:
        t = abs(e[f'tan_{VIEW}'])
        if not TAN_LO <= t <= TAN_HI:
            continue
        r = extent(e, s, nsmp)
        if r:
            rows.append((t, np.sign(e[f'tan_{VIEW}']), r[0], r[1]))
    R = np.array(rows)
    t, sg, L, Lraw = R.T
    A = np.vstack([t, np.ones_like(t)]).T
    # robust slope: iterate a 3-MAD trim of the residuals
    ok = np.ones(len(t), bool)
    for _ in range(5):
        c = np.linalg.lstsq(A[ok], L[ok], rcond=None)[0]
        res = L - A @ c; mad = 1.4826 * np.median(np.abs(res[ok] - np.median(res[ok])))
        ok = np.abs(res) < 3 * mad
    rng = np.random.default_rng(2); bs = []
    idx = np.flatnonzero(ok)
    for _ in range(300):
        j = rng.choice(idx, len(idx)); bs.append(np.linalg.lstsq(A[j], L[j], rcond=None)[0][0])
    out = dict(key=a.key, view=VIEW, cross_max=CROSS_MAX, n=int(len(t)), n_used=int(ok.sum()), footprint_sigma=s, gap_slope=float(c[0]),
               gap_slope_err=float(np.std(bs)), intercept_mm=float(c[1]),
               gap_ratio_median=float(np.median(L / t)),
               by_sign={str(int(k)): float(np.linalg.lstsq(A[ok & (sg == k)], L[ok & (sg == k)], rcond=None)[0][0])
                        for k in (1, -1) if (ok & (sg == k)).sum() > 30},
               gap_rawends_slope=float(np.linalg.lstsq(A[ok], Lraw[ok], rcond=None)[0][0]))
    print(json.dumps(out, indent=1), flush=True)
    p = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f'bench_gap_{a.key}_{VIEW}.json')
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, 'w'), indent=1)


if __name__ == '__main__':
    main()
