#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bow_check.py -- the bow-on vs bow-off test that NTOF_PASS_PLAN §5 asked for, offline, on the A-C cosmic lines of
every beam-off sub-run (line_residuals.py build).

Pass 1 carried each chamber's bench bow block (tilt a_*, scale s_*, mesh height h_*) to n_TOF.  For each track the
bow-off angle is the constant mapping from the fitted transverse speed w: tan_off = w / (kw v); the bow-on one is
tan_bow(w, U, W) (reproduced here against the stored x/y_tan_theta).  The stage-3 v(t) angle is the stage-2 angle times
a per-track factor (frame sign and v_bundle / (kappa v_t)), which carries over unchanged.  The mesh-height part moves
the track point by tan * h; it is ignored here (positions enter the joined line, 469 mm lever arm: 1 mm = 0.002).

    python -m rerun.ntof.improve.bow_check   -> <out>/bow_check.csv, printed summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from wft import reco as R  # noqa: E402
from wft.calib import CalibrationBundle  # noqa: E402
from rerun.ntof.improve.track_quality import out_dir  # noqa: E402
from rerun.ntof.improve.line_residuals import mad  # noqa: E402

import os
BUNDLES = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1' / 'bundles' / 'pf'


def tan_variants(cal, M, s):
    """bow-on (recomputed), bow-off (constant) and tilt-only / scale-only angles, strip frame, per view."""
    wx, wy = M[f'x_w_{s}'].to_numpy(float), M[f'y_w_{s}'].to_numpy(float)
    px, py = M[f'x_p0_{s}'].to_numpy(float), M[f'y_p0_{s}'].to_numpy(float)
    out = {}
    tcx = (wx * 1e3 - cal.w0.get('x', 0.0)) / (cal.kw['x'] * cal.v_drift)
    tcy = (wy * 1e3 - cal.w0.get('y', 0.0)) / (cal.kw['y'] * cal.v_drift)
    U = (px + R.BOW_MID_MM * tcx - 200.0) / 100.0
    W = (py + R.BOW_MID_MM * tcy - 200.0) / 100.0
    U = np.where(np.abs(U) <= R.BOW_UW_MAX, U, 0.0)
    W = np.where(np.abs(W) <= R.BOW_UW_MAX, W, 0.0)
    b = cal.bow
    for p, tc in (('x', tcx), ('y', tcy)):
        sc = b.get(f's_{p}') or {}
        S = 1.0 + sc.get('su', 0.0) * U + sc.get('sw', 0.0) * W
        a = R._bow_poly(b.get(f'a_{p}') or {}, U, W)
        out[f'on_{p}'] = tc / S - a
        out[f'off_{p}'] = tc
        out[f'tilt_{p}'] = tc - a          # tilt only
        out[f'scale_{p}'] = tc / S         # scale only
        out[f'a_{p}'] = a
        out[f'S_{p}'] = S
    out['U'], out['W'] = U, W
    return out


def main() -> int:
    M = pd.read_parquet(out_dir() / 'lines.parquet')
    need = ['x_w', 'y_w', 'x_tan_theta', 'y_tan_theta']
    if f'x_w_a' not in M:
        sys.exit('lines.parquet lacks x_w/y_w/x_tan_theta: rebuild with line_residuals build (COLS updated)')
    rows = []
    for s, arm in (('a', 'A'), ('c', 'C')):
        cal = CalibrationBundle.load(str(BUNDLES / f'{arm}_E2_cos'))
        V = tan_variants(cal, M, s)
        for p in 'xy':
            stored = M[f'{p}_tan_theta_{s}'].to_numpy(float)
            dev = np.nanmax(np.abs(V[f'on_{p}'] - stored))
            fac = M[f'tan{p}_{s}'].to_numpy(float) / stored      # stage-3 v(t) tan per stage-2 tan
            fac = np.where(np.abs(stored) > 0.02, fac, np.nan)
            fmed = np.nanmedian(fac)
            fac = np.where(np.isfinite(fac), fac, fmed)
            j = M[f'j{p}'].to_numpy(float)
            # global slope d_p/d_z in stage 3 vs the strip-frame tan: tan_p (view) maps to d_p/d_z by the same factor
            # as tanx -> d_x/d_z; take it from the data
            g = (M[f'd_{p}_{s}'] / M[f'd_z_{s}']).to_numpy(float) / M[f'tan{p}_{s}'].to_numpy(float)
            g = np.where(np.isfinite(g) & (np.abs(M[f'tan{p}_{s}']) > 0.02), g, np.nan)
            gmed = np.nanmedian(g)
            sel = np.abs(j) < 0.45
            for var in ('on', 'off', 'tilt', 'scale'):
                slope = V[f'{var}_{p}'] * fac * gmed
                r = slope - j
                x = M[f'{"x" if p == "x" else "y"}_local_{s}'].to_numpy(float)
                rr = r[sel]
                # position structure: robust linear fit of residual against the view's own coordinate and against j
                k = sel & np.isfinite(r) & (np.abs(r) < 0.15)
                A = np.c_[np.ones(k.sum()), j[k], x[k] / 100.0, (x[k] / 100.0) ** 2]
                coef = np.linalg.lstsq(A, r[k], rcond=None)[0]
                # scatter of the per-cell median (50 mm cells, own coordinate x other coordinate)
                o = M[f'{"y" if p == "x" else "x"}_local_{s}'].to_numpy(float)
                cell = (np.digitize(x, np.arange(-150, 151, 50)) * 10 + np.digitize(o, np.arange(-150, 151, 50)))
                med = pd.Series(r[k]).groupby(cell[k]).agg(['median', 'size'])
                med = med[med['size'] >= 100]
                rows.append(dict(arm=arm, view=p, variant=var, n=int(sel.sum()), bias=float(np.nanmedian(rr)),
                                 sigma=mad(rr), tail=float(np.mean(np.abs(rr) > 0.15)),
                                 c0=coef[0], c_j=coef[1], c_pos=coef[2], c_pos2=coef[3],
                                 cell_rms=float(np.sqrt(np.mean(med['median'] ** 2))), n_cells=len(med),
                                 recompute_dev=dev, fac=fmed, g=gmed,
                                 a_rms=float(np.sqrt(np.mean(V[f'a_{p}'][sel] ** 2))),
                                 S_rms=float(np.std(V[f'S_{p}'][sel]))))
    T = pd.DataFrame(rows)
    T.to_csv(out_dir() / 'bow_check.csv', index=False)
    pd.set_option('display.width', 220)
    print(T.round(4).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
