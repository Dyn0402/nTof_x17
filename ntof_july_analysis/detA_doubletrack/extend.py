#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extend.py — re-fit the tracks of ONE Det-A double-track event by road-following
(picking up the hits the RANSAC seed missed), test the extended drift span
against the full-gap expectation for that drift HV, resolve/expose the X/Y
pairing ambiguity, and back-project the 3-D tracks toward the target.

Motivated by run_58 ev1054 @ drift 300 V, where the y-plane fits already span
~1200 ns (~full gap) but the x-plane fits only 600-800 ns -> the x lines are
truncated, and the frozen DriftModel is useless here (its bench curve only
covers E >= 233 V/cm, so every drift HV 200-700 V clamps to 23.31 um/ns).
We therefore use the Garfield Ar/iC4H10 90/10 curve, which covers E >= 40 V/cm.

Usage:
  .venv/bin/python .../extend.py run_58 sngPS_dr300_r580_036 1054
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import dtrack_lib as D  # noqa: E402
import scan as SC  # noqa: E402
from ntof_tracking.reco import io, noise, segments as S, geometry as geo  # noqa: E402

OUT = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'detA_doubletrack', 'extend')
GARFIELD = os.path.join(_REPO, 'garfield_sim', 'results',
                        'drift_9010_contam_cern.json')
GAP_MM = 30.0
ROAD_MM = 4.0            # corridor half-width for hit pickup
ROAD_ITERS = 4
LINE_COL = ['crimson', 'royalblue', 'seagreen', 'darkorange']


def garfield_v(E_Vcm, mixture='Ar90_iso10'):
    """Drift velocity [um/ns] at field E for the run_58 gas (Ar/iso 90/10)."""
    arr = json.load(open(GARFIELD))['mixtures'][mixture]
    E = np.array([p['E_Vcm'] for p in arr])
    V = np.array([p['v_um_per_ns'] for p in arr])
    return float(np.interp(E_Vcm, E, V))


def road_extend(line, pos, time, amp, road=ROAD_MM, iters=ROAD_ITERS):
    """Grow a fitted line along its own extrapolation: collect every clean hit
    within `road` mm of the line ACROSS THE WHOLE WINDOW, refit, repeat until
    the membership stops changing. Returns a new line dict (same schema)."""
    sl, ic = line['slope_mm_ns'], line['intercept_mm']
    idx = np.asarray(line['idx'])
    for _ in range(iters):
        resid = np.abs(pos - (sl * time + ic))
        new = np.flatnonzero(resid <= road)
        if len(new) < 3:
            break
        fit = S.robust_line_fit(time[new], pos[new], w=amp[new])
        if fit is None:
            break
        # keep only the fit's inliers so a stray blob cannot drag the line
        new = new[fit['inliers']]
        if len(new) == len(idx) and set(new) == set(idx):
            idx = new
            sl, ic = fit['slope_mm_ns'], fit['intercept_mm']
            break
        idx = new
        sl, ic = fit['slope_mm_ns'], fit['intercept_mm']
    p, t, a = pos[idx], time[idx], amp[idx]
    fit = S.robust_line_fit(t, p, w=a)
    out = dict(line)
    out.update(idx=idx, slope_mm_ns=sl, intercept_mm=ic,
               n_hits=len(idx), n_strips=int(len(np.unique(p))),
               t0_ns=float(t.min()), t1_ns=float(t.max()),
               tspan_ns=float(np.ptp(t)), pspan_mm=float(np.ptp(p)),
               pos_lo_mm=float(p.min()), pos_hi_mm=float(p.max()),
               q_sum=float(a.sum()),
               r2=float(fit['r2']) if fit else np.nan,
               res_rms_mm=float(fit['res_rms_mm']) if fit else np.nan)
    return out


def main():
    run, subrun, evid = sys.argv[1], sys.argv[2], int(sys.argv[3])
    drift_hv = io.parse_drift_hv(subrun) or 300.0
    E = drift_hv / (GAP_MM / 10.0)
    v_g = garfield_v(E)
    tmax_g = GAP_MM * 1000.0 / v_g
    print(f'{run}/{subrun} ev{evid}   drift={drift_hv:.0f} V  -> E={E:.0f} V/cm')
    print(f'  Garfield Ar/iso 90/10:  v_drift={v_g:.1f} um/ns, '
          f'full-gap T_max={tmax_g:.0f} ns   '
          f'(frozen DriftModel would say {geo.DriftModel.from_drift_hv(drift_hv).v_um_ns:.1f} '
          f'um/ns = CLAMPED, invalid here)')

    hits = SC.load_detA_hits(run, subrun)
    g = hits[hits['eventId'] == evid]
    g = noise.flag_noise(g)
    drift = geo.DriftModel(v_um_ns=v_g, t0_ns=0.0)   # t0 set below

    planes = {}
    for pl in ('x', 'y'):
        gp = g[(g['plane'] == pl) & g['clean']]
        pos = gp['pos_mm'].to_numpy(float)
        tim = gp['time'].to_numpy(float)
        amp = gp['amplitude'].to_numpy(float)
        lines = D.plane_lines(gp)
        ext = [road_extend(l, pos, tim, amp) for l in lines]
        ext = D.distinct_lines(ext)
        planes[pl] = dict(pos=pos, time=tim, amp=amp, orig=lines, ext=ext)
        print(f'\n  plane {pl}: {len(pos)} clean hits, {len(lines)} lines')
        for i, (a, b) in enumerate(zip(lines, ext)):
            print(f'    line {i}: n {a["n_hits"]:>3}->{b["n_hits"]:<3} '
                  f'tspan {a["tspan_ns"]:>5.0f}->{b["tspan_ns"]:<6.0f}ns  '
                  f't[{b["t0_ns"]:.0f},{b["t1_ns"]:.0f}]  '
                  f'pos[{b["pos_lo_mm"]:.0f},{b["pos_hi_mm"]:.0f}]mm  '
                  f'slope {b["slope_mm_ns"]*1000:>6.1f}um/ns r2={b["r2"]:.3f}  '
                  f'| tspan/T_max={b["tspan_ns"]/tmax_g:.2f}')

    # ---- drift-span diagnostic -------------------------------------------
    print(f'\n  --- drift span vs full gap (T_max={tmax_g:.0f} ns @ {v_g:.1f} um/ns) ---')
    for pl in ('x', 'y'):
        for i, b in enumerate(planes[pl]['ext']):
            depth = b['tspan_ns'] * v_g / 1000.0
            print(f'    {pl}{i}: tspan={b["tspan_ns"]:.0f} ns -> drift depth '
                  f'{depth:.1f} mm of the {GAP_MM:.0f} mm gap '
                  f'({100*depth/GAP_MM:.0f} %)'
                  + ('   [EXCEEDS FULL GAP -> not one track]'
                     if b['tspan_ns'] > 1.15 * tmax_g else ''))

    np.save(os.path.join(OUT, 'planes.npy'), planes, allow_pickle=True) \
        if os.path.isdir(OUT) else None
    return planes, tmax_g, v_g, drift_hv, g


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
