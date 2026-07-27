#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ev1054_geom.py — drift calibration check, X/Y pairing ambiguity, and
back-projection to the target for run_58 ev1054 (Det A, drift 300 V).

Three questions:
  1. do the (road-extended) tracks match the full-gap drift time at 300 V?
  2. is the X<->Y pairing ambiguous, and can the time spans resolve it?
  3. do the resulting 3-D tracks point back at the He-3 target?

Drift calibration used here (the frozen DriftModel is INVALID at 300 V: its
bench curve starts at E=233 V/cm so every drift HV 200-700 clamps to
23.31 um/ns). We use the Garfield Ar/iC4H10 90/10 curve at E = 100 V/cm.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import extend as X, dtrack_lib as D, scan as SC  # noqa: E402
from ntof_tracking.reco import io, noise, geometry as geo  # noqa: E402

RUN, SUBRUN, EVID = 'run_58', 'sngPS_dr300_r580_036', 1054
GAP_MM = 30.0
T0_NS = 180.0        # run_58 first arrival ~ sample 3 (60 ns/sample); PLAN


def line_at(l, t):
    return l['slope_mm_ns'] * t + l['intercept_mm']


def main():
    drift_hv = 300.0
    E = drift_hv / (GAP_MM / 10.0)
    v = X.garfield_v(E)                       # um/ns
    v_mm = v / 1000.0
    tmax = GAP_MM * 1000.0 / v
    print(f'=== {RUN} ev{EVID}: drift {drift_hv:.0f} V -> E={E:.0f} V/cm ===')
    print(f'Garfield Ar/iso 90/10: v={v:.1f} um/ns -> full-gap T_max={tmax:.0f} ns')
    print(f'(frozen DriftModel: {geo.DriftModel.from_drift_hv(300).v_um_ns:.1f} '
          f'um/ns -- CLAMPED at the bench-curve edge, invalid below E=233 V/cm)\n')

    hits = SC.load_detA_hits(RUN, SUBRUN)
    g = noise.flag_noise(hits[hits.eventId == EVID])
    P = {}
    for pl in ('x', 'y'):
        gp = g[(g.plane == pl) & g.clean]
        pos = gp.pos_mm.to_numpy(float)
        tim = gp.time.to_numpy(float)
        amp = gp.amplitude.to_numpy(float)
        lines = D.plane_lines(gp)
        P[pl] = [X.road_extend(l, pos, tim, amp) for l in lines]

    # ---- 1. drift-span check -------------------------------------------
    print('--- 1. extended spans vs the full-gap ceiling ---')
    for pl in ('x', 'y'):
        for i, l in enumerate(P[pl]):
            frac = l['tspan_ns'] / tmax
            flag = '  <-- EXCEEDS full gap: end-hits are pickup, not track' \
                if frac > 1.10 else ('  <-- = full-gap crossing' if frac > 0.95 else '')
            print(f'  {pl}{i}: t[{l["t0_ns"]:6.0f},{l["t1_ns"]:6.0f}]  '
                  f'span {l["tspan_ns"]:6.0f} ns = {frac:4.2f} x T_max'
                  f'   v_implied(if full gap)={GAP_MM*1000/l["tspan_ns"]:5.1f} um/ns{flag}')

    # ---- 2. pairing ambiguity ------------------------------------------
    print('\n--- 2. X/Y pairing ambiguity (2 x-lines x 2 y-lines) ---')
    print('    a real 3-D track must have IDENTICAL x and y time spans')

    def iou(a, b):
        lo, hi = max(a['t0_ns'], b['t0_ns']), min(a['t1_ns'], b['t1_ns'])
        inter = max(0.0, hi - lo)
        union = max(a['t1_ns'], b['t1_ns']) - min(a['t0_ns'], b['t0_ns'])
        return inter / union if union > 0 else 0.0

    for hyp, (ix0, iy0, ix1, iy1) in {
            'A: x0-y0, x1-y1': (0, 0, 1, 1),
            'B: x0-y1, x1-y0': (0, 1, 1, 0)}.items():
        pa, pb = (P['x'][ix0], P['y'][iy0]), (P['x'][ix1], P['y'][iy1])
        s = 0.0
        print(f'  {hyp}')
        for k, (lx, ly) in enumerate((pa, pb)):
            d0 = abs(lx['t0_ns'] - ly['t0_ns'])
            d1 = abs(lx['t1_ns'] - ly['t1_ns'])
            s += iou(lx, ly)
            print(f'     trk{k}: x t[{lx["t0_ns"]:.0f},{lx["t1_ns"]:.0f}] vs '
                  f'y t[{ly["t0_ns"]:.0f},{ly["t1_ns"]:.0f}]  '
                  f'|dt0|={d0:5.0f} |dt1|={d1:5.0f}  IoU={iou(lx,ly):.3f}')
        print(f'     -> total IoU {s:.3f}')

    # ---- 3. back-projection to the target ------------------------------
    print('\n--- 3. back-projection (t0=%.0f ns, v=%.1f um/ns) ---' % (T0_NS, v))
    tr = geo.detector_transforms(io.load_run_config(RUN))['mx17_A']
    for hyp, pairs in {
            'A: x0-y0, x1-y1': [(0, 0), (1, 1)],
            'B: x0-y1, x1-y0': [(0, 1), (1, 0)]}.items():
        print(f'  {hyp}')
        for k, (ix, iy) in enumerate(pairs):
            lx, ly = P['x'][ix], P['y'][iy]
            t_lo = max(lx['t0_ns'], ly['t0_ns'])
            t_hi = min(lx['t1_ns'], ly['t1_ns'])
            if t_hi <= t_lo:
                print(f'     trk{k}: no time overlap -> not a 3-D track')
                continue
            p_lo = tr.local_to_global(line_at(lx, t_lo), line_at(ly, t_lo),
                                      (t_lo - T0_NS) * v_mm)
            p_hi = tr.local_to_global(line_at(lx, t_hi), line_at(ly, t_hi),
                                      (t_hi - T0_NS) * v_mm)
            d = p_hi - p_lo
            d = d / np.linalg.norm(d)
            dca_beam = geo.line_line_dist(np.zeros(3), geo.V_HAT, p_lo, d)
            dca_org = geo.point_line_dist(np.zeros(3), p_lo, d)
            b = float(d @ geo.V_HAT)
            den = 1 - b * b
            beam_y = ((float(p_lo @ geo.V_HAT) - b * float(p_lo @ d)) / den
                      if den > 1e-9 else np.nan)
            cross = geo.line_crossings(p_lo, d)
            tgt = [c['name'] for c in cross if c['arm'] == 'target']
            print(f'     trk{k}: DCA(beam axis)={dca_beam:6.0f} mm  '
                  f'DCA(origin)={dca_org:6.0f} mm  beam_y={beam_y:7.0f} mm  '
                  f'target-crossing={tgt if tgt else "NO"}')
    print('\n  He-3 gas extent along beam (y): %s mm' %
          str(np.round(geo.HE3_GAS_Y[[0, -1]], 0)) if hasattr(geo, 'HE3_GAS_Y') else '')


if __name__ == '__main__':
    main()
