#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ev1054_display.py — Det-A-ONLY global event displays for run_58 ev1054, for
BOTH X/Y pairing hypotheses, plus the 3-D closest approach ("where do the two
tracks meet?").

Piggy-backs directly on the existing tracking display library
(ntof_tracking.reco.display): we reuse plot_global_tracks (3-panel 2-D:
top-down Z-X, side Z-Y, side X-Y) and plot_global_3d unchanged, and simply
restrict the drawn geometry to arm A by patching geo.ARMS + display.GLOBAL_VIEWS
for the duration of the call. Nothing in the library is modified on disk.

Drift calibration: the frozen DriftModel is INVALID here (bench curve starts at
E=233 V/cm, so every drift HV 200-700 V clamps to 23.31 um/ns). We use the
data-driven velocity from this event -- all gap ionisation must arrive within
[t0, t0+T_max], and the fitted-track hits span 164->1839 ns, so T_max >= 1675 ns
=> v <= 17.9 um/ns. Garfield pure Ar/iso 90/10 (26.0 um/ns) is shown as the
systematic alternative.
"""
import contextlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import extend as X, dtrack_lib as D, scan as SC  # noqa: E402
from ntof_tracking.reco import io, noise, geometry as geo, display as disp  # noqa: E402

RUN, SUBRUN, EVID = 'run_58', 'sngPS_dr300_r580_036', 1054
OUT = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'detA_doubletrack', 'ev1054')
HYPS = {'A_x0y0_x1y1': [(0, 0), (1, 1)], 'B_x0y1_x1y0': [(0, 1), (1, 0)]}


TRK_COLOR = {}          # filled per hypothesis: track label -> colour


@contextlib.contextmanager
def det_A_only(mark=None):
    """Restrict the shared display library to arm A (geometry + all 3 views),
    colour the two tracks individually (the library colours by DETECTOR, and
    here both tracks are Det A), and -- if `mark` is given -- stamp the 3-D
    closest-approach point on every panel by wrapping the library's _save."""
    arms, views = geo.ARMS, disp.GLOBAL_VIEWS
    det_col, save = disp.DET_COLOR, disp._save

    def _save_marked(fig, out_dir, name):
        if mark is not None:
            axes = list(fig.axes)
            for i, ax in enumerate(axes):
                if hasattr(ax, 'zaxis'):                      # 3-D panel
                    ax.scatter(*[[c] for c in disp._p3(mark)], marker='*',
                               s=220, color='k', depthshade=False, zorder=20)
                elif i < len(disp.GLOBAL_VIEWS):              # 2-D projection
                    idx = disp.GLOBAL_VIEWS[i][1]
                    ax.plot(*[[v] for v in disp._proj(mark, idx)], marker='*',
                            ms=17, color='k', mec='w', mew=.8, zorder=20,
                            ls='none', label='3-D closest approach')
                    ax.legend(fontsize=7, loc='lower left')
        return save(fig, out_dir, name)

    try:
        geo.ARMS = ['A']
        disp.GLOBAL_VIEWS = [(t, i, {'A': m['A']}) for t, i, m in views]
        disp.DET_COLOR = dict(TRK_COLOR)
        disp._save = _save_marked
        yield
    finally:
        geo.ARMS, disp.GLOBAL_VIEWS = arms, views
        disp.DET_COLOR, disp._save = det_col, save


def closest_approach(p1, d1, p2, d2):
    """3-D closest approach of two infinite lines -> (dist, midpoint, s1, s2),
    with s1/s2 the signed distances along each direction from p1/p2."""
    w0 = np.asarray(p1, float) - np.asarray(p2, float)
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:                      # parallel
        return np.nan, None, np.nan, np.nan
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    q1, q2 = p1 + s * d1, p2 + t * d2
    return float(np.linalg.norm(q1 - q2)), 0.5 * (q1 + q2), float(s), float(t)


def build(v_um_ns, t0_ns):
    """Extended per-plane lines -> global 3-D segments for both pairings."""
    hits = SC.load_detA_hits(RUN, SUBRUN)
    g = noise.flag_noise(hits[hits.eventId == EVID])
    P = {}
    for pl in ('x', 'y'):
        gp = g[(g.plane == pl) & g.clean]
        pos, tim, amp = (gp.pos_mm.to_numpy(float), gp.time.to_numpy(float),
                         gp.amplitude.to_numpy(float))
        P[pl] = [X.road_extend(l, pos, tim, amp) for l in D.plane_lines(gp)]
    tr = geo.detector_transforms(io.load_run_config(RUN))['mx17_A']
    vm = v_um_ns / 1000.0
    f = lambda l, t: l['slope_mm_ns'] * t + l['intercept_mm']
    out = {}
    for hyp, prs in HYPS.items():
        gsegs = []
        for k, (ix, iy) in enumerate(prs):
            lx, ly = P['x'][ix], P['y'][iy]
            t_lo = max(lx['t0_ns'], ly['t0_ns'])
            t_hi = min(lx['t1_ns'], ly['t1_ns'])
            # 'det' doubles as the display label/colour key: the library keys
            # colour by detector, but both tracks here are Det A.
            seg = dict(det=f'trk{k} (x{ix}-y{iy})', eventId=EVID,
                       p_lo_local=np.array([f(lx, t_lo), f(ly, t_lo),
                                            (t_lo - t0_ns) * vm]),
                       p_hi_local=np.array([f(lx, t_hi), f(ly, t_hi),
                                            (t_hi - t0_ns) * vm]),
                       label=f'x{ix}-y{iy}')
            gsegs.append(geo.segment_to_global(seg, tr))
        out[hyp] = gsegs
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    v_data = 30000.0 / 1675.0            # data-driven upper bound, 17.9 um/ns
    t0 = 164.0
    print(f'drift calibration used: v={v_data:.1f} um/ns (data-driven), t0={t0:.0f} ns\n')
    built = build(v_data, t0)
    for hyp, gsegs in built.items():
        print(f'=== pairing {hyp} ===')
        for s in gsegs:
            print(f"  {s['label']}: dca_beam={s['dca_beam_axis_mm']:6.0f} mm  "
                  f"beam_y={s['beam_y_mm']:7.0f} mm  "
                  f"vert={s['angle_to_vertical_deg']:5.1f} deg")
        g1, g2 = gsegs
        dist, mid, s1, s2 = closest_approach(
            g1['p_lo_global'], g1['dir_global'],
            g2['p_lo_global'], g2['dir_global'])
        # is the meeting point inside the Det A gas, or out toward the beam?
        r_beam = float(np.hypot(mid[0], mid[2]))
        print(f'  --> tracks MEET: 3-D closest approach = {dist:.0f} mm')
        print(f'      meeting point (X,Y,Z) = ({mid[0]:.0f}, {mid[1]:.0f}, {mid[2]:.0f}) mm'
              f'   radius from beam axis = {r_beam:.0f} mm')
        print(f'      (He-3 target: R={geo.HE3_R_MAX:.0f} mm, y in '
              f'[{geo.HE3_GAS_Y[0]:.0f},{geo.HE3_GAS_Y[-1]:.0f}]; '
              f'Det A strip plane at Z={geo.detector_transforms(io.load_run_config(RUN))["mx17_A"].center[2]:.0f} mm)')
        TRK_COLOR.clear()
        TRK_COLOR.update({g['det']: c for g, c in zip(gsegs, ('#1f77b4', '#ff7f0e'))})
        with det_A_only(mark=mid):
            p2 = disp.plot_global_tracks(
                gsegs, f'run_58 ev1054 Det A only — pairing {hyp} '
                       f'(v={v_data:.1f} um/ns): 3-D DCA={dist:.0f} mm',
                OUT, f'ev1054_{hyp}_3panel.png')
            p3 = disp.plot_global_3d(
                gsegs, f'run_58 ev1054 Det A only — pairing {hyp} '
                       f'(v={v_data:.1f} um/ns): 3-D DCA={dist:.0f} mm',
                OUT, f'ev1054_{hyp}_3d.png')
        print(f'      -> {p2}\n      -> {p3}\n')


if __name__ == '__main__':
    main()
