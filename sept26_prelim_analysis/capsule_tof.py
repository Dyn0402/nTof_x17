#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capsule_tof.py -- where the tracks come from, against time since the flash and
against height along the beam: is the < 12 ms window junk or the thermal peak,
and where is the capsule (offset, tilt) really?

Every position here is a POINTING estimator (the band crossing: the zero of the
robust line of displacement-to-the-axis against position on the plane), never
the shape of a distribution -- see CLAUDE.md "Take positions from pointing".
The crossing is scale-free: a wrong angle scale changes the slope of the band,
not where it crosses zero.

For an A/C track the x crossing is where it meets the plane z = 0 (global x of
the source); for B/D the z crossing at x = 0.  Height = the track's global y
where it meets that plane (this one does carry the y angle scale, so it is used
only to bin).

Steps (all write CSVs to <out>/capsule_tof/):

  tof      per ToF bin and arm: tracks / ms, the x (or z) crossing, the fraction
           of tracks whose line passes within R_FOCUS of the measured source
           against a shuffled-angle null (the capsule-pointing excess), and the
           y band crossing (source height)
  tilt     source x (A, C) and z (D, B) crossing in bins of height, >= 12 ms
  beam     the beam's own capture-time profile (EAR2 flux x 1/v), per ToF bin
  g4       the same estimators on Geant4 (nose-first, centred capsule) through
           the real reconstruction (ntof_cosmics/g4_digi g4_<arm>_is2_full) and
           on its truth gap lines -- what a centred capsule looks like

    python -m sept26_prelim_analysis.capsule_tof [tof|tilt|g4|all]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from sept26_prelim_analysis import paths

CF = Path.home() / 'x17_local' / 'stage3_is2_v2_cf' / 'tracks_campaign.parquet'
FULLPASS = Path('/media/dylan/data/x17/sept26_prelim/stage3_fullpass/tracks_campaign.parquet')
G4DIGI = Path('/media/dylan/data/x17/ntof_cosmics/g4_digi')
FLUX = Path.home() / 'CLionProjects/MX17_Full_Geant/data/fluxEAR2-Ph3_in_different_units.root'
FLIGHT_M = 19.5
PRE_ACCESS = ('run_79', 'run_81')
TOF_EDGES_MS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75], float)
#: height bins for `tilt`, in true y (+y up). Before the 2026-10-14 regeneration
#: stage 3 was mirrored and these were -60..+120; mirrored with it.
Y_EDGES = np.arange(-120.0, 61.0, 20.0)
R_FOCUS = 15.0
#: the plane's distance from the axis (geometry.MM_DIST_Z + W_STRIP)
D_PLANE = 234.6
#: the axis foot in each sim arm's own u (ntof_cosmics/g4_digi/analyse.py)
FOOT_SIM = {'A': 16.35, 'C': 16.4}
HALF = 199.29
COLS = ['run', 'subrun', 'event_id', 'arm', 'gated_reco', 'gated', 'tanx', 'tany', 'x_t0', 'y_t0',
        'angle_calibrated', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 't_since_flash_ns',
        'wall_A', 'wall_B', 'wall_C', 'wall_D']
#: which global coordinate an arm's x view measures, and its normal
VIEW = {'A': ('x', 'z'), 'C': ('x', 'z'), 'B': ('z', 'x'), 'D': ('z', 'x')}


def load(src: Path, arms) -> pd.DataFrame:
    f = ds.dataset(src)
    have = set(f.schema.names)
    d = f.to_table(columns=[c for c in COLS if c in have],
                   filter=ds.field('arm').isin(list(arms))).to_pandas()
    d = d[~d.run.astype(str).isin(PRE_ACCESS)]
    gate = d.gated_reco if 'gated_reco' in d else d.gated
    ok = (gate.astype('boolean').fillna(False).astype(bool)
          & d.angle_calibrated.astype('boolean').fillna(False).astype(bool)
          & (d.tanx.abs() < 0.6) & (d.tany.abs() < 0.6) & (d.x_t0 <= 300) & (d.y_t0 <= 300))
    d = d[ok].copy()
    # true y whatever the table's vintage (COORDINATES.md): tables built before
    # the 2026-10-13 fix store y mirrored and carry no y_convention flag
    from sept26_prelim_analysis.build_tracks import y_up
    d = y_up(d, src)
    d['ms'] = d.t_since_flash_ns / 1e6
    # crossing quantities in global coordinates
    pos = np.empty(len(d))
    disp = np.empty(len(d))
    hgt = np.empty(len(d))
    for arm, (c, n) in VIEW.items():
        m = (d.arm == arm).to_numpy()
        if not m.any():
            continue
        P = {k: d.loc[m, f'p0_{k}'].to_numpy(float) for k in 'xyz'}
        D = {k: d.loc[m, f'd_{k}'].to_numpy(float) for k in 'xyz'}
        t = -P[n] / D[n]                         # line parameter at the plane n = 0
        pos[m] = P[c]                            # position on the chamber, global
        disp[m] = t * D[c]                       # displacement from there to n = 0
        hgt[m] = P['y'] + t * D['y']
    d['pos'], d['disp'], d['hgt'] = pos, disp, hgt
    d['wall'] = False
    for arm in d.arm.unique():
        m = d.arm == arm
        d.loc[m, 'wall'] = d.loc[m, f'wall_{arm}'].astype('boolean').fillna(False).astype(bool)
    d = d[np.isfinite(d.pos) & np.isfinite(d.disp) & np.isfinite(d.hgt)]
    return d.reset_index(drop=True)


def crossing(pos, disp, n_iter=4, clip=3.0):
    """Zero crossing of the robust line disp = a (pos - x_s): x where disp = 0
    (the source's coordinate along this axis), with a bootstrap-free error."""
    pos, disp = np.asarray(pos, float), np.asarray(disp, float)
    if len(pos) < 50:
        return np.nan, np.nan, len(pos)
    # disp + pos = x at the plane n = 0; a line through the source has
    # pos + disp = x_s exactly (displacement is minus the slope times distance)
    keep = np.ones(len(pos), bool)
    for _ in range(n_iter):
        a, b = np.polyfit(pos[keep], disp[keep], 1)
        r = disp - (a * pos + b)
        s = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < clip * s
    a, b = np.polyfit(pos[keep], disp[keep], 1)
    x0 = -b / a
    r = disp[keep] - (a * pos[keep] + b)
    sig = np.std(r)
    err = sig / abs(a) / np.sqrt(keep.sum()) * np.sqrt(1 + (x0 - pos[keep].mean()) ** 2
                                                       / max(pos[keep].var(), 1e-9))
    return float(x0), float(err), int(keep.sum())


def focus_excess(pos, disp, x_s, rng):
    """Fraction of lines passing within R_FOCUS of x_s (miss in the view's own
    plane, measured along the source axis), minus the shuffled-angle null."""
    miss = np.abs(pos + disp - x_s)
    f = np.mean(miss < R_FOCUS)
    sh = rng.permutation(len(disp))
    f0 = np.mean(np.abs(pos + disp[sh] - x_s) < R_FOCUS)
    return float(f), float(f0)


def tof(out: Path) -> pd.DataFrame:
    d = load(CF, ('A', 'C'))
    rng = np.random.default_rng(1)
    rows = []
    for arm, g in d.groupby('arm'):
        late = g[g.ms >= 12]
        xs, _, _ = crossing(late.pos, late.disp)
        i = np.digitize(g.ms, TOF_EDGES_MS) - 1
        for k in range(len(TOF_EDGES_MS) - 1):
            s = g[i == k]
            w = TOF_EDGES_MS[k + 1] - TOF_EDGES_MS[k]
            x0, xe, _ = crossing(s.pos, s.disp)
            f, f0 = focus_excess(s.pos.to_numpy(), s.disp.to_numpy(), xs, rng)
            # height: y band crossing (scale-free): y_plane against displacement
            # to the axis along y, the same zero-crossing estimator on (v, dy)
            sy = s[np.abs(s.pos + s.disp - xs) < R_FOCUS]
            vy = sy.p0_y.to_numpy()
            dy = (sy.hgt - sy.p0_y).to_numpy()
            y0, ye, _ = crossing(vy, dy)
            rows.append(dict(arm=arm, lo=TOF_EDGES_MS[k], hi=TOF_EDGES_MS[k + 1], n=len(s),
                             per_ms=len(s) / w, x_cross=x0, x_err=xe, x_late=xs,
                             f_focus=f, f_null=f0, excess=f - f0,
                             focus_per_ms=(f - f0) * len(s) / w,
                             wall_frac=float(s.wall.mean()) if len(s) else np.nan,
                             y_cross=y0, y_err=ye, n_y=len(sy)))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'tof.csv', index=False)
    return R


def tilt(out: Path) -> pd.DataFrame:
    rows = []
    for src, arms, tag in ((CF, ('A', 'C'), 'is2_v2_cf'), (FULLPASS, ('A', 'C', 'B', 'D'), 'fullpass')):
        d = load(src, arms)
        d = d[d.ms >= 12]
        for arm, g in d.groupby('arm'):
            xs, xe, n = crossing(g.pos, g.disp)
            rows.append(dict(chain=tag, arm=arm, axis=VIEW[arm][0], lo=-999, hi=999, n=n,
                             cross=xs, err=xe))
            sel = g[np.abs(g.pos + g.disp - xs) < 3 * R_FOCUS]
            i = np.digitize(sel.hgt, Y_EDGES) - 1
            for k in range(len(Y_EDGES) - 1):
                s = sel[i == k]
                # the crossing on ALL tracks of this height bin (not the
                # focus-selected ones, which would pull it to xs)
                h = g[(g.hgt >= Y_EDGES[k]) & (g.hgt < Y_EDGES[k + 1])]
                c, e, n = crossing(h.pos, h.disp)
                rows.append(dict(chain=tag, arm=arm, axis=VIEW[arm][0], lo=Y_EDGES[k],
                                 hi=Y_EDGES[k + 1], n=n, cross=c, err=e, n_focus=len(s)))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'tilt.csv', index=False)
    return R


def g4(out: Path) -> pd.DataFrame:
    """Band crossings on Geant4 (centred, nose-first) through the reco and on truth."""
    rows = []
    for arm in ('A', 'C'):
        parts = sorted(glob.glob(str(G4DIGI / f'g4_{arm}_is2_full' / '*.parquet')))
        R = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        for c in ('x_quality_ok', 'y_quality_ok'):
            R[c] = R[c].astype(object).fillna(False).astype(bool)
        foot = FOOT_SIM[arm]
        # truth gap line (sim local u, v at the mesh; tan along u, v)
        u_t, v_t = R.u_mesh.to_numpy(float), R.v_mesh.to_numpy(float)
        tu, tv = R.tan_u.to_numpy(float), R.tan_v.to_numpy(float)
        ok = np.isfinite(u_t) & np.isfinite(tu) & (np.abs(tu) < 0.6) & (np.abs(tv) < 0.6)
        xu, eu, nu = crossing(u_t[ok], -tu[ok] * D_PLANE)
        yv, ev, nv = crossing(v_t[ok], -tv[ok] * D_PLANE)
        rows.append(dict(arm=arm, kind='truth gap line', n=nu, u_cross=xu - foot, u_err=eu,
                         v_cross=yv, v_err=ev))
        # reco (sign conventions as analyse.py: xl = -(x_p0 - HALF); y via
        # digitise.y_view, which knows the digitiser's vintage)
        q = R.x_quality_ok & R.y_quality_ok & (R.x_tan_theta.abs() < 0.6) & (R.y_tan_theta.abs() < 0.6)
        r = R[q]
        xl = -(r.x_p0.to_numpy(float) - HALF)
        from ntof_cosmics.g4_digi.digitise import y_view
        yl = y_view(r)[0]
        # reco tan sign: fit both and keep the one that focuses (|slope| ~ 1)
        best = None
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                a = crossing(xl, -sx * r.x_tan_theta.to_numpy(float) * D_PLANE)
                b = crossing(yl, -sy * r.y_tan_theta.to_numpy(float) * D_PLANE)
                c = np.corrcoef(xl, sx * r.x_tan_theta)[0, 1] + np.corrcoef(yl, sy * r.y_tan_theta)[0, 1]
                if best is None or c > best[0]:
                    best = (c, a, b)
        _, (xu, eu, nu), (yv, ev, nv) = best
        # capsule-pointing purity, the same definition as the data's (tof step)
        rng = np.random.default_rng(1)
        txr = r.x_tan_theta.to_numpy(float) * D_PLANE
        f, f0 = max((focus_excess(xl, -sg * txr, xu, rng) for sg in (1.0, -1.0)),
                    key=lambda z: z[0])
        rows.append(dict(arm=arm, kind='reconstructed (is2, raw tan)', n=nu, u_cross=xu - foot,
                         u_err=eu, v_cross=yv, v_err=ev, f_focus=f, f_null=f0, excess=f - f0))
    G = pd.DataFrame(rows)
    G.to_csv(out / 'g4.csv', index=False)
    return G


def beam(out: Path) -> pd.DataFrame:
    """Where in ToF the beam's own captures fall: the EAR2 evaluated flux (the
    file the Geant4 gun samples) times a thin 1/v absorber, per ToF bin."""
    import uproot
    v, e = uproot.open(FLUX)['flux_n_pulse_NOisolet_100bpd'].to_numpy()
    E = np.sqrt(e[:-1] * e[1:])
    t_ms = FLIGHT_M / np.sqrt(2 * E * 1.602177e-19 / 1.674927e-27) * 1e3
    w = v * np.sqrt(0.0253 / E)
    win = t_ms >= TOF_EDGES_MS[0]
    rows = []
    for lo, hi in zip(TOF_EDGES_MS[:-1], TOF_EDGES_MS[1:]):
        m = (t_ms >= lo) & (t_ms < hi)
        rows.append(dict(lo=lo, hi=hi, capture_share=w[m].sum() / w[win].sum(),
                         per_ms=w[m].sum() / w[win].sum() / (hi - lo),
                         e_lo_meV=1e3 * E[m].min() if m.any() else np.nan,
                         e_hi_meV=1e3 * E[m].max() if m.any() else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'beam_capture.csv', index=False)
    return R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', nargs='?', default='all', choices=['tof', 'tilt', 'g4', 'beam', 'all'])
    a = ap.parse_args()
    out = paths.out('capsule_tof')
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 220)
    for name, fn in (('beam', beam), ('g4', g4), ('tof', tof), ('tilt', tilt)):
        if a.step in (name, 'all'):
            print(f'\n== {name}\n{fn(out).round(3).to_string(index=False)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
