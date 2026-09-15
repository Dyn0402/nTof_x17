#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_overhead_figure.py -- the station from above: three confirmed fans and the
capsule they cross at.  run_145.

The Athens rebuild of the MPGD2026 closing figure
(`mpgd26/make_run145_pointing.py`, ``run145_overhead_AC``).  That builder is
left as it was, because it is the record of what the Prague slide showed.  What
changed since, and is applied here:

  * **The sample is the full pass**, all three sub-runs of run_145
    (``<out>/reco_fullpass``), not the August blind pass on two.  The August
    tree is not even staged on the Windows box.
  * **The confirmed tracks are `k_arm.coincident_tracks`** -- the sample every
    published crossing and every ``k`` is measured on -- so this figure cannot
    describe a population the numbers do not.  That brings in two cuts the
    Prague figure did not have: the **hot-strip trigger cut**
    (`hot_seed_strata`, 2026-09-08; removes only D triggers) and the **charge
    window** (25-75 %, the known noise-fit and saturation tails).
  * **The pair-vertex cleaning** (`pair_vertex_imaging.z_image`, 2026-09-12):
    the slope must be measured (|tan| >= 0.08, below which `wft` piles tracks
    at tan ~ 0 and a drawn line is an invented angle) and the track must not
    sit in a noisy readout column found in this run.
  * **Inside the active area in v** (|y_local| <= 170 mm).  A fifth of arm A's
    track table rails just outside it (`det_a_scint`, 2026-09-10).
  * **The angle scale is run_145's own `k`** from `<out>/kcal`, not the
    imaging summary's ``k_phys``.
  * **Chamber D is drawn.**  It sits on the +X axis, so its fan is the one that
    locates the capsule in Z; A and C locate it in X.  **B is drawn hatched and
    carries no fan**: without field-shaping rings it has no time-to-depth
    ladder and no certified ``k``, so its tracks have no angle to draw.
  * **The capsule is drawn where it was measured**: the scale-free band
    crossings over 33 runs (`imaging_campaign/per_arm.csv`), X from the mean of
    A and C, Z from D.  The beam axis is the thin cross.

The numbers in the boxes are the band zero crossing of exactly the tracks drawn
-- scale-free, so they do not depend on ``k``.  The fans' *focus* does: with a
wrong ``k`` each fan would still cross at the right place but converge at the
wrong depth.  run_145's ``k`` sits at the peak of the runs 128-147 excursion and
arm A's wall says its scale is 33 % out, so the drawn focus is illustrative and
the crossings are the measurement.

Writes ``figures/overhead_run145.{png,pdf,csv}`` and ``overhead_run145.json``.

    X17_ROOT=D:/x17 ../.venv/Scripts/python.exe make_overhead_figure.py
    ... --no-charge-window --no-clean     # the looser Prague-like selection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.collections import LineCollection                   # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle           # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths                            # noqa: E402

# `ntof_tracking.reco` resolves the July tree through `common.beam_july_paths`,
# which reads its own variable; point it where `paths` already resolved.
os.environ.setdefault('X17_BEAM_JULY', str(paths.spell('beam_july')))

import plotstyle as P                                               # noqa: E402

RUN = 'run_145'
FAN_ARMS = ('A', 'C', 'D')
#: B gets no fan -- no certified k, so no per-track angle -- but its band
#: crossing is scale-free and exists, so it is measured and shown as that.
CROSS_ARMS = FAN_ARMS + ('B',)
PLANE_HALF = 199.29          # strip plane half-width [mm]
ACTIVE_V = 170.0             # MM_SIZE_V / 2
CAPSULE_R = 10.0
TAN_MIN_SLOPE = 0.08         # wft/reco.py
LEVER = dict(lo=30.0, hi=130.0)
D_PERP_MM = 234.6            # strips to the beam axis, along the normal
ACTIVE_U = 190.0             # MM_SIZE_U / 2
#: The He-3 gas is 80 mm long along the beam, and its nominal centroid sits
#: here in the detector frame (`y_image`).  Not at zero, and not the thing the
#: chambers actually see -- which is the point of the y figure.
GAS_CENTROID_Y = 0.8
HIST_RANGE, HIST_BINS = (-160.0, 160.0), 90


# --------------------------------------------------------------------- sample
def subruns() -> list:
    d = paths.root('out') / 'reco_fullpass' / RUN
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and p.name.startswith('stat090_'))


def k_run() -> dict:
    j = json.loads((paths.root('out') / 'kcal' / f'k_arm_{RUN}.json').read_text())
    return {a: float(v['k']) for a, v in j['arms'].items()
            if v.get('k') is not None and a in FAN_ARMS}


def noisy_columns(subs, col: str = 'x_local') -> pd.DataFrame:
    """Noisy readout columns of this run, per arm, on its whole gated track
    table -- the `z_image` finder, so the definition is the one already
    published.  ``col`` selects the view: ``x_local`` or ``y_local``."""
    from pair_vertex_imaging import z_image as ZI
    from ntof_tracking import run145_target_imaging as TI
    t = pd.concat([pd.read_parquet(
        paths.root('out') / 'stage3_fullpass' / f'tracks_{RUN}_{s}.parquet',
        columns=['arm', 'gated', col, 'x_p0']) for s in subs])
    t = t[t.gated]
    if col != 'x_local':
        _, tab = ZI.hot_columns(t[col].to_numpy(float), t.arm.to_numpy(), RUN)
        return tab.dropna(subset=['lo'])
    # the finder bins stage-3 x_local; the confirmed sample carries TI.local_x.
    # They must be one frame or the mask lands on the wrong strips.
    ok = np.isfinite(t.x_local) & np.isfinite(t.x_p0)
    dev = np.max(np.abs(t.x_local[ok] - TI.local_x(t.x_p0[ok])))
    if dev > 1e-3:
        raise AssertionError(f'stage-3 x_local is not TI.local_x ({dev:.3g} mm)')
    _, tab = ZI.hot_columns(t.x_local.to_numpy(float), t.arm.to_numpy(), RUN)
    return tab.dropna(subset=['lo'])


def arm_sample(arm, subs, k, hot_tab, charge_window=True, clean=True,
               hot_tab_y=None) -> dict:
    """Confirmed tracks of one arm, cleaned, as global lines, with the funnel.

    ``hot_tab_y`` turns on the **y** cleaning as well -- `y_image`'s
    "y-clean" tier: the y slope must be measured and the track must not sit in
    a noisy y column.  The y figure needs it; the overhead figure does not,
    because a track with no measured y slope still has a perfectly good x one.
    """
    from sept26_prelim_analysis import k_arm as K
    from sept26_prelim_analysis import source_imaging as SI

    merged = str(paths.root('out') / 'reco_fullpass' / RUN)
    tr = SI.transforms(RUN)[f'mx17_{arm}']
    cols = {c: [] for c in ('xl', 'yl', 'tx', 'ty')}
    funnel = dict(coincident=0, hotstrip_dropped=0, charge_window=0)
    saved = K.CHARGE_WINDOW
    try:
        if not charge_window:
            K.CHARGE_WINDOW = (0.0, 100.0)
        for s in subs:
            S = K.coincident_tracks(RUN, s, arm, merged)
            funnel['coincident'] += S['n_coincident'] + S['n_hotstrip']
            funnel['hotstrip_dropped'] += S['n_hotstrip']
            funnel['charge_window'] += len(S['xl'])
            for c in cols:
                cols[c].append(S[c])
    finally:
        K.CHARGE_WINDOW = saved
    xl, yl, tx, ty = (np.concatenate(cols[c]) for c in ('xl', 'yl', 'tx', 'ty'))
    foot = S['foot_x']

    rel = np.abs(tx) >= TAN_MIN_SLOPE
    hot = np.zeros(len(xl), bool)
    for lo, hi in hot_tab.loc[hot_tab.arm == arm, ['lo', 'hi']].itertuples(False):
        hot |= (xl >= lo) & (xl < hi)
    fid = np.abs(yl) <= ACTIVE_V
    keep = np.ones(len(xl), bool)
    if clean:
        keep = rel.copy()
        funnel['slope_measured'] = int(keep.sum())
        keep &= ~hot
        funnel['not_noisy_column'] = int(keep.sum())
        keep &= fid
        funnel['inside_active_v'] = int(keep.sum())
        if hot_tab_y is not None:
            keep &= np.abs(ty) >= TAN_MIN_SLOPE
            funnel['y_slope_measured'] = int(keep.sum())
            hy = np.zeros(len(yl), bool)
            for lo, hi in hot_tab_y.loc[hot_tab_y.arm == arm,
                                        ['lo', 'hi']].itertuples(False):
                hy |= (yl >= lo) & (yl < hi)
            keep &= ~hy
            funnel['not_noisy_y_column'] = int(keep.sum())
    funnel['drawn'] = int(keep.sum())

    xl, yl, tx, ty = xl[keep], yl[keep], k * tx[keep], k * ty[keep]
    # the line, exactly as TI.track_lines builds it: position at the strips,
    # one drift-gap step inward along (tan_x, tan_y)
    P0 = tr.local_to_global(xl, yl, np.zeros_like(xl))
    P1 = tr.local_to_global(xl - 30.0 * tx, yl - 30.0 * ty, np.full_like(xl, 30.0))
    D = P1 - P0
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    # scale-free crossing of exactly these tracks (k cancels; raw tan is fine)
    c = SI.crossing(xl, tx / k, dict(foot=foot, **LEVER))
    axis, mm = SI.to_global(RUN, arm, c['x0'])
    return dict(arm=arm, P0=P0, D=D, center=tr.center, k=k, funnel=funnel,
                what=tr.R @ np.array([0.0, 0.0, 1.0]),
                xl=xl, tan_raw=tx / k, yl=yl, tan_y_raw=ty / k,
                crossing=dict(axis=axis, mm=mm, err=c['err'], n=c['n']),
                y_crossing=y_crossing(yl, ty / k))


def y_crossing(yl, ty, n_boot: int = 200, seed: int = 1) -> dict:
    """The y band's zero crossing: the ensemble source height, scale-free.

    `y_image`'s own construction, and deliberately not `source_imaging.crossing`
    -- that one fits over a lever window measured from a perpendicular foot,
    which is an x-plane idea.  In y there is no pinwheel and no foot: a track
    from a source at ``y_s`` has ``tan_y * L = s (y_s - v)``, so a robust line
    of ``tan_y * L`` against ``v`` crosses zero at ``y_s`` whatever the scale
    ``s`` is.  ``s`` itself is returned because it is the bracket, not a
    calibration: the band pulls it below 1 and the focus scan above 1, and
    `y_image` adopts neither.
    """
    from ntof_tracking import run145_target_imaging as TI
    # q is the y DISPLACEMENT from the plane to the target, and it is MINUS
    # tan_y * L: `track_lines` applies the in-plane sign to the position and
    # leaves the tan as reconstructed, so one step inward moves y by -tan_y.
    # Get this backwards and the crossing still comes out right (it is a ratio)
    # while the band slope reports the wrong sign -- which is how it was caught.
    q, v = -np.asarray(ty, float) * D_PERP_MM, np.asarray(yl, float)
    ok = np.isfinite(q) & np.isfinite(v)
    q, v = q[ok], v[ok]
    if len(v) < 200:
        return dict(n=int(len(v)), mm=float('nan'), err=float('nan'),
                    scale=float('nan'))
    sl, ic = TI._robust_line(v, q)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(v), len(v))
        s2, i2 = TI._robust_line(v[i], q[i])
        bs.append(-i2 / s2 if s2 else np.nan)
    return dict(n=int(len(v)), mm=float(-ic / sl) if sl else float('nan'),
                err=float(np.nanstd(bs)), scale=float(-sl))


def dead_spans(subs) -> dict:
    """Where each plane is dead, in its own local x -- found from the full
    occupancy by `source_imaging.dead_ranges`, not transcribed.  Drawn because
    chamber D's is a quarter of its plane and it is the hole in D's fan."""
    from sept26_prelim_analysis import source_imaging as SI
    from ntof_tracking import run145_target_imaging as TI
    merged = str(paths.root('out') / 'reco_fullpass' / RUN)
    out = {}
    for arm in ('A', 'B', 'C', 'D'):
        occ = SI.plane_occupancy(RUN, subs, arm, merged)
        out[arm] = [tuple(sorted(TI.IN_PLANE_SIGN
                                 * (np.array([lo, hi]) - TI.STRIP_MAP_HALF)))
                    for lo, hi in SI.dead_ranges(occ)]
    return out


def plastics(trs) -> dict:
    """The plastic bars, per arm, **from the DAQ's own run_config.json**.

    The config places every scintillator in the GLOBAL frame, so this needs no
    depth constant and no offset convention: each bar's centre is projected
    onto the arm's own (u, outward) axes and that is where it is drawn.  It
    also self-checks against the convention the coincidence uses (below), and
    that is what makes it better than the constants in `geometry.py`, whose
    active-PVT mid-plane sits 2.5 mm short of the surveyed bar centre -- it
    carries the sim's 20 mm bar and the config's own description says 25 mm.

    Returns per arm: ``u`` the two bar centres in the arm's in-plane
    coordinate, ``depth`` their distance past the strip plane, ``half_u``, and
    the (Z, X) corner lists to draw.
    """
    from ntof_tracking.reco import geometry as G
    from ntof_tracking import run145_target_imaging as TI
    cfg = json.loads((paths.root('runs') / RUN / 'run_config.json').read_text())
    acc = {}
    for det in cfg.get('detectors', []):
        name = det.get('name', '')
        if not name.startswith('plastic_'):
            continue
        arm = name.split('_')[1]
        if arm not in trs_arms(trs):
            continue
        c = det['det_center_coords']
        acc.setdefault(arm, []).append(
            np.array([c['x'], c.get('y', 0.0), c['z']], float))

    out = {}
    for arm, centres in acc.items():
        tr = trs[f'mx17_{arm}']
        uhat = tr.R @ np.array([1.0, 0.0, 0.0])
        what = tr.R @ np.array([0.0, 0.0, 1.0])       # outward, past the strips
        u = sorted(float((c - tr.center) @ uhat) for c in centres)
        depth = float(np.mean([(c - tr.center) @ what for c in centres]))
        # Two checks, and they are the reason to read the config at all.
        # The bars are centred on the MM, so in the plane's own coordinate
        # their midpoint is zero -- and it is the BEAM-AXIS foot that sits a
        # pinwheel away from it, which is the offset every pointing quantity
        # in this analysis is measured from.
        mid = 0.5 * (u[0] + u[1])
        if abs(mid) > 1.0:
            raise AssertionError(f'{arm}: plastic midpoint is {mid:.2f} mm, '
                                 f'not on the MM centre')
        if abs(abs(u[1] - u[0]) / 2 - G.PLASTIC_U_OFFSET) > 1.0:
            raise AssertionError(
                f'{arm}: bar centres {u} are not '
                f'+-{G.PLASTIC_U_OFFSET} mm apart from the MM centre')
        _ = TI.PINWHEEL[arm]
        # 20 mm thick, about the surveyed centre: the sim's 2026-07-20
        # correction ("20 mm, not 25"), which postdates the config's text
        half_t = G.PLASTIC_THICK / 2.0
        polys = [_box(tr, (uc - G.PLASTIC_HALF_U, uc + G.PLASTIC_HALF_U),
                      (0, 0), (depth - half_t, depth + half_t), OVERHEAD)
                 for uc in u]
        side = _box(tr, (0, 0), (-G.PLASTIC_HALF_V, G.PLASTIC_HALF_V),
                    (depth - half_t, depth + half_t), SIDE)
        out[arm] = dict(u=u, depth=depth, half_u=G.PLASTIC_HALF_U, polys=polys,
                        side=side)
    return out


def save(fig, base: str) -> None:
    """Write PNG + PDF, via a temp file in the same directory.

    On Windows an image viewer holding the PNG open makes a direct rewrite fail
    with ``OSError: [Errno 22] Invalid argument`` -- after the figure has been
    built, which on this analysis is minutes of work thrown away.  Writing
    beside it and renaming over the top succeeds while the old file is open,
    and if even that is refused the new file is kept and named.
    """
    for ext in ('png', 'pdf'):
        dst, tmp = f'{base}.{ext}', f'{base}.new.{ext}'
        fig.savefig(tmp, bbox_inches=fig.bbox_inches, pad_inches=0.0)
        try:
            os.replace(tmp, dst)
        except OSError as exc:
            print(f'  !! could not replace {dst} ({exc}); left it at {tmp}')
    plt.close(fig)
    print(f'  -> {base}.png')


def trs_arms(trs) -> tuple:
    return tuple(k.split('_')[1] for k in trs)


def sipm_wall(trs) -> dict:
    """The SiPM trigger wall, per arm, from the same run_config survey.

    16 instrumented bars of 25 x 500 mm, **centred on the structure** rather
    than on the MM -- which is why the wall carries the pinwheel term and the
    plastics do not (`run145_target_imaging`).  Read from the config so that
    stays a fact about the apparatus and not a constant to maintain: the
    surveyed strips-to-wall distance comes out 97.4 mm, the lever
    `det_a_scint` measures the angle scale on.

    The active scintillator is 3 mm thick, which is a third of a pixel at this
    scale, so the bars are DRAWN thicker; their u segmentation, which is what a
    reader is meant to see, is exact.
    """
    from ntof_tracking.reco import geometry as G
    cfg = json.loads((paths.root('runs') / RUN / 'run_config.json').read_text())
    acc = {}
    for det in cfg.get('detectors', []):
        name = det.get('name', '')
        if not name.startswith('sipm_'):
            continue
        arm = name.split('_')[1]
        if arm not in trs_arms(trs):
            continue
        c = det['det_center_coords']
        acc.setdefault(arm, []).append(
            np.array([c['x'], c.get('y', 0.0), c['z']], float))

    out = {}
    for arm, centres in acc.items():
        tr = trs[f'mx17_{arm}']
        uhat = tr.R @ np.array([1.0, 0.0, 0.0])
        what = tr.R @ np.array([0.0, 0.0, 1.0])
        u = sorted(float((c - tr.center) @ uhat) for c in centres)
        depth = float(np.mean([(c - tr.center) @ what for c in centres]))
        dt = (depth - SCINT_WALL_MM / 2, depth + SCINT_WALL_MM / 2)

        def rect(u0, u1):
            return _box(tr, (u0, u1), (0, 0), dt, OVERHEAD)

        # each bar inset by a hair so the bars read as separate: a drawn gap,
        # 1.5 mm a side, not a physical one (the wrapped bars touch)
        polys = [rect(uc - G.SIPM_BAR_W / 2 + 1.5, uc + G.SIPM_BAR_W / 2 - 1.5)
                 for uc in u]
        # The 16 bars are read out in FOUR GROUPS OF FOUR -- that is the
        # granularity the coincidence actually has (`_wall_seg_u`), so it is
        # the granularity the figure should show.
        groups = [rect(u[i] - G.SIPM_BAR_W / 2 + 0.75,
                       u[i + 3] + G.SIPM_BAR_W / 2 - 0.75)
                  for i in range(0, len(u) - 3, 4)]
        side = _box(tr, (0, 0), (-G.SIPM_HALF_V, G.SIPM_HALF_V), dt, SIDE)
        out[arm] = dict(u=u, depth=depth, n_bars=len(u), polys=polys,
                        groups=groups, half_v=G.SIPM_HALF_V, side=side)
    return out


def on_bar(d, bars, scale=1.0) -> float:
    """Fraction of the drawn tracks whose extrapolation lands on a bar.

    ``scale`` = 1 is the raw tan the coincidence actually tested; ``scale`` =
    k is what the figure draws.  The difference between the two IS the angle
    scale, seen at a 190 mm lever.
    """
    u = d['xl'] + bars['depth'] * scale * d['tan_raw']
    hit = np.zeros(len(u), bool)
    for uc in bars['u']:
        hit |= np.abs(u - uc) <= bars['half_u']
    return float(hit.mean())


def campaign_capsule() -> dict:
    """Where the capsule is, from 33 runs of scale-free band crossings."""
    pa = pd.read_csv(paths.root('out') / 'imaging_campaign' / 'per_arm.csv'
                     ).set_index('arm')
    return dict(X=0.5 * (pa.loc['A', 'median_mm'] + pa.loc['C', 'median_mm']),
                Z=float(pa.loc['D', 'median_mm']),
                X_sd=float(np.hypot(pa.loc['A', 'std_mm'], pa.loc['C', 'std_mm']) / 2),
                Z_sd=float(pa.loc['D', 'std_mm']), n_runs=int(pa.loc['A', 'n_runs']))


# --------------------------------------------------------------------- figure
def _at_target(d):
    """A/C: global X where the line crosses Z = 0.  D: global Z at X = 0."""
    P0, D = d['P0'], d['D']
    if d['arm'] in ('A', 'C'):
        return P0[:, 0] - P0[:, 2] / D[:, 2] * D[:, 0]
    return P0[:, 2] - P0[:, 0] / D[:, 0] * D[:, 2]


def _fan(ax, d, color, alpha, depth=None):
    """Each track from the plastic bar it fired, through the strip plane, to
    its closest approach to the beam axis -- overhead frame (Z across, X up).

    The outward half is the same line, extrapolated: the track is measured in
    the chamber, and the plastic is 186-190 mm behind it.  Nothing is fitted to
    the plastic; the bars are where the selection already said the track went.
    """
    P0, D = d['P0'], d['D']
    if depth:
        # D points inward (toward the target), so step back along -D until the
        # outward distance from the plane is the surveyed strips-to-plastic
        dn = D @ d['what']
        t = np.where(np.abs(dn) > 1e-6, depth / np.abs(dn), 0.0)
        P0 = P0 - t[:, None] * D
    p, dd = P0[:, [0, 2]], D[:, [0, 2]]
    q = d['P0'][:, [0, 2]]
    s = np.clip(-np.einsum('ij,ij->i', q, dd) / np.einsum('ij,ij->i', dd, dd),
                0.0, None)
    end = q + s[:, None] * dd
    seg = np.stack([p[:, ::-1], end[:, ::-1]], axis=1)
    ax.add_collection(LineCollection(seg, colors=color, linewidths=0.35,
                                     alpha=alpha, zorder=3))


# --------------------------------------------------------- drawn to scale
#: Everything in both views is drawn at its real size, from these and from the
#: run_config survey -- nothing is widened to be visible.
DRIFT_GAP_MM = 30.1       # mylar front -> strip plane (`geometry.W_STRIP`)
SCINT_WALL_MM = 3.0       # active SiPM scintillator (`SIPM_SCINT_W0/W1`)
#: the chambers' serial names, for the measured active-area record
DET_OF_ARM = {'A': 'mx17_3', 'B': 'mx17_2', 'C': 'mx17_6', 'D': 'mx17_7'}
OVERHEAD, SIDE = (2, 0), (2, 1)     # (horizontal, vertical) global axes


def _local_poly(tr, corners, proj):
    """(u, v, d) corners in one arm's frame -> 2-D points in a view.

    ``d`` is measured OUTWARD from the strip plane, so the drift gap is
    d in [-30.1, 0] (it sits on the target side: the mylar faces the capsule)
    and the wall and plastics are at positive d.  Built from the same
    transform the tracks use, so hardware and tracks cannot disagree about
    where anything is.
    """
    uhat = tr.R @ np.array([1.0, 0.0, 0.0])
    vhat = tr.R @ np.array([0.0, 1.0, 0.0])
    what = tr.R @ np.array([0.0, 0.0, 1.0])
    pts = [tr.center + u * uhat + v * vhat + d * what for u, v, d in corners]
    return [(p[proj[0]], p[proj[1]]) for p in pts]


def _box(tr, u, v, d, proj):
    """A rectangle spanning [u0,u1] x [v0,v1] x [d0,d1], seen in ``proj``."""
    (u0, u1), (v0, v1), (d0, d1) = u, v, d
    if proj == OVERHEAD:
        c = [(u0, 0, d0), (u1, 0, d0), (u1, 0, d1), (u0, 0, d1)]
    else:
        c = [(0, v0, d0), (0, v1, d0), (0, v1, d1), (0, v0, d1)]
    return _local_poly(tr, c, proj)


def efficient_v(arm) -> tuple:
    """The chamber's measured efficient v span, in its own local coordinate.

    `common.mx17_active_area`: the Y strips are passivated ~18-20 mm at each
    end, measured per chamber on the June bench and confirmed on n_TOF beam
    data (run_79, 1-2 mm).  The strip pattern itself runs the full 398.6 mm.
    """
    from common.mx17_active_area import TRUE_ACTIVE_BY_DET
    from sept26_prelim_analysis.build_tracks import IN_PLANE_SIGN_Y
    from ntof_tracking import run145_target_imaging as TI
    y0, y1 = TRUE_ACTIVE_BY_DET[DET_OF_ARM[arm]]['y']
    return tuple(sorted(IN_PLANE_SIGN_Y * (np.array([y0, y1])
                                           - TI.STRIP_MAP_HALF)))


def _plane(ax, arm, tr, dead=(), hatched=False):
    """The chamber from above: its 30.1 mm drift gap over the full strip width,
    with dead readout marked across that gap."""
    kw = (dict(facecolor='#eceff3', edgecolor=P.MUTED, lw=0.9, hatch='///')
          if hatched else dict(facecolor=P.LINE, edgecolor=P.MUTED, lw=0.8))
    ax.add_patch(Polygon(_box(tr, (-PLANE_HALF, PLANE_HALF), (0, 0),
                              (-DRIFT_GAP_MM, 0.0), OVERHEAD),
                         closed=True, zorder=4, **kw))
    for lo, hi in dead:
        ax.add_patch(Polygon(_box(tr, (lo, hi), (0, 0), (-DRIFT_GAP_MM, 0.0),
                                  OVERHEAD), closed=True,
                             facecolor=P.BAND_DEAD, edgecolor='none',
                             alpha=0.85, zorder=6))


def figure(arms, cap, dead, bars, sipm, out_base):
    from sept26_prelim_analysis import source_imaging as SI
    trs = SI.transforms(RUN)
    col = P.DET_COLOR

    # The plastics reach ~200 mm past each strip plane, so the view has to hold
    # 434 mm on the three instrumented sides; B's side needs only its own plane.
    XLIM, YLO, YHI = 472.0, -300.0, 452.0
    fig = plt.figure(figsize=(13.35, 6.22))        # the slide's 2.15:1 hole
    H = 0.955
    wl = H * 6.22 / 13.35 * (2 * XLIM) / (YHI - YLO)
    axL = fig.add_axes([0.004, 0.02, wl, H])
    x0 = 0.004 + wl + 0.070
    wr = 0.985 - x0
    axX = fig.add_axes([x0, 0.585, wr, 0.355])
    axZ = fig.add_axes([x0, 0.115, wr, 0.355])

    # ---------------------------------------------------------- the overhead
    axL.set_xlim(-XLIM, XLIM)
    axL.set_ylim(YLO, YHI)
    axL.set_aspect('equal')
    axL.axis('off')
    for a in ('A', 'B', 'C', 'D'):
        _plane(axL, a, trs[f'mx17_{a}'], dead.get(a, ()), hatched=(a == 'B'))
    for a in FAN_ARMS:
        d = arms[a]
        # the two layers this arm's tracks had to hit to be in the sample at
        # all: the segmented SiPM wall first, the plastic bars behind it
        for poly in sipm[a]['groups']:         # white behind each group ...
            axL.add_patch(Polygon(poly, closed=True, facecolor='white',
                                  edgecolor='none', zorder=5))
        for poly in sipm[a]['polys']:          # ... the bars, spaced ...
            axL.add_patch(Polygon(poly, closed=True, facecolor=col[a],
                                  edgecolor='none', zorder=6))
        for poly in sipm[a]['groups']:         # ... one read-out group of four
            axL.add_patch(Polygon(poly, closed=True, facecolor='none',
                                  edgecolor='black', lw=0.35, zorder=7))
        for poly in bars[a]['polys']:
            axL.add_patch(Polygon(poly, closed=True, facecolor=col[a],
                                  edgecolor=col[a], lw=0.8, alpha=0.22,
                                  zorder=4))
        _fan(axL, d, col[a], float(np.clip(240.0 / len(d['P0']), 0.03, 0.12)),
             depth=bars[a]['depth'])

    axL.plot([-26, 26], [0, 0], color=P.MUTED, lw=0.9, zorder=8)
    axL.plot([0, 0], [-26, 26], color=P.MUTED, lw=0.9, zorder=8)
    axL.add_patch(Circle((cap['Z'], cap['X']), CAPSULE_R, facecolor='none',
                         edgecolor=P.TRACK, lw=2.2, zorder=9))
    # the capsule label lives in B's empty half, where nothing is drawn
    axL.annotate('inferred ³He position', xy=(cap['Z'], cap['X'] - CAPSULE_R),
                 xytext=(0, -150), fontsize=11.5, color=P.INK,
                 fontweight='bold', ha='center', va='top',
                 bbox=dict(facecolor=P.SURFACE, edgecolor='none', alpha=0.85,
                           pad=2.0),
                 arrowprops=dict(arrowstyle='-', color=P.INK, lw=1.0,
                                 shrinkB=4), zorder=10)

    lab = dict(fontsize=13, fontweight='bold', ha='center', va='center')
    axL.text(465, 0, 'arm A', rotation=-90, color=col['A'], **lab)
    axL.text(-465, 0, 'arm C', rotation=90, color=col['C'], **lab)
    axL.text(-300, 428, 'arm D', color=col['D'], **lab)
    axL.text(0, -252, 'arm B · no drift field, no angle', color=P.MUTED,
             **{**lab, 'fontsize': 11, 'va': 'top'})
    axL.plot([-464, -364], [-292, -292], color=P.INK, lw=2.4, zorder=10,
             solid_capstyle='butt')
    axL.text(-414, -287, '100 mm', ha='center', va='bottom', fontsize=10,
             color=P.INK)
    n_all = sum(len(arms[a]['P0']) for a in FAN_ARMS)
    axL.text(468, 300, f'{n_all:,} confirmed tracks\nrun_145, full pass',
             ha='right', va='top', fontsize=10, color=P.MUTED,
             fontweight='bold', linespacing=1.3)

    # the dead marks, named once, with arrows to two of D's
    if dead.get('D'):
        uhat = trs['mx17_D'].R @ np.array([1.0, 0.0, 0.0])
        mids = sorted((trs['mx17_D'].center + 0.5 * (lo + hi) * uhat)[2]
                      for lo, hi in dead['D'])
        tip = (-330.0, 330.0)
        x_strips = float(trs['mx17_D'].center[0])     # the plane's outer face
        for z in (mids[0], mids[len(mids) // 2]):
            axL.annotate('', xy=(z, x_strips + 2.0), xytext=tip,
                         arrowprops=dict(arrowstyle='->', color=P.BAND_DEAD,
                                         lw=1.0, shrinkA=6, shrinkB=2),
                         zorder=10)
        axL.text(tip[0], tip[1] + 10, 'dead readout', ha='center', va='bottom',
                 fontsize=11, color=P.BAND_DEAD, fontweight='bold', zorder=10)

    # ------------------------------------------------ what each fan measures
    rows = {}

    def panel(ax, arm_list, centre, coord):
        ax.axvspan(centre - CAPSULE_R, centre + CAPSULE_R, color=P.TRACK,
                   alpha=0.13, zorder=1)
        ax.axvline(0, color=P.MUTED, lw=0.9, ls=(0, (5, 4)), zorder=2)
        lines = []
        for a in arm_list:
            v = _at_target(arms[a])
            h, e = np.histogram(v, bins=HIST_BINS, range=HIST_RANGE)
            c = 0.5 * (e[:-1] + e[1:])
            rows.setdefault('centre_mm', c)
            rows[f'{a}_{coord}'] = h
            ax.step(c, h, where='mid', color=col[a], lw=2.0, zorder=4,
                    label=f'arm {a}  ({len(v):,})')
            cr = arms[a]['crossing']
            lines.append(f"{a}  {cr['mm']:+.1f} ± {cr['err']:.1f} mm")
        ax.set_xlim(*HIST_RANGE)
        ax.set_ylim(0, None)
        ax.set_ylabel('tracks / 3.6 mm')
        ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.92,
                  facecolor=P.SURFACE, edgecolor='none')
        ax.grid(alpha=0.3)
        P.strip(ax)
        ax.text(0.985, 0.94, f'inferred ³He {coord} position:\n'
                + '\n'.join(lines),
                transform=ax.transAxes, ha='right', va='top', fontsize=10.5,
                color=P.INK, linespacing=1.45,
                bbox=dict(facecolor=P.SURFACE, edgecolor=P.LINE, pad=4.0))
        ax.text(0.985, 0.60, 'shaded: the 10 mm bore', transform=ax.transAxes,
                ha='right', va='top', fontsize=9.5, color=P.TRACK,
                fontweight='bold')

    panel(axX, ('A', 'C'), cap['X'], 'X')
    axX.set_xlabel('back-projected to the target,  global X  [mm]', labelpad=2)
    panel(axZ, ('D',), cap['Z'], 'Z')
    axZ.set_xlabel('back-projected to the target,  global Z  [mm]', labelpad=2)
    axX.text(0.5, 1.03, 'A and C see the capsule in X,  D sees it in Z',
             transform=axX.transAxes, ha='center', va='bottom', fontsize=10.5,
             color=P.TRACK, fontweight='bold')

    save(fig, out_base)
    pd.DataFrame(rows).to_csv(f'{out_base}.csv', index=False)


# ------------------------------------------------------------- the y figure
def _at_target_y(d):
    """Global Y where the track crosses the target plane -- Z = 0 for A and C,
    X = 0 for D.  Identically ``v + tan_y * 234.6``, which is what the y band
    is a fit to; computed from the 3D line so the two cannot drift apart."""
    P0, D = d['P0'], d['D']
    t = (-P0[:, 2] / D[:, 2] if d['arm'] in ('A', 'C') else -P0[:, 0] / D[:, 0])
    return P0[:, 1] + t * D[:, 1]


def _capsule_profile():
    """The He-3 gas polycone in the (transverse, Y) plane, as a closed path."""
    from ntof_tracking.reco import geometry as G
    y, r = G.HE3_GAS_Y, G.HE3_GAS_R
    return (list(r) + list(-r[::-1]), list(y) + list(y[::-1]))


def figure_y(arms, bars, sipm, out_base, norm=True):
    """The station from the SIDE: the beam axis up the page, arms A and C
    across it, and the 80 mm gas column the tracks are supposed to come from."""
    from sept26_prelim_analysis import source_imaging as SI
    trs = SI.transforms(RUN)
    col = P.DET_COLOR

    # tall enough for the full 500 mm SiPM wall, the tallest thing in view
    XLIM, YLIM = 472.0, 268.0

    # THE CAPSULE HEIGHT IS A CALIBRATION.  Its placement along the beam was
    # never surveyed; the three chambers measure it.  Inverse-variance mean of
    # their scale-free y band crossings, then the polycone is placed so its gas
    # centroid sits there -- the same offset for every chamber.
    yc = np.array([arms[a]['y_crossing']['mm'] for a in ('A', 'C', 'D')])
    ye = np.array([arms[a]['y_crossing']['err'] for a in ('A', 'C', 'D')])
    w = 1.0 / ye ** 2
    y_meas, y_meas_err = float((w * yc).sum() / w.sum()), float(w.sum() ** -0.5)
    y0 = y_meas - GAS_CENTROID_Y                  # shift applied to the CAD
    fig = plt.figure(figsize=(13.35, 6.22))
    wl = 0.545
    hl = wl * 13.35 / 6.22 * (2 * YLIM) / (2 * XLIM)
    axL = fig.add_axes([0.004, 0.5 - hl / 2, wl, hl])
    x0 = 0.004 + wl + 0.070
    wr = 0.985 - x0
    axY = fig.add_axes([x0, 0.135, wr, 0.76])

    axL.set_xlim(-XLIM, XLIM)
    axL.set_ylim(-YLIM, YLIM)
    axL.set_aspect('equal')
    axL.axis('off')

    # the gas column's y span, all the way across: the fans have to narrow into
    # THIS, and 340 mm of plane arriving at an 80 mm band is the whole picture
    from ntof_tracking.reco import geometry as G
    axL.axhspan(float(G.HE3_GAS_Y[0]) + y0, float(G.HE3_GAS_Y[-1]) + y0,
                color=P.TRACK, alpha=0.08, zorder=1)
    # the beam, straight up the page through the capsule
    axL.plot([0, 0], [-YLIM + 14, YLIM - 14], color=P.MUTED, lw=1.0,
             ls=(0, (5, 4)), alpha=0.6, zorder=1)
    for a in ('A', 'C'):
        tr = trs[f'mx17_{a}']
        # The chamber: its 30.1 mm drift gap over the full 398.6 mm of strips,
        # with the measured passivated ends -- where the strips exist and the
        # chamber is blind -- shaded darker.
        gap = (-DRIFT_GAP_MM, 0.0)
        axL.add_patch(Polygon(_box(tr, (0, 0), (-PLANE_HALF, PLANE_HALF), gap,
                                   SIDE), closed=True, facecolor=P.LINE,
                              edgecolor=P.MUTED, lw=0.8, zorder=4))
        v0, v1 = efficient_v(a)
        for span in ((-PLANE_HALF, v0), (v1, PLANE_HALF)):
            axL.add_patch(Polygon(_box(tr, (0, 0), span, gap, SIDE),
                                  closed=True, facecolor=P.MUTED,
                                  edgecolor='none', alpha=0.55, zorder=5))
        # Seen from the side each layer is one rectangle: the wall's 16 bars
        # are segmented in u, which runs into the page here.
        # both scintillator layers ON TOP of the tracks, outlined in black
        from matplotlib.colors import to_rgba
        for poly in (sipm[a]['side'], bars[a]['side']):
            axL.add_patch(Polygon(poly, closed=True,
                                  facecolor=to_rgba(col[a], 0.75),
                                  edgecolor='black', lw=0.7, zorder=7))
        d = arms[a]
        # plastic -> strips -> the beam-axis plane, in (Z, Y)
        dn = d['D'] @ d['what']
        t0 = np.where(np.abs(dn) > 1e-6, bars[a]['depth'] / np.abs(dn), 0.0)
        start = d['P0'] - t0[:, None] * d['D']
        t1 = -d['P0'][:, 2] / d['D'][:, 2]
        end = d['P0'] + t1[:, None] * d['D']
        seg = np.stack([start[:, [2, 1]], end[:, [2, 1]]], axis=1)
        axL.add_collection(LineCollection(
            seg, colors=col[a], linewidths=0.35, zorder=3,
            alpha=float(np.clip(340.0 / len(seg), 0.04, 0.14))))

    zz, yy = _capsule_profile()
    yy = np.asarray(yy) + y0
    axL.fill(zz, yy, facecolor=P.TRACK, alpha=0.35, edgecolor=P.TRACK, lw=1.8,
             zorder=9)
    axL.text(0, float(yy.max()) + 12, '³He capsule (measured)', ha='center',
             va='bottom', fontsize=10.5, color=P.TRACK, fontweight='bold',
             zorder=11, bbox=dict(facecolor=P.SURFACE, edgecolor='none',
                                  alpha=0.8, pad=2.0))

    lab = dict(fontsize=13, fontweight='bold', va='top')
    axL.text(468, YLIM - 4, 'arm A', color=col['A'], ha='right', **lab)
    axL.text(-468, YLIM - 4, 'arm C', color=col['C'], ha='left', **lab)
    axL.plot([-464, -364], [-YLIM + 10, -YLIM + 10], color=P.INK, lw=2.4,
             zorder=10, solid_capstyle='butt')
    axL.text(-414, -YLIM + 15, '100 mm', ha='center', va='bottom', fontsize=10,
             color=P.INK)

    # The panel is to scale and therefore short: everything that is a caption
    # rather than a thing in the apparatus lives in the margins above and below
    # it, where it costs no drawing area.
    n_all = sum(len(arms[a]['P0']) for a in ('A', 'C'))
    top, bot = 0.5 + hl / 2 + 0.025, 0.5 - hl / 2 - 0.030
    fig.text(0.004, top, 'seen from the side · the beam runs up the page',
             ha='left', va='bottom', fontsize=10, color=P.MUTED)
    fig.text(0.004 + wl, top,
             f'{n_all:,} confirmed tracks · run_145, full pass', ha='right',
             va='bottom', fontsize=10, color=P.MUTED, fontweight='bold')
    fig.text(0.004, bot, 'everything to scale  ·  dark ends of each chamber: '
             'passivated readout', ha='left', va='top', fontsize=10,
             color=P.MUTED)

    # ------------------------------------------------------- what they measure
    from ntof_tracking.reco import geometry as G
    rows = {}
    ylo, yhi = float(G.HE3_GAS_Y[0]) + y0, float(G.HE3_GAS_Y[-1]) + y0

    # one panel, all three chambers: the agreement IS the result, and it reads
    # better with the three curves on one axis than in three boxes
    axY.axvspan(ylo, yhi, color=P.TRACK, alpha=0.13, zorder=1)
    lines = []
    for a in ('A', 'C', 'D'):
        v = _at_target_y(arms[a])
        h, e = np.histogram(v, bins=HIST_BINS, range=HIST_RANGE)
        c = 0.5 * (e[:-1] + e[1:])
        rows.setdefault('centre_mm', c)
        rows[f'{a}_Y'] = h
        axY.step(c, h / h.sum() if norm else h, where='mid', color=col[a],
                 lw=2.0, zorder=4, label=f'arm {a}  ({len(v):,})')
        cr = arms[a]['y_crossing']
        lines.append(f"{a}  {cr['mm']:+.1f} ± {cr['err']:.1f} mm")
    axY.axvline(y_meas, color=P.INK, lw=1.6, ls=(0, (6, 4)), zorder=5)
    axY.set_xlim(*HIST_RANGE)
    axY.set_ylim(0, 1.38 * axY.get_ylim()[1])   # headroom for the box
    axY.set_ylabel('fraction of tracks / 3.6 mm' if norm
                   else 'tracks / 3.6 mm')
    axY.set_xlabel('back-projected to the target,  global Y  [mm]', labelpad=2)
    leg = axY.legend(loc='upper left', fontsize=11, frameon=True,
                     framealpha=0.92, facecolor=P.SURFACE, edgecolor='none')
    leg.set_zorder(12)
    axY.grid(alpha=0.3)
    P.strip(axY)
    axY.text(0.985, 0.94, 'inferred ³He Y position:\n' + '\n'.join(lines)
             + f'\ncombined:  {y_meas:+.1f} ± {y_meas_err:.1f} mm',
             transform=axY.transAxes, ha='right', va='top', fontsize=10.5,
             color=P.INK, linespacing=1.45, zorder=12,
             bbox=dict(boxstyle='round,pad=0.45', facecolor=P.SURFACE,
                       edgecolor='none', alpha=0.8))
    axY.text(0.015, 0.36, 'shaded:\nthe 80 mm gas column',
             transform=axY.transAxes, ha='left', va='top', fontsize=9.5,
             color=P.TRACK, fontweight='bold', linespacing=1.35)
    axY.text(0.5, 1.02, 'all three chambers agree on the capsule height',
             transform=axY.transAxes, ha='center', va='bottom',
             fontsize=10.5, color=P.TRACK, fontweight='bold')

    save(fig, out_base)
    pd.DataFrame(rows).to_csv(f'{out_base}.csv', index=False)
    return dict(capsule_gas_centroid_y_mm=y_meas, err_mm=y_meas_err,
                shift_applied_to_cad_mm=y0,
                nominal_cad_centroid_y_mm=GAS_CENTROID_Y,
                method='inverse-variance mean of the A, C, D y band crossings')


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-charge-window', action='store_true',
                    help='keep the whole charge range of the confirmed sample')
    ap.add_argument('--no-clean', action='store_true',
                    help='skip slope-measured / noisy-column / active-v cuts')
    ap.add_argument('--out', default=str(HERE / 'figures' / 'overhead_run145'))
    ap.add_argument('--projection', choices=('overhead', 'y', 'both'),
                    default='overhead',
                    help='overhead: the XZ fans (the slide-42 figure).  '
                         'y: the side view along the beam, on the y-cleaned '
                         'sample.  both: one after the other.')
    ap.add_argument('--out-y', default=str(HERE / 'figures' / 'sideview_run145'))
    a = ap.parse_args()

    P.use()
    subs = subruns()
    K = k_run()
    hot = noisy_columns(subs)
    from sept26_prelim_analysis import source_imaging as SI
    bars = plastics(SI.transforms(RUN))
    sipm = sipm_wall(SI.transforms(RUN))
    cap = campaign_capsule()
    info = {}

    if a.projection in ('y', 'both'):
        hot_y = noisy_columns(subs, 'y_local')
        yarms = {arm: arm_sample(arm, subs, K.get(arm, 1.0), hot,
                                 charge_window=not a.no_charge_window,
                                 clean=not a.no_clean, hot_tab_y=hot_y)
                 for arm in FAN_ARMS}
        os.makedirs(os.path.dirname(a.out_y), exist_ok=True)
        # both variants, because which one reads better is a judgement: the
        # normalised one shows the three chambers agreeing on the SHAPE, the
        # counts one shows what each chamber actually contributes
        ycal = figure_y(yarms, bars, sipm, a.out_y, norm=True)
        figure_y(yarms, bars, sipm, f'{a.out_y}_counts', norm=False)
        info['capsule_y_calibration'] = ycal
        info['y'] = {arm: dict(funnel=d['funnel'], y_crossing=d['y_crossing'],
                               median_y_at_target_mm=float(
                                   np.median(_at_target_y(d))))
                     for arm, d in yarms.items()}
        info['y_noisy_columns'] = {arm: int((hot_y.arm == arm).sum())
                                   for arm in FAN_ARMS}
        with open(f'{a.out_y}.json', 'w') as f:
            json.dump(info['y'] | dict(gas_centroid_y_mm=GAS_CENTROID_Y,
                                       noisy_y_columns=info['y_noisy_columns'],
                                       capsule_y_calibration=ycal),
                      f, indent=1, default=float)
        if a.projection == 'y':
            print(json.dumps(info, indent=1, default=float))
            return 0

    arms = {arm: arm_sample(arm, subs, K.get(arm, 1.0), hot,
                            charge_window=not a.no_charge_window,
                            clean=not a.no_clean) for arm in CROSS_ARMS}
    dead = dead_spans(subs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    figure(arms, cap, dead, bars, sipm, a.out)

    info = dict(info, run=RUN, subruns=subs, k=K, capsule_campaign=cap,
                dead_spans_local_mm={a_: [list(s) for s in v]
                                     for a_, v in dead.items()},
                plastics={a_: dict(u_mm=b['u'], depth_mm=b['depth'])
                          for a_, b in bars.items()},
                sipm_wall={a_: dict(n_bars=w['n_bars'], depth_mm=w['depth'],
                                    u_first_last=[w['u'][0], w['u'][-1]])
                           for a_, w in sipm.items()},
                # what the overshoot past a bar edge is: the coincidence tested
                # the raw tan, the figure draws k * tan
                on_bar={a_: dict(at_k1=on_bar(arms[a_], bars[a_], 1.0),
                                 at_k=on_bar(arms[a_], bars[a_], arms[a_]['k']))
                        for a_ in FAN_ARMS},
                charge_window=not a.no_charge_window, clean=not a.no_clean,
                noisy_columns={arm: int((hot.arm == arm).sum()) for arm in FAN_ARMS},
                arms={arm: dict(funnel=d['funnel'], crossing=d['crossing'],
                                median_at_target_mm=float(np.median(
                                    (v := _at_target(d))[np.abs(v) < 90])))
                      for arm, d in arms.items()})
    with open(f'{a.out}.json', 'w') as f:
        json.dump(info, f, indent=1, default=float)
    print(json.dumps(info, indent=1, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
