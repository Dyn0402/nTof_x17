#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run79_wall_segment_gif.py -- an animated tour of the wall-pointing result:
four SiPM segments, one at a time, each with the bundle of waveform-first
tracks that fired it.

WHAT THE PICTURE CLAIMS
-----------------------
Every run_79 DREAM trigger is a SiPM-wall AND plastic coincidence in one arm.
Through the n_TOF merge we know WHICH of arm A's four wall segments fired.
The wall is 16 read bars of 25 mm summed in 4 groups of 100 mm, 96 mm beyond
the strip plane -- external position truth at 100 mm granularity, on a lever
arm the chamber knows nothing about.

So: take the tracks reconstructed in chamber A (waveform-first, `wft/` via
`wft_beam.py`), split them by the segment their trigger came from, and draw
each bundle in the 3D model. Two things should be visible and both are:

  * the bundle lands on the segment that fired -- the four fans are separated
    along the wall in the same order as the bar groups (ordering correlation
    -0.98, spread 203 mm; RUN79_PRELIM_2026-07-30.md section 4);
  * each bundle sweeps back through the He-3 capsule -- a waist at the target,
    which is the same statement as the tan-vs-position slope of section 3 but
    drawn instead of fitted.

The animation cycles: all four fans faintly, then each segment in turn goes
opaque while the other three stay ghosted.

WHAT IT DOES *NOT* CLAIM
------------------------
Everything in RUN79_PRELIM_2026-07-30.md section 7 applies -- this draws that
document's data, it does not add to it. In particular the reconstruction is
the PRELIMINARY transferred-bench one (no in-situ calibration), the angle
scale carries the v = 36.0 um/ns rescaling, and the wall read-out order is the
'descending' one the same data picked. A visible waist at the capsule is NOT a
resolution measurement: the target has size, the beam scatters, and the
selection here is a display cut.

Usage:
    python -m ntof_tracking.run79_wall_segment_gif                    # the gif
    python -m ntof_tracking.run79_wall_segment_gif --stills-only      # 5 pngs
    python -m ntof_tracking.run79_wall_segment_gif --frames-per-phase 8 \
        --size 900 520 --out /tmp/quick                               # a draft

    # a messaging-app cut: 5.45 s, under the 6 s cap those apps impose
    python -m ntof_tracking.run79_wall_segment_gif --compact --no-stills \
        --size 1000 520 --frames-per-phase 22 --fps 20 --gif-colors 80 \
        --name wall_segment_tour_whatsapp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ntof_tracking.reco import geometry as geo                      # noqa: E402
from ntof_tracking.run79_merge_prelim import (                      # noqa: E402
    STRIP_MAP_HALF, N_WALL_SEG, OUT_BASE, wall_segment_u)

RUN_CONFIG = '/media/dylan/data/x17/beam_july/runs/{run}/run_config.json'
MERGED = OUT_BASE / 'stat090_0000' / 'mx17_A' / 'merged_prelim.parquet'

# ---------------------------------------------------------------- selection
TAN_SANE = 0.9          # railed fits only; the fit itself bounds nothing
N_STRIPS = (6, 60)      # very loose -- this is a display cut, not an analysis

# --------------------------------------------------------------- appearance
# Four segment hues, validated all-pairs against BOTH surfaces below (OKLab
# x100: worst CVD dE 17.8, worst normal-vision dE 23.5). Two of the four sit
# below 3:1 contrast on their surface, so the relief rule applies -- hence the
# always-on legend and the named active segment in the header.
SEG_COLOR = ['#d77804', '#b60254', '#5f46fd', '#07a3cd']

# A bundle is ~600 translucent lines that all converge on the same place, so
# what a pixel shows is 1 - (1 - alpha)^overlap, not alpha. That is why the
# ghosted segments are a THINNED copy of the bundle (GHOST_FRAC) rather than
# the whole thing at low alpha -- see apply_weights.
GHOST_FRAC = 0.08
THEME = {
    'light': dict(
        surface='#f4f4f2', ink='#141413', muted='#6d6c66', hair='#c9c8c2',
        mm='#7d8ea0', bar_edge='#5c646d',
        plastic='#b3aca0', capsule='#7fa9bd', vessel='#a8a8a4', beam='#9a7268',
        track_hi=0.28, gap_hi=0.36, ghost=0.050, bar_lo=0.05, bar_hi=0.55,
        pt_lo=0.05, pt_hi=0.90),
    'dark': dict(
        surface='#12141a', ink='#f2f2ef', muted='#9b9a92', hair='#3a3c44',
        mm='#5d6b7a', bar_edge='#aab1b9',
        plastic='#6a6459', capsule='#4e7f95', vessel='#6e6e6b', beam='#a07d70',
        track_hi=0.32, gap_hi=0.40, ghost=0.065, bar_lo=0.06, bar_hi=0.60,
        pt_lo=0.06, pt_hi=0.90),
}

# depths, from the strip plane, of the things we draw the tracks against
WALL_DEPTH = geo.SIPM_SCINT_W0 - geo.W_STRIP          # 95.9 mm, past the strips
# Every line is cut at the plane through the target, so the inward ends pile up
# on one surface instead of fanning out again behind it: that pile-up IS the
# pointing statement, and it keeps the drawn volume the size of the apparatus.
W_IN = {'A': 234.6, 'B': 234.1, 'C': 234.6, 'D': 234.1}
GAP = geo.DRIFT_GAP


# ------------------------------------------------------------------ the data
def load(merged: str, arm: str):
    """The merged table, the wall mapping it measured, and the display cut."""
    d = pd.read_parquet(merged)
    sj = Path(merged).with_name('merged_prelim.summary.json')
    summ = json.load(open(sj)) if sj.exists() else {}
    mapping = (summ.get('wall_pointing') or {}).get('mapping', 'descending')

    keep = (d['x_ok'] & d['y_ok'] & d['x_quality_ok'] & d['y_quality_ok']
            & np.isfinite(d['wal_detn'])
            & (d['x_tan_theta'].abs() < TAN_SANE)
            & (d['y_tan_theta'].abs() < TAN_SANE)
            & d['x_n_strips'].between(*N_STRIPS)
            & d['y_n_strips'].between(*N_STRIPS))
    d = d[keep].copy()
    d['seg'] = ((d['wal_detn'] - 1) // 2).astype(int)
    d = d[d['seg'].between(0, N_WALL_SEG - 1)]
    return d, mapping, summ


def track_points(d: pd.DataFrame, tr, w_lo: float, w_hi: float):
    """Global-frame endpoints of every track between two drift depths.

    The fit gives, per plane, a position at the mesh (`*_p0`, strip-map
    coordinates) and tan = d(position)/d(depth) with depth increasing TOWARD
    the target. So the local point at depth w is (u0 + tx w, v0 + ty w, -w),
    and w < 0 is the outward extrapolation past the strips -- the wall side.
    Vectorised copy of run79_event_display.track_line; same convention, and
    `u_wall` in the merged table is the global X of the w = -96.4 point.
    """
    u0 = d['x_p0'].to_numpy(float) - STRIP_MAP_HALF
    v0 = d['y_p0'].to_numpy(float) - STRIP_MAP_HALF
    tx = d['x_tan_theta'].to_numpy(float)
    ty = d['y_tan_theta'].to_numpy(float)

    def at(w):
        return tr.local_to_global(u0 + tx * w, v0 + ty * w, np.full(u0.shape, w))

    return at(w_lo), at(w_hi)


def line_mesh(p0: np.ndarray, p1: np.ndarray):
    """One PolyData holding N independent 2-point lines (one actor for the
    whole bundle -- an actor per track would be thousands of draw calls)."""
    import pyvista as pv
    n = len(p0)
    pts = np.empty((2 * n, 3), float)
    pts[0::2], pts[1::2] = p0, p1
    cells = np.column_stack([np.full(n, 2), np.arange(0, 2 * n, 2),
                             np.arange(1, 2 * n, 2)]).ravel()
    return pv.PolyData(pts, lines=cells)


# -------------------------------------------------------------- the geometry
def capsule_profiles(th: dict):
    """(r, y, colour, opacity) polycone profiles of the He-3 capsule: the Al
    vessel from the Geant scripts if they are reachable, the active gas from
    reco.geometry either way."""
    out = []
    try:
        sys.path.insert(0, os.path.expanduser(
            '~/CLionProjects/MX17_Full_Geant/scripts'))
        import plot_geometry as pg                                # noqa: E402
        out.append((pg.RO_AL * 10.0, pg.Z_AL * 10.0, th['vessel'], 0.30))
    except Exception:
        pass
    out.append((geo.HE3_GAS_R, geo.HE3_GAS_Y, th['capsule'], 0.92))
    return out


def _box(pl, c0, c1, **kw):
    import pyvista as pv
    lo, hi = np.minimum(c0, c1), np.maximum(c0, c1)
    return pl.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1],
                                      lo[2], hi[2])), **kw)


def build_scene(pl, d: pd.DataFrame, arm: str, tr, mapping: str, th: dict,
                show_plastics=False):
    """Add every mesh of one panel and hand back the actors we animate.

    Only arm `arm` is drawn: the other three walls are what makes the existing
    3D displays unreadable, and they carry nothing here.
    """
    import pyvista as pv
    act = {'bars': {}, 'tracks': {}, 'gap': {}, 'pts': {}, 'ghost': {}}

    # --- the chamber's drift gas, and (faint) the plastics behind the wall
    for el in geo.arm_active_volumes(arm):
        if el['kind'] == 'ls':
            continue                       # the LS slab swallows the whole view
        if el['kind'] == 'plastic' and not show_plastics:
            continue
        if el['kind'] == 'sipm':
            continue                       # drawn below, grouped by segment
        ff = geo.arm_front_face(arm, el['on'])
        w, u = geo.W_HAT[arm], geo.U_HAT[arm]
        c0 = ff + w * el['w0'] + u * el['u_lo'] - geo.V_HAT * el['half_v']
        c1 = ff + w * el['w1'] + u * el['u_hi'] + geo.V_HAT * el['half_v']
        if el['kind'] == 'mm':
            _box(pl, c0, c1, color=th['mm'], opacity=0.13,
                 show_edges=True, edge_color=th['hair'], line_width=0.8)
        else:
            _box(pl, c0, c1, color=th['plastic'], opacity=0.07)

    # --- the SiPM wall, one actor per n_TOF segment (4 bars each)
    ff = geo.arm_front_face(arm, 'struct')
    w, u = geo.W_HAT[arm], geo.U_HAT[arm]
    for s in range(N_WALL_SEG):
        grp = (N_WALL_SEG - 1 - s) if mapping == 'descending' else s
        for bar in [grp * 4 + 1 + i for i in range(4)]:
            u_lo, u_hi = geo.sipm_bar_u(bar)
            # a 1 mm gap between bars so the four groups read as groups
            c0 = (ff + w * geo.SIPM_SCINT_W0 + u * (u_lo + 0.5)
                  - geo.V_HAT * geo.SIPM_HALF_V)
            c1 = (ff + w * geo.SIPM_SCINT_W1 + u * (u_hi - 0.5)
                  + geo.V_HAT * geo.SIPM_HALF_V)
            a = _box(pl, c0, c1, color=SEG_COLOR[s], opacity=th['bar_lo'],
                     show_edges=True, edge_color=th['bar_edge'],
                     line_width=0.6)
            act['bars'].setdefault(s, []).append(a)

    # --- the tracks, split by the segment whose trigger they carry
    for s in range(N_WALL_SEG):
        ds = d[d['seg'] == s]
        if not len(ds):
            continue
        p_wall, p_in = track_points(ds, tr, -WALL_DEPTH, W_IN[arm])
        act['tracks'][s] = pl.add_mesh(
            line_mesh(p_wall, p_in), color=SEG_COLOR[s],
            opacity=th['track_hi'], line_width=1.0, lighting=False)
        # The ghost is a fixed random THINNING of the same bundle, not the same
        # bundle at lower alpha. Lowering alpha does not work: 600 lines all
        # converging on the target overlap ~200-deep there, so even alpha 0.005
        # composites to ~0.6 in the core and three "ghosted" segments stay as
        # loud as the active one. Removing 91 % of the lines does work.
        idx = np.random.default_rng(1234 + s).permutation(len(ds))[
            :max(12, int(round(GHOST_FRAC * len(ds))))]
        act['ghost'][s] = pl.add_mesh(
            line_mesh(p_wall[idx], p_in[idx]), color=SEG_COLOR[s],
            opacity=th['ghost'], line_width=1.0, lighting=False)
        p_a, p_b = track_points(ds, tr, 0.0, GAP)
        act['gap'][s] = pl.add_mesh(
            line_mesh(p_a, p_b), color=SEG_COLOR[s],
            opacity=th['gap_hi'], line_width=2.6, lighting=False)
        # Where each bundle crosses the wall. (The target-plane crossings were
        # drawn too and taken out again: spread over +-300 mm they do not
        # overlap, so every one of the 2 267 stayed individually visible and
        # the speckle buried the lines. The waist reads from the line density.)
        act['pts'][s] = pl.add_mesh(
            pv.PolyData(p_wall), color=SEG_COLOR[s], opacity=th['pt_lo'],
            point_size=3.0, render_points_as_spheres=True, lighting=False)

    # --- the target
    for prof_r, prof_y, col, op in capsule_profiles(th):
        pts = np.column_stack([prof_r, np.zeros_like(prof_r), prof_y])
        mesh = pv.lines_from_points(pts).extrude_rotate(resolution=64,
                                                        capping=True)
        mesh.rotate_x(-90.0, inplace=True)          # profile axis -> +Y (beam)
        pl.add_mesh(mesh, color=col, opacity=op, smooth_shading=True)
    # the beam, up the EAR2 flight path
    pl.add_mesh(pv.Arrow(start=(0, -330, 0), direction=(0, 1, 0), scale=660,
                         tip_length=0.05, tip_radius=0.008, shaft_radius=0.0016),
                color=th['beam'], opacity=0.45)
    return act


# ------------------------------------------------------------- the animation
def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def phase_weights(i: int, n_frames: int, per: int, phases: list,
                  cross: float = 0.34) -> np.ndarray:
    """Per-segment emphasis in [0, 1] at frame i.

    Each phase holds its target for the first (1 - cross) of its frames and
    eases into the next one over the rest, so the loop closes on itself.
    """
    def target(k):
        ph = phases[k % len(phases)]
        if ph == 'all':
            return np.full(N_WALL_SEG, 0.42)
        w = np.zeros(N_WALL_SEG)
        w[ph] = 1.0
        return w

    k, frac = divmod(i, per)
    f = frac / max(per - 1, 1)
    if f <= 1.0 - cross:
        return target(k)
    return (target(k) + (target(k + 1) - target(k))
            * smoothstep((f - (1.0 - cross)) / cross))


def _rgb(hexcol: str) -> np.ndarray:
    h = hexcol.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _set(a, opacity: float, **kw):
    """Opacity, plus the properties that go with it. An actor at zero alpha is
    still composited every frame, so hide it outright."""
    if a is None:
        return
    a.visibility = opacity > 0.002
    a.prop.opacity = max(opacity, 0.002)
    for k, v in kw.items():
        setattr(a.prop, k, v)


def apply_weights(act: dict, w: np.ndarray, th: dict):
    """Emphasis in [0, 1] -> actor properties.

    The full bundle fades IN with the weight while the thinned ghost fades OUT,
    so a segment's turn swaps a sparse sketch for the whole population rather
    than brightening a haze that was already saturated.
    """
    lerp = lambda lo, hi, x: lo + (hi - lo) * x       # noqa: E731
    grey = _rgb(th['hair'])
    for s, ws in enumerate(w):
        col = tuple(_rgb(SEG_COLOR[s]))
        ghost_col = tuple(lerp(grey, _rgb(SEG_COLOR[s]), 0.62))
        for a in act['bars'].get(s, []):
            _set(a, lerp(th['bar_lo'], th['bar_hi'], ws),
                 color=col if ws > 0.5 else ghost_col)
        _set(act['tracks'].get(s), th['track_hi'] * ws,
             line_width=lerp(1.0, 1.5, ws), color=col)
        _set(act['gap'].get(s), th['gap_hi'] * ws,
             line_width=lerp(1.8, 3.0, ws), color=col)
        _set(act['ghost'].get(s), th['ghost'] * (1.0 - ws),
             line_width=1.0, color=ghost_col)
        _set(act['pts'].get(s), lerp(th['pt_lo'], th['pt_hi'], ws),
             point_size=lerp(2.0, 6.0, ws),
             color=col if ws > 0.5 else ghost_col)


def set_cameras(pl, az_deg: float, elev_deg: float, focal, radius: float,
                scale: float):
    """Panel 0 orbits; panel 1 is a fixed beam's-eye view (looking UP the
    flight path, which is how EAR2 sees it), in parallel projection so the
    convergence on the capsule is geometric and not perspective."""
    az, el = np.radians(az_deg), np.radians(elev_deg)
    focal = np.asarray(focal, float)
    pos = focal + radius * np.array(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    pl.subplot(0, 0)
    pl.camera_position = [tuple(pos), tuple(focal), (0, 1, 0)]
    # Looking along +Y with up = +Z puts the wall at the top of the panel and
    # global +X to the right, so the four segments read left-to-right in the
    # same order as the legend.
    pl.subplot(0, 1)
    f2 = focal + np.array([0.0, 0.0, 25.0])
    pl.camera_position = [(f2[0], f2[1] - 2000.0, f2[2]), tuple(f2), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = scale


# ------------------------------------------------------------- the overlay
def _font(size, bold=False):
    from matplotlib import font_manager
    from PIL import ImageFont
    name = 'DejaVu Sans'
    p = font_manager.findfont(font_manager.FontProperties(
        family=name, weight='bold' if bold else 'normal'))
    return ImageFont.truetype(p, size)


def _compose_compact(canvas, dr, w, h, head_h, sz, active, stats, th, arm):
    """The phone layout: two lines of header, a legend row, nothing else.

    Everything cut here is explanation the full-size figure carries and a
    6-second clip on a small screen cannot -- the viewer of this one is meant
    to see the four fans move, not to read the method.
    """
    f_t, f_s, f_l, f_c = (_font(sz(19), True), _font(sz(15)),
                          _font(sz(12), True), _font(sz(11)))
    pad = sz(24)
    dr.text((pad, sz(5)),
            f'run_79 / mx17_{arm} -- tracks by the SiPM wall segment that fired',
            font=f_t, fill=th['ink'])
    cv = stats['convergence']
    if active == 'all':
        sub = (f'all {N_WALL_SEG} bundles: {cv["spread_wall_mm"]:.0f} mm apart '
               f'at the wall, {cv["spread_target_mm"]:.0f} mm at the target')
        col = th['ink']
    else:
        st = stats['seg'][active]
        sub = (f'segment {active} -- {st["n"]:,} tracks, bars {st["bars"]}, '
               f'{st["inside"]:.0%} inside the group')
        col = SEG_COLOR[active]
    dr.text((pad, sz(33)), sub, font=f_s, fill=col)
    dr.line([(pad, head_h - 1), (w - pad, head_h - 1)], fill=th['hair'])

    y = h + head_h + sz(11)
    x = pad
    for s in range(N_WALL_SEG):
        on = (active == 'all') or (s == active)
        dr.rounded_rectangle([x, y + sz(1), x + sz(22), y + sz(13)],
                             radius=sz(3),
                             fill=SEG_COLOR[s] if on else th['hair'])
        dr.text((x + sz(29), y), f'seg {s}', font=f_l if on else f_c,
                fill=th['ink'] if on else th['muted'])
        x += sz(95)
    dr.text((x + sz(14), y + sz(1)), 'PRELIMINARY -- waveform-first tracks, '
                                     'arm A only', font=f_c, fill=th['muted'])
    return canvas


def frame_heights(w: int, compact: bool = False):
    """Header/footer heights for a canvas `w` wide (everything scales off the
    width so a draft render and the final one look the same).

    `compact` is the phone layout: type ~1.45x bigger relative to the frame and
    the explanatory lines dropped. Scaling the normal layout down to a width a
    phone will actually display leaves the caption at ~9 px, which is not a
    caption.
    """
    k = w / 1400.0
    if compact:
        k *= 1.45
        return int(round(64 * k)), int(round(41 * k)), k
    return int(round(104 * k)), int(round(112 * k)), k


def compose(img: np.ndarray, active, stats: dict, th: dict, arm: str = 'A',
            compact: bool = False):
    """Frame the render with a header (what is highlighted) and a footer
    (the always-visible legend the palette's relief rule requires)."""
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    head_h, foot_h, k = frame_heights(w, compact)
    canvas = Image.new('RGB', (w, h + head_h + foot_h), th['surface'])
    canvas.paste(Image.fromarray(img), (0, head_h))
    dr = ImageDraw.Draw(canvas)
    sz = lambda p: max(8, int(round(p * k)))                     # noqa: E731
    if compact:
        return _compose_compact(canvas, dr, w, h, head_h, sz, active, stats,
                                th, arm)
    f_t, f_s, f_l, f_c = (_font(sz(25), True), _font(sz(16)),
                          _font(sz(14), True), _font(sz(13)))
    pad = sz(26)

    dr.text((pad, sz(15)),
            f'run_79 / mx17_{arm} -- tracks, by the SiPM wall segment that '
            f'triggered them', font=f_t, fill=th['ink'])
    cv = stats['convergence']
    if active == 'all':
        sub = (f'all {N_WALL_SEG} bundles -- {stats["n_all"]:,} tracks: '
               f'{cv["spread_wall_mm"]:.0f} mm apart at the wall, '
               f'{cv["spread_target_mm"]:.0f} mm apart at the target plane')
        col = th['ink']
    else:
        st = stats['seg'][active]
        sub = (f'segment {active}   {st["n"]:,} tracks   '
               f'bars {st["bars"]}  (u = {st["lo"]:+.0f} .. {st["hi"]:+.0f} mm)'
               f'   median crossing u = {st["med"]:+.0f} mm '
               f'({st["inside"]:.0%} inside the group)'
               f'   ->  X at the target plane {st["med_target"]:+.0f} mm')
        col = SEG_COLOR[active]
    dr.text((pad, sz(52)), sub, font=f_s, fill=col)
    dr.text((pad, sz(77)), 'PRELIMINARY -- transferred bench calibration, '
                           'angles rescaled to v = 36.0 um/ns; see '
                           'RUN79_PRELIM_2026-07-30.md',
            font=f_c, fill=th['muted'])
    dr.line([(pad, head_h - 1), (w - pad, head_h - 1)], fill=th['hair'])

    y = h + head_h + sz(18)
    dr.line([(pad, y - sz(12)), (w - pad, y - sz(12))], fill=th['hair'])
    x = pad
    for s in range(N_WALL_SEG):
        on = (active == 'all') or (s == active)
        dr.rounded_rectangle([x, y + sz(2), x + sz(26), y + sz(16)],
                             radius=sz(3),
                             fill=SEG_COLOR[s] if on else th['hair'])
        dr.text((x + sz(34), y), f'segment {s}', font=f_l if on else f_c,
                fill=th['ink'] if on else th['muted'])
        x += sz(150)
    dr.text((x + sz(20), y),
            f'wall read-out order: {stats["mapping"]}   |   '
            f'{stats["n_all"]:,} tracks over {stats["n_bunch"]} bunches, '
            f'n_TOF run {stats["ntof_run"]}',
            font=f_c, fill=th['muted'])
    dr.text((pad, y + sz(27)),
            "left: orbiting view.    right: beam's-eye, looking up the flight "
            'path, parallel projection.    Other three arms suppressed.',
            font=f_c, fill=th['muted'])
    dr.text((pad, y + sz(45)),
            'Thick = the piece measured inside the 30 mm drift gap;  thin = '
            'the same straight line continued out to the wall (dots) and back '
            'to the plane through the target.', font=f_c, fill=th['muted'])
    return canvas


# -------------------------------------------------------------------- driver
def segment_stats(d: pd.DataFrame, arm: str, tr, mapping: str) -> dict:
    """Per segment: where the bundle crosses the wall, and where it crosses the
    plane through the target.

    The pair is the whole point of the figure. The wall crossings are the four
    bar groups, 100 mm apart by construction; the target-plane crossings are
    where those same lines have got to 331 mm back up the arm. If the tracking
    and the matcher are both right, the first set separates and the second set
    piles up -- and the ratio of the two spreads is a number, not an impression.
    """
    out = {}
    for s in range(N_WALL_SEG):
        ds = d[d['seg'] == s]
        grp = (N_WALL_SEG - 1 - s) if mapping == 'descending' else s
        lo, hi = wall_segment_u(grp)
        p_wall, p_tgt = track_points(ds, tr, -WALL_DEPTH, W_IN[arm])
        u = p_wall[:, 0]                    # global X at the wall == u_wall
        out[s] = dict(n=int(len(ds)), lo=lo, hi=hi,
                      bars=f'{grp * 4 + 1}-{grp * 4 + 4}',
                      med=float(np.median(u)) if len(u) else np.nan,
                      inside=float(np.mean((u >= lo) & (u <= hi))) if len(u)
                      else np.nan,
                      med_target=float(np.median(p_tgt[:, 0])) if len(u)
                      else np.nan,
                      iqr_target=[float(np.percentile(p_tgt[:, 0], q))
                                  for q in (25, 75)] if len(u) else [np.nan] * 2)
    return out


def convergence(seg: dict) -> dict:
    """How far the four bundle medians are apart at each end."""
    w = [seg[s]['med'] for s in seg if np.isfinite(seg[s]['med'])]
    t = [seg[s]['med_target'] for s in seg if np.isfinite(seg[s]['med_target'])]
    return dict(spread_wall_mm=float(np.ptp(w)) if len(w) > 1 else np.nan,
                spread_target_mm=float(np.ptp(t)) if len(t) > 1 else np.nan)


def convergence_null(d: pd.DataFrame, arm: str, tr, mapping: str,
                     seed: int = 7) -> dict:
    """The same two spreads with the fired-segment label randomly reassigned.

    Without this the figure is only suggestive: four subsamples of anything
    have four slightly different medians, and 'they converge' needs a scale to
    be measured against. Shuffling the label keeps the tracks and destroys only
    the matcher's information, so the wall spread it leaves is the noise floor
    that the real 251 mm has to beat.
    """
    sh = d.copy()
    sh['wal_detn'] = np.random.default_rng(seed).permutation(
        sh['wal_detn'].to_numpy())
    sh['seg'] = ((sh['wal_detn'] - 1) // 2).astype(int)
    return convergence(segment_stats(sh, arm, tr, mapping))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged', default=str(MERGED))
    ap.add_argument('--arm', default='A')
    ap.add_argument('--run', default='run_79')
    ap.add_argument('--out', default=None)
    ap.add_argument('--theme', default='light', choices=sorted(THEME))
    ap.add_argument('--size', nargs=2, type=int, default=(1400, 720),
                    metavar=('W', 'H'), help='the 3D render, before the frame')
    ap.add_argument('--frames-per-phase', type=int, default=26)
    ap.add_argument('--fps', type=float, default=14.0)
    ap.add_argument('--orbit', type=float, default=18.0,
                    help='azimuth rock amplitude [deg] about --az0')
    ap.add_argument('--full-orbit', action='store_true',
                    help='sweep a complete revolution instead of rocking')
    ap.add_argument('--az0', type=float, default=45.0)
    ap.add_argument('--elev', type=float, default=32.0)
    ap.add_argument('--stills-only', action='store_true')
    ap.add_argument('--no-stills', action='store_true')
    ap.add_argument('--gif-colors', type=int, default=128)
    ap.add_argument('--compact', action='store_true',
                    help='phone layout: bigger type, no explanatory lines '
                         '(pair with a short --frames-per-phase and a high '
                         '--fps for a messaging-app clip)')
    ap.add_argument('--name', default='wall_segment_tour',
                    help='output stem, so a smaller share copy can sit beside '
                         'the full-size one in the same directory')
    a = ap.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True
    from PIL import Image

    th = THEME[a.theme]
    d, mapping, summ = load(a.merged, a.arm)
    cfg = json.load(open(RUN_CONFIG.format(run=a.run)))
    tr = geo.detector_transforms(cfg)[f'mx17_{a.arm}']

    seg = segment_stats(d, a.arm, tr, mapping)
    stats = dict(seg=seg, convergence=convergence(seg),
                 convergence_null=convergence_null(d, a.arm, tr, mapping),
                 mapping=mapping,
                 n_all=int(len(d)),
                 n_bunch=int(d['BunchNumber'].nunique()),
                 ntof_run=summ.get('ntof_run', 224572))
    print(f'[gif] {len(d):,} tracks pass the display cut '
          f'({d["BunchNumber"].nunique()} bunches), wall order "{mapping}"')
    for s, st in seg.items():
        print(f'    segment {s}: n = {st["n"]:5,}  bars {st["bars"]:>5}  '
              f'u {st["lo"]:+7.1f} .. {st["hi"]:+7.1f}  median at the wall '
              f'{st["med"]:+7.1f} mm ({st["inside"]:.0%} inside)   at the '
              f'target plane {st["med_target"]:+6.1f} mm')
    cv, nl = stats['convergence'], stats['convergence_null']
    print(f'    the four bundle medians are {cv["spread_wall_mm"]:.0f} mm apart '
          f'at the wall and {cv["spread_target_mm"]:.0f} mm apart at the '
          f'target plane')
    print(f'    null (segment label shuffled): {nl["spread_wall_mm"]:.0f} mm at '
          f'the wall, {nl["spread_target_mm"]:.0f} mm at the target plane')

    pl = pv.Plotter(off_screen=True, window_size=tuple(a.size), shape=(1, 2),
                    border=False)
    pl.set_background(th['surface'])
    acts = []
    for col in range(2):
        pl.subplot(0, col)
        pl.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
        acts.append(build_scene(pl, d, a.arm, tr, mapping, th))

    focal = np.array([0.0, 0.0, 145.0])
    radius, par_scale = 1520.0, 345.0
    phases = ['all', 0, 1, 2, 3]
    per = a.frames_per_phase
    n_frames = per * len(phases)

    out = Path(a.out or (Path(a.merged).parent / 'wall_segment_tour'))
    out.mkdir(parents=True, exist_ok=True)
    # Written before anything is drawn, and in --stills-only too: the numbers
    # in it are the figure's caption, and they should not depend on having sat
    # through the render.
    with open(out / f'{a.name}.json', 'w') as f:
        json.dump(dict(status='PRELIMINARY', source=str(a.merged),
                       arm=a.arm, mapping=mapping, theme=a.theme,
                       n_frames=n_frames, fps=a.fps,
                       duration_s=round(n_frames / a.fps, 2),
                       compact=a.compact,
                       phases=[str(p) for p in phases],
                       selection=dict(tan_max=TAN_SANE,
                                      n_strips=list(N_STRIPS),
                                      ghost_frac=GHOST_FRAC,
                                      both_planes=True, quality_ok=True,
                                      wall_matched=True),
                       stats=stats), f, indent=1, default=str)

    def azimuth(i):
        """A gentle rock, not a full revolution: a complete orbit spends a
        third of its time looking at the wall face-on, where the fan is
        foreshortened into the bar it came from and the picture says nothing.
        A sine closes the loop as cleanly as a full turn does."""
        if a.full_orbit:
            return a.az0 + 360.0 * i / n_frames
        return a.az0 + a.orbit * np.sin(2 * np.pi * i / n_frames)

    def render(i):
        w = phase_weights(i, n_frames, per, phases)
        for act in acts:
            apply_weights(act, w, th)
        set_cameras(pl, azimuth(i), a.elev, focal, radius, par_scale)
        # REQUIRED. Without an explicit render the off-screen buffer is stale
        # and every screenshot after the first returns the opening frame --
        # silently, so the animation looks like the emphasis is simply not
        # being applied. (It was: the actor flags were correct all along.)
        pl.render()
        return pl.screenshot(return_img=True)

    # --- the stills: the middle of each phase, where nothing is cross-fading
    if not a.no_stills:
        for k, ph in enumerate(phases):
            i = int((k + 0.4) * per)
            img = render(i)
            name = 'all' if ph == 'all' else f'seg{ph}'
            p = out / f'{a.name}_{name}.png'
            compose(img, ph, stats, th, a.arm, a.compact).save(p)
            print('wrote', p)
    if a.stills_only:
        pl.close()
        return 0

    frames = []
    for i in range(n_frames):
        k = i // per
        img = render(i)
        frames.append(compose(img, phases[k], stats, th, a.arm, a.compact))
        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f'    frame {i + 1}/{n_frames}', flush=True)
    pl.close()

    # One palette for the whole animation: a per-frame palette makes the
    # background and the ghosted fans shimmer between frames.
    sample = Image.new('RGB', (frames[0].width,
                               frames[0].height * min(6, len(frames))))
    for j, i in enumerate(np.linspace(0, len(frames) - 1,
                                      min(6, len(frames))).astype(int)):
        sample.paste(frames[i], (0, j * frames[0].height))
    pal = sample.quantize(colors=a.gif_colors, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]

    gif = out / f'{a.name}.gif'
    q[0].save(gif, save_all=True, append_images=q[1:], loop=0,
              duration=int(round(1000.0 / a.fps)), optimize=True, disposal=1)
    print(f'wrote {gif}  ({gif.stat().st_size / 1e6:.1f} MB, '
          f'{n_frames} frames, {frames[0].width}x{frames[0].height})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
