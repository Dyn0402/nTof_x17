#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_bench.py -- the Saclay cosmic test bench, in 3-D.

The stack, bottom to top (all z from the run config's ``bench_geometry`` and
``detectors``, except the scintillators -- see geometry.ASSUMPTIONS):

    z = -110 mm   trigger scintillator, 60 x 60 cm      (drawn, not configured)
    z =   24 mm   M3 reference Micromegas, 50 x 50 cm
    z =  144 mm   M3 reference Micromegas
    z =  232 mm   P1, the lower test slot
    z =  702 mm   P2, the upper test slot
    z = 1185 mm   M3 reference Micromegas
    z = 1302 mm   M3 reference Micromegas
    z = 1420 mm   trigger scintillator, 60 x 60 cm

A chamber in either test slot may be an MX17 (40 x 40 cm, 30 mm drift gap) or a
P2 BASKET fan, which is what ``mx17_det3_p2_det1_overnight_6-27-26`` actually
ran: P2_1 in the lower slot, mx17_3 in the upper one.

Frame: +Z vertically up (the bench's own frame, as used by every run config and
by the whole analysis chain), X/Y transverse.
"""
from __future__ import annotations

import numpy as np

import geometry as G
import meshes as M
import style as S
import scenes_sps as SPS

RNG = np.random.default_rng(20260807)

FLOOR_Z = -400.0            # drawn floor, below the bottom paddle
FLOOR_HALF = 3200.0


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def add_m3(p, z, label=None):
    """One M3 reference Micromegas: 50 x 50 cm active in a light frame."""
    parts = M.rect_chamber(
        center=(0.0, 0.0, z),
        pcb_size=(G.M3_ACTIVE_MM + 24, G.M3_ACTIVE_MM + 24),
        active_size=(G.M3_ACTIVE_MM, G.M3_ACTIVE_MM),
        frame_size=(G.M3_FRAME_MM, G.M3_FRAME_MM),
        pcb_thick=G.M3_THICK_MM, normal='z', drift_gap=None,
        n_strips=56)
    p.add_mesh(parts['frame'], **S.mat('alu', S.COL['alu']))
    p.add_mesh(parts['pcb'], **S.mat('pcb', S.COL['m3_pcb']))
    p.add_mesh(parts['active'], **S.mat('copper', S.COL['copper']))
    p.add_mesh(parts['strips'], **S.mat('mesh', S.COL['copper_hot']))
    parts['outline'] = M.rect_outline((0.0, 0.0, z), G.M3_FRAME_MM,
                                      G.M3_FRAME_MM, normal='z')
    return parts


def add_mx17(p, z, drift_dir=+1):
    """An MX17 chamber in a test slot, drift volume facing the incoming muons."""
    parts = M.rect_chamber(
        center=(0.0, 0.0, z),
        pcb_size=(G.MX17_PCB_MM, G.MX17_PCB_MM),
        active_size=(G.MX17_ACTIVE_MM, G.MX17_ACTIVE_MM),
        frame_size=(G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM,
                    G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM),
        pcb_thick=8.0, normal='z',
        drift_gap=G.MX17_DRIFT_GAP_MM, drift_dir=drift_dir,
        n_strips=64)
    p.add_mesh(parts['frame'], **S.mat('alu', S.COL['alu']))
    p.add_mesh(parts['pcb'], **S.mat('pcb', S.COL['pcb']))
    p.add_mesh(parts['active'], **S.mat('copper', S.COL['copper']))
    p.add_mesh(parts['strips'], **S.mat('mesh', S.COL['copper_hot']))
    p.add_mesh(parts['gas'], **S.mat('gas', S.COL['gas']))
    # the drift volume is the point of a micro-TPC chamber: outline it so the
    # 30 mm gap reads as a volume rather than a pane of glass
    p.add_mesh(parts['gas'].extract_feature_edges(), color=S.COL['gas'],
               line_width=2.0, lighting=False, opacity=0.75)
    p.add_mesh(parts['cathode'], **S.mat('plastic', '#e2e9f0', opacity=0.16))
    w = G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM
    parts['outline'] = M.rect_outline((0.0, 0.0, z), w, w, normal='z')
    return parts


def add_p2_flat(p, z, pads_lab, sectors):
    """A P2 BASKET in a test slot, lying flat (fan in the horizontal plane).

    On the bench the boards lie horizontally in the rail levels, so the fan's
    own (x, height) plane maps onto the bench's (x, y) and the chamber normal
    is +Z -- the same ``fan_chamber`` code as the SPS scene.  The one thing that
    does *not* carry over is the origin: in the SPS lab frame the fan's height
    coordinate is measured from the table (130 -> 683 mm), so used unshifted it
    would sit half a metre off the bench axis.  Here it is centred on the slot,
    which is what a board resting in a rail level does.
    """
    fr_lo, fr_hi, _, _ = SPS.p2_frame_extent()
    y_mid = G.P2_APEX_HEIGHT - (fr_lo + fr_hi) / 2

    m = SPS.p2_meshes(z, pads_lab, sectors)
    for key in ('frame', 'pcb', 'pads', 'pads_live'):
        m[key] = m[key].translate((0.0, -y_mid, 0.0), inplace=False)

    p.add_mesh(m['frame'], **S.mat('alu_matte', S.COL['alu']))
    p.add_mesh(m['pcb'], **S.mat('pcb', S.COL['pcb']))
    p.add_mesh(m['pads'], **S.mat('copper', S.COL['copper_dead'],
                                  metallic=0.55, roughness=0.62))
    p.add_mesh(m['pads_live'], **S.mat('copper', S.COL['copper']))

    o = SPS.p2_outline3d(z)
    o[:, 1] -= y_mid
    m['outline'] = o
    return m


def add_scintillator(p, z, pmt_side=+1):
    """A 60 x 60 cm trigger paddle with a light guide and PMT."""
    out = {}
    slab = M.slab((0.0, 0.0, z), G.SCINT_MM, G.SCINT_MM, G.SCINT_THICK_MM,
                  normal='z')
    p.add_mesh(slab, **S.mat('scint', S.COL['scint']))
    # light guide: a tapered wedge off one edge, then the PMT can
    gx = pmt_side * (G.SCINT_MM / 2 + 85)
    guide = M.slab((gx, 0.0, z), 170, G.SCINT_MM * 0.5, G.SCINT_THICK_MM,
                   normal='z')
    p.add_mesh(guide, **S.mat('scint', S.COL['scint'], opacity=0.34))
    # PMT: glass envelope, then the base / divider can behind it
    p.add_mesh(M.cylinder((pmt_side * (G.SCINT_MM / 2 + 215), 0.0, z),
                          (1, 0, 0), radius=38, height=110),
               **S.mat('plastic', '#c9d3dd', opacity=0.55, specular=0.9,
                       specular_power=70))
    p.add_mesh(M.cylinder((pmt_side * (G.SCINT_MM / 2 + 330), 0.0, z),
                          (1, 0, 0), radius=32, height=130),
               **S.mat('plastic', S.COL['pmt']))
    out['outline'] = M.rect_outline((0.0, 0.0, z), G.SCINT_MM, G.SCINT_MM,
                                    normal='z')
    return out


# --------------------------------------------------------------------------- #
# Bench structure
# --------------------------------------------------------------------------- #
def add_structure(p, theme, floor=True):
    """Four uprights and the rail levels they carry (drawn, not surveyed)."""
    a = G.BENCH_POST_XY
    s = G.BENCH_POST_SECTION
    z0, z1 = G.BENCH_POST_Z
    for sx in (-a, a):
        for sy in (-a, a):
            p.add_mesh(M.slab((sx, sy, (z0 + z1) / 2), s, s, z1 - z0,
                              normal='z'),
                       **S.mat('alu_matte', S.COL['alu_dark']))
    # cross-rails at the top and bottom of the frame
    for zc in (z0 + s / 2, z1 - s / 2):
        for sy in (-a, a):
            p.add_mesh(M.slab((0.0, sy, zc), 2 * a + s, s, s, normal='z'),
                       **S.mat('alu_matte', S.COL['alu_dark']))
        for sx in (-a, a):
            p.add_mesh(M.slab((sx, 0.0, zc), s, 2 * a + s, s, normal='z'),
                       **S.mat('alu_matte', S.COL['alu_dark']))
    if floor:
        p.add_mesh(M.slab((0.0, 0.0, FLOOR_Z - 25), 2 * FLOOR_HALF,
                          2 * FLOOR_HALF, 50, normal='z'),
                   **S.mat('alu_matte', S.THEMES[theme]['floor'],
                           specular=0.05, specular_power=90, ambient=0.30))


def add_level_rails(p, z_lo=None, z_hi=1150.0):
    """Short brackets on the four uprights at every rail level.

    ``bench_geometry`` gives the levels as ``bottom_level_z`` + k x
    ``level_z_spacing`` (82 mm, then every 97 mm); drawing them is what makes
    the bench read as the modular rack it is, and it is also what fixes the
    test slots P1 = 227 mm and P2 = 697 mm as *levels* rather than free heights.
    """
    z_lo = G.BENCH_BOTTOM_LEVEL_Z if z_lo is None else z_lo
    a, s = G.BENCH_POST_XY, G.BENCH_POST_SECTION
    z = z_lo
    while z <= z_hi:
        for sx in (-a, a):
            for sy in (-a, a):
                p.add_mesh(M.slab((sx - np.sign(sx) * s * 0.62, sy, z),
                                  s * 0.5, s * 0.92, 7, normal='z'),
                           **S.mat('alu_matte', S.COL['alu_dark'],
                                   ambient=0.30))
        z += G.BENCH_LEVEL_SPACING


def add_shelf(p, z, half_width, drop=15.0, section=20.0):
    """The two rails a plane of half-width ``half_width`` rests on.

    The uprights stand just outside the widest element, so a plane is carried
    by a pair of rails spanning post to post -- a real load path, and the thing
    that stops every plane in the stack from looking suspended in mid-air.
    """
    a = G.BENCH_POST_XY
    zc = z - drop
    inset = min(half_width * 0.82, a - section)
    for sy in (-inset, inset):
        p.add_mesh(M.slab((0.0, sy, zc), 2 * a, section, section, normal='z'),
                   **S.mat('alu_matte', S.COL['alu_dark'], ambient=0.30))


# --------------------------------------------------------------------------- #
# Cosmic muons
# --------------------------------------------------------------------------- #
def cosmic_tracks(n=9, accept_only=True, max_try=20000):
    """Muons that fire the top+bottom scintillator coincidence.

    Zenith angles are drawn from the standard cos^2(theta) sea-level
    distribution and azimuth uniformly; a track is kept only if it crosses both
    60 x 60 cm paddles, which is exactly the bench's trigger condition and is
    what makes the drawn tracks steep-limited rather than isotropic.
    """
    z_top, z_bot = G.BENCH_SCINT_Z['top'], G.BENCH_SCINT_Z['bottom']
    half = G.SCINT_MM / 2
    out = []
    for _ in range(max_try):
        if len(out) >= n:
            break
        # sample cos^2(theta) by rejection, theta in [0, pi/2)
        while True:
            th = np.arccos(RNG.uniform(0, 1))
            if RNG.uniform() < np.cos(th) ** 2:
                break
        ph = RNG.uniform(0, 2 * np.pi)
        # entry point on the top paddle
        x0, y0 = RNG.uniform(-half, half, 2)
        tx = np.tan(th) * np.cos(ph)
        ty = np.tan(th) * np.sin(ph)
        dz = z_bot - z_top
        x1, y1 = x0 + tx * dz, y0 + ty * dz
        if accept_only and (abs(x1) > half or abs(y1) > half):
            continue
        pad = 90.0
        out.append((np.array([x0 - tx * pad, y0 - ty * pad, z_top + pad]),
                    np.array([x1 + tx * pad, y1 + ty * pad, z_bot - pad])))
    return out


def add_tracks(p, tracks, radius=3.4, color=None):
    color = color or S.COL['track_mu']
    for a, b in tracks:
        p.add_mesh(M.tube(a, b, radius), **S.mat('glow', color))
    return tracks
