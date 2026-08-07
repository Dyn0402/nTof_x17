#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_sps.py -- the SPS H4 beam-test setup, in 3-D.

Builds the six-station telescope on the H4 rail:

    z =    0 mm   EIC uRWELL (front reference)
    z =  320 mm   P2 BASKET  IN
    z =  630 mm   P2 BASKET  MID
    z =  940 mm   P2 BASKET  OUT
    z = 1155 mm   MX17 "Detector E"          (optional; z is a placeholder)
    z = 1370 mm   EIC uRWELL (back reference)

The P2 chambers are drawn as the real fan they are -- the annulus sector back-
solved from the P2 BASKET Gerber map, with all 1280 pads as their true rotated
rectangles, mounted the way the P2 group states (bisector vertical, apex up,
130 mm from the lowest active point to the table).  Beam particles run along
+Z through the measured trigger aperture.

Frame: +Z downstream, +Y up, Y = 0 at the table top.
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv

import geometry as G
import meshes as M
import style as S

RNG = np.random.default_rng(20260807)

# Drawn margins around the metallised fan (visual; the Gerber board outline is
# not in the data we hold).
P2_PCB_MARGIN_R = 16.0        # board beyond the metallised radius, both ends
P2_PCB_MARGIN_PHI = 2.0       # deg
P2_FRAME_WIDTH_R = 26.0
P2_FRAME_WIDTH_PHI = 2.8      # deg
P2_FRAME_DEPTH = 26.0
PAD_GAP_FRACTION = 0.16       # cosmetic etch gap, see shrink_pads()

TABLE_Z = (-430.0, 1820.0)
TABLE_X = (-560.0, 560.0)


# --------------------------------------------------------------------------- #
def p2_frame_extent():
    """(r_lo, r_hi, phi_lo, phi_hi) of the drawn aluminium frame."""
    r_lo, r_hi = G.P2_R_ACTIVE
    p_lo, p_hi = G.P2_PHI_ACTIVE
    return (r_lo - P2_PCB_MARGIN_R - P2_FRAME_WIDTH_R,
            r_hi + P2_PCB_MARGIN_R + P2_FRAME_WIDTH_R,
            p_lo - P2_PCB_MARGIN_PHI - P2_FRAME_WIDTH_PHI,
            p_hi + P2_PCB_MARGIN_PHI + P2_FRAME_WIDTH_PHI)


def p2_outline3d(z, n=90):
    """Closed 3-D outline of one P2 chamber, for the cast-shadow projection."""
    fr_lo, fr_hi, fp_lo, fp_hi = p2_frame_extent()
    inner, outer = G.p2_band(fr_lo, fr_hi, fp_lo, fp_hi, n=n)
    ring = np.concatenate([outer, inner[::-1]])
    return np.stack([ring[:, 0], ring[:, 1], np.full(len(ring), z)], axis=1)


def shift(parts, x=0.0, y=0.0):
    """Translate a station's meshes and its shadow outline transversely."""
    if not (x or y):
        return parts
    for k, mesh in parts.items():
        if k == 'outline':
            continue
        parts[k] = mesh.translate((x, y, 0.0), inplace=False,
                                  transform_all_input_vectors=True)
    o = np.asarray(parts['outline'], float).copy()
    o[:, 0] += x
    o[:, 1] += y
    parts['outline'] = o
    return parts


def p2_meshes(z, pads_lab, sector_of_pad):
    """All meshes for one P2 BASKET station at rail position ``z``."""
    r_lo, r_hi = G.P2_R_ACTIVE
    p_lo, p_hi = G.P2_PHI_ACTIVE

    br_lo, br_hi = r_lo - P2_PCB_MARGIN_R, r_hi + P2_PCB_MARGIN_R
    bp_lo, bp_hi = p_lo - P2_PCB_MARGIN_PHI, p_hi + P2_PCB_MARGIN_PHI
    fr_lo, fr_hi, fp_lo, fp_hi = p2_frame_extent()

    pcb_band = G.p2_band(br_lo, br_hi, bp_lo, bp_hi)
    frame_bands = [
        G.p2_band(br_hi, fr_hi, fp_lo, fp_hi),          # outer rim
        G.p2_band(fr_lo, br_lo, fp_lo, fp_hi),          # inner rim
        G.p2_band(fr_lo, fr_hi, fp_lo, bp_lo),          # side bar, low phi
        G.p2_band(fr_lo, fr_hi, bp_hi, fp_hi),          # side bar, high phi
    ]

    lo, hi = G.P2_INSTRUMENTED_SECTORS
    live = (sector_of_pad >= lo) & (sector_of_pad <= hi)
    pads = shrink_pads(pads_lab, PAD_GAP_FRACTION)

    parts = M.fan_chamber(pcb_band, frame_bands, pads[~live], z,
                          G.P2_THICK_MM, frame_depth=P2_FRAME_DEPTH)
    parts['pads_live'] = M.quads_mesh(pads[live], z + G.P2_THICK_MM / 2 + 0.55)
    return parts


def shrink_pads(polys, frac):
    """Shrink each pad rectangle about its own centre.

    The map's pad_w/pad_h tile the fan with no gap, so drawn at full size the
    1280 pads read as one solid copper sheet.  Real boards carry a
    solder-mask/etch gap between pads; shrinking by ``frac`` restores the pad
    structure that makes a P2 BASKET recognisable.  Purely cosmetic -- the pad
    *centres, angles and count* are the measured ones.
    """
    c = polys.mean(axis=1, keepdims=True)
    return c + (polys - c) * (1.0 - frac)


def add_p2(p, z, pads_lab, sector_of_pad, spot=None, x=0.0, y=0.0):
    """Draw one P2 BASKET chamber; ``spot`` optionally shades the live pads by
    the measured illumination (0..1 per live pad)."""
    m = p2_meshes(z, pads_lab, sector_of_pad)
    m['outline'] = p2_outline3d(z)
    m = shift(m, x, y)
    p.add_mesh(m['frame'], **S.mat('alu_matte', S.COL['alu']))
    p.add_mesh(m['pcb'], **S.mat('pcb', S.COL['pcb']))
    p.add_mesh(m['pads'], **S.mat('copper', S.COL['copper_dead'],
                                  metallic=0.55, roughness=0.62))
    if spot is None:
        p.add_mesh(m['pads_live'], **S.mat('copper', S.COL['copper']))
    else:
        m['pads_live'].cell_data['illum'] = spot
        p.add_mesh(m['pads_live'], scalars='illum',
                   cmap=S.illumination_cmap(), clim=(0, 1),
                   show_scalar_bar=False,
                   **S.mat('copper', None, metallic=0.30, roughness=0.5))

    # support: a column from the table up to the bottom of the fan frame
    foot = G.P2_APEX_HEIGHT - p2_frame_extent()[1] + y
    if foot > 20:
        p.add_mesh(M.slab((x, foot / 2, z), 150, foot, 70, normal='z'),
                   **S.mat('alu_matte', S.COL['alu_dark']))
        p.add_mesh(M.slab((x, 14.0, z), 300, 28, 190, normal='y'),
                   **S.mat('alu_matte', S.COL['alu_dark']))
    return m


def add_urwell(p, z, label_side=+1, x=0.0, y=0.0):
    parts = M.rect_chamber(
        center=(x, G.SPS_BEAM_HEIGHT + y, z),
        pcb_size=(G.URW_PCB_MM, G.URW_PCB_MM),
        active_size=(G.URW_ACTIVE_MM, G.URW_ACTIVE_MM),
        frame_size=(G.URW_PCB_MM + 56, G.URW_PCB_MM + 56),
        pcb_thick=G.URW_THICK_MM, normal='z', drift_gap=None,
        n_strips=40)
    p.add_mesh(parts['frame'], **S.mat('alu', S.COL['alu']))
    p.add_mesh(parts['pcb'], **S.mat('pcb', S.COL['pcb_urwell']))
    p.add_mesh(parts['active'], **S.mat('copper', S.COL['copper']))
    p.add_mesh(parts['strips'], **S.mat('mesh', S.COL['copper_hot']))
    # stand: a post down to the table
    post_h = G.SPS_BEAM_HEIGHT + y - G.URW_PCB_MM / 2 - 28
    p.add_mesh(M.slab((x, post_h / 2, z), 70, post_h, 55, normal='z'),
               **S.mat('alu_matte', S.COL['alu_dark']))
    w = G.URW_PCB_MM + 56
    parts['outline'] = M.rect_outline((x, G.SPS_BEAM_HEIGHT + y, z), w, w,
                                      normal='z')
    return parts


def add_mx17(p, z, drift_dir=-1, yaw=0.0, x=0.0, y=0.0):
    """MX17 on the rail.  ``yaw`` is det_orientation.y -- rotation about the
    vertical through the chamber centre, i.e. the angle the chamber presents to
    the beam."""
    parts = M.rect_chamber(
        center=(x, G.SPS_BEAM_HEIGHT + y, z),
        pcb_size=(G.MX17_PCB_MM, G.MX17_PCB_MM),
        active_size=(G.MX17_ACTIVE_MM, G.MX17_ACTIVE_MM),
        frame_size=(G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM,
                    G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM),
        pcb_thick=8.0, normal='z',
        drift_gap=G.MX17_DRIFT_GAP_MM, drift_dir=drift_dir,
        n_strips=64)

    if abs(yaw) > 1e-9:
        pivot = (x, G.SPS_BEAM_HEIGHT + y, z)
        for k, mesh in parts.items():
            # transform_all_input_vectors: without it the stored Normals array
            # is left behind and the yawed chamber renders black
            parts[k] = mesh.rotate_y(yaw, point=pivot, inplace=False,
                                     transform_all_input_vectors=True)

    p.add_mesh(parts['frame'], **S.mat('alu', S.COL['alu']))
    p.add_mesh(parts['pcb'], **S.mat('pcb', S.COL['pcb']))
    p.add_mesh(parts['active'], **S.mat('copper', S.COL['copper']))
    p.add_mesh(parts['strips'], **S.mat('mesh', S.COL['copper_hot']))
    p.add_mesh(parts['gas'], **S.mat('gas', S.COL['gas']))
    p.add_mesh(parts['cathode'], **S.mat('plastic', '#dfe6ee', opacity=0.35))
    post_h = G.SPS_BEAM_HEIGHT + y - G.MX17_PCB_MM / 2 - G.MX17_FRAME_MM
    if post_h > 20:
        p.add_mesh(M.slab((x, post_h / 2, z), 120, post_h, 60, normal='z'),
                   **S.mat('alu_matte', S.COL['alu_dark']))
    w = G.MX17_PCB_MM + 2 * G.MX17_FRAME_MM
    centre = np.array([x, G.SPS_BEAM_HEIGHT + y, z])
    out = M.rect_outline(tuple(centre), w, w, normal='z')
    if abs(yaw) > 1e-9:
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        d = out - centre
        out = np.stack([c * d[:, 0] + s * d[:, 2], d[:, 1],
                        -s * d[:, 0] + c * d[:, 2]], axis=1) + centre
    parts['outline'] = out
    return parts


def add_table(p, theme, legs=True):
    cx, cz = (TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Z[0] + TABLE_Z[1]) / 2
    w, l = TABLE_X[1] - TABLE_X[0], TABLE_Z[1] - TABLE_Z[0]
    top = M.slab((cx, -25.0, cz), w, l, 50, normal='y')
    # a big flat plane facing the rig turns into one giant specular highlight
    # unless the lobe is tightened right down
    p.add_mesh(top, **S.mat('alu_matte', S.THEMES[theme]['floor'],
                            specular=0.05, specular_power=90, ambient=0.30))
    for xr in (-300.0, 300.0):
        rail = M.slab((xr, 12.0, cz), 80, l, 24, normal='y')
        p.add_mesh(rail, **S.mat('alu_matte', S.COL['alu_dark']))
    if legs:
        for sx in (TABLE_X[0] + 130, TABLE_X[1] - 130):
            for sz in (TABLE_Z[0] + 150, TABLE_Z[1] - 150):
                p.add_mesh(M.slab((sx, -430.0, sz), 80, 760, 80, normal='y'),
                           **S.mat('alu_matte', S.COL['alu_dark'],
                                   roughness=0.7))


def beam_tracks(n=14, z0=None, z1=None):
    """Parallel H4 tracks through the measured trigger aperture.

    Vertically the illumination is a hard-edged 125 mm slab (the external
    trigger scintillator, SPS_BEAM_GEOMETRY sect. 3b) -> uniform.
    Horizontally it is the beam itself, sigma = 28.6 mm -> Gaussian.
    Divergence < 0.5 mrad.
    """
    z0 = TABLE_Z[0] + 60 if z0 is None else z0
    z1 = TABLE_Z[1] - 60 if z1 is None else z1
    lo, hi = G.SPS_TRIGGER_SLAB
    ys = RNG.uniform(lo, hi, n)
    xs = RNG.normal(G.sps_beam_centre_lab()[0], G.SPS_BEAM_SIGMA_H, n)
    div = G.SPS_BEAM_DIVERGENCE_MRAD * 1e-3
    tx, ty = RNG.normal(0, div, n), RNG.normal(0, div, n)
    out = []
    for x, y, sx, sy in zip(xs, ys, tx, ty):
        out.append((np.array([x + sx * z0, y + sy * z0, z0]),
                    np.array([x + sx * z1, y + sy * z1, z1])))
    return out


def add_tracks(p, tracks, radius=3.0, color=None):
    color = color or S.COL['track_beam']
    for a, b in tracks:
        p.add_mesh(M.tube(a, b, radius), **S.mat('glow', color))
    return tracks


def add_beam_envelope(p, z0=None, z1=None):
    """The trigger acceptance: 2 sigma horizontally x the 125 mm slab."""
    z0 = TABLE_Z[0] + 60 if z0 is None else z0
    z1 = TABLE_Z[1] - 60 if z1 is None else z1
    lo, hi = G.SPS_TRIGGER_SLAB
    box = M.slab((G.sps_beam_centre_lab()[0], (lo + hi) / 2,
                  (z0 + z1) / 2),
                 4 * G.SPS_BEAM_SIGMA_H, hi - lo, z1 - z0, normal='z')
    p.add_mesh(box, **S.mat('envelope', S.COL['beam_env']))


# --------------------------------------------------------------------------- #
def load_pads_lab():
    """1280 P2 pads as lab-frame (x, height) rectangles, plus their sector."""
    m, polys, apex = G.load_p2_pads()
    lab = G.p2_pad_to_lab(polys, apex=apex)
    return lab, m.sector.values, m


def load_illumination(det='P2_MID', pattern='eff_nominal'):
    """Measured per-pad illumination, normalised to its peak.

    The stage-22 ``n_tag`` column is the number of tagged tracks pointing at
    each pad -- the *illumination*, unbiased by the probe plane's efficiency
    (SPS_BEAM_GEOMETRY_2026-07-31.md sect. 1).  Summed over the 10
    ``eff_nominal_1`` sub-runs = 15.1 M tagged tracks.

    Returns a pandas Series indexed by channel_id, or None if the maps are
    not present.
    """
    import glob
    import os
    import pandas as pd

    data = os.path.join(G.REPO, 'sps_beam_test_26', 'det4_sps_assessment',
                        'sps_beam_data')
    files = sorted(glob.glob(
        os.path.join(data, f'eff_map_{det}_{pattern}_*_spark_vetoed.csv')))
    if not files:
        return None
    tot = None
    for f in files:
        s = pd.read_csv(f).set_index('channel_id')['n_tag']
        tot = s if tot is None else tot.add(s, fill_value=0)
    return tot / tot.max()
