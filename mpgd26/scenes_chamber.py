#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_chamber.py -- one MX17 chamber, pulled apart.

The "how a resistive-strip micro-TPC chamber works" figure: the layers of a
single MX17 stacked along the drift axis and separated so each one is visible,
with a muon crossing the drift volume and its ionisation drifting down to the
mesh.

Layer stack, top (cathode) to bottom (laminate).  THE READOUT SIDE IS THE
AS-BUILT MX17 BOARD (re-sourced 2026-08-17 from MX17_Geant, which reads it from
the gerbers and from shared/MX17ModuleGeometry.hh -- the same header
MX17_Full_Geant uses, mirrored in scripts/model/mx17_model.py):

  drift cathode        50 um kapton + 9 um Cu cladding            [CAD]
  drift volume         30 mm, kapton back to mesh front           [CAD/run config]
  micromesh            woven, 19 um wire / 48 um opening          [placeholder]
  amplification gap    150 um, mesh to the resistive film         [garfield_sim/
                                                                   mm_config.py]
  (1) ESL resistive strips  550 um wide, 250 um gaps -> 0.80 mm   [confirmed
                       pitch, a 10 um screen-printed film. NOT the   2026-08-06]
                       0.78 mm readout pitch; the pads show through.
  (2) L4 readout PADS  0.68 mm square on the 0.78 mm grid --      [gerber]
                       the layer this figure was missing until
                       2026-08-17.  88 % Cu over the active area.
  (3) L5 Y strips      0.5 mm wide on 0.78 mm, running along x    [gerber]
  (4) L6 X strips      0.5 mm wide on 0.78 mm, running along y    [gerber]
  laminate             1.70 mm for the WHOLE board (mesh front to [CAD]
                       laminate back), 5 Cu layers of 26 um in it

The numbering and the colours are those of the board-peel figure that sits
beside this one on the slide (MX17_Geant scripts/model/plot_mx17_model.py), so
the two pictures are describing the same object in the same terms.

Only the *thicknesses along the drift axis* are exaggerated -- an exploded view
of a 10 um film under a 30 mm gap is otherwise invisible -- and the exaggeration
is not uniform, so no scale can be read off the vertical axis.  Everything in
the plane (pitches, widths, pad size, gaps) is real.

Frame: +Z along the drift direction (up), X/Y transverse.
"""
from __future__ import annotations

import numpy as np

import geometry as G
import meshes as M
import style as S

RNG = np.random.default_rng(20260807)

# --- the real stack, mm ------------------------------------------------------
AMP_GAP_MM = 0.150            # garfield_sim/mm_config.py: GAP_CM = 0.0150
N_STRIPS = 512                # mx17_m1_map.csv
# N_STRIPS strip centres span N_STRIPS-1 pitches, not N_STRIPS. Dividing by the
# strip COUNT is the off-by-one that produced the old 0.7785 mm.
STRIP_PITCH_MM = G.MX17_ACTIVE_MM / (N_STRIPS - 1)   # 0.780 mm, the design pitch
STRIP_WIDTH_MM = 0.500        # gerber: 0.5 mm copper on the 0.78 mm pitch

# The screen-printed ESL film has its OWN pitch -- 0.80 mm, not the 0.78 mm of
# the readout -- so the resistive strips walk slowly across the pads underneath.
# That is not a detail to smooth over: it is why the pads show through the gaps
# in the board-peel figure next to this one.
RESIST_PITCH_MM = 0.800
RESIST_WIDTH_MM = 0.550       # 550 um wide, 250 um gaps (confirmed 2026-08-06)

PAD_PITCH_MM = 0.780          # L4: 0.68 mm square pads on the readout grid
PAD_SIZE_MM = 0.680

MESH_WIRE_MM, MESH_OPEN_MM = 0.019, 0.048   # placeholder weave, NEEDED_INPUTS
PCB_TOTAL_MM = 1.70           # the whole board, mesh front to laminate back

# The four readout layers keep the colours of the board-peel figure beside this
# one (plot_mx17_model.py), so a colour means the same layer in both pictures.
COL_RESIST = '#1c1c1c'        # (1) ESL resistive strips -- black
COL_PADS = '#e09a55'          # (2) L4 pads
COL_L5 = '#b87333'            # (3) L5 Y strips
COL_L6 = '#8a5a28'            # (4) L6 X strips

# --- what the figure draws (a WINDOW on the chamber, not the whole 40 cm) ----
# A RECTANGLE since 2026-08-17, not a square (Dylan: "instead of squares, we
# should look at a rectangular subset of the detector to use the width of the
# screen").  The stack is ~110 mm tall as drawn and the slide gives the figure
# about 62 % of the page width, so a square window is the wrong shape twice
# over: it wastes the width, and paying for that width with height is what made
# the layers small.  A wide window fills the frame and every layer gets bigger.
#
# It is still a WINDOW at the real pitch -- a cut-out of a 400 mm chamber -- so
# nothing about the drawing became less true; there is just more of it.
#
# ZOOMED IN on 2026-08-17 (Dylan: "zoom in a bit more to see the strip
# structure"): 60 x 18 mm rather than 120 x 30.  The pitch is 0.78 mm, so what
# is on screen is ~77 strips across the window instead of 154 -- twice the
# pixels per strip, which is the difference between strip structure and moire.
# EXPLODE came down with it: the drawn stack height and the window width set the
# figure's shape between them, so zooming in without closing the gaps turns a
# landscape figure back into a portrait one.
#
# DEEPER on 2026-08-17 (Dylan: "make the planes extend further back into the
# page, keeping the current perspective"): 18 -> 34 mm along the strips.  At
# 18 mm the layers read as ribbons -- the depth direction had barely more
# extent than the exploded gaps between the layers, so the eye saw a stack of
# edges rather than a stack of PLANES.  34 mm is where the far edge of the
# stack reaches the top-right of the frame without the near corner of the
# laminate leaving the bottom of it (measured at this camera; 48 mm clips).
#
# ZOOMED AGAIN on 2026-08-17 (Dylan: "zoom in on this track horizontally to
# better show the detail on the strips/pads, while keeping the same width"):
# 60 -> 44 mm across.  The frame width is unchanged -- make_chamber.VIEW's
# view_angle came down 17.8 -> 16.6 with it -- so this is a pure magnification:
# 56 strips across the window instead of 77, each ~1.4x bigger on the projected
# slide.  The DEPTH is deliberately untouched at 34 mm, so the planes still go
# as far back into the page as they did; only the strip direction is zoomed.
# The track had to be re-centred for it (add_track_and_drift), because at 44 mm
# the old placement put the muon's bottom end past the left edge.
WIN_MM = (44.0, 34.0)         # (x, y) size of the cut-out shown


def _win(window):
    """Accept a scalar (square, the old call) or an (x, y) pair."""
    if np.isscalar(window):
        return float(window), float(window)
    wx, wy = window
    return float(wx), float(wy)


def n_across(extent):
    """Strips fitting across ``extent`` at the real pitch."""
    return max(2, int(round(extent / STRIP_PITCH_MM)))


# Exploded spacing along z, mm (drawn, not physical).  Came down 19 -> 7.5 with
# the zoom: see the WIN_MM note -- the two together set the figure's shape.
EXPLODE = 7.5


def layers(explode=EXPLODE, window=WIN_MM):
    """(name, z_centre, thickness, label) for each layer, bottom to top.

    ``z`` is the *drawn* height; the physical stack is in the label text.
    """
    # Drawn thicknesses.  The laminate is the only one that is nearly honest:
    # the whole board is 1.70 mm, and it used to be drawn 7 mm thick against a
    # 1.6 mm strip layer, which read as a slab of FR4 carrying some foil.  It is
    # now the thinnest structural thing in the picture, which is what it is.
    t_pcb, t_cu, t_resist, t_mesh, t_cath = 2.0, 0.9, 0.8, 1.1, 0.8
    z = 0.0
    out = []

    # The label text is deliberately SHORT (shortened 2026-08-17): it now sits
    # on the render beside the layer, so every character of it is width taken
    # from the picture.  What was in the parentheses -- spark protection, the
    # mylar, "amplification" -- is in the caption and in the speaker's mouth.
    #
    # The layer ORDER and the numbering are the board's, not a cartoon's: the
    # ESL film is on top, then the L4 pads, then L5 (Y, running along x), then
    # L6 (X, running along y).  X above Y, which is what this figure drew until
    # 2026-08-17, is the wrong way round.
    out.append(('pcb', z, t_pcb,
                f'Readout PCB\n{PCB_TOTAL_MM:.2f} mm'))
    z += t_pcb / 2 + explode + t_cu / 2
    out.append(('strips_x', z, t_cu,
                'L6  X strips\nalong y'))
    z += t_cu / 2 + explode * 0.55 + t_cu / 2
    out.append(('strips_y', z, t_cu,
                f'L5  Y strips\n{N_STRIPS} × {STRIP_PITCH_MM:.2f} mm'))
    z += t_cu / 2 + explode * 0.55 + t_cu / 2
    out.append(('pads', z, t_cu,
                f'L4  pads\n{PAD_SIZE_MM:.2f} mm square'))
    z += t_cu / 2 + explode * 0.55 + t_resist / 2
    out.append(('resist', z, t_resist,
                f'Resistive strips\n{RESIST_WIDTH_MM * 1000:.0f} µm / '
                f'{RESIST_PITCH_MM:.2f} mm'))
    z += t_resist / 2 + explode * 0.55 + t_mesh / 2
    out.append(('mesh', z, t_mesh,
                f'Micromesh\n{AMP_GAP_MM * 1000:.0f} µm gap'))
    z_gas0 = z + t_mesh / 2 + explode
    out.append(('gas', z_gas0 + G.MX17_DRIFT_GAP_MM / 2, G.MX17_DRIFT_GAP_MM,
                f'Drift volume\n{G.MX17_DRIFT_GAP_MM:.0f} mm, Argon/Isobutane'))
    z = z_gas0 + G.MX17_DRIFT_GAP_MM + t_cath / 2
    out.append(('cathode', z, t_cath, 'Drift cathode'))
    return out


def build(p, explode=EXPLODE, window=WIN_MM, track=True):
    """Draw the exploded chamber; returns {layer: anchor point} for labelling."""
    L = {n: (z, t, lab) for n, z, t, lab in layers(explode, window)}
    anchors = {}
    wx, wy = _win(window)
    hx, hy = wx / 2, wy / 2

    # The anchor is the layer's LEFT front corner, because the labels sit down
    # the left-hand side of the render (2026-08-17).  Anchoring on the right,
    # as the portrait version did, would send every leader line straight across
    # the picture.
    def anchor(name, z):
        anchors[name] = (-hx * 0.99, -hy * 0.99, z)

    # readout PCB -- the laminate the four copper layers live in
    z, t, _ = L['pcb']
    p.add_mesh(M.slab((0, 0, z), wx, wy, t), **S.mat('pcb', S.COL['pcb']))
    anchor('pcb', z)

    # L6 X strips (along y) then L5 Y strips (along x): the board's order, and
    # the direction convention of the gerbers -- a "Y strip" MEASURES y, so it
    # runs along x.  Each sits on a sliver of laminate so it reads as copper on
    # a board rather than as a floating grid.
    for name, along, col in (('strips_x', 'v', COL_L6),
                             ('strips_y', 'u', COL_L5)):
        z, t, _ = L[name]
        p.add_mesh(M.slab((0, 0, z - t / 2), wx, wy, t * 0.30),
                   **S.mat('pcb', '#123f38'))
        # the count follows the extent the strips are pitched ACROSS, so the
        # pitch on the page is the real 0.78 mm in both views
        n = n_across(wy if along == 'u' else wx)
        strips = M.strip_lines(n, (-hx, hx), (-hy, hy),
                               z, along=along, width=STRIP_WIDTH_MM)
        p.add_mesh(strips, **S.mat('copper', col))
        anchor(name, z)

    # L4 pads: 0.68 mm square copper on the 0.78 mm readout grid -- the layer
    # the drift charge actually lands on, and what this figure was missing
    z, t, _ = L['pads']
    p.add_mesh(M.slab((0, 0, z - t / 2), wx, wy, t * 0.30),
               **S.mat('pcb', '#123f38'))
    p.add_mesh(M.pad_grid((-hx, hx), (-hy, hy), PAD_PITCH_MM, PAD_SIZE_MM, z),
               **S.mat('copper', COL_PADS))
    anchor('pads', z)

    # the ESL film: black strips on their own 0.80 mm pitch over a kapton
    # sheet -- deliberately NOT the readout pitch, which is why the pads show
    # through the gaps in the board-peel figure beside this one
    z, t, _ = L['resist']
    # the 50 um kapton the film is printed on: muted on purpose, so it reads as
    # the substrate under the black strips and not as a layer of its own
    p.add_mesh(M.slab((0, 0, z - t / 2), wx, wy, t * 0.34),
               **S.mat('plastic', '#d8bd85', opacity=0.75))
    p.add_mesh(M.strip_lines(int(round(wx / RESIST_PITCH_MM)),
                             (-hx, hx), (-hy, hy), z,
                             along='v', width=RESIST_WIDTH_MM),
               **S.mat('plastic', COL_RESIST, specular=0.25))
    anchor('resist', z)

    # micromesh: a woven grid, drawn as two crossed sets of fine bars.  The
    # bars keep the same drawn pitch in x and y, so a wide window gets more of
    # them rather than stretched ones
    z, t, _ = L['mesh']
    mesh_pitch = 30.0 / 26
    for along in ('u', 'v'):
        n_mesh = max(4, int(round((wy if along == 'u' else wx) / mesh_pitch)))
        p.add_mesh(M.strip_lines(n_mesh, (-hx, hx), (-hy, hy),
                                 z + (0.35 if along == 'u' else -0.35),
                                 along=along, width=mesh_pitch * 0.40),
                   **S.mat('mesh', '#6f7883'))
    anchor('mesh', z)

    # drift volume + cathode
    z, t, _ = L['gas']
    gas = M.slab((0, 0, z), wx, wy, t)
    p.add_mesh(gas, **S.mat('gas', S.COL['gas'], opacity=0.11))
    p.add_mesh(gas.extract_feature_edges(), color=S.COL['gas'], line_width=2.4,
               lighting=False, opacity=0.8)
    anchor('gas', z)

    zc, tc, _ = L['cathode']
    p.add_mesh(M.slab((0, 0, zc), wx, wy, tc),
               **S.mat('plastic', '#dfe7ef', opacity=0.42))
    anchor('cathode', zc)

    if track:
        add_track_and_drift(p, L, window)
    return anchors


def add_track_and_drift(p, L, window):
    """A muon through the drift volume, plus the ionisation drifting to the mesh.

    This is the micro-TPC picture the whole reconstruction rests on: each
    primary cluster drifts down at v_drift, so its arrival time measures its
    depth, and the depth-vs-position slope is the track angle.
    """
    wx, wy = _win(window)
    z0, t0, _ = L['gas']
    z_bot, z_top = z0 - t0 / 2, z0 + t0 / 2
    z_mesh = L['mesh'][0]

    # the muon: enters the top of the drift volume at an angle.  The track is
    # CENTRED on the window rather than offset from it (2026-08-17): the window
    # is now barely wider than the track's own horizontal span, so the old
    # "-0.30 x half-window" placement ran it off the left edge.
    tanx, tany = 0.42, 0.14
    span = tanx * (z_top - z_bot)                 # how far it walks in x
    x_top, y_top = 0.5 * span, -0.15 * min(wy, 30.0)
    a = np.array([x_top, y_top, z_top])
    b = np.array([x_top + tanx * (z_bot - z_top),
                  y_top + tany * (z_bot - z_top), z_bot])
    ext = 0.28 * (z_top - z_bot)
    d = (b - a) / np.linalg.norm(b - a)
    # THIN since 2026-08-17 (Dylan: "make the track and drift lines
    # significantly smaller").  The old 0.9 mm tube was 1.2 strip pitches
    # across -- drawn at the same scale as the structure it is supposed to be
    # crossing, which made the muon read as a rod lying on the chamber.  A real
    # MIP leaves a track whose transverse size is far below the pitch, and the
    # picture is now zoomed in far enough that the difference is visible.
    p.add_mesh(M.tube(a - d * ext, b + d * ext, 0.30),
               **S.mat('glow', S.COL['track_mu']))

    # primary ionisation clusters along it, and their drift down to the mesh.
    # More of them, each smaller: at the old size and spacing the clusters
    # merged into a dashed rod.
    n = 15
    for f in np.linspace(0.05, 0.95, n):
        q = a + (b - a) * f
        p.add_mesh(M.cylinder(tuple(q), (0, 0, 1), radius=0.42, height=0.45),
                   **S.mat('glow', '#ffd166'))
        # drift line: straight down to the mesh plane
        p.add_mesh(M.tube(q, (q[0], q[1], z_mesh + 1.2), 0.10),
                   **S.mat('glow', '#8fd8ea', ambient=0.55))
