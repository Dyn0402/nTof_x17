#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_target.py -- the n_TOF Target #3 spallation target, in detail.

The facility figure (``scenes_ear2.py``) draws the target because the neutrons
come from it, at a scale where the whole 20 m beam line fits on one slide.  At
that scale the target is 130 px tall and the things that make it interesting --
9.85 mm cooling plates, a 4 mm window, a 3 mm channel -- are a quarter of a pixel
each.  This module draws the same object at a scale where they are not, for the
two backup slides that answer "how does that target actually work?".

Two views, and they are deliberately different KINDS of picture:

``build_layers``
    The whole assembly, cut open on a vertical plane along the proton beam, from
    the cradle up to the vacuum window.  It answers "what is it made of, in what
    order" -- which for this talk means the stack a neutron on its way to EAR2
    crosses, because that stack is what sets the EAR2 spectrum.

``build_cooling``
    Two lead slices and the anti-creep plate between them, at ~6x that, with the
    machined channels, a wedge obstruction and the nitrogen path drawn.  It
    answers "how is a 2.7 kW lead block that melts at 327 C and creeps at 135 C
    kept alive by a gas".

SINGLE SOURCE OF TRUTH: every shared dimension is imported from
``scenes_ear2``, never re-typed.  The two figures are of the same object and a
divergence between them would be worse than either being wrong on its own.

Provenance -- all of it from the design paper:

  R. Esposito et al. (for the n_TOF Collaboration), "Design of the
  third-generation lead-based neutron spallation target for the neutron
  time-of-flight facility at CERN", Phys. Rev. Accel. Beams 24 (2021) 093001,
  arXiv:2106.11242

      Sec. III A   six lead slices, the anti-creep plates, the 9.85 mm thickness
      Sec. III B   the vessel, its two windows, the two moderators
      Sec. II B    the 4 cm water layer
      Sec. II C    the 5 cm lead plate above the core
      Sec. IV A    the cradle plenums, the flow deflectors, the wedge
                   obstructions, and the 10 deg beam-to-target angle
      Sec. IV B    the flow rates, pressure drops, velocities, temperatures
      Sec. V       the thermal, stress and creep results
      Sec. VI      the nitrogen cooling station

and the hemispherical vacuum window from J. A. Pavon-Rodriguez et al. (n_TOF),
Eur. Phys. J. A 61 (2025), arXiv:2505.00042, Sec. 3.

What is drawn rather than sourced is in ``ASSUMPTIONS`` and on both slides.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

import meshes as M
import style as S
import scenes_ear2 as E

COL = dict(E.COL)
COL.update(
    # Nitrogen: it has to be legible against BOTH the dark lead and the bright
    # aluminium, and it must not be the moderator's blue -- one is a coolant at
    # 20 C and the other is what makes the neutron spectrum, and a reader who
    # conflates them has lost the plot of the whole figure.
    n2='#6ec8a8',
    n2_hot='#e08a5a',                     # the same gas, on the way out warm
    tc='#c2434a',                         # thermocouples
    # Three metals in one picture that must not merge.  Lead is
    # E.COL['lead'] = #7d838c; the plate is pushed BRIGHTER than the
    # shared aluminium and the groove interiors much DARKER than either,
    # because at the shared values (#c8ccd2 on #98a1ac on #7d838c) all
    # three land within 0.2 in value and the figure is one grey slab.
    plate='#dde1e7',
    groove='#4c545f',
    wedge='#c98b52',                      # a warm metal, so a wedge is
                                          # neither plate nor groove
)

# 10 deg in the HORIZONTAL plane, i.e. a yaw about the vertical axis.  Sourced,
# and not a detail: it is there "to reduce the EAR1 background caused by gamma
# rays and high-energy charged particles" (Sec. IV A), and it is why the beam
# spot on the lead is an ellipse and why the temperature maps in the paper are
# not symmetric about the beam axis.
TGT_YAW = 10.0

# Channels machined in the anti-creep plates.  The DEPTH is sourced -- 3 mm, from
# the creep section, where the 0.64 mm of lead that flows into a channel over
# twice the target lifetime is compared against it.  The COUNT and the width are
# not: the paper shows rows of them in Figs. 11-12 and only ever counts them
# relatively ("three channels obstructed", "five channels"), so 13 is a drawn
# number chosen to look like the figures.  See ASSUMPTIONS.
CH_DEPTH = 3.0
N_CHANNELS = 13
CH_FRAC = 0.62                            # channel width as a fraction of pitch

# Wedge-shaped obstructions go in the channels FARTHER from the beam axis, to
# push flow towards the channels that are on it.  The paper does not say how many
# carry a wedge; the drawn split is the outer third each side.
WEDGE_FRAC = 0.34

N_TC = 6                                  # radiation-hard thermocouples, sourced

ASSUMPTIONS = [
    'The COUNT and width of the cooling channels are drawn, not sourced: the '
    'design paper shows rows of them but only ever counts them relatively '
    '("three channels obstructed", "five channels"). Their 3 mm DEPTH is '
    'sourced, from the creep analysis. How many channels carry a wedge is also '
    'drawn -- the paper says only that the wedges go in the channels farther '
    'from the beam axis.',
    'Wall thicknesses the paper does not give are drawn: the vessel wall (only '
    'the 3 mm proton window and the 4 mm neutron window are quoted), the '
    'moderator cans\' walls and their plan size, and how far the EAR1 moderator '
    'extends along the beam. The radius of the hemispherical aluminium vacuum '
    'window is drawn at the pipe bore.',
    'The cradle is drawn as a plenum box with two arteries and the flow '
    'deflectors indicated. Its real internal shape is a CFD-optimised curved '
    'volume (paper Fig. 10b) and is not reproduced; what the figure claims is '
    '"the gas arrives here, is slowed down, and is turned up into the '
    'channels".',
    'The six thermocouples are drawn on the slice faces. The paper says they '
    'are "placed inside the target, directly in contact with the lead slices" '
    'and does not give positions.',
    'The nitrogen arrows are a drawn flow topology -- in through the cradle, up '
    'the channels, out of the top of the vessel. Directions are from the '
    'paper; the arrows are not a velocity field.',
    'Lead is drawn as a solid grey. The real slices are cast high-purity lead '
    '(UNS L50006, >= 99.98 wt%) and the drawing says nothing about their '
    'surface or grain.',
]

CITATION = ('R. Esposito et al. (n_TOF), Phys. Rev. Accel. Beams 24 (2021) '
            '093001, arXiv:2106.11242 · hemispherical vacuum window from '
            'J. A. Pavon-Rodriguez et al. (n_TOF), Eur. Phys. J. A 61 (2025), '
            'arXiv:2505.00042')

# --------------------------------------------------------------------------- #
# The cutaway, same idiom as scenes_ear2: a vertical plane through the beam axis
# with the near half removed, so you look at real inner walls.
# --------------------------------------------------------------------------- #
_CUT = None


def set_cut(normal):
    global _CUT
    if normal is None:
        _CUT = None
        return
    n = np.array([normal[0], 0.0, normal[2]], float)
    _CUT = tuple(n / np.linalg.norm(n))


def cut(mesh):
    if _CUT is None or mesh is None:
        return mesh
    out = mesh.clip(normal=_CUT, origin=(0, 0, 0), invert=True)
    return out if out.n_points else None


def add(p, mesh, **kw):
    """``add_mesh``, tolerant of the cutaway removing a part entirely."""
    if mesh is not None and mesh.n_points:
        p.add_mesh(mesh, **kw)


def _yaw(mesh, deg=TGT_YAW):
    """Rotate about the vertical axis -- the 10 deg beam-to-target angle."""
    return mesh.rotate_y(deg, point=(0.0, 0.0, 0.0), inplace=False)


def box(x0, x1, y0, y1, z_size):
    return M.slab(((x0 + x1) / 2, (y0 + y1) / 2, 0.0),
                  x1 - x0, z_size, y1 - y0, normal='y')


def slice_edges():
    """(x0, x1) of each lead slice and of each anti-creep plate, along the beam.

    One function, used by both views and by ``make_target.py``'s anchors, so the
    stack cannot be laid out twice and differ.
    """
    lead, plates = [], []
    x = -E.TGT_LEN / 2.0
    for i, t in enumerate(E.TGT_SLICES):
        lead.append((x, x + t))
        x += t
        if i < len(E.TGT_SLICES) - 1:
            plates.append((x, x + E.ACP_T))
            x += E.ACP_T
    return lead, plates


# --------------------------------------------------------------------------- #
# View A -- the layers
# --------------------------------------------------------------------------- #
def build_layers(p, cut_normal=None):
    """The whole assembly, cut open along the beam.  Returns label anchors."""
    set_cut(cut_normal)
    lead, plates = slice_edges()
    xv = E.TGT_LEN / 2.0 + E.VES_GAP
    half = E.TGT_XY / 2.0

    for x0, x1 in lead:
        add(p, cut(_yaw(box(x0, x1, -half, half, E.TGT_XY))),
            **S.mat('alu_matte', COL['lead'], opacity=1.0))
    for x0, x1 in plates:
        # PBR here, unlike everywhere else in these two figures.  The cutaway
        # leaves all six slices and all five plates on ONE flat plane, so the only
        # thing separating them is shade -- and at this light rig a matte
        # #dde1e7 plate and a matte #7d838c slice both saturate towards the same
        # pale grey and the core comes out as one blank block.  The metallic
        # highlight is what makes 9.85 mm of aluminium visible between 50 mm of
        # lead.  It costs some apparent thickness; the plate's real thickness is
        # shown in the cooling view, and labelled here.
        add(p, cut(_yaw(box(x0, x1, -half, half, E.TGT_XY))),
            **S.mat('alu', COL['al'], opacity=1.0))

    # the vessel, translucent, and its two named windows at full strength
    add(p, cut(_yaw(box(-xv, xv, -E.Y_VES, E.Y_VES, E.TGT_XY + 2 * E.VES_GAP))),
        **S.mat('alu_matte', COL['steel'], opacity=0.13))
    # the 3 mm proton window: a thinned patch on the upstream face, drawn as an
    # inset panel because a 3 mm wall against an 18 mm one is the whole point
    add(p, cut(_yaw(box(-xv - 3.0, -xv, -180.0, 180.0, 360.0))),
        **S.mat('alu_matte', COL['flange'], opacity=0.95))
    # the 4 mm neutron window, welded to the top, which carries the lead plate
    add(p, cut(_yaw(box(-xv, xv, E.Y_VES, E.Y_NWIN, E.TGT_XY + 2 * E.VES_GAP))),
        **S.mat('alu_matte', COL['steel'], opacity=1.0))
    mh = E.MOD_XY / 2.0
    add(p, cut(_yaw(box(-mh, mh, E.Y_NWIN, E.Y_PB, E.MOD_XY))),
        **S.mat('alu_matte', COL['lead'], opacity=1.0))

    # the EAR2 moderator: aluminium can, 4 cm of water.  Inset from the vessel
    # top by a ledge (E.MOD_XY), which is where the gas outlets go.
    add(p, cut(_yaw(box(-mh, mh, E.Y_PB, E.Y_MOD, E.MOD_XY))),
        **S.mat('alu', COL['al'], opacity=0.30))
    add(p, cut(_yaw(box(-mh + E.MOD_WALL, mh - E.MOD_WALL,
                        E.Y_PB + E.MOD_WALL, E.Y_MOD - E.MOD_WALL,
                        E.MOD_XY - 2 * E.MOD_WALL))),
        **S.mat('gas', COL['water'], opacity=0.72))
    dome = pv.Sphere(radius=E.R_VACWIN, center=(0, E.Y_MOD, 0),
                     theta_resolution=56, phi_resolution=56).clip(
                         normal=(0, -1, 0), origin=(0, E.Y_MOD, 0), invert=True)
    add(p, cut(dome), **S.mat('alu', COL['al'], opacity=0.42,
                              smooth_shading=True))

    # the EAR1 moderator, downstream, quiet on purpose (see scenes_ear2)
    add(p, cut(_yaw(box(xv, xv + E.EAR1_MOD_X, -half, half, E.TGT_XY))),
        **S.mat('alu', COL['al'], opacity=0.20))

    # the cradle, and the nitrogen through it
    yc0 = -E.Y_VES - E.CRADLE_T
    add(p, cut(_yaw(box(-xv, xv, yc0, -E.Y_VES, E.TGT_XY + 2 * E.VES_GAP))),
        **S.mat('alu', COL['al'], opacity=0.55))
    for zs in (-1, 1):                    # the two gas distribution arteries
        art = pv.Cylinder(center=(0.0, yc0 + E.CRADLE_T * 0.42,
                                  zs * E.TGT_XY * 0.28),
                          direction=(1, 0, 0), radius=26.0,
                          height=2 * xv * 0.92, resolution=32)
        add(p, cut(_yaw(art)), **S.mat('glow', COL['n2'], opacity=0.85))
    _n2_in(p, -xv, yc0 + E.CRADLE_T * 0.42)
    _n2_out(p, xv, E.Y_VES + E.NWIN_T)

    # the protons: along +X, NOT yawed -- the target is what is turned
    for m in M.tracks_with_heads([((-1250.0, 0.0, 0.0), (-xv - 6.0, 0.0, 0.0))],
                                 radius=18.0, head_len=120.0, head_radius=42.0):
        add(p, m, **S.mat('glow', COL['proton'], opacity=1.0))

    # the neutrons that this figure is about: up, out through the moderator
    for j in (-1, 0, 1):
        a = np.array([j * 78.0, 130.0, j * 34.0])
        b = np.array([j * 52.0, E.Y_MOD + E.R_VACWIN + 130.0, j * 22.0])
        for m in M.tracks_with_heads([(a, b)], radius=7.5, head_len=70.0,
                                     head_radius=19.0):
            add(p, m, **S.mat('glow', COL['neutron'], opacity=1.0))

    x2 = lead[1]                          # the hottest slice, 85-89 C
    return dict(
        protons=(-820.0, 0.0, 0.0),
        yaw=(-xv - 40.0, -half * 0.62, 0.0),
        slices=(sum(x2) / 2, -half * 0.34, 0.0),
        thick=(sum(lead[5]) / 2, half * 0.30, 0.0),
        plates=(sum(plates[2]) / 2, half * 0.66, 0.0),
        vessel=(-xv * 0.72, -E.Y_VES, 0.0),
        pwin=(-xv - 3.0, 120.0, 0.0),
        nwin=(half * 0.66, E.Y_VES + E.NWIN_T / 2, 0.0),
        pbplate=(-half * 0.55, (E.Y_NWIN + E.Y_PB) / 2, 0.0),
        moderator=(half * 0.30, (E.Y_PB + E.Y_MOD) / 2 + 6.0, 0.0),
        vacwin=(E.R_VACWIN * 0.52, E.Y_MOD + E.R_VACWIN * 0.72, 0.0),
        ear1=(xv + E.EAR1_MOD_X * 0.55, half * 0.30, 0.0),
        cradle=(0.0, yc0 + E.CRADLE_T * 0.42, 0.0),
        n2_in=(-xv - 210.0, yc0 + E.CRADLE_T * 0.42, 0.0),
        n2_out=(-xv - 200.0, E.Y_VES + 190.0, E.TGT_XY * 0.26),
        neutrons=(70.0, E.Y_MOD + E.R_VACWIN + 150.0, 0.0),
    )


def _n2_in(p, x, y):
    """Cold nitrogen arriving at the cradle."""
    for m in M.tracks_with_heads([((x - 330.0, y, 0.0), (x - 40.0, y, 0.0))],
                                 radius=15.0, head_len=80.0, head_radius=36.0):
        add(p, m, **S.mat('glow', COL['n2'], opacity=1.0))


def _n2_out(p, x, y):
    """... and leaving the top of the vessel warm, through the gas outlets."""
    for zs in (-0.22, 0.22):
        a = np.array([-x + 16.0, y + 6.0, zs * E.TGT_XY])
        b = np.array([-x - 260.0, y + 220.0, zs * E.TGT_XY * 1.35])
        for m in M.tracks_with_heads([(a, b)], radius=11.0, head_len=64.0,
                                     head_radius=26.0):
            add(p, m, **S.mat('glow', COL['n2_hot'], opacity=1.0))


# --------------------------------------------------------------------------- #
# View B -- the cooling, at the scale of one plate
# --------------------------------------------------------------------------- #
def _wedge(t, z, w, y0=-E.TGT_XY * 0.34, h=230.0):
    """One wedge-shaped obstruction in a channel.

    Drawn as a lens that **narrows the channel over a stretch of its height**,
    with the gas passing either side of it, because that is the mechanism the
    paper describes -- "the nitrogen flowing through the narrow sections beside
    the obstruction wedges can reach velocities between 70 m/s and 87 m/s".

    The alternative, a ramp tapering in DEPTH, is what the real part looks like in
    the paper's Fig. 12 and was tried first.  It cannot be seen: the groove is
    drawn as a solid dark box, so a wedge sitting inside it is occluded down to
    the 0.3 mm of apex that protrudes, which renders as a sliver.  Narrowing the
    width is the same statement -- restrict this channel, so the ones on the beam
    axis get the flow -- in a form the reader can actually see.  ASSUMPTIONS says
    which one is drawn.
    """
    zc, hw = z, w * 0.40
    poly = np.array([[y0 - h / 2, zc],
                     [y0 - h * 0.18, zc + hw],
                     [y0 + h * 0.18, zc + hw],
                     [y0 + h / 2, zc],
                     [y0 + h * 0.18, zc - hw],
                     [y0 - h * 0.18, zc - hw]])
    # Front face 0.45 mm proud of the groove's own face (which sits at
    # t + 0.3), or the solid dark groove box occludes it and we are back to a
    # sliver.
    return M.polygon_prism(poly, t - CH_DEPTH + 0.75, CH_DEPTH,
                           normal_axis='x')


def build_cooling(p, cut_normal=None):
    """One anti-creep plate, exploded off its lead slice.  Returns label anchors.

    Drawn at the scale the paper's own Figs. 11-12 use, because the mechanism is
    a 3 mm groove in a 9.85 mm plate and there is no honest way to show that on
    the same picture as a 20 m beam line.  In order, what it has to say: the
    channels are milled in the ALUMINIUM, not in the lead; the ones on the beam
    axis are left clear while the ones away from it are throttled by a wedge, so
    the fastest gas is where the beam is; and the plate's other job is to stop
    the lead creeping into the grooves it just made.

    Two idioms were tried and dropped before this one, and both failures are
    instructive.  (1) The **beam-axis cutaway** the other views use: that plane
    runs diagonally across a 10 mm plate and slices it into a wedge -- right for a
    tube, wrong for a face.  (2) A **sandwich with the downstream slice cut
    back**: legible in principle, but lead, plate and groove are three greys
    within 0.2 of each other in value, so the whole thing came out as one slab.
    Hence: **exploded**, so nothing occludes anything, and the groove interiors
    pushed to a much darker slate than either metal.
    """
    set_cut(None)
    half = E.TGT_XY / 2.0
    t = E.ACP_T
    pb = 90.0                             # a slice of lead, exploded off behind
    gap = 190.0                           # the explosion gap

    # the lead slice, behind, at low relief -- it is here to say "this plate
    # lives between two of these", not to be looked at
    add(p, box(-gap - pb, -gap, -half, half, E.TGT_XY),
        **S.mat('alu_matte', COL['lead'], opacity=0.92))

    # the plate itself, bright, and the grooves in it dark
    # Phong, not PBR.  ``S.mat('alu')`` drives VTK's PBR path off the studio
    # cubemap, and a flat 600 mm plate then comes out as polished chrome whose
    # reflection streaks are indistinguishable from the 3 mm slots milled in it --
    # the same trap scenes_ear2 documents for the inside of the beam pipe.
    add(p, box(0.0, t, -half, half, E.TGT_XY),
        **S.mat('alu_matte', COL['plate'], opacity=1.0))

    pitch = E.TGT_XY / N_CHANNELS
    w = pitch * CH_FRAC
    n_edge = int(round(N_CHANNELS * WEDGE_FRAC / 2.0))
    mid = N_CHANNELS // 2
    zc = -half + pitch * (mid + 0.5)
    for i in range(N_CHANNELS):
        z = -half + pitch * (i + 0.5)
        outer = i < n_edge or i >= N_CHANNELS - n_edge
        add(p, M.slab((t - CH_DEPTH / 2 + 0.3, 0.0, z), CH_DEPTH, w,
                      E.TGT_XY, normal='y'),
            **S.mat('plastic', COL['groove'], opacity=1.0))
        if outer:
            # the wedge-shaped obstruction, in a warm metal so that it cannot be
            # mistaken for either the plate or the groove
            add(p, _wedge(t, z, w), **S.mat('plastic', COL['wedge'],
                                            opacity=1.0))

    # Arrows in five channels, not thirteen: three on the axis drawn long and
    # bright, two outer ones short, which is the whole statement.
    for idx, fast in ((mid - 1, True), (mid, True), (mid + 1, True),
                      (1, False), (N_CHANNELS - 2, False)):
        z = -half + pitch * (idx + 0.5)
        y1 = half * (0.90 if fast else 0.44)
        for m in M.tracks_with_heads([((t + 11.0, -half * 0.88, z),
                                       (t + 11.0, y1, z))],
                                     radius=6.0 if fast else 4.4,
                                     head_len=38.0, head_radius=15.0):
            add(p, m, **S.mat('glow', COL['n2'] if fast else COL['n2_hot'],
                              opacity=1.0))

    # NO proton beam in this view, deliberately.  The 10 deg beam-to-target
    # angle is a yaw about the VERTICAL axis, so it is an angle you can only see
    # from above; on a view looking at the plate face it foreshortens to nothing
    # and a beam drawn here would read as perpendicular, which is the opposite of
    # the truth.  It is shown in ``build_layers``, where the horizontal proton
    # arrow and the yawed stack are both in frame, and stated on the slide.
    # a thermocouple, in contact with the lead
    # z > 0, i.e. screen-LEFT, which is the only side of the exploded lead slice
    # the plate does not cover -- at z < 0 the thermocouple is behind the plate
    # and its leader points at nothing.  Its label is in the left column to match.
    tc = pv.Cylinder(center=(-gap - pb * 0.5, -half * 0.66, half * 0.62),
                     direction=(0, 0, 1), radius=10.0, height=260.0,
                     resolution=24)
    add(p, tc, **S.mat('glow', COL['tc'], opacity=1.0))

    # ANCHOR SIDES: at this azimuth VTK's screen-right is (n_z, 0, -n_x) with
    # n_x > n_z, so **+Z projects to screen-LEFT**.  Anchors for the left-hand
    # label column therefore need z > 0 and the right-hand ones z < 0.  Getting
    # this backwards is not subtle -- every leader then crosses the whole plate to
    # reach its part, which is what the first version did.
    return dict(
        plate=(t, half * 0.96, half * 0.60),
        lead_up=(-gap - pb * 0.5, half * 0.52, half * 0.82),
        wedge=(t + 11.0, -half * 0.34, half - pitch * 0.5),
        channel=(t + 11.0, half * 0.62, zc - pitch),
        flow=(t + 11.0, half * 0.26, -half + pitch * 1.5),
        creep=(t - CH_DEPTH / 2, -half * 0.78, zc - pitch * 2.0),
        tc=(-gap - pb * 0.5, -half * 0.66, half * 0.62),
    )


def layers_center_scale():
    """Focal point and characteristic size for view A [mm]."""
    top = E.Y_MOD + E.R_VACWIN
    bot = -E.Y_VES - E.CRADLE_T
    return (55.0, (top + bot) / 2.0 + 30.0, 0.0), 1180.0


def cooling_center_scale():
    """... and for view B.  Pulled upstream of the plate so the exploded lead
    slice behind it is inside the frame rather than clipped by the margin."""
    return (-70.0, 0.0, 0.0), 800.0
