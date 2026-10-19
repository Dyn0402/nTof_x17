#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ntof_plan.py -- the setup and one event, straight down the beam, as a
drawing rather than as a render.

    ../.venv/bin/python make_ntof_plan.py

Writes ``figures/ntof_plan_light.png`` and ``.pdf`` (same drawing, live text).
The ``_light`` suffix is what ``make_report.py`` looks for; the deck is
light-only, so there is no second theme.

WHY A DRAWING AND NOT ANOTHER FRAME
-----------------------------------
The build-up sequence (``make_ntof.py``) already ends on a near-overhead
camera, and it is a *photograph*: it has perspective, the four arms sit at four
different distances from the lens, and every length on it is foreshortened by a
different amount.  That is the right thing for "what does this look like", and
the wrong thing for "how far is the trigger wall from the target".

This figure is the orthographic complement.  Beam along +Y, so the plan is the
X-Z plane at 1:1 in both directions: a length measured on it with a ruler is a
real length, the four arms are drawn identically because they *are* identical,
and the layer stack can carry a dimension chain.  Nothing is exaggerated -- the
23 mm capsule really is that small a dot at the middle of a 1.1 m apparatus,
and that is one of the things the drawing is for.

Geometry comes from ``scenes_ntof`` (and so from the Geant4 repository's
``plot_geometry.py``, and so from ``SimConfig.hh``), and the event from the
same ``data/ntof_event.json`` the 3-D frames use, so this cannot drift away
from them.

WHAT IS PROJECTED AWAY
----------------------
The beam axis.  Both legs also *rise* along the beam as they cross the
apparatus -- they leave the capsule going upward and reach the liquid ~135 mm
above the vertex -- and none of that is visible here.  The opening angle drawn
between them is therefore the projected one, and the box on the figure quotes
both it and the true space angle.  The event's bremsstrahlung gammas and its
knock-on electrons are not drawn, exactly as on the 3-D frames.

The drawn angle is WIDER than the space angle here (122 deg vs 110), which is
not a general rule and is worth knowing before it is asked from the floor.
Projection is not a bound in either direction:

    cos(theta_3D) = a_perp . b_perp + a_y b_y        (unit vectors)
    cos(theta_2D) = a_perp . b_perp / (|a_perp| |b_perp|)

Both legs of this event leave going UP (+y components +0.284 and +0.407, i.e.
polar 73.5 and 66.0 deg from the beam), so a_y b_y > 0 and it is the one term
pulling cos(theta) up, i.e. making the space angle less obtuse.  Drop it, then
renormalise by |a_perp||b_perp| < 1, and the drawn angle opens to 121.7 deg.
Had the two legs gone to opposite sides of the drawing plane the projection
would have SHRUNK the angle instead.  None of this is scattering: both numbers
come from the same primary directions at the vertex, and the 110.1 deg the
generator recorded reproduces to 110.2 deg from the drawn direction vectors.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plotstyle as P            # noqa: E402
import scenes_ntof as N          # noqa: E402

FIG = os.path.join(HERE, 'figures')
PG = N.PG
CM = N.CM
COL = N.COL

# Half-width of the drawing [mm].  Set by the longest thing in the plane, which
# is not a detector: it is the PMT of an arm whose vessel is laid on its side,
# reaching 495 mm out along the arm's own transverse axis.
HALF = 580.0

# The plan is seen FROM ABOVE, so with +X to the right, +Z runs down the page
# (the axis is inverted below).  Getting this backwards mirrors the pinwheel
# and swaps the arm letters, which is exactly the kind of error a figure like
# this exists to prevent, so it is stated once here and never re-derived.
ARM_LETTER = N.ARM_LETTER


# --------------------------------------------------------------------------- #
# Arm-local (u, w) -> plan (x, z)
# --------------------------------------------------------------------------- #
def place(arm, uw):
    """An arm-local (u, w) polyline as plan-view (x, z) points [mm]."""
    o = np.asarray(arm['ff_struct'], float) * CM
    u = np.asarray(arm['u_hat'], float)
    w = np.asarray(arm['w_hat'], float)
    uw = np.atleast_2d(np.asarray(uw, float))
    p = o[None, :] + uw[:, :1] * u[None, :] + uw[:, 1:] * w[None, :]
    return p[:, [0, 2]]


def quad(arm, u_lo, u_hi, w_lo, w_hi):
    """The rectangle u in [u_lo, u_hi], w in [w_lo, w_hi], placed."""
    return place(arm, [(u_lo, w_lo), (u_hi, w_lo), (u_hi, w_hi), (u_lo, w_hi)])


def patch(ax, pts, fc, ec=None, lw=0.8, alpha=1.0, z=2, ls='-'):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, alpha=alpha,
                         edgecolor=ec or 'none', linewidth=lw, zorder=z,
                         linestyle=ls, joinstyle='round'))


# --------------------------------------------------------------------------- #
# The apparatus, layer by layer -- the same order as the build-up frames
# --------------------------------------------------------------------------- #
EDGE = '#33404f'          # one outline colour for every solid, as on a drawing


def draw_chamber(ax, arm):
    u0 = pin_u = N.pin_u(arm)
    hu = N.MM_HU
    w_gas_f = (PG.tMylar + PG.tAlWin + PG.tKapCath + PG.tCuCath) * CM
    w_gas_b = w_gas_f + PG.tDrift * CM

    # The frame cheeks, which is all a plan view sees of a frame ring.  In this
    # projection they are the 20.3 mm strips between the active area and the
    # 440 mm outer square -- and their outer edge is the thing that has to clear
    # the neighbouring arm's window, so this is the one view where the frame
    # size is not decoration.  See scenes_ntof.FRAME_HU.
    for s in (-1, +1):
        patch(ax, quad(arm, u0 + s * hu, u0 + s * (hu + N.FRAME_W),
                       0.0, PG.w_PCB_b * CM),
              COL['frame'], EDGE, lw=0.6, z=2)
    # 30 mm of drift gas: the layer the talk is about, so it is the one filled
    # at full strength
    patch(ax, quad(arm, u0 - hu, u0 + hu, w_gas_f, w_gas_b),
          COL['gas'], EDGE, lw=0.8, alpha=0.55, z=3)
    patch(ax, quad(arm, u0 - hu, u0 + hu, PG.w_PCB_f * CM, PG.w_PCB_b * CM),
          COL['pcb'], EDGE, lw=0.8, z=3)
    # No line for the mylar entrance window.  It is 40 um -- a quarter of the
    # width of the stroke that would draw it -- and in the one colour it could
    # honestly be drawn in, red, it reads as a highlight box around the target
    # in almost the positron's own colour.  The gas's own front edge IS the
    # window, and the note says so.
    return pin_u


def draw_sipm(ax, arm):
    hu_w = PG.sipm_wall_hu * CM
    for s in (-1, +1):                                   # container cheeks
        patch(ax, quad(arm, s * hu_w, s * (hu_w + 12.0),
                       PG.w_sipm_f * CM, PG.w_sipm_b * CM),
              COL['frame_dark'], EDGE, lw=0.6, alpha=0.55, z=2)
    w_f, w_b = PG.w_sipm_sc_f * CM, PG.w_sipm_sc_b * CM
    hb = PG.bw * CM / 2
    for i, u_i in PG._sipm_bar_offsets():
        live = i in PG.SIPM_READOUT
        patch(ax, quad(arm, u_i * CM - hb, u_i * CM + hb, w_f, w_b),
              COL['sipm'] if live else COL['sipm_dead'], EDGE, lw=0.5,
              alpha=1.0 if live else 0.55, z=3)


def draw_plastics(ax, arm):
    i = arm['id']
    u0 = N.pin_u(arm)
    w_f, w_b = PG.w_bsc_f[i] * CM, PG.w_bsc_b[i] * CM
    hu = PG.bsc_u * CM / 2
    for s in (-1, +1):
        patch(ax, quad(arm, u0 + s * PG.bsc_u_offset * CM - hu,
                       u0 + s * PG.bsc_u_offset * CM + hu, w_f, w_b),
              COL['plastic'], EDGE, lw=0.8, z=3)


def _pillow(hu, ht, cap, w_mid, n=40):
    """The vessel's own (u, w) cross-section -- the same curve as the 3-D scene."""
    th = np.linspace(0, np.pi, n)
    u_arc = hu * np.cos(th)
    front = np.column_stack([u_arc, w_mid - ht - cap * np.sin(th)])
    back = np.column_stack([-u_arc, w_mid + ht + cap * np.sin(th)])
    return np.vstack([front, back])


def draw_ls(ax, arm):
    """The vessel from above.

    Two of the four arms carry theirs laid on its side (``LS_ROT`` = -90), and
    from above the two orientations genuinely look different: upright, the plan
    sees the pillow cross-section, with its domed edges; laid over, it sees the
    flat side of a 451 mm slab, and the neck and PMT come out sideways into the
    plane instead of pointing up the beam.  Both are drawn as they are.
    """
    i = arm['id']
    w_mid = PG.w_LS_mid[i] * CM
    hu_o, hv_o = PG.lsUo * CM, PG.lsVo * CM
    ht_o, cap, wall = PG.lsTo * CM, PG.hCap * CM, PG.lsWall * CM
    du, dv = PG.LS_OFF_U[i] * CM, PG.LS_OFF_V[i] * CM
    laid = abs(PG.LS_ROT[i]) > 1.0

    if laid:
        # rotating the vessel about the depth axis puts its long axis in the
        # plane, so the silhouette is the slab's flank, offset by the surveyed
        # (u, v) shift with u and v exchanged by the same rotation
        off = np.array([-dv, 0.0])
        outer = np.array([(-hv_o, w_mid - ht_o - cap), (hv_o, w_mid - ht_o - cap),
                          (hv_o, w_mid + ht_o + cap), (-hv_o, w_mid + ht_o + cap)])
        inner = outer + np.array([[wall, wall], [-wall, wall],
                                  [-wall, -wall], [wall, -wall]])
    else:
        off = np.array([du, 0.0])
        outer = _pillow(hu_o, ht_o, cap, w_mid)
        inner = _pillow(hu_o - wall, ht_o - wall, cap, w_mid)

    patch(ax, place(arm, outer + off), COL['ls_shell'], EDGE, lw=0.8,
          alpha=0.30, z=2)
    patch(ax, place(arm, inner + off), COL['ls_liquid'], None, lw=0.0,
          alpha=0.80, z=3)

    if not laid:
        return
    # funnel -> neck -> PMT, out along the arm's own transverse axis
    a0 = hv_o
    a1 = a0 + PG.lsFunL * CM
    a2 = a1 + PG.lsNkL * CM
    r = PG.lsNkR * CM
    patch(ax, place(arm, np.array(
        [(a0, w_mid - ht_o - cap), (a1, w_mid - r), (a1, w_mid + r),
         (a0, w_mid + ht_o + cap)]) + off),
        COL['ls_shell'], EDGE, lw=0.7, alpha=0.30, z=2)
    patch(ax, quad(arm, a1 + off[0], a2 + off[0], w_mid - r, w_mid + r),
          COL['ls_shell'], EDGE, lw=0.7, alpha=0.30, z=2)
    r_p = PG.CFG['ls_pmt_r_cm'] * CM
    patch(ax, quad(arm, PG.pmtFaceV * CM + off[0],
                   (PG.pmtFaceV + PG.CFG['ls_pmt_len_cm']) * CM + off[0],
                   w_mid - r_p, w_mid + r_p),
          COL['pmt'], EDGE, lw=0.7, alpha=0.95, z=4)


# The standoff circle -- one faint dashed circle every arm's mylar window sits
# on -- was drawn here until 2026-08-12.  Dropped on Dylan's call: the four
# windows are visibly on one circle without it, the dimension chain already
# states the 204.5 mm, and at this scale the circle passes close enough to the
# chamber frames to read as a part rather than as a construction line.


def draw_capsule(ax):
    """The target, at 1:1 with everything else -- which is the point of it."""
    from matplotlib.patches import Circle
    for r, c, a in ((PG.RO_CFRP.max() * CM, COL['cfrp'], 1.0),
                    (PG.RO_AL.max() * CM, COL['al'], 1.0),
                    (PG.RO_GAS.max() * CM, COL['he3'], 1.0)):
        ax.add_patch(Circle((0, 0), r, facecolor=c, edgecolor=EDGE, lw=0.4,
                            zorder=6, alpha=a))


# --------------------------------------------------------------------------- #
# The event
# --------------------------------------------------------------------------- #
def draw_event(ax, ev):
    from matplotlib.patches import Circle

    v = np.asarray(ev['pair']['vertex'], float)
    for leg in ev['pair']['legs']:
        p = N.clip_to_scene(np.asarray(leg['points'], float))
        c = N.track_color(leg['particle'])
        ax.plot(p[:, 0], p[:, 2], color=c, lw=2.6, zorder=7,
                solid_capstyle='round', solid_joinstyle='round')

    # Deposits are drawn in the LEG's colour, not the layer's.  On the 3-D
    # frames a deposit is coloured by the layer it lands in, which works
    # because it is a lit sphere sitting in front of that layer; on a flat
    # drawing a lavender dot inside a lavender bar is not there at all.  In the
    # leg's colour, every marker also says which particle left it.
    for d in ev['pair']['deposits']:
        col = N.track_color(d['particle'])
        if d['layer'] == 'mm' and 'a' in d:
            # 30 mm of ionisation along the track, which is the thing the
            # micro-TPC actually images -- a segment, not a point
            a, b = np.asarray(d['a']), np.asarray(d['b'])
            ax.plot([a[0], b[0]], [a[2], b[2]],
                    color=N.DEPOSIT_COL['mm'], lw=4.0, zorder=8,
                    solid_capstyle='round')
            continue
        r = 3.0 + 8.0 * (min(d['edep_MeV'], 2.0) / 2.0) ** (1 / 3)
        ax.add_patch(Circle((d['p'][0], d['p'][2]), r, facecolor=col,
                            edgecolor=P.SURFACE, lw=0.5, alpha=0.9, zorder=8))

    # The vertex, small on purpose: the capsule under it is only 23 mm across,
    # and a marker big enough to see from the back of a room would cover the
    # one object on this drawing whose size is the point.
    ax.add_patch(Circle((v[0], v[2]), 3.6, facecolor='#ffffff',
                        edgecolor=P.INK, lw=0.9, zorder=9))


def leg_summary(ev):
    """(particle, energy, arm letter, exit direction) for the two legs."""
    out = []
    for leg, E, arm in zip(ev['pair']['legs'], ev['pair']['E_MeV'],
                           ev['pair']['arms']):
        p = np.asarray(leg['points'], float)
        d = p[1] - p[0]
        out.append(dict(particle=leg['particle'], E=E, arm=ARM_LETTER[arm],
                        dir=d / np.linalg.norm(d), end=p[-1]))
    return out


def projected_opening(legs):
    """The opening angle as the DRAWING shows it, i.e. in the X-Z plane."""
    a, b = [np.array([l['dir'][0], l['dir'][2]]) for l in legs]
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def deposit_table(ev):
    """MeV per (layer, arm letter), summed over the whole event."""
    tot = {}
    for d in ev['pair']['deposits']:
        tot[(d['layer'], ARM_LETTER[d['arm']])] = \
            tot.get((d['layer'], ARM_LETTER[d['arm']]), 0.0) + d['edep_MeV']
    return tot


# --------------------------------------------------------------------------- #
# Type on the drawing
# --------------------------------------------------------------------------- #
def draw_labels(ax, ev, legs):
    """Arm letters, the layer key, the target callout and the event box."""
    import matplotlib.patheffects as pe

    halo = [pe.withStroke(linewidth=2.6, foreground=P.SURFACE, alpha=0.92)]

    # --- arm letters, on the axis just outside the last vessel --------------
    for arm in N.ARMS:
        w = (PG.w_LS_mid[arm['id']] + PG.lsTo + PG.hCap) * CM + 30.0
        (x, z), = place(arm, [(0.0, w)])
        hit = arm['id'] in ev['pair']['arms']
        ax.text(x, z, f"arm {ARM_LETTER[arm['id']]}", ha='center', va='center',
                fontsize=13, fontweight='bold',
                color=P.INK if hit else P.MUTED, zorder=10,
                path_effects=halo, rotation=_arm_rot(arm))

    # --- the target, to scale ----------------------------------------------
    # Just the name.  The Ø23 mm and the "drawn to scale" were saying in words
    # what the drawing says by being 1:1 -- and the whole point of this figure
    # is that you can measure the capsule off it.
    ax.annotate('³He',
                xy=(-13.0, 13.0), xytext=(-46.0, 62.0),
                ha='right', va='center', fontsize=10.5, color=P.INK,
                zorder=10, path_effects=halo,
                arrowprops=dict(arrowstyle='-', lw=1.0, color=P.MUTED,
                                shrinkA=2, shrinkB=3))

    # --- the beam, which is the axis of the whole drawing -------------------
    # The free corner is only ~320 mm wide (x < -262 is clear of every arm),
    # and only down to the vessel of the arm at the top of the frame, so this
    # block is kept short and pushed right up against the edge.
    ax.text(-HALF + 18, -HALF + 8,
            'Seen from above.\nThe neutron beam\ncomes out of the page.',
            ha='left', va='top', fontsize=10.0, color=P.MUTED, zorder=10,
            linespacing=1.35)

    # --- which leg is which, named ALONG the leg -----------------------------
    # Not at the far end: out there the type would land on the vessel the leg
    # stops in.  The 200 mm inside the standoff circle is the one part of the
    # plan with nothing in it, so the legs are named as they cross it -- set
    # along the track and offset off it, the way a drawing labels a line, which
    # is also the only way two 200 mm-long words fit in a 400 mm-wide hole.
    for l in legs:
        sym = 'e⁻' if l['particle'] == 'e-' else 'e⁺'
        d = np.array([l['dir'][0], l['dir'][2]])
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])              # +90 deg in DATA coordinates
        p = d * 120.0 + n * 26.0
        # z runs down the page, so a data-frame angle is the negative of the
        # screen angle; then flip through 180 deg if it would set the type
        # upside down
        rot = np.degrees(np.arctan2(-d[1], d[0]))
        rot = (rot + 90) % 180 - 90
        ax.text(p[0], p[1], f"{sym}  {l['E']:.1f} MeV",
                ha='center', va='center', fontsize=11.0, fontweight='bold',
                color=N.track_color(l['particle']), zorder=10,
                rotation=rot, rotation_mode='anchor', path_effects=halo)


def _arm_rot(arm):
    """Read the arm letter along the arm, not across it."""
    w = np.asarray(arm['w_hat'], float)
    return {(1, 0): 90, (-1, 0): 90, (0, 1): 0, (0, -1): 0}[
        (int(round(w[0])), int(round(w[2])))]


def key_handles():
    """Proxy artists for the layer key, which lives UNDER the drawing.

    Inside the frame it would have to go in a corner, and the corners are the
    only places left for the two things that have to be near the apparatus --
    the orientation note and the event.  A key is the one block of type that
    does not care where it is.
    """
    from matplotlib.patches import Patch
    rows = [(COL['gas'], 0.55, '30 mm drift gas'),
            (COL['pcb'], 1.0, 'readout board'),
            (COL['sipm'], 1.0, 'SiPM trigger bars'),
            (COL['plastic'], 1.0, 'plastic scintillator'),
            (COL['ls_liquid'], 0.80, 'liquid scintillator')]
    return [Patch(facecolor=c, alpha=a, edgecolor=EDGE, lw=0.6, label=t)
            for c, a, t in rows]


def _arm_sum(tot, letter):
    return sum(v for (_, arm), v in tot.items() if arm == letter)


def draw_event_box(ax, ev, legs):
    """The event, as numbers, in the corner the tracks leave free."""
    tot = deposit_table(ev)
    a, b = legs[0]['arm'], legs[1]['arm']
    s = sum(tot.values())
    # Short lines on purpose.  Everything outside x = -262 mm is clear of all
    # four arms at any z (the widest thing on the drawing is the SiPM container,
    # 500 mm plus its 12 mm cheeks), and that is a 300 mm-wide column measured
    # from the type's left margin.  About 24 characters at this size; a longer
    # line runs out over the vessel of the arm at the bottom of the frame.
    lines = [
        (f'Geant4 event #{ev["provenance"]["pair_event"]}', True),
        (f'{ev["pair"]["opening_deg"]:.0f}° opening in space', False),
        (f'({projected_opening(legs):.0f}° as drawn)', False),
        (f'{s:.1f} of {sum(ev["pair"]["E_MeV"]):.1f} MeV seen', True),
        (f'arm {a} {_arm_sum(tot, a):.1f}, arm {b} {_arm_sum(tot, b):.1f} MeV',
         False),
    ]
    dz = 26.0
    z0 = HALF - 18 - dz * (len(lines) - 1)
    for k, (text, bold) in enumerate(lines):
        ax.text(-HALF + 18, z0 + k * dz, text, ha='left', va='center',
                fontsize=9.6, color=P.INK if bold else P.MUTED,
                fontweight='bold' if bold else 'normal', zorder=10)


def draw_dimensions(ax, arm):
    """A dimension chain out along one arm, in the plane, at 1:1.

    This is the thing a render cannot do, and the reason the drawing exists:
    on a perspective frame these four radii are four different amounts of
    foreshortened, so they can only be written in a caption.  Here they can be
    measured off the page.

    The numbers are computed from the same geometry that drew the layers, so a
    survey change moves the tick AND its label together.
    """
    u_line = -(PG.lsUo * CM + 58.0)                # just outside the vessels
    r0 = abs(float(np.asarray(arm['ff_struct'], float) @
                   np.asarray(arm['w_hat'], float)) * CM)        # 204.5
    stops = [0.0, PG.w_sipm_sc_f * CM, PG.w_bsc_f[arm['id']] * CM,
             PG.w_LS_mid[arm['id']] * CM]
    (x_a, z_a), (x_b, z_b) = place(arm, [(u_line, stops[0]),
                                         (u_line, stops[-1])])
    ax.annotate('', xy=(x_b, z_b), xytext=(x_a, z_a), zorder=9,
                arrowprops=dict(arrowstyle='<->', lw=0.9, color=P.MUTED,
                                shrinkA=0, shrinkB=0))
    for w in stops:
        (x, z), = place(arm, [(u_line, w)])
        ax.plot([x, x], [z - 7, z + 7], color=P.MUTED, lw=0.9, zorder=9)
        ax.text(x, z - 12, f'{r0 + w:.0f}', ha='center', va='bottom',
                fontsize=9.5, color=P.MUTED, zorder=10)
    # Above the chain, not below it.  Below, it lands on the trigger-wall
    # strip of the very arm being dimensioned, which runs the full width of
    # that arm; above, it is out past every layer.
    # ... and pushed out to the frame edge rather than centred on the chain,
    # because centred it reaches back over the end of that arm's trigger wall.
    (_, z), = place(arm, [(u_line, 0.0)])
    ax.text(-HALF + 18, z - 34, 'mm from beam axis', ha='left',
            va='bottom', fontsize=9.0, color=P.MUTED, zorder=10)


# --------------------------------------------------------------------------- #
def figure(ev, bare=False):
    """The drawing.  ``bare`` drops the headline and the provenance note.

    Same split as the X17 diagrams: a slide has its own title bar and its own
    caption, and repeating them inside the figure costs the drawing the height
    it needs.  The report keeps the titled version, which is the one that has
    to stand on its own.
    """
    import matplotlib.pyplot as plt

    P.use()
    # 7.8 in, not the 9.6 the deck's charts use.  Type is in POINTS, so the
    # figure's size is what sets how big the annotation is against the drawing;
    # on a slide this figure lives in a half-width column, and at 9-plus inches
    # the dimension numbers arrive at the back of the room as grey specks.
    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    ax.set_aspect('equal')
    ax.grid(False)

    for arm in N.ARMS:
        draw_chamber(ax, arm)
        draw_sipm(ax, arm)
        draw_plastics(ax, arm)
        draw_ls(ax, arm)
    draw_capsule(ax)
    draw_event(ax, ev)

    legs = leg_summary(ev)
    draw_dimensions(ax, N.ARMS[1])          # arm B: the one no leg crosses
    draw_labels(ax, ev, legs)
    draw_event_box(ax, ev, legs)

    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.invert_yaxis()                       # from ABOVE: +Z runs down the page
    ax.set_xlabel('x  [mm]')
    ax.set_ylabel('z  [mm]')
    P.strip(ax)
    # Three columns, not five: laid out in one row the key is wider than the
    # drawing it belongs to, and a tight bounding box then pads the whole
    # figure out to the key's width and shrinks the drawing to pay for it.
    ax.legend(handles=key_handles(), loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, -0.085), frameon=False, fontsize=10,
              handlelength=1.4, handleheight=1.0, columnspacing=1.4,
              borderpad=0.0)

    if bare:
        return fig
    head = 'The four arms and one pair, straight down the beam'
    P.title(ax, head,
            'orthographic — every length on this drawing is 1:1 in both axes')
    # The house title pad is set for the deck's wide, short charts; this one is
    # square, so the deck line lands on the headline's descenders.
    ax.set_title(head, loc='left', color=P.INK, pad=32)
    P.note(fig,
           'Geometry from the Geant4 SimConfig via MX17_Full_Geant/scripts/'
           'plot_geometry.py; per-arm distances and vessel rotations as '
           'surveyed 2026-07-17/18. Event: the same simulated pair as the 3-D '
           'build-up (data/ntof_event.json). The beam axis is projected '
           'away, so the legs also rise ~135 mm along it before reaching the '
           'liquid; both legs rise, which is why the drawn opening angle is '
           'the wider of the two. '
           'Markers are energy deposits, in the depositing leg’s colour '
           'and sized by how much; bremsstrahlung gammas and knock-on '
           'electrons are not drawn.')
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(FIG,
                                                  'ntof_plan_light.png'))
    ap.add_argument('--no-pdf', action='store_true')
    ap.add_argument('--bare', action='store_true',
                    help='no headline and no provenance note -- the version '
                         'the slide uses, since the slide carries both')
    args = ap.parse_args()

    if args.bare and args.out == os.path.join(FIG, 'ntof_plan_light.png'):
        args.out = os.path.join(FIG, 'ntof_plan_bare_light.png')
    ev = N.load_event()
    fig = figure(ev, bare=args.bare)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(args.out)
    print(f'  -> {args.out}')
    if not args.no_pdf:
        pdf = os.path.splitext(args.out)[0] + '.pdf'
        fig.savefig(pdf)
        print(f'  -> {pdf}')
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == '__main__':
    main()
