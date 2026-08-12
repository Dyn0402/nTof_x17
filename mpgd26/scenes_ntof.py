#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_ntof.py -- the n_TOF EAR2 setup, built up layer by layer.

Four MX17 chambers in a pinwheel around the ³He capsule, then the SiPM trigger
wall, the plastic scintillators and the liquid-scintillator calorimeters, with
a neutron arriving up the beam and one real Geant4 e⁺e⁻ pair leaving.

Frame: the simulation's own -- **beam along +Y** (EAR2 is a vertical beam line,
so +Y is up), X and Z transverse, origin at the capsule centre, millimetres.

Geometry provenance
-------------------
Every dimension is imported from ``MX17_Full_Geant/scripts/plot_geometry.py``,
which is itself written against ``SimConfig.hh`` -- so this figure and the
simulation cannot drift apart, and the measured records behind them
(GEOMETRY_COORDINATE_CONVENTION.md, GEOMETRY_CHANGE_CHECKLIST.md) apply to the
figure unchanged.  Set ``MX17_FULL_GEANT`` if that repository is not at
``~/CLionProjects/MX17_Full_Geant``.

One number is typed here rather than imported: the chambers' 440 mm outer frame
square, which ``plot_geometry`` does not carry.  It is measured too -- CAD plus
two gerber cross-checks, ``MX17_Full_Geant/docs/HANDOFF_ARM_GEOMETRY.md`` -- and
it is what sets the clearance between neighbouring arms, so see the comment at
``FRAME_HU`` before changing it.

Three things here are drawn rather than measured, and all three are flagged in
the report: the liquid-scintillator dome is extruded at constant height along
the vessel's long axis (the real fill bulges away at the edges); the neutron
beam envelope is drawn at the radius the simulation's own sampled start points
occupy, which is the beam profile, not the collimator; and the neutron itself
is a straight line up that axis to the pair vertex rather than a transported
Geant4 trajectory (see ``add_neutron``).

Arms are built once in a local (u, v, w) frame -- u across the chamber, v along
the beam, w into the stack -- and placed by a rigid transform per arm.  The
pinwheel is right-handed for all four, so one transform serves everything from
the mylar window to the PMT.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pyvista as pv

import meshes as M
import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
EVENT_JSON = os.path.join(DATA, 'ntof_event.json')

CM = 10.0                      # plot_geometry works in cm, this scene in mm


# --------------------------------------------------------------------------- #
# The simulation's geometry module
# --------------------------------------------------------------------------- #
def _load_plot_geometry():
    root = os.environ.get('MX17_FULL_GEANT',
                          os.path.expanduser('~/CLionProjects/MX17_Full_Geant'))
    scripts = os.path.join(root, 'scripts')
    if not os.path.isfile(os.path.join(scripts, 'plot_geometry.py')):
        raise SystemExit(
            f'cannot find plot_geometry.py under {scripts!r}.\n'
            f'This scene imports the detector geometry from the Geant4 '
            f'repository rather than re-typing it; point MX17_FULL_GEANT at '
            f'your checkout of MX17_Full_Geant.')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import plot_geometry as PG                                  # noqa: E402
    return PG


PG = _load_plot_geometry()

ARMS = PG.ARM_DEF                      # 0=D(+X) 1=B(-X) 2=A(+Z) 3=C(-Z)
ARM_LETTER = {0: 'D', 1: 'B', 2: 'A', 3: 'C'}     # as the experiment names them

# Arms between the camera and the target.  They are drawn see-through so the
# capsule, the tracks and the two arms the pair actually crossed stay visible --
# the alternative (a clipping plane) cuts the near chambers in half and reads as
# a broken detector rather than a full one.
#
# 1 = B (-X) is the arm the camera looks straight through; 2 = A (+Z) is the one
# on the right of the frame, and it is ghosted too because at this azimuth its
# stack is seen almost edge-on and so presents a solid slab of scintillator
# across the near half of the picture.  The simulated pair crossed arms 0 (D)
# and 3 (C), which are exactly the two that stay solid -- keep this, the camera
# azimuth and extract_ntof_event.py's --prefer-arms in step with each other.
NEAR_ARMS = (1, 2)


def set_near_arms(arms):
    """Which arms the camera looks through, and so which are ghosted."""
    global NEAR_ARMS
    NEAR_ARMS = tuple(arms)

# --------------------------------------------------------------------------- #
# Palette.  e+/e-/neutron/gamma are the same colours as the physics-case
# diagram (scenes_x17.palette), so a track means the same thing on every slide.
# --------------------------------------------------------------------------- #
COL = dict(
    gas=S.COL['gas'],
    mylar='#c0392b',
    pcb=S.COL['pcb'],
    frame=S.COL['alu'],
    frame_dark=S.COL['alu_dark'],
    sipm='#f0c040',
    sipm_dead='#8d8672',
    # Lavender, not the package's usual scintillator blue: the plastics sit two
    # layers out from a 30 mm slab of drift gas drawn in #6fd0e8, and a light
    # blue beside a light cyan reads as the same material twice.  Lavender is
    # the one pastel left that is far in hue from all four of the gas, the gold
    # trigger bars, the orange liquid and the red/crimson tracks.
    plastic='#b49ae6',
    ls_liquid='#e0703c',
    ls_shell='#3a4049',
    pmt=S.COL['pmt'],
    he3='#8ad0f0',
    al='#c8ccd2',
    cfrp='#2b2f36',
    beam='#9aa4b0',           # grey: the beam is scene furniture, not a signal
    beam_dart='#6b7580',      # the direction dart, a shade down so it reads
    neutron='#8b96a3',
    positron='#d6402c',
    electron='#a02c52',
    gamma='#c1841c',
    product='#3f8f6b',
)
# Deposit markers.  Each layer's marker is its own colour EXCEPT the drift gas:
# a marker in the gas colour on top of the gas is invisible, and that layer is
# the one the talk is actually about, so it gets an ice-blue that reads against
# the teal of the drift volume.
DEPOSIT_COL = dict(mm='#bdf0ff', sipm=COL['sipm'], plastic=COL['plastic'],
                   ls=COL['ls_liquid'])


# --------------------------------------------------------------------------- #
# Arm placement
# --------------------------------------------------------------------------- #
def arm_transform(arm):
    """4x4 mapping the arm-local frame (u, v, w) onto the world.

    The origin is the arm's *mechanical* front face (the SiPM wall's reference);
    the chambers, plastics and vessels carry the per-arm pinwheel shift as a
    local-u offset, exactly as the survey quotes it.
    """
    T = np.eye(4)
    T[:3, 0] = arm['u_hat']
    T[:3, 1] = (0.0, 1.0, 0.0)
    T[:3, 2] = arm['w_hat']
    T[:3, 3] = np.asarray(arm['ff_struct'], float) * CM
    return T


def pin_u(arm):
    """The pinwheel shift as a local-u offset [mm] (ff = ff_struct - u * shift)."""
    return -PG.PINWHEEL[arm['id']] * CM


def place(mesh, T):
    return mesh.transform(T, inplace=False)


def lbox(u, v, w, hu, hv, hw):
    """A local-frame box from its centre and half-sizes [mm]."""
    return pv.Box(bounds=(u - hu, u + hu, v - hv, v + hv, w - hw, w + hw))


# --------------------------------------------------------------------------- #
# One chamber
# --------------------------------------------------------------------------- #
MM_HU = PG.HW_U['mm'] * CM
MM_HV = PG.HW_V['mm'] * CM

# The frame/flange outer profile is a 440 mm square CENTRED on the active axis
# (as-built, CAD + two gerber cross-checks; MX17_Full_Geant/docs/
# HANDOFF_ARM_GEOMETRY.md).  It is a fixed square, not a constant cheek around
# the active area -- and the difference started to matter on 2026-08-11, when
# the measured active area (39.9 x 36.0 cm) replaced the unsourced 38 x 34.  A
# constant 30 mm ring around the old size happened to give exactly 440; around
# the new one it gives 490, and 490 does not fit: the pinwheel has only
# 15.5-17.3 mm of tangential shift to buy clearance with, and 440 is precisely
# what that clears.  Drawing the ring instead of the square puts ~10 mm of
# interpenetration between neighbouring arms into the plan view.
#
# Because the active area is no longer square, neither is the cheek: 20.3 mm
# across the chamber, 40.0 mm along the beam, the second of which is where the
# passivated band lives.
FRAME_HU = FRAME_HV = 220.0
FRAME_W = FRAME_HU - MM_HU     # the cheek a plan view sees, 20.3 mm
FRAME_D = 34.0                 # frame depth


def chamber_meshes(arm):
    """Mylar window, drift volume, readout board and frame, in the local frame."""
    u0 = pin_u(arm)
    w_gas_f = (PG.tMylar + PG.tAlWin + PG.tKapCath + PG.tCuCath) * CM
    w_gas_b = w_gas_f + PG.tDrift * CM
    w_pcb_f, w_pcb_b = PG.w_PCB_f * CM, PG.w_PCB_b * CM

    out = dict(
        window=lbox(u0, 0, PG.tMylar * CM / 2, MM_HU, MM_HV, PG.tMylar * CM),
        gas=lbox(u0, 0, (w_gas_f + w_gas_b) / 2, MM_HU, MM_HV,
                 (w_gas_b - w_gas_f) / 2),
        pcb=lbox(u0, 0, (w_pcb_f + w_pcb_b) / 2, MM_HU, MM_HV,
                 (w_pcb_b - w_pcb_f) / 2),
        frame=M.frame_ring((u0, 0, (w_gas_f + w_pcb_b) / 2),
                           2 * FRAME_HU, 2 * FRAME_HV,
                           2 * MM_HU, 2 * MM_HV, FRAME_D, normal='z'),
    )
    return out


# --------------------------------------------------------------------------- #
# SiPM trigger wall
# --------------------------------------------------------------------------- #
def sipm_meshes(arm):
    """The 20 bars (16 instrumented) and the container they sit in."""
    hv = PG.HW_V['sipm'] * CM
    hu_bar = PG.bw * CM / 2
    w_f, w_b = PG.w_sipm_sc_f * CM, PG.w_sipm_sc_b * CM
    bars_live, bars_dead = [], []
    for i, u_i in PG._sipm_bar_offsets():
        b = lbox(u_i * CM, 0, (w_f + w_b) / 2, hu_bar, hv, (w_b - w_f) / 2)
        (bars_live if i in PG.SIPM_READOUT else bars_dead).append(b)
    cw_f, cw_b = PG.w_sipm_f * CM, PG.w_sipm_b * CM
    return dict(
        bars=bars_live[0].merge(bars_live[1:]),
        bars_dead=bars_dead[0].merge(bars_dead[1:]) if bars_dead else None,
        container=M.frame_ring((0, 0, (cw_f + cw_b) / 2),
                               2 * PG.sipm_wall_hu * CM + 24, 2 * hv + 24,
                               2 * PG.sipm_wall_hu * CM, 2 * hv,
                               cw_b - cw_f, normal='z'),
    )


# --------------------------------------------------------------------------- #
# Plastic scintillators
# --------------------------------------------------------------------------- #
def plastic_meshes(arm):
    """Two wrapped bars per arm, centred on the chamber (pinwheel included)."""
    i = arm['id']
    u0 = pin_u(arm)
    w_f, w_b = PG.w_bsc_f[i] * CM, PG.w_bsc_b[i] * CM
    hu, hv = PG.bsc_u * CM / 2, PG.bsc_v * CM / 2
    hw = PG.bsc_th * CM / 2
    du = PG.bsc_u_offset * CM
    wrap = (PG.bscTape_hw - PG.bsc_th / 2) * CM
    bars, tape = [], []
    for s in (-1, +1):
        bars.append(lbox(u0 + s * du, 0, (w_f + w_b) / 2, hu, hv, hw))
        tape.append(M.frame_ring((u0 + s * du, 0, (w_f + w_b) / 2),
                                 2 * (hu + wrap), 2 * (hv + wrap),
                                 2 * hu, 2 * hv, 2 * hw, normal='z'))
    return dict(bars=bars[0].merge(bars[1:]), tape=tape[0].merge(tape[1:]))


# --------------------------------------------------------------------------- #
# Liquid-scintillator calorimeter
# --------------------------------------------------------------------------- #
def _funnel_hexahedron(a0, a1, hu0, hw0, hu1, hw1, w_mid):
    """The loft from the slab end (a0) to the neck (a1), in local (u, a, w)."""
    corners = []
    for a, hu, hw in ((a0, hu0, hw0), (a1, hu1, hw1)):
        for su, sw in ((-1, -1), (+1, -1), (+1, +1), (-1, +1)):
            corners.append((su * hu, a, w_mid + sw * hw))
    return pv.UnstructuredGrid(np.array([8, 0, 1, 2, 3, 4, 5, 6, 7]),
                               np.array([pv.CellType.HEXAHEDRON]),
                               np.array(corners, float))


def ls_meshes(arm):
    """Vessel, liquid, neck and PMT for one arm.

    Built with the vessel's long axis along local +v, then rotated by the
    surveyed ``ls_rot_deg`` about the depth axis -- 0 leaves the neck up
    (arms B, C), -90 lays it over with the neck toward +u (arms A, D).
    """
    i = arm['id']
    w_mid = PG.w_LS_mid[i] * CM
    hu_o, hv_o, ht_o = PG.lsUo * CM, PG.lsVo * CM, PG.lsTo * CM
    wall = PG.lsWall * CM
    hcap = PG.hCap * CM

    # slab: the (u, w) pillow outline extruded along the long axis
    def pillow(hu, ht, cap, half_len, shrink=0.0):
        th = np.linspace(0, np.pi, 48)
        u_arc = (hu - shrink) * np.cos(th)
        front = np.column_stack([u_arc, w_mid - ht - cap * np.sin(th)])
        back = np.column_stack([-u_arc, w_mid + ht + cap * np.sin(th)])
        poly = np.vstack([front, back])
        return M.polygon_prism(poly, -half_len, 2 * half_len,
                               normal_axis='y')

    slab = pillow(hu_o, ht_o, hcap, hv_o)
    liquid = pillow(hu_o - wall, ht_o - wall, hcap, hv_o - wall / 2)

    a_neck0 = hv_o + PG.lsFunL * CM
    a_neck1 = a_neck0 + PG.lsNkL * CM
    funnel = _funnel_hexahedron(hv_o, a_neck0, hu_o, ht_o,
                                PG.lsNkR * CM, PG.lsNkR * CM, w_mid)
    neck = pv.Cylinder(center=(0, (a_neck0 + a_neck1) / 2, w_mid),
                       direction=(0, 1, 0), radius=PG.lsNkR * CM,
                       height=a_neck1 - a_neck0, resolution=48)
    pmt_a0 = PG.pmtFaceV * CM
    pmt = pv.Cylinder(center=(0, pmt_a0 + PG.CFG['ls_pmt_len_cm'] * CM / 2,
                              w_mid),
                      direction=(0, 1, 0), radius=PG.CFG['ls_pmt_r_cm'] * CM,
                      height=PG.CFG['ls_pmt_len_cm'] * CM, resolution=40)

    parts = dict(shell=slab.merge([funnel.extract_surface(), neck]),
                 liquid=liquid, pmt=pmt)

    # surveyed placement: rotate about the depth axis, then offset in (u, v)
    R = np.eye(4)
    th = np.radians(PG.LS_ROT[i])
    c, s = np.cos(th), np.sin(th)
    R[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    R[0, 3] = PG.LS_OFF_U[i] * CM
    R[1, 3] = PG.LS_OFF_V[i] * CM
    return {k: v.transform(R, inplace=False) for k, v in parts.items()}


# --------------------------------------------------------------------------- #
# He-3 capsule and the beam
# --------------------------------------------------------------------------- #
def polycone(z_cm, r_cm, resolution=96):
    """A G4Polycone as a surface of revolution about the beam axis.

    The profile is quoted along the capsule's own axis, which the sim mounts
    nose-first onto the beam: local +z becomes world +y.
    """
    pts = np.column_stack([np.asarray(r_cm) * CM, np.zeros(len(z_cm)),
                           np.asarray(z_cm) * CM])
    mesh = pv.lines_from_points(pts).extrude_rotate(resolution=resolution,
                                                    capping=True)
    mesh.rotate_x(-90.0, inplace=True)
    return mesh


def capsule_meshes():
    return dict(
        cfrp=polycone(PG.Z_AL, PG.RO_CFRP),
        al=polycone(PG.Z_AL, PG.RO_AL),
        gas=polycone(PG.Z_GAS, PG.RO_GAS),
    )


# The beam itself is just a grey column at the radius the primaries occupy --
# no arrowhead on it.  An arrowhead big enough to read at slide size is bigger
# than the 23 mm target it is pointing at, which makes the target look like a
# bead on a dart.  The direction is carried instead by a slim dart lying ON the
# axis, INSIDE the column: its length is set by the frame (so it reads the same
# on a 23 mm close-up and a 1.2 m hero) but its girth by the beam, so it can
# never grow out through the column it belongs to.
BEAM_Y = (-1500.0, 620.0)   # bottom well past any frame, so it runs off it
CAPSULE_TOP_Y = 52.0           # just past the vessel neck (Z_AL max = 5.1 cm)
# Where the column stops once the detector is there.  It ends at the vessel's
# NOSE, not past its neck: the column is wider than the capsule, so drawn up
# alongside it, it wraps the thing the frame is about in a translucent grey
# sleeve and the capsule reads as a haze instead of as a machined object.  The
# beam arrives from below, meets the target, and that is where the picture of
# it ends.
CAPSULE_NOSE_Y = float(PG.Z_AL.min()) * CM        # -35 mm
BEAM_STUB_Y = CAPSULE_NOSE_Y - 80.0     # a beam arriving, not a stalk it sits on


def beam_mesh(radius_mm, y0=BEAM_Y[0], y1=BEAM_Y[1]):
    # a high facet count on purpose: at the silhouette the facets go edge-on,
    # and a coarse cylinder stipples its own outline there
    return pv.Cylinder(center=(0, (y0 + y1) / 2, 0), direction=(0, 1, 0),
                       radius=radius_mm, height=y1 - y0, resolution=256)


def beam_arrow(length, radius_mm, y_mid):
    """(shaft, head) for the direction dart inside the beam column.

    The head is sized from the BEAM and only allowed to grow so far, because on
    the hero frames a head in proportion to the dart's own length would be a
    wedge wider than the target; on the close-ups it stays comfortably inside
    the column.  The head is 1.2x as long as it is wide, which is what makes it
    read as a dart rather than as a drawing pin at close-up sizes.
    """
    hr = float(np.clip(0.11 * length, 0.40 * radius_mm, 1.10 * radius_mm))
    hl = 2.4 * hr
    r = 0.34 * hr
    y0, y1 = y_mid - length / 2, y_mid + length / 2
    shaft = pv.Cylinder(center=(0, (y0 + y1 - hl) / 2, 0), direction=(0, 1, 0),
                        radius=r, height=y1 - hl - y0, resolution=32)
    head = pv.Cone(center=(0, y1 - hl / 2, 0), direction=(0, 1, 0),
                   height=hl, radius=hr, resolution=48)
    return shaft, head


# --------------------------------------------------------------------------- #
# The event
# --------------------------------------------------------------------------- #
def load_event(path=EVENT_JSON):
    if not os.path.isfile(path):
        raise SystemExit(
            f'{path} not found -- build it first with\n'
            f'  tools/extract_ntof_event.py --pairs <pairs_traj.csv> '
            f'--neutrons <neutrons_traj.csv>')
    with open(path) as f:
        return json.load(f)


# Nothing is drawn beyond this radius: bremsstrahlung gammas leave the setup
# and fly to the world boundary, and a 3 m streak across the slide says nothing
# about a 1.2 m detector.
CLIP_R = 620.0


def clip_to_scene(points, r=CLIP_R):
    """Truncate a track at the first point that leaves the drawn scene."""
    p = np.asarray(points, float)
    out = np.linalg.norm(p, axis=1) > r
    if not out.any():
        return p
    return p[:max(2, int(np.argmax(out)) + 1)]


def tube(points, radius, n_max=1200, clip=True):
    """A polyline as a tube, with degenerate steps dropped.

    Geant4 writes a step every time a process wins, so a low-energy electron
    can produce hundreds of sub-micron steps in one place; those collapse the
    tube filter's normals and show up as black speckle.
    """
    p = clip_to_scene(points) if clip else np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-3]
    p = p[keep]
    if len(p) < 2:
        return None
    if len(p) > n_max:                       # keep the ends, thin the middle
        idx = np.unique(np.r_[0, np.linspace(0, len(p) - 1, n_max).astype(int),
                              len(p) - 1])
        p = p[idx]
    return pv.lines_from_points(p).tube(radius=radius, n_sides=12)


def track_color(particle):
    return {'e-': COL['electron'], 'e+': COL['positron'],
            'gamma': COL['gamma'], 'neutron': COL['neutron']}.get(
                particle, COL['product'])


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
GHOST_COL = '#7b8695'      # one neutral colour for every ghosted outline
GHOST_FACE = 0.05          # how much of a near arm's face survives
GHOST_EDGE = 0.38          # ... and how strongly its outline is drawn

# "Bare" mode: frames, boards, wrappings, vessel shells and PMTs drop to a
# whisper and only the ACTIVE volumes keep their colour.  Straight down the
# beam the four arms are seen through each other, and the aluminium that is
# perfectly legible from three-quarters becomes an opaque lid over everything
# it is supposed to hold; with it out of the way, what is left on the picture
# is exactly the material that measures something.
BARE = False
BARE_ALPHA = 0.10


def set_bare(on):
    global BARE
    BARE = bool(on)


def _add(p, mesh, arm_id, mat_name, color, opacity=1.0, active=False):
    """Draw one part of one arm, or nothing if that arm is being ghosted.

    ``active`` marks the sensitive volumes, the only ones that keep their
    colour in ``BARE`` mode.

    The camera looks through two of the four arms, and they cannot simply be
    drawn faint: four half-transparent layers still hide what is behind them,
    and they hide it in a different colour at every overlap.  Those arms are
    therefore not drawn part by part at all -- ``ghost`` replaces each of their
    layers with one outline (see below), which says "this is here, you are
    looking through it" in a single line weight and a single colour.
    """
    if arm_id in NEAR_ARMS:
        return
    if BARE and not active:
        p.add_mesh(mesh, color=GHOST_COL, opacity=BARE_ALPHA, pbr=False,
                   ambient=0.6, diffuse=0.4, specular=0.0)
        return
    p.add_mesh(mesh, **S.mat(mat_name, color, opacity=opacity))


def ghost(p, mesh, T):
    """One near-arm layer, as a face-on whisper and its silhouette."""
    m = place(mesh, T)
    p.add_mesh(m, color=GHOST_COL, opacity=GHOST_FACE, pbr=False, ambient=0.6,
               diffuse=0.4, specular=0.0)
    edges = m.extract_feature_edges(feature_angle=25, boundary_edges=True,
                                    non_manifold_edges=False,
                                    manifold_edges=False)
    if edges.n_points:
        p.add_mesh(edges, color=GHOST_COL, opacity=GHOST_EDGE, lighting=False,
                   line_width=1.5 * getattr(p, '_mpgd_scale', 1.0))


def add_capsule(p, theme='light', cut_normal=None):
    """The vessel, cut open along the beam axis.

    Drawn whole, the capsule is a bright blob: the Al wall is 0.6 mm on a
    10 mm bore, so at any opacity that lets the gas through, the vessel itself
    disappears.  Cutting the wrap and the vessel on a plane through the axis --
    the half facing the camera removed -- gives back both: solid metal around a
    lit gas core, which is what the object actually is.

    ``cut_normal=None`` leaves it whole, which is what the overhead camera asks
    for: looking down the beam there is nothing to see *into*, and the cut --
    made on a plane that contains the view direction -- simply takes away the
    half of the vessel nearest the bottom of the frame, so the capsule reads as
    broken rather than as sectioned.
    """
    m = capsule_meshes()
    if cut_normal is not None:
        n = np.asarray(cut_normal, float)
        n /= np.linalg.norm(n)
        for k in ('cfrp', 'al', 'gas'):
            m[k] = m[k].clip(normal=tuple(n), origin=(0, 0, 0), invert=True)
    if BARE:
        # The one exception to the bare rule.  Everything else that drops to a
        # whisper up here is a box the eye can still infer from its neighbours;
        # the capsule is 23 mm on a 1.2 m frame and whispered it is a smudge at
        # the exact point the whole picture converges on.  Solid, it is a small
        # dark disc where the two legs meet -- which is what it is.
        for k, mat, col in (('cfrp', 'plastic', COL['cfrp']),
                            ('al', 'alu', COL['al']),
                            ('gas', 'glow', COL['he3'])):
            p.add_mesh(m[k], **S.mat(mat, col, opacity=1.0,
                                     smooth_shading=True))
        return
    p.add_mesh(m['cfrp'], **S.mat('plastic', COL['cfrp'], opacity=0.92,
                                  smooth_shading=True))
    p.add_mesh(m['al'], **S.mat('alu', COL['al'], opacity=0.95,
                                smooth_shading=True))
    # Kept well short of opaque on purpose.  The vessel is cut open towards the
    # camera, so a track crosses the cut plane somewhere inside the gas -- at a
    # high opacity it is drawn in front of the gas on one side of that plane and
    # vanishes behind it on the other, which reads as a rendering fault rather
    # than as depth.  At this opacity the two regimes differ by a tint.
    p.add_mesh(m['gas'], **S.mat('glow', COL['he3'], opacity=0.42,
                                 smooth_shading=True))


def add_beam(p, radius_mm, arrow=None, y_top=BEAM_Y[1], y_bot=BEAM_Y[0]):
    """The beam column, optionally cut down to a stub around the target.

    Once the chambers are in, a column drawn the full height of the frame runs
    through the middle of the apparatus -- above the target it reads as a pole
    holding the thing up, below it as a rod hanging off it.  The build frames
    therefore keep only the piece at the vessel: the beam has been established
    by then, and what the picture is about from there on is what leaves.
    """
    # Front faces only.  A long thin translucent cylinder shows its own back
    # wall through its front one, and where the two meet at the silhouette the
    # depth-peeled result stipples along the edge; culling the back leaves one
    # clean soft band.  Opacity is tuned against a TRANSPARENT frame -- on a
    # slide, a 7 % envelope disappears entirely.
    p.add_mesh(beam_mesh(radius_mm, y0=y_bot, y1=y_top), culling='back',
               smooth_shading=True,
               **S.mat('envelope', COL['beam'], opacity=0.26))
    if arrow is None:
        return
    for m in beam_arrow(radius_mm=radius_mm, **arrow):
        p.add_mesh(m, **S.mat('alu_matte', COL['beam_dart'], opacity=1.0,
                              smooth_shading=True))


def add_chambers(p):
    for arm in ARMS:
        T = arm_transform(arm)
        m = chamber_meshes(arm)
        a = arm['id']
        if a in NEAR_ARMS:
            u0 = pin_u(arm)
            ghost(p, lbox(u0, 0, PG.w_PCB_b * CM / 2, FRAME_HU,
                          FRAME_HV, PG.w_PCB_b * CM / 2), T)
            continue
        _add(p, place(m['frame'], T), a, 'alu', COL['frame'])
        _add(p, place(m['pcb'], T), a, 'pcb', COL['pcb'])
        # the drift volume is the chamber's message, so it is the one thing
        # given real presence: from the target you look straight into 30 mm of
        # lit gas, with the readout board behind it
        _add(p, place(m['gas'], T), a, 'gas', COL['gas'], opacity=0.46,
             active=True)
        # the entrance window is 40 um of mylar: a tint on the front face, not
        # a wall -- drawn any heavier it hides the capsule behind it
        _add(p, place(m['window'], T), a, 'plastic', COL['mylar'],
             opacity=0.10)


def add_sipm(p):
    for arm in ARMS:
        T = arm_transform(arm)
        m = sipm_meshes(arm)
        a = arm['id']
        if a in NEAR_ARMS:
            cw_f, cw_b = PG.w_sipm_f * CM, PG.w_sipm_b * CM
            ghost(p, lbox(0, 0, (cw_f + cw_b) / 2, PG.sipm_wall_hu * CM,
                          PG.HW_V['sipm'] * CM, (cw_b - cw_f) / 2), T)
            continue
        _add(p, place(m['bars'], T), a, 'plastic', COL['sipm'], active=True)
        if m['bars_dead'] is not None:
            _add(p, place(m['bars_dead'], T), a, 'plastic', COL['sipm_dead'],
                 opacity=0.55, active=True)
        _add(p, place(m['container'], T), a, 'alu_matte', COL['frame_dark'],
             opacity=0.55)


def add_plastics(p):
    for arm in ARMS:
        T = arm_transform(arm)
        m = plastic_meshes(arm)
        a = arm['id']
        if a in NEAR_ARMS:
            i = arm['id']
            w_f, w_b = PG.w_bsc_f[i] * CM, PG.w_bsc_b[i] * CM
            ghost(p, lbox(pin_u(arm), 0, (w_f + w_b) / 2,
                          PG.bsc_u_offset * CM + PG.bsc_u * CM / 2,
                          PG.bsc_v * CM / 2, (w_b - w_f) / 2), T)
            continue
        _add(p, place(m['tape'], T), a, 'plastic', '#20242b', opacity=0.85)
        _add(p, place(m['bars'], T), a, 'plastic', COL['plastic'],
             opacity=0.92, active=True)


def add_ls(p):
    for arm in ARMS:
        T = arm_transform(arm)
        m = ls_meshes(arm)
        a = arm['id']
        if a in NEAR_ARMS:
            ghost(p, m['shell'], T)
            continue
        # CFRP shell solid, liquid translucent inside it: the vessel reads as
        # a filled tank rather than as a coloured slab
        _add(p, place(m['shell'], T), a, 'plastic', COL['ls_shell'],
             opacity=0.30)
        _add(p, place(m['liquid'], T), a, 'scint', COL['ls_liquid'],
             opacity=0.72, active=True)
        _add(p, place(m['pmt'], T), a, 'alu_matte', COL['pmt'], opacity=0.9)


def add_neutron(p, ev, radius=3.2, scale=1.0, y_from=BEAM_Y[0]):
    """The neutron arriving: one straight line up the beam to the vertex.

    Deliberately NOT the transported Geant4 history.  That history is a real
    one and it is still in the JSON, but it belonged to a different event from
    the pair, so to draw it at all it had to be translated onto the pair's
    vertex -- and what it then showed was its own in-gas scattering, which is
    a detail about that neutron rather than about this figure.  A neutron
    arriving up the beam line is the true and legible statement, so that is
    what is drawn: straight, vertical, from below the frame to the vertex.
    """
    v = np.asarray(ev['pair']['vertex'], float)
    t = tube(np.array([[v[0], y_from, v[2]], v]), radius * scale, clip=False)
    if t is not None:
        p.add_mesh(t, **S.mat('glow', COL['neutron'], opacity=0.95))
    p.add_mesh(pv.Sphere(radius=6.5 * scale, center=v),
               **S.mat('glow', '#ffffff', opacity=0.95))


# Only the charged pair is drawn.  The event's secondaries are three
# bremsstrahlung gammas and one knock-on electron; the gammas are neutral, so
# they leave the setup in a straight line from wherever they were radiated and
# read as stray rays crossing the picture with no visible cause.  The slide is
# about where the e+ and the e- go, so that is all it shows.
DRAWN_PARTICLES = ('e+', 'e-')
DETECTOR_LAYERS = ('mm', 'sipm', 'plastic', 'ls')
MIN_TRACK_MM = 25.0        # below this a truncated track is a speck, not a track


def truncate_at_next_layer(track, layers):
    """The track's points, cut where it first enters a layer not built yet.

    So a leg grows with the apparatus: on the frame that has only the chambers
    it reaches the chambers and stops, and it only runs on to the trigger wall
    on the frame that puts the trigger wall there.  Drawn full length from the
    start, the tracks say "we measure all of this" three slides before any of
    it exists, and the layer being added stops being the thing that changes.

    ``track['layers']`` is one entry per STEP and ``points`` one per step
    boundary, so entering step i means keeping i + 1 points.
    """
    pts = track['points']
    for i, lay in enumerate(track.get('layers', ())):
        if lay in DETECTOR_LAYERS and lay not in layers:
            # A secondary BORN inside a layer that is not there yet has nothing
            # to draw, and one that only got a few millimetres before reaching
            # it has nothing worth drawing: either way what lands on the frame
            # is a speck floating in empty space, which is exactly what the
            # truncation is there to avoid.
            kept = pts[:i + 1]
            if len(kept) < 2:
                return None
            a, b = np.asarray(kept[0]), np.asarray(kept[-1])
            return kept if np.linalg.norm(b - a) >= MIN_TRACK_MM else None
    return pts


def add_pair(p, ev, layers, radius=4.4, scale=1.0):
    """The two legs, their charged secondaries, and the deposits so far.

    ``layers`` is the set of detector layers currently built; a deposit in a
    layer that is not on the slide yet would be a glow floating in mid-air,
    and a leg is cut where it reaches the first layer that is not there yet.
    """
    for leg in ev['pair']['legs']:
        pts = truncate_at_next_layer(leg, layers)
        t = None if pts is None else tube(pts, radius * scale)
        if t is not None:
            p.add_mesh(t, **S.mat('glow', track_color(leg['particle']),
                                  opacity=1.0))
    for sec in ev['pair']['shower']:
        if sec['particle'] not in DRAWN_PARTICLES:
            continue
        pts = truncate_at_next_layer(sec, layers)
        t = None if pts is None else tube(pts, radius * scale * 0.30)
        if t is not None:
            p.add_mesh(t, **S.mat('glow', track_color(sec['particle']),
                                  opacity=0.55))
    for d in ev['pair']['deposits']:
        if d['layer'] not in layers:
            continue
        col = DEPOSIT_COL[d['layer']]
        r = (5.0 + 15.0 * (min(d['edep_MeV'], 2.0) / 2.0) ** (1 / 3)) * scale
        if d['layer'] == 'mm' and 'a' in d and 'b' in d:
            # The drift gap's deposit is not a point.  It is 30 mm of ionisation
            # left along the track -- the thing the micro-TPC actually images --
            # so it is drawn as the trail: a bright tube over the crossing, at a
            # radius that reads at slide size rather than at 23 keV's worth.
            t = tube([d['a'], d['b']], max(r, 3.2 * scale * 2.2), clip=False)
            if t is not None:
                p.add_mesh(t, **S.mat('glow', col, opacity=1.0))
                continue
        p.add_mesh(pv.Sphere(radius=r, center=d['p'], theta_resolution=24,
                             phi_resolution=24),
                   **S.mat('glow', col, opacity=0.92))


# --------------------------------------------------------------------------- #
# Where a leader line should point when a layer is being named on the frame.
# One point per (layer, arm), taken from the same geometry the meshes are built
# from, so a label can never end up pointing at where a layer used to be.
#
# A layer is one object per arm except the plastics, which are two separate
# bars: ``LAYER_PARTS`` says how many, so a label can put a line on each of
# them rather than one line on the pair's centre of mass, which lands in the
# gap between them.
LAYER_PARTS = dict(mm=1, sipm=1, plastic=2, ls=1)


def layer_anchor(layer, arm_id, v_frac=0.0, part=0):
    """The centre of one arm's ``layer`` in world coordinates [mm].

    ``v_frac`` slides the point along the beam direction as a fraction of that
    layer's own half-height, which is how a leader is kept off the tracks
    crossing the middle of it.  ``part`` picks between the sub-objects a layer
    is made of (see ``LAYER_PARTS``).
    """
    arm = ARMS[arm_id]
    i = arm['id']
    u0 = pin_u(arm)
    if layer == 'mm':
        w_f = (PG.tMylar + PG.tAlWin + PG.tKapCath + PG.tCuCath) * CM
        loc = np.array([u0, 0.0, w_f + PG.tDrift * CM / 2])
        hv = MM_HV
    elif layer == 'sipm':
        loc = np.array([0.0, 0.0,
                        (PG.w_sipm_sc_f + PG.w_sipm_sc_b) * CM / 2])
        hv = PG.HW_V['sipm'] * CM
    elif layer == 'plastic':
        s = (+1.0, -1.0)[part]                 # the two bars, as plastic_meshes
        loc = np.array([u0 + s * PG.bsc_u_offset * CM, 0.0,
                        (PG.w_bsc_f[i] + PG.w_bsc_b[i]) * CM / 2])
        hv = PG.bsc_v * CM / 2
    elif layer == 'ls':
        loc = np.array([PG.LS_OFF_U[i] * CM, PG.LS_OFF_V[i] * CM,
                        PG.w_LS_mid[i] * CM])
        # the vessel is laid over on two of the four arms, so its "long axis"
        # is not the beam axis there and the offset has to follow the rotation
        th = np.radians(PG.LS_ROT[i])
        loc = loc + PG.lsVo * CM * v_frac * np.array([-np.sin(th), np.cos(th),
                                                      0.0])
        hv = 0.0
    else:
        raise KeyError(layer)
    loc[1] += v_frac * hv
    return (arm_transform(arm) @ np.r_[loc, 1.0])[:3]


PARTS = ('capsule', 'beam', 'neutron', 'mm', 'sipm', 'plastic', 'ls', 'pair')
LAYER_OF_PART = dict(mm='mm', sipm='sipm', plastic='plastic', ls='ls')


def build(p, show=PARTS, event=None, theme='light', track_scale=1.0,
          cut_normal=None, beam_arrow_kw=None, bare=False):
    """Add everything in ``show`` to an existing plotter.

    ``track_scale`` sizes the tracks against the frame: a 4 mm tube reads as a
    line across the whole setup and as a pipe across the 23 mm capsule, so the
    close-up act asks for thinner ones.  ``beam_arrow_kw`` is the little
    direction arrow beside the beam, sized against the frame by the caller.
    """
    show = set(show)
    ev = event if event is not None else load_event()
    set_bare(bare)
    built = {LAYER_OF_PART[k] for k in show if k in LAYER_OF_PART}

    if 'beam' in show:
        add_beam(p, ev.get('beam', {}).get('r90_mm', 20.0),
                 arrow=beam_arrow_kw,
                 y_top=CAPSULE_NOSE_Y if built else BEAM_Y[1],
                 y_bot=BEAM_STUB_Y if built else BEAM_Y[0])
    if 'capsule' in show:
        add_capsule(p, theme, cut_normal=cut_normal)
    if 'mm' in show:
        add_chambers(p)
    if 'sipm' in show:
        add_sipm(p)
    if 'plastic' in show:
        add_plastics(p)
    if 'ls' in show:
        add_ls(p)
    if 'neutron' in show:
        add_neutron(p, ev, scale=track_scale,
                    y_from=BEAM_STUB_Y if built else BEAM_Y[0])
    if 'pair' in show:
        add_pair(p, ev, built, scale=track_scale)
    return p


def scene_scale():
    """Characteristic radius of the full setup [mm], for the light rig."""
    return float(PG.dist + PG.stack_depth) * CM
